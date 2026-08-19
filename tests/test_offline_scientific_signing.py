from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.experiments.e0_authority_artifact import (
    E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    E0FinalCellCompletion,
    E0FinalCompletionReceipt,
    e0_final_completion_receipt_to_dict,
)
from lightcone_spec.experiments.e4_stage_authority import (
    E4_PROFILER_COMPLETION_PROTOCOL_SHA256,
    E4ProfilerCompletionReceipt,
    E4ProfilerTerminalCompletion,
    e4_profiler_completion_receipt_to_dict,
)
from lightcone_spec.experiments.formal_materialization_shards import (
    publish_formal_materialization_shard_index,
)
from lightcone_spec.experiments.formal_protocol import ProtocolLock, content_sha256
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_to_dict,
    stage_coverage_receipt_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.experiments.stage_decisions import E1SurvivorSelectionReceipt
from lightcone_spec.experiments.stage_materialization import (
    E1Geometry,
    GpuHourEstimate,
    MaterializedCell,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime import offline_signer
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.attester_bundle import (
    AttestationNoncePolicy,
    TrustedAttesterPolicyBundle,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseEd25519Root,
    SourceReleaseRootBinding,
    deployment_policy_subject_sha256,
)
from lightcone_spec.runtime.relocatable_evidence import (
    materialize_relocatable_evidence_bundle,
)
from lightcone_spec.runtime.scientific_signing import (
    SCIENTIFIC_ARTIFACT_TYPES,
    finalize_scientific_candidate,
    rebuild_scientific_signed_proof_wrapper,
    scientific_payload_sha256,
    sign_scientific_candidate,
)
from lightcone_spec.runtime.scientific_source_validation import (
    publish_scientific_source_validation_artifact,
)

NOW_NS = 2_000_000_000


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _public(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _root_binding(private_key: Ed25519PrivateKey) -> SourceReleaseRootBinding:
    public = _public(private_key)
    spki = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    root = SourceReleaseEd25519Root(
        schema_version=1,
        kind="lightcone_source_release_ed25519_root",
        root_id="lightcone-release-root-2026q3",
        key_id="lightcone-release-root-key-2026q3",
        algorithm="Ed25519",
        public_key_base64=base64.b64encode(public).decode("ascii"),
        public_key_sha256=hashlib.sha256(public).hexdigest(),
        spki_sha256=hashlib.sha256(spki).hexdigest(),
    )
    return SourceReleaseRootBinding(
        root=root,
        path="/validation/scientific-root.json",
        sidecar_path="/validation/scientific-root.json.sha256",
        semantic_sha256=root.sha256,
        file_sha256=_sha("root-file"),
        sidecar_file_sha256=_sha("root-sidecar"),
    )


def _bundle(private_key: Ed25519PrivateKey) -> TrustedAttesterPolicyBundle:
    public = _public(private_key)
    fingerprint = hashlib.sha256(public).hexdigest()
    return TrustedAttesterPolicyBundle(
        schema_version=1,
        kind="lightcone_trusted_attester_policy_bundle",
        bundle_id="scientific-signing-bundle-v1",
        valid_from_ns=1,
        expires_ns=20_000_000_000,
        nonce_policy=AttestationNoncePolicy(
            schema_version=1,
            kind="lightcone_attestation_nonce_policy",
            nonce_bytes=32,
            minimum_lifetime_ns=1_000_000_000,
            maximum_lifetime_ns=600_000_000_000,
            maximum_clock_skew_ns=30_000_000_000,
            replay_policy="external_single_use_store",
            subject_binding_required=True,
        ),
        hardware_envelope_sha256_allowlist=(_sha("hardware"),),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id="scientific-signing-policy-v1",
            trusted_attesters=(("formal-signer", "formal-signer-key", fingerprint),),
            public_keys=((fingerprint, base64.b64encode(public).decode("ascii")),),
        ),
    )


def _install_root(
    monkeypatch: pytest.MonkeyPatch, binding: SourceReleaseRootBinding
) -> None:
    from lightcone_spec.runtime import release_trust_root

    monkeypatch.setattr(
        offline_signer, "load_source_release_ed25519_root", lambda: binding
    )
    monkeypatch.setattr(
        release_trust_root, "load_source_release_ed25519_root", lambda: binding
    )


def _authorization(
    root_private: Ed25519PrivateKey,
    signer_private: Ed25519PrivateKey,
) -> DeploymentPolicyAuthorization:
    bundle = _bundle(signer_private)
    inventory = _sha("inventory")
    subject = deployment_policy_subject_sha256(
        root_manifest_sha256=_root_binding(root_private).semantic_sha256,
        inventory_sha256=inventory,
        bundle_sha256=bundle.sha256,
    )
    challenge = AttestationChallenge.issue(
        challenge_id="scientific-deployment",
        subject_sha256=subject,
        lifetime_s=300,
        now_ns=NOW_NS,
    )
    return offline_signer.sign_deployment_policy_authorization(
        bundle=bundle,
        inventory_sha256=inventory,
        challenge=challenge,
        private_key=root_private,
        now_ns=NOW_NS,
    )


def _protocol_lock() -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-scientific-signing-test",
        code_git_head=_sha("head")[:40],
        code_git_tree=_sha("tree")[:40],
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=_sha("registry"),
        english_protocol_sha256=_sha("protocol-en"),
        chinese_protocol_sha256=_sha("protocol-zh"),
        tts_calibration_authority_sha256=_sha("tts-authority"),
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-anchors"),
        e2_recipe_grid_authority_sha256=_sha("e2-grid"),
        formal_runtime_authority_manifest_sha256=_sha("runtime-authority"),
        offline_release_trust_root_sha256=_sha("offline-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-models"),
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


def _stage_payloads(lock: ProtocolLock) -> tuple[object, object, object]:
    cell = MaterializedCell(
        stage="E3a",
        method_role="Target-only",
        model="Qwen/Qwen3.6-35B-A3B",
        backend="dflash",
        task="capacity_probe",
        publication_policy="fixed_barrier",
        recipe_sha256=None,
        dimensions=(),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("preflight-upstream"),),
        source_decision_sha256=_sha("e3a-source"),
        materialization_rule="signed E3a capacity slice",
        expected_cell_count=1,
        cells=(cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E3a",
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=(
            StageCellDisposition(
                stage="E3a",
                cell_id=cell.cell_id,
                status="COMPLETE",
                reason_code="terminal_complete",
                terminal_receipt_sha256=_sha("terminal"),
            ),
        ),
    )
    e1_receipt = E1SurvivorSelectionReceipt(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        e1_materialization_receipt_sha256=_sha("e1-materialization"),
        e1_coverage_receipt_sha256=_sha("e1-coverage"),
        e3a_selection_receipt_sha256=_sha("e3a-selection"),
        staged_pareto_evidence_manifest_sha256=_sha("pareto-manifest"),
        staged_pareto_artifact_sha256=_sha("pareto-artifact"),
        inventory_sha256=_sha("e1-inventory"),
        model="Qwen/Qwen3.6-35B-A3B",
        frozen_tts_recipe_sha256=_sha("tts-recipe"),
        surviving_geometries=(E1Geometry("last1", "full", None, None),),
    )
    return materialization, coverage, e1_receipt


def _json_tree(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _portable_proof_bundle(
    tmp_path: Path,
    proof: Path,
    *,
    label: str,
) -> Path:
    tmp_path.chmod(0o700)
    local_root = (tmp_path / label).resolve()
    local_root.mkdir(mode=0o700)
    return Path(
        materialize_relocatable_evidence_bundle(
            remote_root=tmp_path.resolve(),
            entry_paths=(proof,),
            local_root=local_root,
        ).absolute_path
    )


def _sign_and_finalize(
    artifact_type: str,
    payload: object,
    *,
    authorization: DeploymentPolicyAuthorization,
    signer_private: Ed25519PrivateKey,
    ledger: Path,
    source_validation_artifact_path: str | None = None,
) -> object:
    payload_json = _json_tree(payload)
    payload_digest = scientific_payload_sha256(
        artifact_type=artifact_type,
        payload_json=payload_json,
        source_validation_artifact_path=source_validation_artifact_path,
        now_ns=NOW_NS,
    )
    challenge = AttestationChallenge.issue(
        challenge_id=f"typed-{artifact_type}",
        subject_sha256=payload_digest,
        lifetime_s=300,
        now_ns=NOW_NS,
    )
    candidate = sign_scientific_candidate(
        artifact_type=artifact_type,
        payload_json=payload_json,
        deployment_policy_authorization=authorization,
        challenge=challenge,
        private_key=signer_private,
        attester_id="formal-signer",
        key_id="formal-signer-key",
        now_ns=NOW_NS,
        source_validation_artifact_path=source_validation_artifact_path,
    )
    finalized = finalize_scientific_candidate(
        artifact_type=artifact_type,
        candidate_json=candidate,
        deployment_policy_authorization=authorization,
        challenge_ledger=ledger,
        now_ns=NOW_NS,
    )
    if finalized.get("kind") != "lightcone_scientific_signed_proof_wrapper":
        return finalized
    compact = (
        ledger.parent / f"compact-{artifact_type}-{finalized['wrapper_sha256']}.json"
    ).resolve()
    publish_canonical_json_no_replace(compact, finalized)
    return rebuild_scientific_signed_proof_wrapper(compact, now_ns=NOW_NS)


def test_preflight_materialization_signing_replays_exact_bootstrap_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_control_attestation import HARDWARE_SHA256, INVENTORY_SHA256
    from test_formal_cli_replay import _control as registry_control
    from test_formal_cli_replay import (
        _deployment_authorization as registry_authorization,
    )
    from test_formal_cli_replay import _lock as registry_lock

    from lightcone_spec.experiments.formal_initial_stage_proof import (
        bind_formal_initial_stage_materialization_proof_artifact,
        publish_formal_initial_stage_materialization_proof_artifact,
    )
    from lightcone_spec.experiments.formal_registry import (
        extend_formal_registry_verification_receipt,
        reserve_formal_registry_verification_receipt,
    )
    from lightcone_spec.experiments.formal_registry_layers import (
        bind_formal_registry_layer_artifact,
        load_formal_registry_verification_receipt_path,
        publish_formal_registry_layer_artifact,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry
    from lightcone_spec.experiments.stage_materialization import (
        materialize_preflight,
    )
    from lightcone_spec.runtime import scientific_source_validation as source_module
    from lightcone_spec.runtime.control_attestation import ChallengeReplayStore

    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    _install_root(monkeypatch, root_binding)
    scientific_only_bundle = _bundle(signer_private)
    signer_fingerprint = hashlib.sha256(_public(signer_private)).hexdigest()
    attester_bundle = replace(
        scientific_only_bundle,
        hardware_envelope_sha256_allowlist=(HARDWARE_SHA256,),
        trusted_attester_policy=TrustedAttesterPolicy(
            policy_id=scientific_only_bundle.trusted_attester_policy.policy_id,
            trusted_attesters=tuple(
                sorted(
                    (
                        *scientific_only_bundle.trusted_attester_policy.trusted_attesters,
                        (
                            "validation-signer",
                            "validation-signer-key",
                            signer_fingerprint,
                        ),
                    )
                )
            ),
            public_keys=scientific_only_bundle.trusted_attester_policy.public_keys,
        ),
    )
    lock = registry_lock(root_binding.semantic_sha256)
    root_authorization = registry_authorization(
        root_private,
        binding=root_binding,
        bundle=attester_bundle,
        nonce_byte=b"j",
    )
    protocol_proof = (tmp_path / "unit-protocol-proof.json").resolve()
    publish_canonical_json_no_replace(
        protocol_proof,
        {"schema_version": 1, "kind": "unit_protocol_lock_reducer_proof"},
    )
    protocol_bundle = _portable_proof_bundle(
        tmp_path,
        protocol_proof,
        label="unit-protocol-proof-bundle",
    )
    monkeypatch.setattr(
        source_module,
        "_protocol_lock_expected_payload",
        lambda _path, *, now_ns: lock,
    )
    protocol_source = (tmp_path / "unit-protocol-source.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="protocol-lock",
        proof_bundle_path=protocol_bundle,
        now_ns=NOW_NS,
        output_path=protocol_source,
    )
    protocol_ledger = (tmp_path / "unit-protocol-ledger").resolve()
    protocol_ledger.mkdir(mode=0o700)
    signed_lock = _sign_and_finalize(
        "protocol-lock",
        protocol_lock_to_dict(lock),
        authorization=root_authorization,
        signer_private=signer_private,
        ledger=protocol_ledger,
        source_validation_artifact_path=str(protocol_source),
    )
    protocol_wrappers = tuple(tmp_path.glob("compact-protocol-lock-*.json"))
    assert len(protocol_wrappers) == 1
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed_lock.sha256, "dispatch"),),
        }
    )
    control = registry_control(
        signer_private,
        binding=root_binding,
        bundle=attester_bundle,
        authorization=root_authorization,
        artifact_type="dispatch",
        artifact_sha256=signed_lock.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=lineage,
        nonce_byte=b"k",
    )
    replay = (tmp_path / "registry-replay").resolve()
    replay.mkdir(mode=0o700)
    registry_store = ChallengeReplayStore(str(replay))
    root_receipt = reserve_formal_registry_verification_receipt(
        signed_lock,
        control_attestation=control,
        expected_inventory_sha256=INVENTORY_SHA256,
        replay_store=registry_store,
        now_ns=NOW_NS,
    )
    root_layer = (tmp_path / "root-layer.json").resolve()
    publish_formal_registry_layer_artifact(
        bind_formal_registry_layer_artifact(
            root_receipt,
            prior_layer_path=None,
            signed_protocol_lock_path=protocol_wrappers[0],
            signed_materialization_paths=(),
            signed_coverage_paths=(),
            formal_stage_prefix_paths=(),
        ),
        root_layer,
    )
    proof = bind_formal_initial_stage_materialization_proof_artifact(
        phase="preflight",
        registry_layer_path=root_layer,
        now_ns=NOW_NS,
    )
    proof_path = (tmp_path / "preflight-materialization-proof.json").resolve()
    publish_formal_initial_stage_materialization_proof_artifact(proof, proof_path)
    proof_bundle = _portable_proof_bundle(
        tmp_path,
        proof_path,
        label="preflight-materialization-proof-bundle",
    )
    source = (tmp_path / "preflight-materialization-source.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="stage-materialization",
        proof_bundle_path=proof_bundle,
        now_ns=NOW_NS,
        output_path=source,
    )
    materialization = materialize_preflight(
        protocol_lock_sha256=lock.sha256,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    signing_ledger = (tmp_path / "preflight-materialization-ledger").resolve()
    signing_ledger.mkdir(mode=0o700)
    signed = _sign_and_finalize(
        "stage-materialization",
        stage_materialization_receipt_to_dict(materialization),
        authorization=root_authorization,
        signer_private=signer_private,
        ledger=signing_ledger,
        source_validation_artifact_path=str(source),
    )
    assert signed.payload == materialization
    wrappers = tuple(tmp_path.glob("compact-stage-materialization-*.json"))
    assert len(wrappers) == 1
    extension_lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed.sha256, "dispatch"),),
            "prior_registry_verification_receipt_sha256": root_receipt.sha256,
        }
    )
    extension_authorization = registry_authorization(
        root_private,
        binding=root_binding,
        bundle=attester_bundle,
        nonce_byte=b"m",
    )
    extension_control = registry_control(
        signer_private,
        binding=root_binding,
        bundle=attester_bundle,
        authorization=extension_authorization,
        artifact_type="dispatch",
        artifact_sha256=signed.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=extension_lineage,
        nonce_byte=b"l",
    )
    appended = extend_formal_registry_verification_receipt(
        root_receipt,
        appended_signed_materializations=(signed,),
        control_attestations=(extension_control,),
        replay_store=registry_store,
        now_ns=NOW_NS,
    )
    appended_layer = (tmp_path / "preflight-materialization-layer.json").resolve()
    publish_formal_registry_layer_artifact(
        bind_formal_registry_layer_artifact(
            appended,
            prior_layer_path=root_layer,
            signed_materialization_paths=(wrappers[0],),
            signed_coverage_paths=(),
            formal_stage_prefix_paths=(),
        ),
        appended_layer,
    )
    assert (
        load_formal_registry_verification_receipt_path(
            appended_layer,
            now_ns=NOW_NS,
        )
        == appended
    )

    altered = replace(materialization, materialization_rule="caller-altered")
    altered_ledger = (tmp_path / "altered-materialization-ledger").resolve()
    altered_ledger.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="differs from its proof reducer"):
        _sign_and_finalize(
            "stage-materialization",
            stage_materialization_receipt_to_dict(altered),
            authorization=root_authorization,
            signer_private=signer_private,
            ledger=altered_ledger,
            source_validation_artifact_path=str(source),
        )


def test_typed_ceremony_finalizes_protocol_and_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    _install_root(monkeypatch, _root_binding(root_private))
    authorization = _authorization(root_private, signer_private)
    ledger = (tmp_path / "ledger").resolve()
    ledger.mkdir(mode=0o700)
    lock = _protocol_lock()
    materialization, coverage, e1_receipt = _stage_payloads(lock)

    with pytest.raises(ValueError, match="source validation"):
        _sign_and_finalize(
            "protocol-lock",
            protocol_lock_to_dict(lock),
            authorization=authorization,
            signer_private=signer_private,
            ledger=ledger,
        )
    materialization_index = publish_formal_materialization_shard_index(
        materialization,
        cell_shard_output_paths=(
            (tmp_path / "materialization-cell-shard.json").resolve(),
        ),
        index_output_path=(tmp_path / "materialization-index.json").resolve(),
    )
    materialization_bundle = _portable_proof_bundle(
        tmp_path,
        Path(materialization_index.absolute_path),
        label="offline-materialization-proof-bundle",
    )
    materialization_source = (tmp_path / "materialization-source.json").resolve()
    with pytest.raises(ValueError, match="typed predecessor reducer proof"):
        publish_scientific_source_validation_artifact(
            artifact_type="stage-materialization",
            proof_bundle_path=materialization_bundle,
            now_ns=NOW_NS,
            output_path=materialization_source,
        )
    assert not materialization_source.exists()
    with pytest.raises(ValueError, match="source validation"):
        _sign_and_finalize(
            "stage-coverage",
            stage_coverage_receipt_to_dict(coverage),
            authorization=authorization,
            signer_private=signer_private,
            ledger=ledger,
        )
    with pytest.raises(ValueError, match="source validation"):
        _sign_and_finalize(
            "e1-survivor-selection",
            asdict(e1_receipt),
            authorization=authorization,
            signer_private=signer_private,
            ledger=ledger,
        )

    assert not tuple(ledger.iterdir())


def test_closed_types_wrong_key_type_tamper_replay_and_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(SCIENTIFIC_ARTIFACT_TYPES) == {
        "protocol-lock",
        "tts-calibration-seal",
        "stage-materialization",
        "stage-coverage",
        "stage-gpu-hour-envelope",
        "e3a-staged-selection",
        "e1-survivor-selection",
        "e2-staged-selection",
        "e4-stage-selection",
        "e4-profiler-completion",
        "e0-formal-breadth-fdr",
        "e0-final-completion",
    }
    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    _install_root(monkeypatch, _root_binding(root_private))
    authorization = _authorization(root_private, signer_private)
    lock = _protocol_lock()
    materialization, _coverage, _selection = _stage_payloads(lock)
    from lightcone_spec.runtime import scientific_source_validation

    proof = (tmp_path / "materialization-proof.json").resolve()
    publish_canonical_json_no_replace(
        proof,
        {"schema_version": 1, "kind": "test_materialization_proof"},
    )
    proof_bundle = _portable_proof_bundle(
        tmp_path,
        proof,
        label="materialization-negative-proof-bundle",
    )
    monkeypatch.setattr(
        scientific_source_validation,
        "_expected_payload",
        lambda artifact_type, proof_path, now_ns: materialization,
    )
    source = (tmp_path / "materialization-source.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="stage-materialization",
        proof_bundle_path=proof_bundle,
        now_ns=NOW_NS,
        output_path=source,
    )
    challenge = AttestationChallenge.issue(
        challenge_id="typed-negative",
        subject_sha256=materialization.sha256,
        lifetime_s=300,
        now_ns=NOW_NS,
    )
    call = {
        "artifact_type": "stage-materialization",
        "payload_json": stage_materialization_receipt_to_dict(materialization),
        "deployment_policy_authorization": authorization,
        "challenge": challenge,
        "attester_id": "formal-signer",
        "key_id": "formal-signer-key",
        "now_ns": NOW_NS,
        "source_validation_artifact_path": str(source),
    }
    with pytest.raises(ValueError, match="identity is not authorized"):
        sign_scientific_candidate(**call, private_key=Ed25519PrivateKey.generate())
    with pytest.raises(ValueError, match="not allowlisted"):
        sign_scientific_candidate(
            **{**call, "artifact_type": "generic-json"},
            private_key=signer_private,
        )
    wrong_subject = AttestationChallenge.issue(
        challenge_id="typed-wrong-subject",
        subject_sha256=_sha("foreign-payload"),
        lifetime_s=300,
        now_ns=NOW_NS,
    )
    with pytest.raises(ValueError, match="not payload-bound"):
        sign_scientific_candidate(
            **{**call, "challenge": wrong_subject},
            private_key=signer_private,
        )
    candidate = sign_scientific_candidate(**call, private_key=signer_private)
    foreign_authorization = _authorization(root_private, Ed25519PrivateKey.generate())
    with pytest.raises(ValueError, match="root or policy binding differs"):
        finalize_scientific_candidate(
            artifact_type="stage-materialization",
            candidate_json=candidate,
            deployment_policy_authorization=foreign_authorization,
            challenge_ledger=(tmp_path / "missing-ledger").resolve(),
            now_ns=NOW_NS,
        )
    tampered = {**candidate, "artifact_type": "stage-coverage"}
    with pytest.raises(ValueError, match="digest differs"):
        finalize_scientific_candidate(
            artifact_type="stage-coverage",
            candidate_json=tampered,
            deployment_policy_authorization=authorization,
            challenge_ledger=(tmp_path / "missing-ledger").resolve(),
            now_ns=NOW_NS,
        )
    signature_tamper = copy.deepcopy(candidate)
    attestation = signature_tamper["attestation"]
    assert type(attestation) is dict
    signature = bytearray(base64.b64decode(attestation["signature_base64"]))
    signature[0] ^= 1
    attestation["signature_base64"] = base64.b64encode(signature).decode("ascii")
    signature_tamper["candidate_sha256"] = content_sha256(
        {
            key: value
            for key, value in signature_tamper.items()
            if key != "candidate_sha256"
        }
    )
    ledger = (tmp_path / "ledger").resolve()
    ledger.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="wrapper digest differs"):
        finalize_scientific_candidate(
            artifact_type="stage-materialization",
            candidate_json=signature_tamper,
            deployment_policy_authorization=authorization,
            challenge_ledger=ledger,
            now_ns=NOW_NS,
        )
    finalized = finalize_scientific_candidate(
        artifact_type="stage-materialization",
        candidate_json=candidate,
        deployment_policy_authorization=authorization,
        challenge_ledger=ledger,
        now_ns=NOW_NS,
    )
    with pytest.raises(RuntimeError, match="target already exists"):
        finalize_scientific_candidate(
            artifact_type="stage-materialization",
            candidate_json=candidate,
            deployment_policy_authorization=authorization,
            challenge_ledger=ledger,
            now_ns=NOW_NS,
        )
    duplicate_challenge = AttestationChallenge.issue(
        challenge_id="typed-negative",
        subject_sha256=materialization.sha256,
        lifetime_s=300,
        now_ns=NOW_NS,
    )
    duplicate_candidate = sign_scientific_candidate(
        **{**call, "challenge": duplicate_challenge},
        private_key=signer_private,
    )
    with pytest.raises(RuntimeError, match="target already exists"):
        finalize_scientific_candidate(
            artifact_type="stage-materialization",
            candidate_json=duplicate_candidate,
            deployment_policy_authorization=authorization,
            challenge_ledger=ledger,
            now_ns=NOW_NS,
        )
    output = (tmp_path / "final.json").resolve()
    publish_canonical_json_no_replace(output, finalized)
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_canonical_json_no_replace(output, finalized)


def test_proof_derived_stage_payload_requires_exact_reducer_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightcone_spec.runtime import scientific_source_validation

    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    _install_root(monkeypatch, _root_binding(root_private))
    authorization = _authorization(root_private, signer_private)
    _materialization, coverage, expected = _stage_payloads(_protocol_lock())
    proof = (tmp_path / "formal-prefix.json").resolve()
    publish_canonical_json_no_replace(
        proof,
        {"schema_version": 1, "kind": "test_formal_stage_prefix"},
    )
    monkeypatch.setattr(
        scientific_source_validation,
        "_formal_prefix_expected_payload",
        lambda artifact_type, proof_path, now_ns: expected,
    )
    monkeypatch.setattr(
        scientific_source_validation,
        "_coverage_expected_payload",
        lambda proof_path, now_ns: coverage,
    )
    source = (tmp_path / "source-validation.json").resolve()
    bundle = _portable_proof_bundle(
        tmp_path,
        proof,
        label="offline-proof-bundle",
    )
    publish_scientific_source_validation_artifact(
        artifact_type="e1-survivor-selection",
        proof_bundle_path=bundle,
        now_ns=NOW_NS,
        output_path=source,
    )
    ledger = (tmp_path / "source-ledger").resolve()
    ledger.mkdir(mode=0o700)
    signed = _sign_and_finalize(
        "e1-survivor-selection",
        asdict(expected),
        authorization=authorization,
        signer_private=signer_private,
        ledger=ledger,
        source_validation_artifact_path=str(source),
    )
    assert signed.payload == expected

    coverage_source = (tmp_path / "coverage-source-validation.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="stage-coverage",
        proof_bundle_path=bundle,
        now_ns=NOW_NS,
        output_path=coverage_source,
    )
    signed_coverage = _sign_and_finalize(
        "stage-coverage",
        stage_coverage_receipt_to_dict(coverage),
        authorization=authorization,
        signer_private=signer_private,
        ledger=ledger,
        source_validation_artifact_path=str(coverage_source),
    )
    assert signed_coverage.payload == coverage

    changed = replace(expected, inventory_sha256=_sha("altered-winner-lineage"))
    with pytest.raises(ValueError, match="differs from its proof reducer"):
        scientific_payload_sha256(
            artifact_type="e1-survivor-selection",
            payload_json=_json_tree(asdict(changed)),
            source_validation_artifact_path=str(source),
            now_ns=NOW_NS,
        )


def test_completion_receipts_have_closed_offline_signing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightcone_spec.runtime import scientific_source_validation

    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    _install_root(monkeypatch, _root_binding(root_private))
    authorization = _authorization(root_private, signer_private)
    ledger = (tmp_path / "completion-ledger").resolve()
    ledger.mkdir(mode=0o700)
    e4 = E4ProfilerCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("e4-lock"),
        registry_sha256=_sha("e4-registry"),
        registry_verification_receipt_sha256=_sha("e4-registry-receipt"),
        materialization_receipt_sha256=_sha("e4-materialization"),
        coverage_receipt_sha256=_sha("e4-coverage"),
        signed_local_selection_sha256=_sha("e4-local-selection"),
        terminals=tuple(
            sorted(
                (
                    E4ProfilerTerminalCompletion(
                        materialized_cell_id=_sha(f"e4-cell-{index}"),
                        terminal_receipt_sha256=_sha(f"e4-terminal-{index}"),
                    )
                    for index in range(3)
                ),
                key=lambda row: row.materialized_cell_id,
            )
        ),
        protocol_sha256=E4_PROFILER_COMPLETION_PROTOCOL_SHA256,
    )
    proof = (tmp_path / "completion-proof.json").resolve()
    publish_canonical_json_no_replace(
        proof,
        {"schema_version": 1, "kind": "test_completion_proof"},
    )
    expected_by_type: dict[str, object] = {"e4-profiler-completion": e4}
    monkeypatch.setattr(
        scientific_source_validation,
        "_expected_payload",
        lambda artifact_type, proof_path, now_ns: expected_by_type[artifact_type],
    )
    e4_source = (tmp_path / "e4-completion-source.json").resolve()
    bundle = _portable_proof_bundle(
        tmp_path,
        proof,
        label="offline-completion-proof-bundle",
    )
    publish_scientific_source_validation_artifact(
        artifact_type="e4-profiler-completion",
        proof_bundle_path=bundle,
        now_ns=NOW_NS,
        output_path=e4_source,
    )
    e4_signed = _sign_and_finalize(
        "e4-profiler-completion",
        e4_profiler_completion_receipt_to_dict(e4),
        authorization=authorization,
        signer_private=signer_private,
        ledger=ledger,
        source_validation_artifact_path=str(e4_source),
    )
    assert e4_signed.payload == e4

    e0_cells = tuple(
        sorted(
            (
                E0FinalCellCompletion(
                    materialized_cell_id=_sha(f"e0-cell-{index}"),
                    execution_binding_sha256=_sha(f"e0-binding-{index}"),
                    terminal_receipt_sha256=_sha(f"e0-terminal-{index}"),
                    native_result_proof_semantic_sha256=_sha(f"e0-native-{index}"),
                    stage_itl_proof_semantic_sha256=_sha(f"e0-itl-{index}"),
                )
                for index in range(192)
            ),
            key=lambda row: row.materialized_cell_id,
        )
    )
    e0 = E0FinalCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=_sha("e0-lock"),
        registry_sha256=_sha("e0-registry"),
        prior_registry_verification_receipt_sha256=_sha("e0-prior-registry"),
        current_registry_verification_receipt_sha256=_sha("e0-current-registry"),
        materialization_receipt_sha256=_sha("e0-materialization"),
        coverage_receipt_sha256=_sha("e0-coverage"),
        stage_source_binding_sha256=_sha("e0-stage-source"),
        evidence_manifest_sha256=_sha("e0-evidence"),
        inventory_sha256=_sha("e0-inventory"),
        rebuild_artifact_sha256=_sha("e0-rebuild"),
        selected_final_prefix=tuple(range(4, 16)),
        valid_compatibility_count=1,
        cells=e0_cells,
        protocol_sha256=E0_FINAL_COMPLETION_PROTOCOL_SHA256,
    )
    expected_by_type["e0-final-completion"] = e0
    e0_source = (tmp_path / "e0-completion-source.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="e0-final-completion",
        proof_bundle_path=bundle,
        now_ns=NOW_NS,
        output_path=e0_source,
    )
    e0_signed = _sign_and_finalize(
        "e0-final-completion",
        e0_final_completion_receipt_to_dict(e0),
        authorization=authorization,
        signer_private=signer_private,
        ledger=ledger,
        source_validation_artifact_path=str(e0_source),
    )
    assert e0_signed.payload == e0


def test_scientific_cli_uses_only_inherited_key_fd_and_finalizes_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lightcone_spec.runtime import scientific_source_validation

    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    _install_root(monkeypatch, _root_binding(root_private))
    authorization = _authorization(root_private, signer_private)
    materialization, _coverage, _selection = _stage_payloads(_protocol_lock())
    proof = (tmp_path / "cli-materialization-proof.json").resolve()
    publish_canonical_json_no_replace(
        proof,
        {"schema_version": 1, "kind": "test_cli_materialization_proof"},
    )
    proof_bundle = _portable_proof_bundle(
        tmp_path,
        proof,
        label="cli-materialization-proof-bundle",
    )
    monkeypatch.setattr(
        scientific_source_validation,
        "_expected_payload",
        lambda artifact_type, proof_path, now_ns: materialization,
    )
    source = (tmp_path / "cli-materialization-source.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="stage-materialization",
        proof_bundle_path=proof_bundle,
        now_ns=NOW_NS,
        output_path=source,
    )
    payload_path = (tmp_path / "materialization-payload.json").resolve()
    authorization_path = (tmp_path / "authorization.json").resolve()
    publish_canonical_json_no_replace(
        payload_path,
        stage_materialization_receipt_to_dict(materialization),
    )
    publish_canonical_json_no_replace(authorization_path, authorization.to_dict())
    key_path = (tmp_path / "signer-private-key").resolve()
    key_path.write_bytes(
        signer_private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    descriptor = os.open(key_path, os.O_RDONLY)
    candidate_path = (tmp_path / "candidate.json").resolve()
    argv = [
        "sign-scientific",
        "--artifact-type",
        "stage-materialization",
        "--payload",
        str(payload_path),
        "--source-validation-artifact",
        str(source),
        "--deployment-authorization",
        str(authorization_path),
        "--attester-id",
        "formal-signer",
        "--key-id",
        "formal-signer-key",
        "--challenge-id",
        "typed-cli",
        "--now-ns",
        str(NOW_NS),
        "--key-fd",
        str(descriptor),
        "--output",
        str(candidate_path),
    ]
    assert str(key_path) not in argv
    try:
        assert offline_signer.main(argv) == 0
    finally:
        os.close(descriptor)
    ledger = (tmp_path / "ledger").resolve()
    ledger.mkdir(mode=0o700)
    final_path = (tmp_path / "signed-lock.json").resolve()
    assert (
        offline_signer.main(
            [
                "finalize-scientific",
                "--artifact-type",
                "stage-materialization",
                "--candidate",
                str(candidate_path),
                "--deployment-authorization",
                str(authorization_path),
                "--challenge-ledger",
                str(ledger),
                "--now-ns",
                str(NOW_NS),
                "--output",
                str(final_path),
            ]
        )
        == 0
    )
    signed = rebuild_scientific_signed_proof_wrapper(
        final_path,
        now_ns=NOW_NS,
    )
    assert signed.payload == materialization
