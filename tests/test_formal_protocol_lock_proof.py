from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_content_authorization import _workload_file
from test_content_authorization_operator import (
    _dataset_source,
    _publish,
    _sign_source,
)
from test_formal_method_authority import _plan_source
from test_offline_scientific_signing import (
    NOW_NS,
    _authorization,
    _install_root,
    _root_binding,
    _sign_and_finalize,
)

import lightcone_spec.cli.main as cli_module
import lightcone_spec.experiments.workload_authority as workload_module
import lightcone_spec.runtime.content_authorization as content_module
from lightcone_spec.experiments import formal_protocol_lock_proof as lock_proof_module
from lightcone_spec.experiments.formal_method_authority import (
    TTS_DRAFTER_NATIVE_LOSS_SOURCE,
    TTS_TUNING_WINDOW_SOURCE_KIND,
    build_source_chronobelief_authority_artifact,
    build_source_tts_calibration_authority_artifact,
    publish_chronobelief_authority_artifact,
    publish_tts_calibration_authority_artifact,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_protocol_lock_proof import (
    bind_formal_protocol_lock_source_proof_artifact,
    publish_formal_protocol_lock_git_snapshot,
    publish_formal_protocol_lock_source_proof_artifact,
    revalidate_formal_protocol_lock_source_proof_artifact,
)
from lightcone_spec.experiments.formal_registry import (
    protocol_lock_to_dict,
    publish_formal_runtime_authority_manifest,
    reserve_formal_registry_verification_receipt,
)
from lightcone_spec.experiments.formal_registry_layers import (
    bind_formal_registry_layer_artifact,
    load_formal_registry_verification_receipt_path,
    publish_formal_registry_layer_artifact,
)
from lightcone_spec.experiments.formal_runtime_manifest import (
    FORMAL_RUNTIME_SOURCE_LAYOUT,
    build_source_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.formal_stage_execution import (
    build_source_e1_recipe_anchor_authority_artifact,
    publish_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.runtime import offline_signer
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    attestation_message,
)
from lightcone_spec.runtime.content_authorization import (
    AuthorizedPreparedModel,
    ContentVerificationReceipt,
    DatasetContentReleaseAuthorization,
    PreparedModelContentReleaseAuthorization,
    PreparedModelContentReleaseAuthorizationSource,
    PreparedModelStageMembership,
    ReleaseWorkloadSourceAuthorization,
    ReleaseWorkloadSourceAuthorizationSource,
    TtsCalibrationTuningWindow,
    TtsCalibrationTuningWindowEntry,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayStore,
    ControlArtifactAttestation,
    ControlArtifactSubject,
)
from lightcone_spec.runtime.preflight_runner import (
    derive_burstgpt_shape_authority_from_content_receipt,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.release_trust_root import (
    DeploymentPolicyAuthorization,
    SourceReleaseRootBinding,
)
from lightcone_spec.runtime.relocatable_evidence import (
    materialize_relocatable_evidence_bundle,
)
from lightcone_spec.runtime.scientific_source_validation import (
    publish_scientific_source_validation_artifact,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_body(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def _registry_control(
    private: Ed25519PrivateKey,
    *,
    binding: SourceReleaseRootBinding,
    authorization: DeploymentPolicyAuthorization,
    artifact_sha256: str,
    protocol_sha256: str,
    lineage_sha256: str,
) -> ControlArtifactAttestation:
    subject = ControlArtifactSubject(
        schema_version=1,
        kind="lightcone_control_artifact_subject",
        artifact_type="dispatch",
        artifact_sha256=artifact_sha256,
        protocol_sha256=protocol_sha256,
        registry_sha256=build_industrial_registry().sha256,
        lineage_sha256=lineage_sha256,
    )
    challenge = AttestationChallenge.issue(
        challenge_id="formal-protocol-lock-registry-control",
        subject_sha256=subject.sha256,
        lifetime_s=300,
        now_ns=NOW_NS,
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="formal-signer",
        key_id="formal-signer-key",
        environment="release",
        public_key_base64=base64.b64encode(public).decode("ascii"),
        challenge_sha256=challenge.sha256,
        payload_sha256=artifact_sha256,
        signature_base64=base64.b64encode(
            private.sign(
                attestation_message(
                    challenge,
                    payload_sha256=artifact_sha256,
                )
            )
        ).decode("ascii"),
    )
    return ControlArtifactAttestation(
        schema_version=1,
        kind="lightcone_control_artifact_attestation",
        subject=subject,
        hardware_envelope_sha256=_sha("hardware"),
        trust_anchor_sha256=binding.sha256,
        trust_bundle_sha256=authorization.bundle.sha256,
        trusted_attester_policy_sha256=(
            authorization.bundle.trusted_attester_policy.sha256
        ),
        deployment_policy_authorization=authorization,
        challenge=challenge,
        attestation=attestation,
    )


def _copy_runtime_snapshot_source(repository: Path, *, root_binding: object) -> None:
    project = Path(__file__).resolve().parents[1]
    relative_paths = {
        "pyproject.toml",
        "src/lightcone_spec/experiments/formal_protocol.py",
        "src/lightcone_spec/runtime/release_trust_root.py",
    }
    for layout in FORMAL_RUNTIME_SOURCE_LAYOUT:
        relative_paths.update(layout.runner_sources)
        relative_paths.update(layout.test_nodes)
    for relative in sorted(relative_paths):
        source = project / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    trust = repository / "src/lightcone_spec/runtime/trust"
    trust.mkdir(parents=True, exist_ok=True)
    manifest = trust / "release_ed25519_root_v1.json"
    manifest.write_bytes(_canonical_body(root_binding.root.to_dict()))
    raw_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (trust / "release_ed25519_root_v1.json.sha256").write_text(
        f"{raw_sha256}\n",
        encoding="ascii",
    )
    (repository / "patch.json").write_text('{"patches":[]}\n', encoding="utf-8")
    (repository / "protocol-en.md").write_text("formal protocol\n", encoding="utf-8")
    (repository / "protocol-zh.md").write_text("正式协议\n", encoding="utf-8")


def _publish_method_authorities(
    remote: Path,
) -> tuple[Path, Path, Path, Path, tuple[Path, ...]]:
    tts_plan_path, tts_plan = _plan_source(remote / "tts-plan", scope="all")
    e1_plan_path, e1_plan = _plan_source(remote / "e1-plan", scope="last1")
    tts_pdf = (remote / "tts-v2.pdf").resolve()
    tts_source = (remote / "tts-v2.tex").resolve()
    chrono_pdf = (remote / "chronobelief.pdf").resolve()
    chrono_tex = (remote / "chronobelief.tex").resolve()
    tts_pdf.write_bytes(b"%PDF-1.7\nTTS v2 fixture\n")
    tts_source.write_text("TTS v2 source\n", encoding="utf-8")
    chrono_pdf.write_bytes(b"%PDF-1.7\nChronoBelief fixture\n")
    chrono_tex.write_text("equations 5.5--5.8\n", encoding="utf-8")
    tuning = (remote / "tts-window.json").resolve()
    loss = (remote / "tts-loss.json").resolve()
    workload_fixture = remote / "tts-window-workload"
    workload_fixture.mkdir(mode=0o700)
    workload_path, workload_source = _workload_file(
        workload_fixture,
        "livecodebench_v6_hard",
        count=6,
    )
    workload_envelope = json.loads(workload_path.read_text(encoding="utf-8"))
    samples = workload_module._select_all_rows(
        workload_module.FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"],
        workload_envelope["rows"],
    )
    entries = tuple(
        TtsCalibrationTuningWindowEntry(
            workload_id="livecodebench_v6_hard",
            source_sample_id=sample.sample_id,
            source_descriptor_sha256=workload_source.sha256,
            prompt_sha256=content_sha256(sample.prompt),
        )
        for sample in samples
    )
    window = TtsCalibrationTuningWindow(
        schema_version=2,
        kind=TTS_TUNING_WINDOW_SOURCE_KIND,
        tuning_entries=tuple(sorted(entries[:2], key=lambda row: row.entry_id)),
        excluded_pilot_entries=tuple(sorted(entries[2:], key=lambda row: row.entry_id)),
    )
    publish_canonical_json_no_replace(
        tuning,
        window.to_dict(),
    )
    publish_canonical_json_no_replace(loss, TTS_DRAFTER_NATIVE_LOSS_SOURCE)
    tts_artifact = build_source_tts_calibration_authority_artifact(
        paper_pdf_path=tts_pdf,
        paper_source_path=tts_source,
        tuning_window_path=tuning,
        trainable_plan_authority_path=tts_plan_path,
        drafter_native_loss_path=loss,
    )
    chrono_artifact = build_source_chronobelief_authority_artifact(
        paper_pdf_path=chrono_pdf,
        tex_source_path=chrono_tex,
    )
    e1_artifact = build_source_e1_recipe_anchor_authority_artifact(e1_plan_path)
    tts_output = (remote / "tts-authority.json").resolve()
    chrono_output = (remote / "chronobelief-authority.json").resolve()
    e1_output = (remote / "e1-anchor-authority.json").resolve()
    publish_tts_calibration_authority_artifact(tts_artifact, tts_output)
    publish_chronobelief_authority_artifact(chrono_artifact, chrono_output)
    publish_e1_recipe_anchor_authority_artifact(e1_artifact, e1_output)
    snapshot_roots = tuple(
        Path(snapshot.root)
        for binding in (tts_plan, e1_plan)
        for snapshot in binding.prepared_model_content_authority.prepared_model_set.snapshots
    )
    return tts_output, chrono_output, e1_output, tuning, snapshot_roots


def _publish_content_master(
    remote: Path,
    *,
    root_private: Ed25519PrivateKey,
    root_binding: object,
    tuning_window: Path,
) -> tuple[Path, Path]:
    content = remote / "content"
    content.mkdir(mode=0o700)
    _lcb_path, lcb = _workload_file(
        content,
        "livecodebench_v6_hard",
        count=6,
    )
    _math_path, math = _workload_file(content, "math500_level5")
    workload_source = ReleaseWorkloadSourceAuthorizationSource(
        schema_version=1,
        kind="lightcone_release_workload_source_authorization_source",
        root_manifest_sha256=root_binding.semantic_sha256,
        workload_sources=(lcb, math),
    )
    target_manifest = {"kind": "snapshot", "member": "target"}
    drafter_manifest = {"kind": "snapshot", "member": "drafter"}
    target_path = _publish(content / "target-snapshot.json", target_manifest)
    drafter_path = _publish(content / "drafter-snapshot.json", drafter_manifest)
    target_body = _canonical_body(target_manifest)
    drafter_body = _canonical_body(drafter_manifest)
    models = (
        AuthorizedPreparedModel(
            member_id="shared:drafter",
            backend="DFLASH",
            role="drafter",
            model_id="z-lab/Qwen3-8B-DFlash-b16",
            revision="2" * 40,
            snapshot_manifest_raw_sha256=hashlib.sha256(drafter_body).hexdigest(),
            snapshot_manifest_semantic_sha256=content_module._canonical_sha256(
                drafter_manifest
            ),
        ),
        AuthorizedPreparedModel(
            member_id="shared:target",
            backend="DFLASH",
            role="target",
            model_id="Qwen/Qwen3-8B",
            revision="1" * 40,
            snapshot_manifest_raw_sha256=hashlib.sha256(target_body).hexdigest(),
            snapshot_manifest_semantic_sha256=content_module._canonical_sha256(
                target_manifest
            ),
        ),
        AuthorizedPreparedModel(
            member_id="shared:tokenizer",
            backend="DFLASH",
            role="tokenizer",
            model_id="Qwen/Qwen3-8B",
            revision="1" * 40,
            snapshot_manifest_raw_sha256=hashlib.sha256(target_body).hexdigest(),
            snapshot_manifest_semantic_sha256=content_module._canonical_sha256(
                target_manifest
            ),
        ),
    )
    member_ids = tuple(row.member_id for row in models)
    prepared_source = PreparedModelContentReleaseAuthorizationSource(
        schema_version=1,
        kind="lightcone_prepared_model_content_release_authorization_source",
        root_manifest_sha256=root_binding.semantic_sha256,
        model_lock_sha256=_sha("model-lock"),
        prepared_model_set_sha256=_sha("prepared-set"),
        content_manifest_raw_sha256=_sha("prepared-content-raw"),
        content_manifest_semantic_sha256=_sha("prepared-content-semantic"),
        content_manifest_size=123,
        models=models,
        stage_memberships=tuple(
            PreparedModelStageMembership(stage_id=stage, member_ids=member_ids)
            for stage in ("E0", "E3a", "TTS-Cal")
        ),
    )
    burst_source, burst_binding = _dataset_source(
        content,
        domain="burstgpt_six_source",
        count=6,
        root_sha256=root_binding.semantic_sha256,
    )
    e0_source, e0_binding = _dataset_source(
        content,
        domain="e0_task_native",
        count=1,
        root_sha256=root_binding.semantic_sha256,
    )
    rows = (
        ("workload", workload_source, "workload"),
        ("prepared_model", prepared_source, "prepared"),
        ("dataset", burst_source, "burst"),
        ("dataset", e0_source, "e0"),
    )
    key = content / "root.key"
    key.write_bytes(
        root_private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    key.chmod(0o600)
    authorization_paths: dict[str, Path] = {}
    with key.open("rb") as handle:
        for artifact_type, source, label in rows:
            source_path = _publish(content / f"{label}-source.json", source.to_dict())
            output = (content / f"{label}-authorization.json").resolve()
            _sign_source(
                source_path=source_path,
                output=output,
                artifact_type=artifact_type,
                challenge_id=f"protocol-lock-content-{label}",
                key_fd=handle.fileno(),
                now_ns=NOW_NS,
            )
            authorization_paths[label] = output
    workload_authorization = ReleaseWorkloadSourceAuthorization.from_dict(
        CanonicalJsonProofBinding.bind(authorization_paths["workload"]).reopen()
    )
    prepared_authorization = PreparedModelContentReleaseAuthorization.from_dict(
        CanonicalJsonProofBinding.bind(authorization_paths["prepared"]).reopen()
    )
    burst_authorization = DatasetContentReleaseAuthorization.from_dict(
        CanonicalJsonProofBinding.bind(authorization_paths["burst"]).reopen()
    )
    e0_authorization = DatasetContentReleaseAuthorization.from_dict(
        CanonicalJsonProofBinding.bind(authorization_paths["e0"]).reopen()
    )
    del workload_authorization, prepared_authorization
    burst_binding = replace(
        burst_binding,
        authorization_sha256=burst_authorization.sha256,
    )
    e0_binding = replace(e0_binding, authorization_sha256=e0_authorization.sha256)
    burst_binding_path = _publish(
        content / "burst-path-binding.json", burst_binding.to_dict()
    )
    e0_binding_path = _publish(content / "e0-path-binding.json", e0_binding.to_dict())
    replay = (content / "replay").resolve()
    replay.mkdir(mode=0o700)
    master = (remote / "content-master.json").resolve()
    argv = [
        "verify-content-authorizations",
        "--workload-authorization",
        str(authorization_paths["workload"]),
        "--prepared-model-authorization",
        str(authorization_paths["prepared"]),
        "--burstgpt-authorization",
        str(authorization_paths["burst"]),
        "--e0-dataset-authorization",
        str(authorization_paths["e0"]),
        "--replay-store",
        str(replay),
        "--now-ns",
        str(NOW_NS),
        "--output",
        str(master),
    ]
    for specification in (
        f"dataset:burstgpt_six_source:path_binding={burst_binding_path}",
        f"dataset:e0_task_native:path_binding={e0_binding_path}",
        f"snapshot:shared:drafter={drafter_path}",
        f"snapshot:shared:target={target_path}",
        f"tts_calibration_tuning_window={tuning_window}",
    ):
        argv.extend(("--content-artifact", specification))
    assert cli_module.main(argv) == 0
    receipt = ContentVerificationReceipt.from_dict(
        CanonicalJsonProofBinding.bind(master).reopen()
    )
    burst = derive_burstgpt_shape_authority_from_content_receipt(
        receipt,
        current_ns=NOW_NS,
    )
    burst_output = (remote / "burstgpt-authority.json").resolve()
    publish_canonical_json_no_replace(burst_output, burst.to_dict())
    return master, burst_output


def test_protocol_lock_source_proof_is_portable_and_replayed_before_signing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    remote = (tmp_path / "remote-a").resolve()
    repository = (tmp_path / "repository-a").resolve()
    offline = (tmp_path / "offline-b").resolve()
    remote.mkdir(mode=0o700)
    repository.mkdir(mode=0o700)
    offline.mkdir(mode=0o700)

    root_private = Ed25519PrivateKey.generate()
    signer_private = Ed25519PrivateKey.generate()
    fixture_root = _root_binding(root_private).root
    root_body = _canonical_body(fixture_root.to_dict())
    root_file_sha256 = hashlib.sha256(root_body).hexdigest()
    root_binding = SourceReleaseRootBinding(
        root=fixture_root,
        path="/validation/scientific-root.json",
        sidecar_path="/validation/scientific-root.json.sha256",
        semantic_sha256=fixture_root.sha256,
        file_sha256=root_file_sha256,
        sidecar_file_sha256=hashlib.sha256(
            f"{root_file_sha256}\n".encode("ascii")
        ).hexdigest(),
    )
    monkeypatch.setattr(
        content_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    monkeypatch.setattr(
        offline_signer,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    monkeypatch.setattr(
        lock_proof_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    _install_root(monkeypatch, root_binding)

    _copy_runtime_snapshot_source(repository, root_binding=root_binding)
    subprocess.run(("git", "init", str(repository)), check=True, capture_output=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Protocol Test"),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "protocol@example.invalid",
        ),
        check=True,
    )
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-m", "formal snapshot"),
        check=True,
        capture_output=True,
    )
    runtime = build_source_formal_runtime_authority_manifest(repository)
    runtime_path = (remote / "runtime-authority.json").resolve()
    publish_formal_runtime_authority_manifest(runtime_path, runtime)
    (
        tts_path,
        chrono_path,
        e1_path,
        tuning_path,
        snapshot_roots,
    ) = _publish_method_authorities(remote)
    content_path, burst_path = _publish_content_master(
        remote,
        root_private=root_private,
        root_binding=root_binding,
        tuning_window=tuning_path,
    )
    snapshot_path = (remote / "git-snapshot.json").resolve()
    publish_formal_protocol_lock_git_snapshot(
        project_root=repository,
        chunk_output_directory=remote,
        index_output_path=snapshot_path,
    )
    artifact = bind_formal_protocol_lock_source_proof_artifact(
        protocol_id="formal-protocol-portable-test",
        git_snapshot_path=snapshot_path,
        patch_manifest_relative_path="patch.json",
        english_protocol_relative_path="protocol-en.md",
        chinese_protocol_relative_path="protocol-zh.md",
        runtime_authority_path=runtime_path,
        tts_calibration_authority_path=tts_path,
        chronobelief_authority_path=chrono_path,
        e1_recipe_anchor_authority_path=e1_path,
        content_verification_receipt_path=content_path,
        burstgpt_shape_authority_path=burst_path,
        now_ns=NOW_NS,
    )
    proof_path = (remote / "protocol-lock-proof.json").resolve()
    publish_formal_protocol_lock_source_proof_artifact(artifact, proof_path)
    lock = revalidate_formal_protocol_lock_source_proof_artifact(
        proof_path,
        now_ns=NOW_NS,
    )
    hydrated = (tmp_path / "rehydrated-b").resolve()
    hydrated.mkdir(mode=0o700)
    directory_rebindings: dict[Path, Path] = {}
    for index, source in enumerate(snapshot_roots):
        destination = (
            hydrated / f"model-{index}" / "snapshots" / source.name
        ).resolve()
        destination.parent.mkdir(parents=True, mode=0o700)
        shutil.copytree(source, destination)
        directory_rebindings[source] = destination
    bundle = materialize_relocatable_evidence_bundle(
        remote_root=remote,
        entry_paths=(proof_path,),
        local_root=offline,
        directory_rebindings=directory_rebindings,
    )
    shutil.rmtree(remote)
    shutil.rmtree(repository)

    source_validation = (tmp_path / "protocol-lock-source-validation.json").resolve()
    publish_scientific_source_validation_artifact(
        artifact_type="protocol-lock",
        proof_bundle_path=bundle.absolute_path,
        proof_entry_remote_absolute_path=proof_path,
        now_ns=NOW_NS,
        output_path=source_validation,
    )
    authorization = _authorization(root_private, signer_private)
    ledger = (tmp_path / "scientific-ledger").resolve()
    ledger.mkdir(mode=0o700)
    signed = _sign_and_finalize(
        "protocol-lock",
        protocol_lock_to_dict(lock),
        authorization=authorization,
        signer_private=signer_private,
        ledger=ledger,
        source_validation_artifact_path=str(source_validation),
    )
    assert signed.payload == lock
    wrapper_path = next(tmp_path.glob("compact-protocol-lock-*.json"))
    lineage = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_registry_control_lineage",
            "protocol_lock_sha256": lock.sha256,
            "registry_sha256": build_industrial_registry().sha256,
            "signed_artifacts": ((signed.sha256, "dispatch"),),
        }
    )
    control = _registry_control(
        signer_private,
        binding=root_binding,
        authorization=authorization,
        artifact_sha256=signed.sha256,
        protocol_sha256=lock.sha256,
        lineage_sha256=lineage,
    )
    replay_root = (tmp_path / "registry-replay").resolve()
    replay_root.mkdir(mode=0o700)
    receipt = reserve_formal_registry_verification_receipt(
        signed,
        control_attestation=control,
        expected_inventory_sha256=_sha("inventory"),
        replay_store=ChallengeReplayStore(str(replay_root)),
        now_ns=NOW_NS,
    )
    layer = bind_formal_registry_layer_artifact(
        receipt,
        prior_layer_path=None,
        signed_protocol_lock_path=wrapper_path,
        signed_materialization_paths=(),
        signed_coverage_paths=(),
        formal_stage_prefix_paths=(),
    )
    layer_path = (tmp_path / "formal-registry-root-layer.json").resolve()
    publish_formal_registry_layer_artifact(layer, layer_path)
    assert (
        load_formal_registry_verification_receipt_path(
            layer_path,
            now_ns=NOW_NS + 20_000_000_000,
        )
        == receipt
    )
    assert lock.code_git_head and lock.formal_runtime_authority_manifest_sha256 == (
        runtime.sha256
    )

    altered_ledger = (tmp_path / "altered-ledger").resolve()
    altered_ledger.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="differs from its proof reducer"):
        _sign_and_finalize(
            "protocol-lock",
            protocol_lock_to_dict(replace(lock, protocol_id="altered")),
            authorization=authorization,
            signer_private=signer_private,
            ledger=altered_ledger,
            source_validation_artifact_path=str(source_validation),
        )
    assert content_sha256(lock) == lock.sha256
