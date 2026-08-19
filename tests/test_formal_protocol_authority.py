from __future__ import annotations

import base64
import hashlib
import math
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lightcone_spec.cli.main import _create_protocol_lock, _load_bound_json
from lightcone_spec.config.schema import OptimizerConfig
from lightcone_spec.experiments.formal_protocol import (
    BANNED_MODEL,
    E6_MODELS,
    FORMAL_METHOD_ROLES,
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
    FORMAL_STAGE_DAG,
    CandidateStateReplay,
    CandidateStateTerminalPair,
    ChronoBeliefAuthority,
    ChronoBeliefState,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
    ProtocolLock,
    SignedProtocolLock,
    TtsCalibrationAuthority,
    TtsCalibrationSeal,
    assert_tts_l0_candidate_state_equality,
    chronobelief_reference_transition,
    code_owned_qualification_source_identities,
    content_sha256,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.formal_protocol_lock_proof import (
    FormalProtocolLockGitSnapshotIndex,
    checkout_formal_protocol_lock_git_snapshot,
    publish_formal_protocol_lock_git_snapshot,
)
from lightcone_spec.experiments.formal_registry import (
    formal_runtime_authority_manifest_from_dict,
    formal_runtime_authority_manifest_to_dict,
    protocol_lock_from_dict,
    protocol_lock_to_dict,
    publish_formal_runtime_authority_manifest,
    revalidate_formal_runtime_authority_manifest,
)
from lightcone_spec.experiments.stage_materialization import (
    default_e2_recipe_grid_authority,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
    attestation_message,
)
from lightcone_spec.runtime.compile_cache import (
    PINNED_SGLANG_PATCH_MANIFEST_SHA256,
)
from lightcone_spec.runtime.preflight_runner import (
    BurstGptShapeAuthority,
    BurstGptSourcePin,
)
from lightcone_spec.runtime.proof_artifact import publish_canonical_json_no_replace
from lightcone_spec.runtime.readiness import (
    NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_candidate_terminal_source_round_is_one_based() -> None:
    with pytest.raises(ValueError, match="source round must be positive"):
        CandidateStateTerminalPair(
            source_round=0,
            tts_cell_id=_sha("tts-cell"),
            l0_naive_cell_id=_sha("l0-cell"),
            tts_run_id="tts-run",
            l0_naive_run_id="l0-run",
            tts_native_replay_pointer_sha256=_sha("tts-pointer"),
            l0_naive_native_replay_pointer_sha256=_sha("l0-pointer"),
            proposal_evidence_sha256=_sha("proposal"),
            tts_terminal_receipt_sha256=_sha("tts-terminal"),
            l0_naive_terminal_receipt_sha256=_sha("l0-terminal"),
        )


def _authority() -> TtsCalibrationAuthority:
    return TtsCalibrationAuthority(
        schema_version=1,
        authority_id="tts-primary-source-reconstruction-v2",
        primary_source_id="arXiv:2605.09329",
        primary_source_version="v2",
        paper_pdf_sha256=_sha("tts-pdf"),
        paper_source_sha256=_sha("tts-source"),
        tuning_window_sha256=_sha("disjoint-tuning-window"),
        trainable_plan_sha256=_sha("full-drafter-plan"),
        drafter_native_loss_recipe_sha256=_sha("drafter-native-loss-recipe"),
    )


def _protocol_lock(authority: TtsCalibrationAuthority) -> ProtocolLock:
    return ProtocolLock(
        schema_version=4,
        protocol_id="lightcone-formal-protocol-v1",
        code_git_head="1" * 40,
        code_git_tree="2" * 40,
        patch_manifest_sha256=_sha("patch"),
        registry_sha256=_sha("registry"),
        english_protocol_sha256=_sha("protocol-en"),
        chinese_protocol_sha256=_sha("protocol-zh"),
        tts_calibration_authority_sha256=authority.sha256,
        chronobelief_authority_sha256=_sha("chronobelief"),
        e1_recipe_anchor_authority_sha256=_sha("e1-recipe-anchors"),
        e2_recipe_grid_authority_sha256=_sha("e2-grid"),
        formal_runtime_authority_manifest_sha256=_sha("formal-runtime"),
        offline_release_trust_root_sha256=_sha("release-root"),
        prepared_model_content_authorization_sha256=_sha("prepared-model"),
        formal_workload_e3a_authorization_sha256=_sha("e3a-workload"),
        formal_workload_e0_authorization_sha256=_sha("e0-workload"),
        burstgpt_shape_authorization_sha256=_sha("burstgpt-shape"),
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


def _runtime_manifest() -> FormalRuntimeAuthorityManifest:
    return FormalRuntimeAuthorityManifest(
        schema_version=2,
        authority_id="formal-runtime-authority-test-v1",
        members=tuple(
            FormalRuntimeAuthorityMember(
                member_id=member_id,
                protocol_sha256=_sha(f"{member_id}:protocol"),
                runner_sha256=_sha(f"{member_id}:runner"),
                test_set_sha256=_sha(f"{member_id}:tests"),
                source_sha256=_sha(f"{member_id}:source"),
            )
            for member_id in FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
        ),
    )


def test_protocol_lock_and_runtime_share_one_qualification_authority() -> None:
    native = code_owned_qualification_source_identities()["native_runtime"]
    lock = replace(
        _protocol_lock(_authority()),
        patch_manifest_sha256=PINNED_SGLANG_PATCH_MANIFEST_SHA256,
        native_runtime_qualification_protocol_sha256=native[0],
        native_runtime_qualification_runner_sha256=native[1],
        native_runtime_qualification_test_set_sha256=native[2],
    )

    assert (
        lock.native_runtime_qualification_authority_sha256
        == NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
    )


def test_protocol_lock_qualification_identities_bind_protocol_runner_and_test_set() -> (
    None
):
    identities = code_owned_qualification_source_identities()
    native = identities["native_runtime"]
    compile_identity = identities["compile"]
    exactness = identities["exactness"]
    lock = replace(
        _protocol_lock(_authority()),
        native_runtime_qualification_protocol_sha256=native[0],
        native_runtime_qualification_runner_sha256=native[1],
        native_runtime_qualification_test_set_sha256=native[2],
        compile_qualification_protocol_sha256=compile_identity[0],
        compile_qualification_runner_sha256=compile_identity[1],
        compile_qualification_test_set_sha256=compile_identity[2],
        exactness_qualification_protocol_sha256=exactness[0],
        exactness_qualification_runner_sha256=exactness[1],
        exactness_qualification_test_set_sha256=exactness[2],
    )

    for kind, (protocol, runner, test_set) in identities.items():
        expected = content_sha256(
            {
                "schema_version": 1,
                "kind": f"{kind}_qualification_source_identity",
                "protocol_sha256": protocol,
                "runner_sha256": runner,
                "test_set_sha256": test_set,
                "patch_manifest_sha256": lock.patch_manifest_sha256,
            }
        )
        assert lock._qualification_source_identity(kind) == expected

    mutated = replace(
        lock,
        exactness_qualification_test_set_sha256=_sha("mutated-exactness-tests"),
    )
    assert (
        mutated.exactness_qualification_source_identity_sha256
        != lock.exactness_qualification_source_identity_sha256
    )


def test_formal_runtime_authority_manifest_requires_exact_named_members() -> None:
    manifest = _runtime_manifest()
    assert tuple(row.member_id for row in manifest.members) == (
        FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
    )
    assert manifest.member("all_stage_execution_mapper").member_id == (
        "all_stage_execution_mapper"
    )
    with pytest.raises(ValueError, match="cover every named member exactly"):
        replace(manifest, members=manifest.members[:-1])
    with pytest.raises(ValueError, match="cover every named member exactly"):
        replace(
            manifest,
            members=(manifest.members[1], manifest.members[0], *manifest.members[2:]),
        )
    with pytest.raises(ValueError, match="unregistered"):
        FormalRuntimeAuthorityMember(
            member_id="foreign_reducer",
            protocol_sha256=_sha("foreign-protocol"),
            runner_sha256=_sha("foreign-runner"),
            test_set_sha256=_sha("foreign-tests"),
            source_sha256=_sha("foreign-source"),
        )


def test_protocol_lock_schema4_binds_formal_runtime_manifest() -> None:
    authority = _authority()
    manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(authority),
        formal_runtime_authority_manifest_sha256=manifest.sha256,
    )
    assert lock.formal_runtime_authority_manifest_sha256 == manifest.sha256
    with pytest.raises(ValueError, match="schema 4"):
        replace(lock, schema_version=3)


def test_runtime_manifest_codec_and_no_replace_artifact_are_fail_closed(
    tmp_path,
) -> None:
    manifest = _runtime_manifest()
    payload = formal_runtime_authority_manifest_to_dict(manifest)
    assert formal_runtime_authority_manifest_from_dict(payload) == manifest
    tampered = {**payload, "manifest_sha256": _sha("tampered-manifest")}
    with pytest.raises(ValueError, match="digest differs"):
        formal_runtime_authority_manifest_from_dict(tampered)

    path = tmp_path / "formal-runtime-authority.json"
    binding = publish_formal_runtime_authority_manifest(str(path), manifest)
    assert binding.semantic_sha256 == content_sha256(payload)
    assert (
        revalidate_formal_runtime_authority_manifest(
            str(path), expected_manifest_sha256=manifest.sha256
        )
        == manifest
    )
    with pytest.raises(RuntimeError, match="target already exists"):
        publish_formal_runtime_authority_manifest(str(path), manifest)
    with pytest.raises(ValueError, match="differs from ProtocolLock"):
        revalidate_formal_runtime_authority_manifest(
            str(path), expected_manifest_sha256=_sha("foreign-manifest")
        )


def test_create_protocol_lock_reopens_clean_checkout_and_source_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    patch = root / "patch-manifest.json"
    english = root / "protocol-en.md"
    chinese = root / "protocol-zh.md"
    patch.write_text('{"patches":[]}\n', encoding="utf-8")
    english.write_text("formal protocol en\n", encoding="utf-8")
    chinese.write_text("正式实验协议\n", encoding="utf-8")
    manifest = _runtime_manifest()
    manifest_path = root / "formal-runtime-authority.json"
    publish_formal_runtime_authority_manifest(str(manifest_path), manifest)
    foreign_manifest = replace(
        manifest,
        members=(
            replace(
                manifest.members[0],
                source_sha256=_sha("foreign-runtime-source"),
            ),
            *manifest.members[1:],
        ),
    )
    foreign_manifest_path = root / "foreign-runtime-authority.json"
    publish_formal_runtime_authority_manifest(
        str(foreign_manifest_path),
        foreign_manifest,
    )
    burstgpt_shape = BurstGptShapeAuthority(
        schema_version=1,
        kind="burstgpt_six_source_shape_authority",
        sources=tuple(
            BurstGptSourcePin(
                source_id=f"official-{index}",
                official_sha256=_sha(f"official-{index}"),
            )
            for index in range(6)
        ),
        rows_sha256=_sha("burstgpt-rows"),
        row_count=6,
    )
    burstgpt_shape_path = root / "burstgpt-shape.json"
    publish_canonical_json_no_replace(burstgpt_shape_path, burstgpt_shape.to_dict())

    class PreparedContent:
        authorization_sha256 = _sha("prepared")
        authorization = SimpleNamespace(root_manifest_sha256=_sha("root"))

    class WorkloadContent:
        authorization_sha256 = _sha("e3a-workload")
        authorization = SimpleNamespace(root_manifest_sha256=_sha("root"))

    class DatasetContent:
        authority_domain = "e0_task_native"
        authorization_sha256 = _sha("e0-workload")
        authorization = SimpleNamespace(root_manifest_sha256=_sha("root"))

    verified_content = (PreparedContent(), WorkloadContent(), DatasetContent())
    receipt = SimpleNamespace(
        revalidate_formal_scope=lambda *, current_ns: verified_content
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main.build_source_formal_runtime_authority_manifest",
        lambda project_root: manifest,
    )
    method_authorities = {
        "tts": SimpleNamespace(authority=SimpleNamespace(sha256=_sha("tts"))),
        "chronobelief": SimpleNamespace(
            authority=SimpleNamespace(sha256=_sha("chronobelief"))
        ),
        "e1": SimpleNamespace(authority=SimpleNamespace(sha256=_sha("e1"))),
    }
    monkeypatch.setattr(
        "lightcone_spec.cli.main.load_tts_calibration_authority_artifact",
        lambda path: method_authorities["tts"],
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main.load_chronobelief_authority_artifact",
        lambda path: method_authorities["chronobelief"],
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main.load_e1_recipe_anchor_authority_artifact",
        lambda path: method_authorities["e1"],
    )
    qualification_identities = {
        "native_runtime": (
            _sha("native-protocol"),
            _sha("native-runner"),
            _sha("native-tests"),
        ),
        "compile": (
            _sha("compile-protocol"),
            _sha("compile-runner"),
            _sha("compile-tests"),
        ),
        "exactness": (
            _sha("exactness-protocol"),
            _sha("exactness-runner"),
            _sha("exactness-tests"),
        ),
    }
    monkeypatch.setattr(
        "lightcone_spec.cli.main.code_owned_qualification_source_identities",
        lambda: qualification_identities,
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main._load_content_verification_receipt",
        lambda path, now_ns: (receipt, verified_content),
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main.VerifiedPreparedModelContentRelease",
        PreparedContent,
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main.VerifiedReleaseWorkloadSources",
        WorkloadContent,
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main.VerifiedDatasetContentRelease",
        DatasetContent,
    )
    monkeypatch.setattr(
        "lightcone_spec.runtime.preflight_runner.derive_burstgpt_shape_authority_from_content_receipt",
        lambda receipt, current_ns: burstgpt_shape,
    )

    subprocess.run(("git", "init", str(root)), check=True, capture_output=True)
    subprocess.run(
        ("git", "-C", str(root), "config", "user.name", "Formal Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(root), "config", "user.email", "formal@example.invalid"),
        check=True,
    )
    subprocess.run(("git", "-C", str(root), "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(root),
            "commit",
            "-m",
            "formal source fixture",
        ),
        check=True,
        capture_output=True,
    )
    output = tmp_path / "protocol-lock.json"
    args = SimpleNamespace(
        protocol_id="formal-source-bound-test-v1",
        project_root=str(root),
        code_git_head=None,
        code_git_tree=None,
        patch_manifest=str(patch),
        patch_manifest_sha256=None,
        registry_sha256=None,
        english_protocol=str(english),
        english_protocol_sha256=None,
        chinese_protocol=str(chinese),
        chinese_protocol_sha256=None,
        tts_calibration_authority=str(tmp_path / "tts-authority.json"),
        chronobelief_authority=str(tmp_path / "chronobelief-authority.json"),
        e1_recipe_anchor_authority=str(tmp_path / "e1-authority.json"),
        formal_runtime_authority_manifest=str(manifest_path),
        content_verification_receipt="content-master.json",
        content_verification_now_ns=1,
        burstgpt_shape_authority=str(burstgpt_shape_path),
        output=str(output),
    )
    assert _create_protocol_lock(args) == 0
    payload = _load_bound_json(output)
    assert isinstance(payload, dict)
    lock = protocol_lock_from_dict(payload["payload"])
    assert lock.formal_runtime_authority_manifest_sha256 == manifest.sha256
    assert lock.patch_manifest_sha256 == hashlib.sha256(patch.read_bytes()).hexdigest()
    assert lock.tts_calibration_authority_sha256 == _sha("tts")
    assert lock.chronobelief_authority_sha256 == _sha("chronobelief")
    assert lock.e1_recipe_anchor_authority_sha256 == _sha("e1")
    assert (
        lock.e2_recipe_grid_authority_sha256
        == default_e2_recipe_grid_authority().sha256
    )
    assert (
        lock.native_runtime_qualification_protocol_sha256
        == qualification_identities["native_runtime"][0]
    )

    foreign_runtime = SimpleNamespace(
        **{
            **vars(args),
            "formal_runtime_authority_manifest": str(foreign_manifest_path),
            "output": str(tmp_path / "foreign-protocol-lock.json"),
        }
    )
    with pytest.raises(ValueError, match="differs from source-owned rebuild"):
        _create_protocol_lock(foreign_runtime)

    wrong_head = SimpleNamespace(**{**vars(args), "code_git_head": "0" * 40})
    with pytest.raises(ValueError, match="Git HEAD differs"):
        _create_protocol_lock(wrong_head)
    english.write_text("dirty protocol\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git worktree"):
        _create_protocol_lock(args)


def test_tts_authority_binds_exact_arxiv_v2_primary_source() -> None:
    authority = _authority()
    assert (
        f"{authority.primary_source_id}{authority.primary_source_version}"
        == "arXiv:2605.09329v2"
    )
    with pytest.raises(ValueError, match="exactly arXiv:2605.09329v2"):
        replace(authority, primary_source_version="v1")
    with pytest.raises(ValueError, match="exactly arXiv:2605.09329v2"):
        replace(authority, primary_source_id="arXiv:9999.99999")


def test_tts_authority_requires_literal_no_clip_runtime_recipe() -> None:
    authority = _authority()
    config = OptimizerConfig(
        name="adam",
        learning_rate=authority.learning_rates[0],
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        grad_clip=None,
    )
    authority.validate_runtime_optimizer_config(config)
    with pytest.raises(ValueError, match="frozen no-clip recipe"):
        authority.validate_runtime_optimizer_config(
            OptimizerConfig(**{**config.model_dump(), "grad_clip": 1e30})
        )


def _sign(payload: object, *, now_ns: int = 10_000_000_000):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_base64 = base64.b64encode(public_bytes).decode()
    public_sha256 = hashlib.sha256(public_bytes).hexdigest()
    policy = TrustedAttesterPolicy(
        policy_id="formal-offline-authority-v1",
        trusted_attesters=(
            ("formal-authority-signer", "formal-authority-key", public_sha256),
        ),
        public_keys=((public_sha256, public_base64),),
    )
    payload_sha256 = content_sha256(payload)
    challenge = AttestationChallenge.issue(
        challenge_id="formal-authority-challenge",
        subject_sha256=payload_sha256,
        lifetime_s=60,
        now_ns=now_ns,
    )
    signature = private_key.sign(
        attestation_message(challenge, payload_sha256=payload_sha256)
    )
    attestation = SignedAttestation(
        schema_version=1,
        kind="lightcone_signed_attestation",
        algorithm="Ed25519",
        attester_id="formal-authority-signer",
        key_id="formal-authority-key",
        environment="release",
        public_key_base64=public_base64,
        challenge_sha256=challenge.sha256,
        payload_sha256=payload_sha256,
        signature_base64=base64.b64encode(signature).decode(),
    )
    return payload_sha256, challenge, attestation, policy, now_ns


def test_protocol_lock_fixes_five_roles_dag_models_and_contrasts() -> None:
    authority = _authority()
    lock = _protocol_lock(authority)

    assert (
        lock.method_roles
        == FORMAL_METHOD_ROLES
        == (
            "Target-only",
            "Static",
            "TTS",
            "L0-naive",
            "LightCone",
        )
    )
    assert lock.stage_dag == FORMAL_STAGE_DAG
    assert lock.stage_dag[2] == "TTS-Cal"
    assert lock.primary_holm_family == ("LightCone-Static", "LightCone-TTS")
    assert lock.e6_models == E6_MODELS
    assert BANNED_MODEL not in lock.e6_models
    assert len(lock.code_git_head) == len(lock.code_git_tree) == 40
    assert len(lock.preflight_source_authority_bindings) == 8

    with pytest.raises(ValueError, match="E6 model set"):
        replace(lock, e6_models=(*lock.e6_models, BANNED_MODEL))
    with pytest.raises(ValueError, match="five formal roles"):
        replace(lock, method_roles=lock.method_roles[:-1])
    with pytest.raises(ValueError, match="40-hex Git object ID"):
        replace(lock, code_git_head=_sha("not-a-git-oid"))


def test_protocol_lock_rejects_future_result_control_envelope_fields() -> None:
    lock = _protocol_lock(_authority())
    encoded = protocol_lock_to_dict(lock)
    encoded.pop("native_runtime_qualification_protocol_sha256")
    encoded["native_runtime_qualification_control_envelope_sha256"] = _sha(
        "future-native-control"
    )

    with pytest.raises(ValueError, match="fields differ"):
        protocol_lock_from_dict(encoded)

    encoded = protocol_lock_to_dict(lock)
    encoded.pop("burstgpt_shape_authorization_sha256")
    encoded["burstgpt_shape_control_envelope_sha256"] = _sha("future-burstgpt-control")
    with pytest.raises(ValueError, match="fields differ"):
        protocol_lock_from_dict(encoded)


def test_protocol_lock_signature_is_payload_policy_and_expiry_bound() -> None:
    lock = _protocol_lock(_authority())
    payload_sha256, challenge, attestation, policy, now_ns = _sign(lock)
    signed = SignedProtocolLock(lock, payload_sha256, challenge, attestation)

    assert (
        signed.verify(
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
        is lock
    )
    with pytest.raises(ValueError, match="pinned trust root"):
        signed.verify(
            policy=policy,
            expected_policy_sha256=_sha("wrong-policy"),
            now_ns=now_ns,
        )
    with pytest.raises(ValueError, match="expired"):
        signed.verify(
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns + 61_000_000_000,
        )
    tampered = replace(signed, payload=replace(lock, protocol_id="other-protocol"))
    with pytest.raises(ValueError, match="digest differs"):
        tampered.verify(
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )


def test_tts_calibration_grid_rejects_an_untyped_frozen_seal() -> None:
    authority = _authority()
    assert len(authority.candidate_ids) == 9 * 8
    assert len(set(authority.candidate_ids)) == 72

    with pytest.raises(TypeError, match="raw 288-cell reducer"):
        TtsCalibrationSeal(
            schema_version=2,
            authority_sha256=authority.sha256,
            protocol_lock_sha256=_sha("protocol-lock"),
            materialization_receipt_sha256=_sha("materialization"),
            coverage_receipt_sha256=_sha("coverage"),
            reduction_receipt_sha256=_sha("reduction"),
            raw_manifest_sha256=_sha("manifest"),
            tuning_window_sha256=authority.tuning_window_sha256,
            selected_learning_rate=3e-5,
            selected_stride=15,
            selected_candidate_id=authority.candidate_id(learning_rate=3e-5, stride=15),
            selected_pilot_run_binding_sha256s=tuple(
                _sha(f"pilot-{i}") for i in range(4)
            ),
            _construction_seal=object(),
        )


def test_l0_naive_replay_requires_same_candidate_and_state_bytes() -> None:
    shared = {
        "source_round": 7,
        "source_version": 6,
        "source_state_sha256": _sha("source-state"),
        "trainable_plan_sha256": _sha("plan"),
        "candidate_bytes_sha256": _sha("candidate"),
        "optimizer_state_bytes_sha256": _sha("optimizer-state"),
        "proposal_evidence_sha256": _sha("proposal-evidence"),
    }
    tts = CandidateStateReplay(
        method_role="TTS",
        cell_id=_sha("tts-cell"),
        run_id="tts-controlled-replay",
        native_replay_pointer_sha256=_sha("tts-pointer"),
        publication_policy="fixed_barrier",
        **shared,
    )
    naive = CandidateStateReplay(
        method_role="L0-naive",
        cell_id=_sha("l0-cell"),
        run_id="l0-controlled-replay",
        native_replay_pointer_sha256=_sha("l0-pointer"),
        publication_policy="first_ready",
        **shared,
    )
    assert_tts_l0_candidate_state_equality(tts, naive)

    with pytest.raises(ValueError, match="candidate_bytes_sha256"):
        assert_tts_l0_candidate_state_equality(
            tts,
            replace(naive, candidate_bytes_sha256=_sha("different-candidate")),
        )
    with pytest.raises(ValueError, match="distinct live state identities"):
        assert_tts_l0_candidate_state_equality(
            tts,
            replace(naive, run_id=tts.run_id),
        )


def test_chronobelief_reference_uses_centered_second_moment_age_and_commit_count() -> (
    None
):
    authority = ChronoBeliefAuthority(
        schema_version=1,
        authority_id="chronobelief-equations-5.5-5.8-v1",
        paper_pdf_sha256=_sha("paper-pdf"),
        tex_source_sha256=_sha("paper-tex"),
    )
    assert authority.bias_correction == "standard_update_count"
    assert authority.weight_decay_semantics == "decoupled"
    assert authority.equations == (
        "m_r=beta1*m_(r-1)+(1-beta1)*g_r",
        "s_r=beta2*s_(r-1)+(1-beta2)*(g_r-m_r)^{odot2}",
        "kappa(d_r)=min(1,(beta1/sqrt(beta2))^d_r)",
        (
            "theta_(r+1)=(1-eta*lambda)*theta_r-eta*kappa(d_r)*"
            "mhat_r/(sqrt(shat_r)+epsilon)"
        ),
    )
    assert len(authority.equations) == len(set(authority.equations)) == 4
    state = ChronoBeliefState((2.0, -3.0), (0.0, 0.0), (0.0, 0.0), 0)
    gradients = (0.5, -0.25)
    result = chronobelief_reference_transition(
        state,
        gradients,
        safe_boundary_age=3,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )

    expected_first = tuple(0.1 * value for value in gradients)
    expected_second = tuple(
        0.001 * (gradient - moment) ** 2
        for gradient, moment in zip(gradients, expected_first, strict=True)
    )
    kappa = min(1.0, (0.9 / math.sqrt(0.999)) ** 3)
    expected_parameters = tuple(
        (1.0 - 1e-3 * 0.01) * parameter
        - 1e-3
        * kappa
        * (moment1 / (1 - 0.9))
        / (math.sqrt(moment2 / (1 - 0.999)) + 1e-8)
        for parameter, moment1, moment2 in zip(
            state.parameters, expected_first, expected_second, strict=True
        )
    )
    assert result.update_count == 1
    assert result.first_moments == pytest.approx(expected_first)
    assert result.second_moments == pytest.approx(expected_second)
    assert result.parameters == pytest.approx(expected_parameters)
    assert (
        chronobelief_reference_transition(
            result,
            gradients,
            safe_boundary_age=10,
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
            outcome="skip",
        )
        is result
    )
    assert (
        chronobelief_reference_transition(
            result,
            gradients,
            safe_boundary_age=10,
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.01,
            outcome="abort",
        )
        is result
    )


def test_chronobelief_reference_is_overflow_safe_and_rejects_nonfinite_outputs() -> (
    None
):
    state = ChronoBeliefState((1.0,), (0.0,), (0.0,), 0)
    large_age = chronobelief_reference_transition(
        state,
        (0.25,),
        safe_boundary_age=10**9,
        learning_rate=1e-3,
        beta1=0.99,
        beta2=0.01,
        epsilon=1e-8,
        weight_decay=0.0,
    )
    assert all(
        math.isfinite(value)
        for vector in (
            large_age.parameters,
            large_age.first_moments,
            large_age.second_moments,
        )
        for value in vector
    )
    with pytest.raises(ValueError, match="non-finite state"):
        chronobelief_reference_transition(
            state,
            (1e308,),
            safe_boundary_age=0,
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
        )
    with pytest.raises(ValueError, match="non-finite state"):
        chronobelief_reference_transition(
            ChronoBeliefState((1e308,), (0.0,), (0.0,), 0),
            (1.0,),
            safe_boundary_age=0,
            learning_rate=1e308,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=1e308,
        )


def test_banned_model_validator_scans_nested_free_form_fields() -> None:
    with pytest.raises(ValueError, match="banned E6 model"):
        reject_banned_model_identity(
            {"disposition": {"reason": f"download:{BANNED_MODEL}:ready"}}
        )


def test_protocol_lock_git_snapshot_rebuilds_exact_clean_checkout(tmp_path) -> None:
    repository = tmp_path / "source"
    repository.mkdir(mode=0o700)
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
    (repository / "patch.json").write_text('{"patches":[]}\n', encoding="utf-8")
    (repository / "protocol-en.md").write_text("formal protocol\n", encoding="utf-8")
    (repository / "protocol-zh.md").write_text("正式协议\n", encoding="utf-8")
    subprocess.run(
        ("git", "-C", str(repository), "add", "."),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-m", "fixture"),
        check=True,
        capture_output=True,
    )

    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    binding = publish_formal_protocol_lock_git_snapshot(
        project_root=repository,
        chunk_output_directory=evidence,
        index_output_path=evidence / "git-snapshot.json",
    )
    index = FormalProtocolLockGitSnapshotIndex.from_dict(binding.reopen())
    assert (
        index.git_head
        == subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert all(row.size <= 1024 * 1024 for row in index.chunks)
    with checkout_formal_protocol_lock_git_snapshot(binding.absolute_path) as (
        checkout,
        rebuilt,
    ):
        assert rebuilt == index
        assert (checkout / "protocol-zh.md").read_text(encoding="utf-8") == (
            "正式协议\n"
        )

    with pytest.raises(FileExistsError):
        publish_formal_protocol_lock_git_snapshot(
            project_root=repository,
            chunk_output_directory=evidence,
            index_output_path=evidence / "git-snapshot.json",
        )
