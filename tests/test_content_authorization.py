from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import lightcone_spec.runtime.content_authorization as content_module
from lightcone_spec.experiments import workload_authority as workload_module
from lightcone_spec.locking.prepared_models import (
    PreparedModelContentFile,
    PreparedModelSnapshotContent,
    SafetensorsHeaderBinding,
    SnapshotTensorMetadata,
)
from lightcone_spec.runtime.attestation import AttestationChallenge, attestation_message
from lightcone_spec.runtime.backend import (
    EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256,
    EAGLE3_OFFICIAL_SELECTOR_RELATIVE_PATH,
    Eagle3OfficialSelectorManifest,
    Eagle3OfficialSelectorRow,
    resolve_eagle3_official_selector_content_authority,
)
from lightcone_spec.runtime.compile_runner import (
    revalidate_prepared_content_verification_receipt,
)
from lightcone_spec.runtime.content_authorization import (
    AuthorizedDatasetContentMember,
    AuthorizedPreparedModel,
    AuthorizedWorkloadSource,
    ContentJsonArtifactBinding,
    ContentVerificationReceipt,
    DatasetContentMemberPathBinding,
    DatasetContentReleaseAuthorization,
    PreparedModelContentReleaseAuthorization,
    PreparedModelStageMembership,
    ReleaseWorkloadSourceAuthorization,
    bind_authorized_dataset_content_release,
    build_content_verification_receipt,
    revalidate_authorized_dataset_content_release,
    verify_dataset_content_release_authorization,
    verify_prepared_model_content_release_authorization,
    verify_release_workload_source_authorization,
)
from lightcone_spec.runtime.control_attestation import ChallengeReplayStore
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _install_test_root(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Ed25519PrivateKey, str]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    root_sha256 = _sha("ephemeral-content-root")
    binding = SimpleNamespace(
        semantic_sha256=root_sha256,
        sha256=_sha("ephemeral-content-root-binding"),
        root=SimpleNamespace(
            public_key_base64=base64.b64encode(public).decode("ascii")
        ),
    )
    monkeypatch.setattr(
        content_module, "load_source_release_ed25519_root", lambda: binding
    )
    return private, root_sha256


def _challenge(subject_sha256: str, *, now_ns: int, label: str) -> AttestationChallenge:
    return AttestationChallenge.issue(
        challenge_id=label,
        subject_sha256=subject_sha256,
        lifetime_s=60.0,
        now_ns=now_ns,
    )


def _signature(
    private: Ed25519PrivateKey,
    challenge: AttestationChallenge,
    subject_sha256: str,
) -> str:
    return base64.b64encode(
        private.sign(attestation_message(challenge, payload_sha256=subject_sha256))
    ).decode("ascii")


def _workload_file(
    tmp_path: Path,
    workload_id: str,
    *,
    count: int = 1,
) -> tuple[Path, AuthorizedWorkloadSource]:
    if type(count) is not int or isinstance(count, bool) or count < 1:
        raise ValueError("workload test row count must be positive")
    protocol = workload_module.FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    identity_prefix = "q" if workload_id == "livecodebench_v6_hard" else "m"
    prompt_prefix = (
        "Write a function." if workload_id == "livecodebench_v6_hard" else "Solve x=1."
    )
    rows = [
        {
            protocol.identity_field: f"{identity_prefix}-{index + 1}",
            protocol.prompt_field: f"{prompt_prefix} Case {index + 1}.",
            protocol.filter_field: protocol.filter_value,
        }
        for index in range(count)
    ]
    envelope = {
        "schema_version": 1,
        "repository": protocol.repository,
        "repository_revision": "a" * 40,
        "dataset_config": protocol.dataset_config,
        "split": protocol.split,
        "rows": rows,
    }
    path = (tmp_path / f"{workload_id}.json").resolve()
    body = (
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    path.write_bytes(body)
    samples = workload_module._select_all_rows(protocol, envelope["rows"])
    source = AuthorizedWorkloadSource(
        workload_id=workload_id,
        repository=protocol.repository,
        dataset_config=protocol.dataset_config,
        split=protocol.split,
        repository_revision="a" * 40,
        raw_file_sha256=hashlib.sha256(body).hexdigest(),
        raw_file_size=len(body),
        raw_row_count=count,
        filter_field=protocol.filter_field,
        filter_value=protocol.filter_value,
        prompt_compiler=protocol.prompt_compiler,
        selection_policy=protocol.selection_policy,
        selected_row_count=count,
        selected_rows_sha256=(workload_module.formal_workload_samples_sha256(samples)),
        protocol_sha256=protocol.sha256,
    )
    return path, source


def _workload_authorization(
    private: Ed25519PrivateKey,
    root_sha256: str,
    sources: tuple[AuthorizedWorkloadSource, ...],
    *,
    now_ns: int,
) -> ReleaseWorkloadSourceAuthorization:
    subject = content_module._canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_release_workload_source_subject",
            "root_manifest_sha256": root_sha256,
            "workload_sources": [row.to_dict() for row in sources],
        }
    )
    challenge = _challenge(subject, now_ns=now_ns, label="workload-content")
    return ReleaseWorkloadSourceAuthorization(
        schema_version=1,
        kind="lightcone_release_workload_source_authorization",
        root_manifest_sha256=root_sha256,
        workload_sources=sources,
        challenge=challenge,
        signature_base64=_signature(private, challenge, subject),
    )


def test_workload_authorization_rejects_wrong_key_replay_expiry_and_toctou(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 10_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)
    lcb_path, lcb = _workload_file(tmp_path, "livecodebench_v6_hard")
    _math_path, math = _workload_file(tmp_path, "math500_level5")
    authorization = _workload_authorization(
        private, root_sha256, (lcb, math), now_ns=now_ns
    )
    verified = verify_release_workload_source_authorization(
        authorization, now_ns=now_ns
    )
    assert verified.authorization is authorization
    assert verified.authorization_sha256 == authorization.sha256
    assert verified.challenge_sha256 == authorization.challenge.sha256
    assert verified.root_binding_sha256 == _sha("ephemeral-content-root-binding")
    bound = workload_module.bind_authorized_formal_workload_authority(
        "livecodebench_v6_hard", lcb_path, authorization=verified
    )
    assert (
        len(
            workload_module.require_authorized_formal_workload_authority(
                bound, authorization=verified
            )
        )
        == 1
    )

    with pytest.raises(ValueError, match="already consumed"):
        verify_release_workload_source_authorization(
            authorization,
            now_ns=now_ns,
            consumed_challenge_sha256s=(authorization.challenge.sha256,),
        )
    with pytest.raises(ValueError, match="expired"):
        verify_release_workload_source_authorization(
            authorization, now_ns=authorization.challenge.expires_ns + 1
        )
    with pytest.raises(ValueError, match="signature"):
        verify_release_workload_source_authorization(
            replace(
                authorization,
                signature_base64=base64.b64encode(bytes(64)).decode("ascii"),
            ),
            now_ns=now_ns,
        )

    other_private = Ed25519PrivateKey.generate()
    other_public = other_private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    wrong_binding = SimpleNamespace(
        semantic_sha256=root_sha256,
        sha256=_sha("wrong-binding"),
        root=SimpleNamespace(
            public_key_base64=base64.b64encode(other_public).decode("ascii")
        ),
    )
    monkeypatch.setattr(
        content_module, "load_source_release_ed25519_root", lambda: wrong_binding
    )
    with pytest.raises(ValueError, match="signature"):
        verify_release_workload_source_authorization(authorization, now_ns=now_ns)

    monkeypatch.undo()
    _install_test_root(monkeypatch)
    lcb_path.write_bytes(lcb_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="raw bytes"):
        workload_module.revalidate_authorized_formal_workload_authority(
            bound, authorization=verified
        )


def _prepared_authorization(
    private: Ed25519PrivateKey,
    root_sha256: str,
    *,
    now_ns: int,
) -> PreparedModelContentReleaseAuthorization:
    members = tuple(
        sorted(
            (
                AuthorizedPreparedModel(
                    member_id="e6:qwen35:target",
                    backend="target_only",
                    role="target",
                    model_id="Qwen/Qwen3.5-122B-A10B-FP8",
                    revision="b" * 40,
                    snapshot_manifest_raw_sha256=_sha("qwen35-snapshot"),
                    snapshot_manifest_semantic_sha256=_sha("qwen35-snapshot"),
                ),
                AuthorizedPreparedModel(
                    member_id="e6:qwen35:tokenizer",
                    backend="target_only",
                    role="tokenizer",
                    model_id="Qwen/Qwen3.5-122B-A10B-FP8",
                    revision="b" * 40,
                    snapshot_manifest_raw_sha256=_sha("qwen35-snapshot"),
                    snapshot_manifest_semantic_sha256=_sha("qwen35-snapshot"),
                ),
                AuthorizedPreparedModel(
                    member_id="e6:qwen36:target",
                    backend="target_only",
                    role="target",
                    model_id="Qwen/Qwen3.6-35B-A3B",
                    revision="c" * 40,
                    snapshot_manifest_raw_sha256=_sha("qwen36-snapshot"),
                    snapshot_manifest_semantic_sha256=_sha("qwen36-snapshot"),
                ),
                AuthorizedPreparedModel(
                    member_id="e6:qwen36:tokenizer",
                    backend="target_only",
                    role="tokenizer",
                    model_id="Qwen/Qwen3.6-35B-A3B",
                    revision="c" * 40,
                    snapshot_manifest_raw_sha256=_sha("qwen36-snapshot"),
                    snapshot_manifest_semantic_sha256=_sha("qwen36-snapshot"),
                ),
            ),
            key=lambda row: row.member_id,
        )
    )
    memberships = (
        PreparedModelStageMembership(
            stage_id="E6",
            member_ids=tuple(row.member_id for row in members),
        ),
    )
    subject = content_module._canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_prepared_model_content_release_subject",
            "root_manifest_sha256": root_sha256,
            "model_lock_sha256": _sha("model-lock"),
            "prepared_model_set_sha256": _sha("prepared-set"),
            "content_manifest_raw_sha256": _sha("content-raw"),
            "content_manifest_semantic_sha256": _sha("content-semantic"),
            "content_manifest_size": 123,
            "models": [row.to_dict() for row in members],
            "stage_memberships": [row.to_dict() for row in memberships],
        }
    )
    challenge = _challenge(subject, now_ns=now_ns, label="prepared-content")
    return PreparedModelContentReleaseAuthorization(
        schema_version=1,
        kind="lightcone_prepared_model_content_release_authorization",
        root_manifest_sha256=root_sha256,
        model_lock_sha256=_sha("model-lock"),
        prepared_model_set_sha256=_sha("prepared-set"),
        content_manifest_raw_sha256=_sha("content-raw"),
        content_manifest_semantic_sha256=_sha("content-semantic"),
        content_manifest_size=123,
        models=members,
        stage_memberships=memberships,
        challenge=challenge,
        signature_base64=_signature(private, challenge, subject),
    )


def test_prepared_authorization_covers_two_e6_targets_and_rejects_missing_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ns = 20_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)
    authorization = _prepared_authorization(private, root_sha256, now_ns=now_ns)
    verified = verify_prepared_model_content_release_authorization(
        authorization, now_ns=now_ns
    )
    assert verified.authorization is authorization
    assert verified.authorization_sha256 == authorization.sha256
    assert verified.challenge_sha256 == authorization.challenge.sha256
    assert verified.root_binding_sha256 == _sha("ephemeral-content-root-binding")
    assert verified.stage_ids == ("E6",)
    stage = verified.require_stage("E6")
    assert len(stage) == 4
    assert {row.model_id for row in stage if row.role == "target"} == {
        "Qwen/Qwen3.5-122B-A10B-FP8",
        "Qwen/Qwen3.6-35B-A3B",
    }
    with pytest.raises(ValueError, match="cover members exactly"):
        replace(authorization, models=authorization.models[:-1])
    with pytest.raises(ValueError, match="already consumed"):
        verify_prepared_model_content_release_authorization(
            authorization,
            now_ns=now_ns,
            consumed_challenge_sha256s=(authorization.challenge.sha256,),
        )
    with pytest.raises(ValueError, match="globally banned"):
        replace(
            authorization.models[0],
            model_id="Qwen/Qwen3.5-35B-A3B",
        )
    with pytest.raises(ValueError, match="40-hex"):
        replace(authorization.models[0], revision="d" * 64)
    with pytest.raises(ValueError, match="outside the formal DAG"):
        PreparedModelStageMembership(stage_id="foreign", member_ids=("x",))


def test_eagle3_selector_is_derived_from_prepared_content_not_source_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 25_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)

    def write_canonical(path: Path, value: object) -> bytes:
        body = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        path.write_bytes(body)
        return body

    target_root = (tmp_path / "target").resolve()
    drafter_root = (tmp_path / "drafter").resolve()
    target_root.mkdir()
    drafter_root.mkdir()
    target_member_id = "e0:eagle3:target"
    drafter_member_id = "e0:eagle3:drafter"
    target_model_id = "Qwen/Qwen3-4B"
    drafter_model_id = "official/EAGLE3-Qwen3-4B"
    target_revision = "a" * 40
    drafter_revision = "b" * 40
    tasks = (
        "AIME-2025",
        "Alpaca",
        "Arena-Hard",
        "GSM8K",
        "HumanEval",
        "LiveCodeBench",
        "MATH-500",
        "MBPP",
        "MT-Bench",
    )
    selector = Eagle3OfficialSelectorManifest(
        schema_version=1,
        kind="lightcone_eagle3_official_selector_manifest",
        protocol_sha256=EAGLE3_OFFICIAL_SELECTOR_CONTENT_PROTOCOL_SHA256,
        backend="EAGLE3",
        target_member_id=target_member_id,
        drafter_member_id=drafter_member_id,
        target_model_id=target_model_id,
        drafter_model_id=drafter_model_id,
        target_revision=target_revision,
        drafter_revision=drafter_revision,
        source_repository="https://github.com/official/eagle3",
        source_commit="c" * 40,
        rows=tuple(
            Eagle3OfficialSelectorRow(
                task=task,
                status="COMPATIBLE" if task == "LiveCodeBench" else "N/A",
                interface_sha256=_sha(f"eagle3-interface-{task}"),
                reason_code=(
                    "official_interface_match"
                    if task == "LiveCodeBench"
                    else "official_task_selector_unavailable"
                ),
            )
            for task in tasks
        ),
    )
    selector_path = drafter_root / EAGLE3_OFFICIAL_SELECTOR_RELATIVE_PATH
    selector_body = write_canonical(selector_path, selector.to_dict())

    tensor = SnapshotTensorMetadata(
        name="model.weight",
        shape=(1,),
        dtype="torch.float32",
        data_start=0,
        data_end=4,
    )
    header = SafetensorsHeaderBinding(
        relative_path="model.safetensors",
        file_size=14,
        device=1,
        inode=1,
        mtime_ns=1,
        ctime_ns=1,
        header_size=2,
        header_sha256=_sha("header"),
        raw_sha256=_sha("weights"),
        tensors=(tensor,),
    )
    target_snapshot = PreparedModelSnapshotContent(
        model_id=target_model_id,
        revision=target_revision,
        root=str(target_root),
        profile="test-eagle3-target",
        critical_files=(
            PreparedModelContentFile(
                relative_path="config.json",
                size=2,
                raw_sha256=_sha("target-config"),
            ),
        ),
        weight_kind="single_safetensors",
        weight_headers=(header,),
        tensor_metadata_sha256=content_module._canonical_sha256([tensor.to_dict()]),
    )
    drafter_snapshot = PreparedModelSnapshotContent(
        model_id=drafter_model_id,
        revision=drafter_revision,
        root=str(drafter_root),
        profile="test-eagle3-drafter",
        critical_files=tuple(
            sorted(
                (
                    PreparedModelContentFile(
                        relative_path="config.json",
                        size=2,
                        raw_sha256=_sha("drafter-config"),
                    ),
                    PreparedModelContentFile(
                        relative_path=EAGLE3_OFFICIAL_SELECTOR_RELATIVE_PATH,
                        size=len(selector_body),
                        raw_sha256=hashlib.sha256(selector_body).hexdigest(),
                    ),
                ),
                key=lambda row: row.relative_path,
            )
        ),
        weight_kind="single_safetensors",
        weight_headers=(replace(header, inode=2, raw_sha256=_sha("drafter-weights")),),
        tensor_metadata_sha256=content_module._canonical_sha256([tensor.to_dict()]),
    )
    target_snapshot_path = (tmp_path / "target-snapshot.json").resolve()
    drafter_snapshot_path = (tmp_path / "drafter-snapshot.json").resolve()
    target_snapshot_body = write_canonical(
        target_snapshot_path, target_snapshot.to_dict()
    )
    drafter_snapshot_body = write_canonical(
        drafter_snapshot_path, drafter_snapshot.to_dict()
    )
    target_snapshot_semantic = content_module._canonical_sha256(
        target_snapshot.to_dict()
    )
    drafter_snapshot_semantic = content_module._canonical_sha256(
        drafter_snapshot.to_dict()
    )
    models = tuple(
        sorted(
            (
                AuthorizedPreparedModel(
                    member_id=target_member_id,
                    backend="EAGLE3",
                    role="target",
                    model_id=target_model_id,
                    revision=target_revision,
                    snapshot_manifest_raw_sha256=hashlib.sha256(
                        target_snapshot_body
                    ).hexdigest(),
                    snapshot_manifest_semantic_sha256=target_snapshot_semantic,
                ),
                AuthorizedPreparedModel(
                    member_id=drafter_member_id,
                    backend="EAGLE3",
                    role="drafter",
                    model_id=drafter_model_id,
                    revision=drafter_revision,
                    snapshot_manifest_raw_sha256=hashlib.sha256(
                        drafter_snapshot_body
                    ).hexdigest(),
                    snapshot_manifest_semantic_sha256=drafter_snapshot_semantic,
                ),
                AuthorizedPreparedModel(
                    member_id="e0:eagle3:tokenizer",
                    backend="EAGLE3",
                    role="tokenizer",
                    model_id=target_model_id,
                    revision=target_revision,
                    snapshot_manifest_raw_sha256=hashlib.sha256(
                        target_snapshot_body
                    ).hexdigest(),
                    snapshot_manifest_semantic_sha256=target_snapshot_semantic,
                ),
            ),
            key=lambda row: row.member_id,
        )
    )
    memberships = (
        PreparedModelStageMembership(
            stage_id="E0",
            member_ids=tuple(row.member_id for row in models),
        ),
    )
    authorization_subject = content_module._canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_prepared_model_content_release_subject",
            "root_manifest_sha256": root_sha256,
            "model_lock_sha256": _sha("e0-model-lock"),
            "prepared_model_set_sha256": _sha("e0-prepared-set"),
            "content_manifest_raw_sha256": _sha("e0-content-raw"),
            "content_manifest_semantic_sha256": _sha("e0-content-semantic"),
            "content_manifest_size": 456,
            "models": [row.to_dict() for row in models],
            "stage_memberships": [row.to_dict() for row in memberships],
        }
    )
    challenge = _challenge(
        authorization_subject,
        now_ns=now_ns,
        label="e0-eagle3-prepared-content",
    )
    authorization = PreparedModelContentReleaseAuthorization(
        schema_version=1,
        kind="lightcone_prepared_model_content_release_authorization",
        root_manifest_sha256=root_sha256,
        model_lock_sha256=_sha("e0-model-lock"),
        prepared_model_set_sha256=_sha("e0-prepared-set"),
        content_manifest_raw_sha256=_sha("e0-content-raw"),
        content_manifest_semantic_sha256=_sha("e0-content-semantic"),
        content_manifest_size=456,
        models=models,
        stage_memberships=memberships,
        challenge=challenge,
        signature_base64=_signature(private, challenge, authorization_subject),
    )
    authorization_path = (tmp_path / "e0-prepared-authorization.json").resolve()
    write_canonical(authorization_path, authorization.to_dict())
    authorization_binding = ContentJsonArtifactBinding.from_path(
        "prepared:formal_dag", authorization_path
    )
    content_bindings = (
        ContentJsonArtifactBinding.from_path(
            "eagle3_official_selector:e0:eagle3:drafter", selector_path
        ),
        ContentJsonArtifactBinding.from_path(
            "snapshot:e0:eagle3:drafter", drafter_snapshot_path
        ),
        ContentJsonArtifactBinding.from_path(
            "snapshot:e0:eagle3:target", target_snapshot_path
        ),
    )
    replay_root = (tmp_path / "e0-eagle3-replay").resolve()
    replay_root.mkdir(mode=0o700)
    challenges = tuple(sorted((authorization.challenge.sha256, "f" * 64)))
    reservation_value = {
        "schema_version": 2,
        "kind": "lightcone_control_challenge_reservation",
        "reserved_ns": now_ns,
        "challenge_sha256s": list(challenges),
    }
    reservation_sha256 = content_module._canonical_sha256(reservation_value)
    write_canonical(
        replay_root / f"reservation-{reservation_sha256}.json", reservation_value
    )
    reservation = ChallengeReplayStore(str(replay_root)).bind_reservation(
        reservation_sha256
    )
    receipt = build_content_verification_receipt(
        verified_ns=now_ns,
        authorization_artifacts=(authorization_binding,),
        content_artifacts=content_bindings,
        reservation=reservation,
    )
    receipt_path = (tmp_path / "e0-content-receipt.json").resolve()
    write_canonical(receipt_path, receipt.to_dict())
    verified = resolve_eagle3_official_selector_content_authority(
        CanonicalJsonProofBinding.bind(receipt_path),
        expected_prepared_content_authorization_sha256=authorization.sha256,
        expected_root_manifest_sha256=root_sha256,
        expected_target_member_id=target_member_id,
        expected_drafter_member_id=drafter_member_id,
        expected_task="LiveCodeBench",
        now_ns=now_ns,
    )
    assert verified.status == "COMPATIBLE"
    assert verified.target_member_id == target_member_id
    assert verified.drafter_member_id == drafter_member_id
    assert (
        verified.selector_asset_raw_sha256 == hashlib.sha256(selector_body).hexdigest()
    )
    assert verified.prepared_content_authorization_sha256 == authorization.sha256
    assert verified.root_manifest_sha256 == root_sha256

    selector_path.write_bytes(selector_body + b" ")
    with pytest.raises((ValueError, RuntimeError), match="canonical|changed"):
        resolve_eagle3_official_selector_content_authority(
            CanonicalJsonProofBinding.bind(receipt_path),
            expected_prepared_content_authorization_sha256=authorization.sha256,
            expected_root_manifest_sha256=root_sha256,
            expected_target_member_id=target_member_id,
            expected_drafter_member_id=drafter_member_id,
            expected_task="LiveCodeBench",
            now_ns=now_ns,
        )


def _dataset_authorization(
    private: Ed25519PrivateKey,
    root_sha256: str,
    *,
    domain: str,
    count: int,
    now_ns: int,
) -> DatasetContentReleaseAuthorization:
    members = tuple(
        AuthorizedDatasetContentMember(
            member_id=f"{domain}:{index:02d}",
            source_uri=f"dataset://{domain}/{index}",
            revision=f"{index + 1:040x}",
            data_format="canonical_json_array",
            raw_file_sha256=_sha(f"{domain}-raw-{index}"),
            raw_file_size=100 + index,
            raw_row_count=10 + index,
            selected_rows_raw_sha256=_sha(f"{domain}-rows-raw-{index}"),
            selected_rows_sha256=_sha(f"{domain}-rows-{index}"),
            selected_rows_size=90 + index,
            selected_row_count=5 + index,
            request_shape_raw_sha256=_sha(f"{domain}-shape-raw-{index}"),
            request_shape_sha256=_sha(f"{domain}-shape-{index}"),
            request_shape_size=80 + index,
            protocol_sha256=_sha(f"{domain}-protocol"),
        )
        for index in range(count)
    )
    subject = content_module._canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_dataset_content_release_subject",
            "authority_domain": domain,
            "root_manifest_sha256": root_sha256,
            "members": [row.to_dict() for row in members],
        }
    )
    challenge = _challenge(subject, now_ns=now_ns, label=f"{domain}-content")
    return DatasetContentReleaseAuthorization(
        schema_version=1,
        kind="lightcone_dataset_content_release_authorization",
        authority_domain=domain,
        root_manifest_sha256=root_sha256,
        members=members,
        challenge=challenge,
        signature_base64=_signature(private, challenge, subject),
    )


def test_burstgpt_six_source_and_e0_task_native_are_independent_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ns = 30_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)
    burst = _dataset_authorization(
        private,
        root_sha256,
        domain="burstgpt_six_source",
        count=6,
        now_ns=now_ns,
    )
    e0 = _dataset_authorization(
        private,
        root_sha256,
        domain="e0_task_native",
        count=3,
        now_ns=now_ns,
    )
    verified_burst = verify_dataset_content_release_authorization(
        burst,
        expected_authority_domain="burstgpt_six_source",
        now_ns=now_ns,
    )
    verified_e0 = verify_dataset_content_release_authorization(
        e0,
        expected_authority_domain="e0_task_native",
        now_ns=now_ns,
    )
    assert verified_burst.authorization is burst
    assert verified_burst.challenge_sha256 == burst.challenge.sha256
    assert verified_burst.root_binding_sha256 == _sha("ephemeral-content-root-binding")
    assert (
        len(
            verified_burst.require_members(
                tuple(row.member_id for row in burst.members)
            )
        )
        == 6
    )
    assert verified_burst.authorization_sha256 != verified_e0.authorization_sha256
    with pytest.raises(ValueError, match="another domain"):
        verify_dataset_content_release_authorization(
            burst,
            expected_authority_domain="e0_task_native",
            now_ns=now_ns,
        )
    with pytest.raises(ValueError, match="exactly six"):
        replace(burst, members=burst.members[:-1])
    with pytest.raises(ValueError, match="lacks a required member"):
        verified_e0.require_members(("e0_task_native:missing",))


def test_dataset_path_binding_reopens_bytes_counts_shape_and_toctou(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 40_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)
    raw = [{"request_id": "r0", "prompt": "p"}]
    selected = [{"request_id": "r0", "prompt": "p"}]
    shape = {
        "schema_version": 1,
        "kind": "lightcone_request_shape_manifest",
        "requests": [{"request_id": "r0", "input_tokens": 1}],
    }

    def body(value: object) -> bytes:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )

    raw_body, selected_body, shape_body = body(raw), body(selected), body(shape)
    raw_path = (tmp_path / "raw.json").resolve()
    selected_path = (tmp_path / "selected.json").resolve()
    shape_path = (tmp_path / "shape.json").resolve()
    raw_path.write_bytes(raw_body)
    selected_path.write_bytes(selected_body)
    shape_path.write_bytes(shape_body)
    member = AuthorizedDatasetContentMember(
        member_id="e0_task_native:00",
        source_uri="dataset://e0/0",
        revision="d" * 40,
        data_format="canonical_json_array",
        raw_file_sha256=hashlib.sha256(raw_body).hexdigest(),
        raw_file_size=len(raw_body),
        raw_row_count=1,
        selected_rows_raw_sha256=hashlib.sha256(selected_body).hexdigest(),
        selected_rows_sha256=content_module._canonical_sha256(selected),
        selected_rows_size=len(selected_body),
        selected_row_count=1,
        request_shape_raw_sha256=hashlib.sha256(shape_body).hexdigest(),
        request_shape_sha256=content_module._canonical_sha256(shape),
        request_shape_size=len(shape_body),
        protocol_sha256=_sha("e0-protocol"),
    )
    unsigned = _dataset_authorization(
        private,
        root_sha256,
        domain="e0_task_native",
        count=1,
        now_ns=now_ns,
    )
    subject = content_module._canonical_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_dataset_content_release_subject",
            "authority_domain": "e0_task_native",
            "root_manifest_sha256": root_sha256,
            "members": [member.to_dict()],
        }
    )
    challenge = _challenge(subject, now_ns=now_ns, label="e0-path-content")
    authorization = replace(
        unsigned,
        members=(member,),
        challenge=challenge,
        signature_base64=_signature(private, challenge, subject),
    )
    verified = verify_dataset_content_release_authorization(
        authorization,
        expected_authority_domain="e0_task_native",
        now_ns=now_ns,
    )
    path_row = DatasetContentMemberPathBinding(
        member_id=member.member_id,
        raw_path=str(raw_path),
        selected_rows_path=str(selected_path),
        request_shape_path=str(shape_path),
    )
    binding = bind_authorized_dataset_content_release(
        authorization=verified,
        member_paths=(path_row,),
    )
    assert revalidate_authorized_dataset_content_release(
        binding, authorization=verified
    ) == (member,)
    selected_path.write_bytes(selected_body + b" ")
    with pytest.raises((ValueError, RuntimeError), match="canonical|differ"):
        revalidate_authorized_dataset_content_release(binding, authorization=verified)


def test_content_verification_receipt_survives_expiry_but_reopens_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 50_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)
    _, lcb = _workload_file(tmp_path, "livecodebench_v6_hard")
    _, math = _workload_file(tmp_path, "math500_level5")
    authorization = _workload_authorization(
        private, root_sha256, (lcb, math), now_ns=now_ns
    )

    def write_canonical(path: Path, value: object) -> None:
        path.write_bytes(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )

    authorization_path = (tmp_path / "workload-authorization.json").resolve()
    write_canonical(authorization_path, authorization.to_dict())
    content_path = (tmp_path / "workload-content-binding.json").resolve()
    write_canonical(content_path, {"binding": _sha("workload-binding")})
    authorization_binding = ContentJsonArtifactBinding.from_path(
        "workload:e3a", authorization_path
    )
    content_binding = ContentJsonArtifactBinding.from_path(
        "workload:e3a:path_authority", content_path
    )

    replay_root = (tmp_path / "replay").resolve()
    replay_root.mkdir(mode=0o700)
    challenges = tuple(sorted((authorization.challenge.sha256, "f" * 64)))
    reservation_value = {
        "schema_version": 2,
        "kind": "lightcone_control_challenge_reservation",
        "reserved_ns": now_ns,
        "challenge_sha256s": list(challenges),
    }
    reservation_sha256 = content_module._canonical_sha256(reservation_value)
    write_canonical(
        replay_root / f"reservation-{reservation_sha256}.json",
        reservation_value,
    )
    reservation = ChallengeReplayStore(str(replay_root)).bind_reservation(
        reservation_sha256
    )
    receipt = build_content_verification_receipt(
        verified_ns=now_ns,
        authorization_artifacts=(authorization_binding,),
        content_artifacts=(content_binding,),
        reservation=reservation,
    )
    assert receipt.schema_version == 2
    assert receipt.protocol_sha256 == (
        content_module.CONTENT_VERIFICATION_PROTOCOL_SHA256
    )
    decoded = ContentVerificationReceipt.from_dict(receipt.to_dict())
    verified_rows = decoded.revalidate(
        current_ns=authorization.challenge.expires_ns + 10**12
    )
    assert verified_rows[0].authorization_sha256 == authorization.sha256
    legacy = replace(receipt, schema_version=1, protocol_sha256=None)
    decoded_legacy = ContentVerificationReceipt.from_dict(legacy.to_dict())
    assert decoded_legacy == legacy
    assert (
        decoded_legacy.revalidate(
            current_ns=authorization.challenge.expires_ns + 10**12
        )[0].authorization_sha256
        == authorization.sha256
    )
    with pytest.raises(ValueError, match="reservation time"):
        build_content_verification_receipt(
            verified_ns=now_ns - 1,
            authorization_artifacts=(authorization_binding,),
            content_artifacts=(content_binding,),
            reservation=reservation,
        )

    content_path.write_bytes(content_path.read_bytes() + b" ")
    with pytest.raises((ValueError, RuntimeError), match="canonical|changed"):
        decoded.revalidate(current_ns=authorization.challenge.expires_ns + 10**12)


def test_compile_reopens_durable_prepared_content_without_reusing_live_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 60_000_000_000
    private, root_sha256 = _install_test_root(monkeypatch)
    authorization = _prepared_authorization(
        private,
        root_sha256,
        now_ns=now_ns,
    )

    def write_canonical(path: Path, value: object) -> None:
        path.write_bytes(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )

    authorization_path = (tmp_path / "prepared-authorization.json").resolve()
    write_canonical(authorization_path, authorization.to_dict())
    content_path = (tmp_path / "prepared-content-binding.json").resolve()
    write_canonical(content_path, {"binding": _sha("prepared-content-binding")})
    authorization_binding = ContentJsonArtifactBinding.from_path(
        "prepared:formal_dag",
        authorization_path,
    )
    content_binding = ContentJsonArtifactBinding.from_path(
        "prepared:formal_dag:path_authority",
        content_path,
    )
    replay_root = (tmp_path / "prepared-replay").resolve()
    replay_root.mkdir(mode=0o700)
    challenges = tuple(sorted((authorization.challenge.sha256, "e" * 64)))
    reservation_value = {
        "schema_version": 2,
        "kind": "lightcone_control_challenge_reservation",
        "reserved_ns": now_ns,
        "challenge_sha256s": list(challenges),
    }
    reservation_sha256 = content_module._canonical_sha256(reservation_value)
    write_canonical(
        replay_root / f"reservation-{reservation_sha256}.json",
        reservation_value,
    )
    reservation = ChallengeReplayStore(str(replay_root)).bind_reservation(
        reservation_sha256
    )
    receipt = build_content_verification_receipt(
        verified_ns=now_ns,
        authorization_artifacts=(authorization_binding,),
        content_artifacts=(content_binding,),
        reservation=reservation,
    )
    receipt_path = (tmp_path / "content-verification-receipt.json").resolve()
    write_canonical(receipt_path, receipt.to_dict())

    binding, verified = revalidate_prepared_content_verification_receipt(
        receipt_path,
        current_ns=authorization.challenge.expires_ns + 10**12,
    )
    assert binding.semantic_sha256 == receipt.sha256
    assert verified.authorization_sha256 == authorization.sha256
    assert verified.challenge_sha256 in reservation.challenge_sha256s

    authorization_path.write_bytes(authorization_path.read_bytes() + b" ")
    with pytest.raises((ValueError, RuntimeError), match="canonical|changed"):
        revalidate_prepared_content_verification_receipt(
            receipt_path,
            current_ns=authorization.challenge.expires_ns + 10**12,
        )
