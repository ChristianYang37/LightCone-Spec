from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_content_authorization import _workload_file
from test_formal_dispatch import _protocol_lock

import lightcone_spec.runtime.content_authorization as content_module
from lightcone_spec.cli import main as cli_module
from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments import formal_method_authority as method_module
from lightcone_spec.experiments import formal_stage_execution
from lightcone_spec.experiments import workload_authority as workload_module
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_STAGE_DAG,
    TtsCalibrationAuthority,
    content_sha256,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.orchestration import formal_physical_dispatch as physical_dispatch
from lightcone_spec.runtime import offline_signer
from lightcone_spec.runtime.content_authorization import (
    TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE,
    AuthorizedDatasetContentMember,
    AuthorizedPreparedModel,
    ContentVerificationReceipt,
    DatasetContentMemberPathBinding,
    DatasetContentPathBinding,
    DatasetContentReleaseAuthorization,
    DatasetContentReleaseAuthorizationSource,
    PreparedModelContentReleaseAuthorization,
    PreparedModelContentReleaseAuthorizationSource,
    PreparedModelStageMembership,
    ReleaseWorkloadSourceAuthorization,
    ReleaseWorkloadSourceAuthorizationSource,
    TtsCalibrationTuningWindow,
)
from lightcone_spec.runtime.preflight_runner import BurstGptShapeAuthority
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _publish(path: Path, value: object) -> Path:
    publish_canonical_json_no_replace(path.resolve(), value)
    return path.resolve()


def _install_root(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Ed25519PrivateKey, SimpleNamespace]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    root = SimpleNamespace(
        semantic_sha256=_sha("content-operator-root"),
        sha256=_sha("content-operator-root-binding"),
        root=SimpleNamespace(
            public_key_base64=base64.b64encode(public).decode("ascii"),
            public_key_sha256=hashlib.sha256(public).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        content_module,
        "load_source_release_ed25519_root",
        lambda: root,
    )
    monkeypatch.setattr(
        offline_signer,
        "load_source_release_ed25519_root",
        lambda: root,
    )
    return private, root


def _json_body(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def _dataset_source(
    tmp_path: Path,
    *,
    domain: str,
    count: int,
    root_sha256: str,
) -> tuple[DatasetContentReleaseAuthorizationSource, DatasetContentPathBinding]:
    members = []
    paths = []
    for index in range(count):
        request_id = f"{domain}-{index}"
        raw = [{"prompt": f"prompt-{index}", "request_id": request_id}]
        selected = list(raw)
        shape = {
            "schema_version": 1,
            "kind": "lightcone_request_shape_manifest",
            "requests": [{"input_tokens": index + 1, "request_id": request_id}],
        }
        raw_path = _publish(tmp_path / f"{domain}-{index}-raw.json", raw)
        selected_path = _publish(tmp_path / f"{domain}-{index}-selected.json", selected)
        shape_path = _publish(tmp_path / f"{domain}-{index}-shape.json", shape)
        raw_body = _json_body(raw)
        selected_body = _json_body(selected)
        shape_body = _json_body(shape)
        member_id = f"{domain}:{index:02d}"
        members.append(
            AuthorizedDatasetContentMember(
                member_id=member_id,
                source_uri=f"dataset://{domain}/{index}",
                revision=f"{index + 1:040x}",
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
                protocol_sha256=_sha(f"{domain}-protocol"),
            )
        )
        paths.append(
            DatasetContentMemberPathBinding(
                member_id=member_id,
                raw_path=str(raw_path),
                selected_rows_path=str(selected_path),
                request_shape_path=str(shape_path),
            )
        )
    source = DatasetContentReleaseAuthorizationSource(
        schema_version=1,
        kind="lightcone_dataset_content_release_authorization_source",
        authority_domain=domain,
        root_manifest_sha256=root_sha256,
        members=tuple(members),
    )
    return source, DatasetContentPathBinding(
        schema_version=1,
        kind="lightcone_dataset_content_path_binding",
        authorization_sha256="0" * 64,
        authority_domain=domain,
        members=tuple(paths),
    )


def _sign_source(
    *,
    source_path: Path,
    output: Path,
    artifact_type: str,
    challenge_id: str,
    key_fd: int,
    now_ns: int,
) -> None:
    os.lseek(key_fd, 0, os.SEEK_SET)
    assert (
        offline_signer.main(
            [
                "sign-content-authorization",
                "--artifact-type",
                artifact_type,
                "--source",
                str(source_path),
                "--challenge-id",
                challenge_id,
                "--lifetime-seconds",
                "60",
                "--now-ns",
                str(now_ns),
                "--key-fd",
                str(key_fd),
                "--output",
                str(output.resolve()),
            ]
        )
        == 0
    )


def test_offline_content_ceremony_master_scopes_consumers_and_burst_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now_ns = 100_000_000_000
    private, root = _install_root(monkeypatch)
    lcb_path, lcb = _workload_file(
        tmp_path,
        "livecodebench_v6_hard",
        count=20,
    )
    _math_path, math = _workload_file(tmp_path, "math500_level5")
    workload_source = ReleaseWorkloadSourceAuthorizationSource(
        schema_version=1,
        kind="lightcone_release_workload_source_authorization_source",
        root_manifest_sha256=root.semantic_sha256,
        workload_sources=(lcb, math),
    )

    target_manifest = {"kind": "snapshot", "member": "target"}
    drafter_manifest = {"kind": "snapshot", "member": "drafter"}
    target_path = _publish(tmp_path / "target-snapshot.json", target_manifest)
    drafter_path = _publish(tmp_path / "drafter-snapshot.json", drafter_manifest)
    target_body = _json_body(target_manifest)
    drafter_body = _json_body(drafter_manifest)
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
        root_manifest_sha256=root.semantic_sha256,
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
        tmp_path,
        domain="burstgpt_six_source",
        count=6,
        root_sha256=root.semantic_sha256,
    )
    e0_source, e0_binding = _dataset_source(
        tmp_path,
        domain="e0_task_native",
        count=1,
        root_sha256=root.semantic_sha256,
    )

    source_rows = (
        ("workload", workload_source, "workload"),
        ("prepared_model", prepared_source, "prepared"),
        ("dataset", burst_source, "burst"),
        ("dataset", e0_source, "e0"),
    )
    key_path = tmp_path / "offline-root.key"
    key_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    authorization_paths: dict[str, Path] = {}
    source_paths: dict[str, Path] = {}
    with key_path.open("rb") as key_handle:
        for artifact_type, source, label in source_rows:
            source_path = _publish(tmp_path / f"{label}-source.json", source.to_dict())
            source_paths[label] = source_path
            output = (tmp_path / f"{label}-authorization.json").resolve()
            _sign_source(
                source_path=source_path,
                output=output,
                artifact_type=artifact_type,
                challenge_id=f"content-{label}",
                key_fd=key_handle.fileno(),
                now_ns=now_ns,
            )
            authorization_paths[label] = output
        os.lseek(key_handle.fileno(), 0, os.SEEK_SET)
        with pytest.raises(SystemExit, match="failed closed"):
            offline_signer.main(
                [
                    "sign-content-authorization",
                    "--artifact-type",
                    "dataset",
                    "--source",
                    str(source_paths["workload"]),
                    "--challenge-id",
                    "wrong-type",
                    "--now-ns",
                    str(now_ns),
                    "--key-fd",
                    str(key_handle.fileno()),
                    "--output",
                    str((tmp_path / "wrong-type.json").resolve()),
                ]
            )
        with pytest.raises(SystemExit, match="failed closed"):
            _sign_source(
                source_path=source_paths["workload"],
                output=authorization_paths["workload"],
                artifact_type="workload",
                challenge_id="no-replace",
                key_fd=key_handle.fileno(),
                now_ns=now_ns,
            )

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
    burst_binding = replace(
        burst_binding,
        authorization_sha256=burst_authorization.sha256,
    )
    e0_binding = replace(e0_binding, authorization_sha256=e0_authorization.sha256)
    burst_binding_path = _publish(
        tmp_path / "burst-path-binding.json", burst_binding.to_dict()
    )
    e0_binding_path = _publish(tmp_path / "e0-path-binding.json", e0_binding.to_dict())

    replay_root = (tmp_path / "replay").resolve()
    replay_root.mkdir(mode=0o700)
    master_path = (tmp_path / "content-master.json").resolve()
    content_specs = (
        f"dataset:burstgpt_six_source:path_binding={burst_binding_path}",
        f"dataset:e0_task_native:path_binding={e0_binding_path}",
        f"snapshot:shared:drafter={drafter_path}",
        f"snapshot:shared:target={target_path}",
    )
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
        str(replay_root),
        "--now-ns",
        str(now_ns),
        "--output",
        str(master_path),
    ]
    for specification in content_specs:
        argv.extend(("--content-artifact", specification))
    forbidden_tuning = _publish(
        tmp_path / "forbidden-pre-master-tts-window.json",
        {"post_master_derived": True},
    )
    with pytest.raises(ValueError, match="post-master derived"):
        cli_module.main(
            [
                *argv,
                "--content-artifact",
                f"tts_calibration_tuning_window={forbidden_tuning}",
            ]
        )
    assert cli_module.main(argv) == 0
    master = ContentVerificationReceipt.from_dict(
        CanonicalJsonProofBinding.bind(master_path).reopen()
    )
    assert master.schema_version == 2
    assert master.protocol_sha256 == content_module.CONTENT_VERIFICATION_PROTOCOL_SHA256
    assert len(master.revalidate_formal_scope(current_ns=now_ns)) == 4
    forged_schedule_binding = replace(
        master.content_artifacts[0],
        artifact_id=("derived_formal_serving_request_schedule:" + _sha("future-cell")),
    )
    unused_replay_root = (tmp_path / "unused-derived-replay").resolve()
    unused_replay_root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="post-master derived"):
        content_module.verify_and_reserve_content_authorizations(
            verified_ns=now_ns,
            authorization_artifacts=master.authorization_artifacts,
            content_artifacts=(forged_schedule_binding,),
            replay_store=content_module.ChallengeReplayStore(str(unused_replay_root)),
        )
    assert not tuple(unused_replay_root.iterdir())
    replay_argv = list(argv)
    replay_argv[replay_argv.index("--output") + 1] = str(
        (tmp_path / "other-master.json").resolve()
    )
    with pytest.raises(ValueError, match="already consumed"):
        cli_module.main(replay_argv)

    assert (
        len(
            master.revalidate_formal_scope(
                current_ns=workload_authorization.challenge.expires_ns + 1
            )
        )
        == 4
    )
    workload_output = (tmp_path / "workload-authority.json").resolve()
    assert (
        cli_module.main(
            [
                "bind-formal-workload-authority",
                "--workload",
                "livecodebench_v6_hard",
                "--source",
                str(lcb_path),
                "--content-verification-receipt",
                str(master_path),
                "--now-ns",
                str(now_ns),
                "--output",
                str(workload_output),
            ]
        )
        == 0
    )
    tuning_path = (tmp_path / "tts-window.json").resolve()
    assert (
        cli_module.main(
            [
                "publish-tts-calibration-tuning-window",
                "--tuning-workload-authority",
                str(workload_output),
                "--content-verification-receipt",
                str(master_path),
                "--output",
                str(tuning_path),
            ]
        )
        == 0
    )
    tuning_window = TtsCalibrationTuningWindow.from_dict(
        CanonicalJsonProofBinding.bind(tuning_path).reopen()
    )
    workload_authority = workload_module.formal_workload_authority_from_cli_artifact(
        CanonicalJsonProofBinding.bind(workload_output).reopen()
    )
    assert tuning_window.schema_version == 4
    assert tuning_window.content_verification_receipt_sha256 == master.sha256
    assert tuning_window.content_verification_verified_ns == master.verified_ns
    assert (
        tuning_window.content_verification_reservation_sha256
        == master.reservation.reservation_sha256
    )
    assert len(tuning_window.excluded_pilot_entries) == 4
    assert set(tuning_window.problem_ids) == {
        row.source_row_id for row in workload_authority.samples
    }
    expected_excluded = {
        row.source_row_id
        for row in sorted(
            workload_authority.samples,
            key=lambda row: (
                content_sha256(
                    {
                        "selector_namespace": (
                            TTS_CALIBRATION_TUNING_SELECTOR_NAMESPACE
                        ),
                        "source_problem_id": row.source_row_id,
                    }
                ),
                row.source_row_id,
                row.sample_id,
            ),
        )[:4]
    }
    assert set(tuning_window.excluded_problem_ids or ()) == expected_excluded

    legacy_window = TtsCalibrationTuningWindow(
        schema_version=2,
        kind=tuning_window.kind,
        tuning_entries=tuple(
            sorted(
                (
                    replace(row, source_problem_id=None)
                    for row in tuning_window.tuning_entries
                ),
                key=lambda row: row.entry_id,
            )
        ),
        excluded_pilot_entries=tuple(
            sorted(
                (
                    replace(row, source_problem_id=None)
                    for row in tuning_window.excluded_pilot_entries
                ),
                key=lambda row: row.entry_id,
            )
        ),
    )
    legacy_window_path = _publish(
        tmp_path / "legacy-schema-2-tts-window.json",
        legacy_window.to_dict(),
    )
    legacy_master = replace(
        master,
        schema_version=1,
        protocol_sha256=None,
        content_artifacts=tuple(
            sorted(
                (
                    *master.content_artifacts,
                    content_module.ContentJsonArtifactBinding.from_path(
                        "tts_calibration_tuning_window",
                        legacy_window_path,
                    ),
                ),
                key=lambda row: row.artifact_id,
            )
        ),
    )
    decoded_legacy_master = ContentVerificationReceipt.from_dict(
        legacy_master.to_dict()
    )
    assert decoded_legacy_master == legacy_master
    assert (
        len(
            decoded_legacy_master.revalidate_formal_scope(
                current_ns=workload_authorization.challenge.expires_ns + 1
            )
        )
        == 4
    )
    with pytest.raises(ValueError, match="schema-2 content receipt"):
        method_module.build_code_owned_tts_calibration_tuning_window(
            workload_authority,
            content_verification_receipt=decoded_legacy_master,
        )

    raw_authorization_output = (tmp_path / "raw-authorization-window.json").resolve()
    with pytest.raises(ValueError, match="content verification receipt"):
        cli_module.main(
            [
                "publish-tts-calibration-tuning-window",
                "--tuning-workload-authority",
                str(workload_output),
                "--content-verification-receipt",
                str(authorization_paths["workload"]),
                "--output",
                str(raw_authorization_output),
            ]
        )
    assert not raw_authorization_output.exists()

    tampered_receipt_value = master.to_dict()
    tampered_reservation = tampered_receipt_value["reservation"]
    assert isinstance(tampered_reservation, dict)
    tampered_reservation["raw_sha256"] = "0" * 64
    tampered_receipt_path = _publish(
        tmp_path / "tampered-content-receipt.json",
        tampered_receipt_value,
    )
    with pytest.raises(ValueError, match="reservation binding changed"):
        cli_module.main(
            [
                "publish-tts-calibration-tuning-window",
                "--tuning-workload-authority",
                str(workload_output),
                "--content-verification-receipt",
                str(tampered_receipt_path),
                "--output",
                str((tmp_path / "tampered-receipt-window.json").resolve()),
            ]
        )

    displaced_tuning = tuning_window.tuning_entries[0]
    displaced_pilot = tuning_window.excluded_pilot_entries[0]
    foreign_tuning = tuple(
        sorted(
            (*tuning_window.tuning_entries[1:], displaced_pilot),
            key=lambda row: row.entry_id,
        )
    )
    foreign_excluded = tuple(
        sorted(
            (displaced_tuning, *tuning_window.excluded_pilot_entries[1:]),
            key=lambda row: row.entry_id,
        )
    )
    foreign_window = replace(
        tuning_window,
        tuning_problem_ids=tuple(
            sorted(str(row.source_problem_id) for row in foreign_tuning)
        ),
        excluded_problem_ids=tuple(
            sorted(str(row.source_problem_id) for row in foreign_excluded)
        ),
        tuning_entries=foreign_tuning,
        excluded_pilot_entries=foreign_excluded,
    )
    foreign_window_path = _publish(
        tmp_path / "foreign-tts-window.json",
        foreign_window.to_dict(),
    )
    with pytest.raises(ValueError, match="differs from code-owned selector"):
        method_module._reopen_tuning_window(
            CanonicalJsonProofBinding.bind(foreign_window_path),
            workload_source=CanonicalJsonProofBinding.bind(workload_output),
            content_verification_receipt_source=CanonicalJsonProofBinding.bind(
                master_path
            ),
        )

    tts_authority = TtsCalibrationAuthority(
        schema_version=2,
        authority_id="tts-arxiv-v2-numeric-calibration",
        primary_source_id="arXiv:2605.09329",
        primary_source_version="v2",
        paper_pdf_sha256=_sha("paper-pdf"),
        paper_source_sha256=_sha("paper-source"),
        tuning_window_sha256=tuning_window.sha256,
        trainable_plan_sha256=_sha("trainable-plan"),
        drafter_native_loss_recipe_sha256=_sha("native-loss"),
    )

    scoped: dict[str, ContentVerificationReceipt] = {}
    for stage in ("E3a", "TTS-Cal", "E0"):
        output = (tmp_path / f"content-{stage}.json").resolve()
        assert (
            cli_module.main(
                [
                    "scope-content-verification-receipt",
                    "--master-receipt",
                    str(master_path),
                    "--stage",
                    stage,
                    "--now-ns",
                    str(now_ns),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        scoped[stage] = ContentVerificationReceipt.from_dict(
            CanonicalJsonProofBinding.bind(output).reopen()
        )
        scoped[stage].revalidate_formal_scope(current_ns=now_ns)

    lock = replace(
        _protocol_lock(),
        offline_release_trust_root_sha256=root.semantic_sha256,
        prepared_model_content_authorization_sha256=prepared_authorization.sha256,
        formal_workload_e3a_authorization_sha256=workload_authorization.sha256,
        formal_workload_e0_authorization_sha256=e0_authorization.sha256,
    )
    config = RunConfig(
        method="static",
        model=ModelPair(target_revision="1" * 40, drafter_revision="2" * 40),
        runtime=RuntimeConfig(
            sampling_profile_sha256="3" * 64,
            device_identity="GPU-content-test",
            speculative_num_draft_tokens=16,
        ),
    )
    basic_cell = MaterializedCell(
        stage="E3a",
        method_role="Static",
        model=config.model.target,
        backend="DFLASH",
        task="formal-content-test",
        publication_policy="fixed_barrier",
        recipe_sha256=None,
        dimensions=(),
    )
    prepared_sha256s, workload_sha256s = (
        formal_stage_execution._verified_content_identity(
            receipt=scoped["E3a"],
            protocol_lock=lock,
            stage="E3a",
            cell=basic_cell,
            run_config=config,
            tts_authority=None,
            now_ns=now_ns,
        )
    )
    assert len(prepared_sha256s) == 3
    assert len(workload_sha256s) == 2
    with pytest.raises(ValueError, match="not path-reopened"):
        formal_stage_execution._verified_content_identity(
            receipt=scoped["TTS-Cal"],
            protocol_lock=lock,
            stage="TTS-Cal",
            cell=replace(basic_cell, stage="TTS-Cal"),
            run_config=config,
            tts_authority=tts_authority,
            now_ns=now_ns,
        )
    e0_member = e0_source.members[0]
    _prepared, e0_workload = formal_stage_execution._verified_content_identity(
        receipt=scoped["E0"],
        protocol_lock=lock,
        stage="E0",
        cell=replace(
            basic_cell,
            stage="E0",
            dimensions=(
                ("load", "concurrency_one"),
                ("task_native_workload_sha256", e0_member.request_shape_sha256),
            ),
        ),
        run_config=config,
        tts_authority=None,
        now_ns=now_ns,
    )
    assert e0_authorization.sha256 in e0_workload

    assert all(
        row.artifact_id != "formal_workload_authority:livecodebench_v6_hard"
        for row in scoped["E3a"].content_artifacts
    )
    workload_binding, workload_authority, workload_descriptor_sha256 = (
        physical_dispatch._root_verified_workload_source(
            scoped["E3a"],
            workload_id="livecodebench_v6_hard",
            workload_authority_path=workload_output,
            current_ns=now_ns,
        )
    )
    assert workload_binding.path == str(workload_output)
    assert workload_descriptor_sha256 == lcb.sha256
    schedule_cell = replace(
        basic_cell,
        dimensions=(
            ("concurrency", 1),
            ("context", 4096),
            ("regime", "short_input_long_generation"),
        ),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("preflight-receipt"),),
        source_decision_sha256=_sha("e3a-materialization-source"),
        materialization_rule="content_then_materialization_then_schedule",
        expected_cell_count=1,
        cells=(schedule_cell,),
        gpu_hours=GpuHourEstimate(
            status="UNMEASURED",
            source_pilot_receipt_sha256=None,
            compute_gpu_hours=None,
            reserved_gpu_hours=None,
            estimated_wall_hours=None,
            retry_reserve_gpu_hours=None,
            profile_reserve_gpu_hours=None,
            evidence_reserve_gpu_hours=None,
        ),
    )
    sampling = SamplingProfile()
    schedule_source = physical_dispatch.rebuild_formal_serving_request_schedule_source(
        subject_sha256=_sha("post-master-serving-subject"),
        workload_authority_sha256=workload_authorization.sha256,
        topology_mode="tp1_dp1",
        materialization=materialization,
        materialized_cell_id=schedule_cell.cell_id,
        workload_source=workload_authority,
        workload_source_descriptor_sha256=workload_descriptor_sha256,
        tts_tuning_window=None,
        sampling_profile=sampling,
        max_running_requests=1,
        server_context_limit=4096,
        tokenizer_content_member_id="shared:tokenizer",
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        tokenizer_content_authority_sha256=_sha("target-snapshot"),
    )
    assert {row.source_member_sha256 for row in schedule_source.requests} == {
        lcb.sha256
    }
    assert {row.source_raw_file_sha256 for row in schedule_source.requests} == {
        lcb.raw_file_sha256
    }
    assert {row.source_selected_rows_sha256 for row in schedule_source.requests} == {
        lcb.selected_rows_sha256
    }
    assert (
        physical_dispatch.revalidate_formal_serving_request_schedule_source(
            schedule_source,
            materialization=materialization,
            workload_source=workload_authority,
            workload_source_descriptor_sha256=workload_descriptor_sha256,
            tts_tuning_window=None,
            sampling_profile=sampling,
            server_context_limit=4096,
        )
        == schedule_source
    )

    tts_cell = replace(
        schedule_cell,
        stage="TTS-Cal",
        dimensions=(
            ("arrival", "closed_loop"),
            ("context", 4096),
            ("regime", "short_input_long_generation"),
        ),
    )
    tts_materialization = replace(
        materialization,
        stage="TTS-Cal",
        cells=(tts_cell,),
    )
    tts_source = physical_dispatch.rebuild_formal_serving_request_schedule_source(
        subject_sha256=_sha("post-master-tts-subject"),
        workload_authority_sha256=tts_authority.sha256,
        topology_mode="tp1_dp1",
        materialization=tts_materialization,
        materialized_cell_id=tts_cell.cell_id,
        workload_source=workload_authority,
        workload_source_descriptor_sha256=workload_descriptor_sha256,
        tts_tuning_window=tuning_window,
        sampling_profile=sampling,
        max_running_requests=1,
        server_context_limit=4096,
        tokenizer_content_member_id="shared:tokenizer",
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        tokenizer_content_authority_sha256=_sha("target-snapshot"),
    )
    assert tts_source.tts_tuning_window_sha256 == tuning_window.sha256
    assert tuple(row.source_sample_id for row in tts_source.requests) == tuple(
        row.source_sample_id for row in tuning_window.tuning_entries
    )
    forged_entry = replace(
        tuning_window.tuning_entries[0],
        prompt_sha256=_sha("foreign-tts-prompt"),
    )
    forged_window = replace(
        tuning_window,
        tuning_entries=tuple(
            sorted(
                (forged_entry, *tuning_window.tuning_entries[1:]),
                key=lambda row: row.entry_id,
            )
        ),
    )
    with pytest.raises(ValueError, match="root-authorized prompt"):
        physical_dispatch.rebuild_formal_serving_request_schedule_source(
            subject_sha256=_sha("post-master-tts-subject"),
            workload_authority_sha256=tts_authority.sha256,
            topology_mode="tp1_dp1",
            materialization=tts_materialization,
            materialized_cell_id=tts_cell.cell_id,
            workload_source=workload_authority,
            workload_source_descriptor_sha256=workload_descriptor_sha256,
            tts_tuning_window=forged_window,
            sampling_profile=sampling,
            max_running_requests=1,
            server_context_limit=4096,
            tokenizer_content_member_id="shared:tokenizer",
            tokenizer_model_id="Qwen/Qwen3-8B",
            tokenizer_revision="1" * 40,
            tokenizer_content_authority_sha256=_sha("target-snapshot"),
        )

    def reject_row_tamper(field: str, value: object) -> None:
        payload = schedule_source.to_dict()
        request = payload["requests"][1]
        assert isinstance(request, dict)
        request[field] = value
        if field == "prompt":
            request["prompt_sha256"] = physical_dispatch._sha256(value)
        tampered_source = (
            physical_dispatch.FormalServingRequestScheduleSource.from_dict(payload)
        )
        with pytest.raises(ValueError, match="differs from source reducer"):
            physical_dispatch.revalidate_formal_serving_request_schedule_source(
                tampered_source,
                materialization=materialization,
                workload_source=workload_authority,
                workload_source_descriptor_sha256=workload_descriptor_sha256,
                tts_tuning_window=None,
                sampling_profile=sampling,
                server_context_limit=4096,
            )

    reject_row_tamper("prompt", "caller-forged prompt")
    reject_row_tamper("arrival_us", 1)
    reject_row_tamper(
        "requested_output_tokens",
        schedule_source.requests[1].requested_output_tokens + 1,
    )
    forged_sampling = dict(schedule_source.requests[1].sampling)
    forged_sampling["sampling_seed"] = int(forged_sampling["sampling_seed"]) + 1
    reject_row_tamper("sampling", forged_sampling)

    dp_cell = replace(
        schedule_cell,
        dimensions=tuple(sorted((*schedule_cell.dimensions, ("cohort_count", 4)))),
    )
    dp_materialization = replace(materialization, cells=(dp_cell,))
    dp_source = physical_dispatch.rebuild_formal_serving_request_schedule_source(
        subject_sha256=_sha("post-master-dp2-subject"),
        workload_authority_sha256=workload_authorization.sha256,
        topology_mode="tp1_dp2",
        materialization=dp_materialization,
        materialized_cell_id=dp_cell.cell_id,
        workload_source=workload_authority,
        workload_source_descriptor_sha256=workload_descriptor_sha256,
        tts_tuning_window=None,
        sampling_profile=sampling,
        max_running_requests=1,
        server_context_limit=4096,
        tokenizer_content_member_id="shared:tokenizer",
        tokenizer_model_id="Qwen/Qwen3-8B",
        tokenizer_revision="1" * 40,
        tokenizer_content_authority_sha256=_sha("target-snapshot"),
    )
    dp_payload = dp_source.to_dict()
    for request in dp_payload["requests"]:
        assert isinstance(request, dict)
        request["routed_dp_rank"] = 1 - int(request["routed_dp_rank"])
    forged_dp_source = physical_dispatch.FormalServingRequestScheduleSource.from_dict(
        dp_payload
    )
    with pytest.raises(ValueError, match="differs from source reducer"):
        physical_dispatch.revalidate_formal_serving_request_schedule_source(
            forged_dp_source,
            materialization=dp_materialization,
            workload_source=workload_authority,
            workload_source_descriptor_sha256=workload_descriptor_sha256,
            tts_tuning_window=None,
            sampling_profile=sampling,
            server_context_limit=4096,
        )

    burst_shape_path = (tmp_path / "burst-shape-authority.json").resolve()
    assert (
        cli_module.main(
            [
                "publish-burstgpt-shape-authority",
                "--content-verification-receipt",
                str(master_path),
                "--now-ns",
                str(now_ns),
                "--output",
                str(burst_shape_path),
            ]
        )
        == 0
    )
    burst_shape = BurstGptShapeAuthority.from_dict(
        CanonicalJsonProofBinding.bind(burst_shape_path).reopen()
    )
    assert burst_shape.row_count == 6
    assert tuple(row.official_sha256 for row in burst_shape.sources) == tuple(
        row.raw_file_sha256 for row in burst_source.members
    )

    tampered = replace(
        scoped["E3a"],
        content_artifacts=tuple(
            row
            for row in scoped["E3a"].content_artifacts
            if row.artifact_id != "content:master_verification_receipt"
        ),
    )
    with pytest.raises(ValueError, match="lacks one master"):
        tampered.revalidate_formal_scope(current_ns=now_ns)


def test_generic_coverage_cli_rejects_every_formal_stage_before_dispositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def load(path: str) -> str:
        observed.append(path)
        if path == "caller-dispositions.json":
            raise AssertionError(
                "formal coverage must reject before caller dispositions"
            )
        return path

    monkeypatch.setattr(cli_module, "_load_bound_json", load)
    monkeypatch.setattr(
        cli_module,
        "stage_materialization_receipt_from_dict",
        lambda stage: SimpleNamespace(stage=stage),
    )
    for stage in FORMAL_STAGE_DAG:
        with pytest.raises(ValueError, match="reducer-owned"):
            cli_module._create_stage_coverage(
                SimpleNamespace(
                    materialization=stage,
                    dispositions="caller-dispositions.json",
                    tts_l0_candidate_state_coverage=[],
                    output="forbidden.json",
                )
            )
    assert "caller-dispositions.json" not in observed
