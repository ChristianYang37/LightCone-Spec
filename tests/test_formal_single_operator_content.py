from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import lightcone_spec.experiments.formal_single_operator_content as content_module
import lightcone_spec.experiments.formal_single_operator_loads as loads_module
from lightcone_spec.experiments.formal_single_operator_content import (
    TRUSTED_SINGLE_OPERATOR_CONTENT_REPLAY_PROTOCOL_SHA256,
    TrustedLockedWorkload,
    TrustedModelRuntimeBinding,
    TrustedModelSnapshotMember,
    TrustedModelSnapshotSpec,
    TrustedNamedInputPath,
    TrustedRuntimeObservations,
    TrustedSingleOperatorContentBundle,
    TrustedSingleOperatorContentBundleBinding,
    TrustedSingleOperatorContentPathSpec,
    TrustedSingleOperatorContentReplayAuthorityBinding,
    TrustedSingleOperatorContentReplayBlocked,
    TrustedSourceSnapshot,
    bind_trusted_json_artifact,
    bind_trusted_locked_workload,
    bind_trusted_model_snapshot_member,
    bind_trusted_single_operator_runtime_observations,
    bind_trusted_source_snapshot,
    build_trusted_single_operator_content_bundle,
    load_trusted_single_operator_content_bundle,
    load_trusted_single_operator_content_path_spec,
    publish_runtime_bound_trusted_single_operator_content_from_spec,
    publish_trusted_preflight_workload_authority_from_content,
    publish_trusted_single_operator_content_bundle,
    publish_trusted_single_operator_content_replay_authority_from_spec,
    revalidate_trusted_json_artifact,
    revalidate_trusted_model_snapshot_member,
    revalidate_trusted_single_operator_content_bundle,
    revalidate_trusted_single_operator_content_replay_authority,
)
from lightcone_spec.experiments.formal_single_operator_loads import (
    BURSTGPT_V2_ACTIVE_ASSET,
    OfficialBurstGptAsset,
)
from lightcone_spec.experiments.workload_authority import (
    RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA,
    RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA,
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(tmp_path: Path) -> Path:
    root = (tmp_path / "source").resolve()
    patch_root = root / "patches" / "sglang"
    patch_root.mkdir(parents=True)
    patches: list[dict[str, object]] = []
    for index in range(1, 9):
        name = f"000{index}-test.patch"
        body = f"patch-{index}\n".encode()
        (patch_root / name).write_bytes(body)
        patches.append(
            {
                "file": name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "files": [f"python/example_{index}.py"],
            }
        )
    manifest = {
        "schema_version": 2,
        "upstream": {
            "repository": "https://github.com/sgl-project/sglang.git",
            "commit": "1" * 40,
        },
        "expected_tree": "2" * 40,
        "patches": patches,
    }
    (patch_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root.parent, "init", str(root))
    _git(root, "add", ".")
    subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Content Test",
            "-c",
            "user.email=content@example.invalid",
            "commit",
            "-m",
            "fixture",
        ),
        check=True,
        capture_output=True,
    )
    return root


def _model_spec(tmp_path: Path, role: str) -> TrustedModelSnapshotSpec:
    root = (tmp_path / f"{role}-snapshot").resolve()
    (root / "nested").mkdir(parents=True)
    (root / "config.json").write_text('{"model":"fixture"}\n', encoding="utf-8")
    (root / "nested" / "weights.bin").write_bytes(role.encode())
    (root / "empty.marker").write_bytes(b"")
    return TrustedModelSnapshotSpec(
        model_id=f"fixture/{role}",
        revision=f"{role}-revision",
        role=role,  # type: ignore[arg-type]
        stages=("E0",),
        local_snapshot_path=str(root),
    )


def _fake_locked_workload(tmp_path: Path, workload_id: str) -> TrustedLockedWorkload:
    path = (tmp_path / f"{workload_id}.jsonl").resolve()
    path.write_text("{}\n", encoding="utf-8")
    metadata = (
        RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA
        if workload_id == "livecodebench_v6_hard"
        else RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA
    )
    selected_ids = (
        metadata.selected_question_ids
        if workload_id == "livecodebench_v6_hard"
        else metadata.selected_source_row_ids
    )
    return TrustedLockedWorkload(
        workload_id=workload_id,  # type: ignore[arg-type]
        raw_source_path=str(path),
        raw_file_size=path.stat().st_size,
        raw_file_sha256=metadata.raw_file_sha256,
        repository_revision=metadata.repository_revision,
        raw_row_count=metadata.raw_row_count,
        selected_row_count=metadata.selected_row_count,
        selected_source_row_ids=selected_ids,
        selected_raw_rows_sha256=metadata.selected_raw_rows_sha256,
        formal_samples_sha256=metadata.formal_samples_sha256,
        protocol_sha256=metadata.protocol_sha256,
        source_lock_sha256=metadata.source_lock_sha256,
        authority_sha256="a" * 64,
        verification_metadata_sha256=metadata.sha256,
        verification_metadata=metadata,
    )


def _tiny_burstgpt_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    names = (
        "BurstGPT_1.csv",
        "BurstGPT_2.csv",
        "BurstGPT_3.csv",
        "BurstGPT_without_fails_1.csv",
        "BurstGPT_without_fails_2.csv",
        BURSTGPT_V2_ACTIVE_ASSET,
    )
    paths: dict[str, Path] = {}
    pins: list[OfficialBurstGptAsset] = []
    for name in sorted(names):
        path = (tmp_path / name).resolve()
        body = (
            b"Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            b"0.000001,fixture,1,2,3,ok\n"
            b"0.000002,fixture,2,3,5,ok\n"
        )
        path.write_bytes(body)
        paths[name] = path
        pins.append(
            OfficialBurstGptAsset(
                name=name,
                size=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
            )
        )
    pinned = tuple(sorted(pins, key=lambda row: row.name))
    monkeypatch.setattr(loads_module, "BURSTGPT_V2_ASSETS", pinned)
    monkeypatch.setattr(content_module, "BURSTGPT_V2_ASSETS", pinned)
    return paths


def _content_replay_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, tuple[TrustedModelSnapshotSpec, ...], tuple[Path, ...]]:
    repository = _source_repository(tmp_path)
    roots: list[Path] = []
    specs: list[TrustedModelSnapshotSpec] = []
    for index in range(19):
        root = (tmp_path / f"replay-model-{index:02d}").resolve()
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"model": index}) + "\n", encoding="utf-8"
        )
        (root / "weights.bin").write_bytes(f"weights-{index}".encode())
        roots.append(root)
        roles = ("target", "tokenizer") if index < 6 else ("drafter",)
        for role in roles:
            specs.append(
                TrustedModelSnapshotSpec(
                    model_id=f"fixture/model-{index:02d}",
                    revision=f"revision-{index:02d}",
                    role=role,  # type: ignore[arg-type]
                    stages=("E0",),
                    local_snapshot_path=str(root),
                )
            )
    ordered_specs = tuple(
        sorted(
            specs,
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )
    livecodebench = (tmp_path / "replay-livecodebench.jsonl").resolve()
    math500 = (tmp_path / "replay-math500.jsonl").resolve()
    inventory = (tmp_path / "replay-inventory.json").resolve()
    livecodebench.write_text("{}\n", encoding="utf-8")
    math500.write_text("{}\n", encoding="utf-8")
    inventory.write_text("{}\n", encoding="utf-8")
    burst = _tiny_burstgpt_assets(tmp_path, monkeypatch)
    receipt_path = (tmp_path / "content-replay-authority.json").resolve()
    spec = TrustedSingleOperatorContentPathSpec(
        schema_version=2,
        kind="trusted_single_operator_content_path_spec",
        repository_root=str(repository),
        model_specs=ordered_specs,
        livecodebench_raw_path=str(livecodebench),
        math500_raw_path=str(math500),
        burstgpt_asset_paths=tuple(
            TrustedNamedInputPath(name=name, absolute_path=str(path))
            for name, path in sorted(burst.items())
        ),
        e0_task_native_specs=(),
        inventory_path=str(inventory),
        doctor_path=str((tmp_path / "future-replay-doctor.json").resolve()),
        content_replay_authority_path=str(receipt_path),
    )
    spec_path = (tmp_path / "content-replay-path-spec.json").resolve()
    spec_path.write_text(
        json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return spec_path, receipt_path, ordered_specs, tuple(roots)


def _symlink_content_replay_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path]:
    repository = _source_repository(tmp_path)
    cache = (tmp_path / "models--fixture--symlink").resolve()
    revision = "f" * 40
    snapshot = cache / "snapshots" / revision
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    first = blobs / ("a" * 64)
    second = blobs / ("b" * 64)
    first.write_bytes(b"first")
    second.write_bytes(b"other")
    link = snapshot / "model.safetensors"
    link.symlink_to(f"../../blobs/{first.name}")
    livecodebench = (tmp_path / "symlink-livecodebench.jsonl").resolve()
    math500 = (tmp_path / "symlink-math500.jsonl").resolve()
    inventory = (tmp_path / "symlink-inventory.json").resolve()
    for path in (livecodebench, math500, inventory):
        path.write_text("{}\n", encoding="utf-8")
    burst = _tiny_burstgpt_assets(tmp_path, monkeypatch)
    receipt_path = (tmp_path / "symlink-content-replay.json").resolve()
    spec = TrustedSingleOperatorContentPathSpec(
        schema_version=2,
        kind="trusted_single_operator_content_path_spec",
        repository_root=str(repository),
        model_specs=(
            TrustedModelSnapshotSpec(
                model_id="fixture/symlink",
                revision=revision,
                role="drafter",
                stages=("E0",),
                local_snapshot_path=str(snapshot),
                storage_mode="huggingface_cache_symlinks",
                content_cache_root=str(cache),
            ),
        ),
        livecodebench_raw_path=str(livecodebench),
        math500_raw_path=str(math500),
        burstgpt_asset_paths=tuple(
            TrustedNamedInputPath(name=name, absolute_path=str(path))
            for name, path in sorted(burst.items())
        ),
        e0_task_native_specs=(),
        inventory_path=str(inventory),
        doctor_path=str((tmp_path / "future-symlink-doctor.json").resolve()),
        content_replay_authority_path=str(receipt_path),
    )
    spec_path = (tmp_path / "symlink-content-path-spec.json").resolve()
    spec_path.write_text(
        json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return spec_path, receipt_path, first, second, link


def test_model_snapshot_scans_all_regular_files_and_rejects_aliases(
    tmp_path: Path,
) -> None:
    spec = _model_spec(tmp_path, "target")
    member = bind_trusted_model_snapshot_member(spec)

    assert tuple(row.relative_path for row in member.files) == (
        "config.json",
        "empty.marker",
        "nested/weights.bin",
    )
    assert TrustedModelSnapshotMember.from_dict(member.to_dict()) == member
    assert member.tree_sha256 != member.content_sha256

    (Path(spec.local_snapshot_path) / "config.json").write_text(
        '{"model":"changed"}\n',
        encoding="utf-8",
    )
    assert bind_trusted_model_snapshot_member(spec) != member

    link = Path(spec.local_snapshot_path) / "link"
    link.symlink_to(Path(spec.local_snapshot_path) / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        bind_trusted_model_snapshot_member(spec)


def test_model_runtime_bindings_separate_builtin_e6_from_external_e0(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "e6-target").resolve()
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    e6 = TrustedModelRuntimeBinding(
        stage="E6",
        target_model_id="Qwen/Qwen3.6-35B-A3B",
        backend="NEXTN",
        draft_depth=1,
    )
    member = bind_trusted_model_snapshot_member(
        TrustedModelSnapshotSpec(
            model_id="Qwen/Qwen3.6-35B-A3B",
            revision="1" * 40,
            role="target",
            stages=("E6",),
            local_snapshot_path=str(root),
            runtime_bindings=(e6,),
        )
    )
    assert member.runtime_bindings == (e6,)
    assert TrustedModelSnapshotMember.from_dict(member.to_dict()) == member

    with pytest.raises(ValueError, match="runtime bindings"):
        TrustedModelSnapshotSpec(
            model_id="fixture/external-drafter",
            revision="2" * 40,
            role="drafter",
            stages=("E6",),
            local_snapshot_path=str(root),
            runtime_bindings=(e6,),
        )
    e0 = TrustedModelRuntimeBinding(
        stage="E0",
        target_model_id="Qwen/Qwen3-4B",
        backend="DFLASH",
        draft_depth=1,
    )
    with pytest.raises(ValueError, match="runtime bindings"):
        TrustedModelSnapshotSpec(
            model_id="Qwen/Qwen3-4B",
            revision="3" * 40,
            role="target",
            stages=("E0",),
            local_snapshot_path=str(root),
            runtime_bindings=(e0,),
        )


def test_preflight_runtime_binding_requires_real_registered_drafter(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "preflight-dspark-drafter").resolve()
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    dspark = TrustedModelRuntimeBinding(
        stage="preflight",
        target_model_id="Qwen/Qwen3-8B",
        backend="DSPARK",
        draft_depth=15,
    )
    member = bind_trusted_model_snapshot_member(
        TrustedModelSnapshotSpec(
            model_id="z-lab/Qwen3-8B-DSpark",
            revision="4" * 40,
            role="drafter",
            stages=("preflight",),
            local_snapshot_path=str(root),
            runtime_bindings=(dspark,),
        )
    )
    assert member.runtime_bindings == (dspark,)

    with pytest.raises(ValueError, match="runtime bindings"):
        TrustedModelSnapshotSpec(
            model_id="z-lab/Qwen3-8B-DSpark",
            revision="4" * 40,
            role="target",
            stages=("preflight",),
            local_snapshot_path=str(root),
            runtime_bindings=(dspark,),
        )
    with pytest.raises(ValueError, match="preflight runtime binding"):
        TrustedModelRuntimeBinding(
            stage="preflight",
            target_model_id="Qwen/Qwen3-8B",
            backend="DSPARK",
            draft_depth=14,
        )


def test_model_snapshot_rejects_duplicate_hardlinks(tmp_path: Path) -> None:
    spec = _model_spec(tmp_path, "drafter")
    original = Path(spec.local_snapshot_path) / "config.json"
    os.link(original, Path(spec.local_snapshot_path) / "duplicate.json")

    with pytest.raises(ValueError, match="duplicate hard-linked"):
        bind_trusted_model_snapshot_member(spec)


def test_huggingface_snapshot_symlink_farm_is_bound_and_revalidated(
    tmp_path: Path,
) -> None:
    cache_root = (tmp_path / "models--fixture--target").resolve()
    revision = "1" * 40
    snapshot = cache_root / "snapshots" / revision
    blobs = cache_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    first_blob = blobs / ("a" * 64)
    first_blob.write_bytes(b"immutable-model-bytes")
    config_link = snapshot / "config.json"
    config_link.symlink_to(f"../../blobs/{first_blob.name}")
    spec = TrustedModelSnapshotSpec(
        model_id="fixture/target",
        revision=revision,
        role="target",
        stages=("E0",),
        local_snapshot_path=str(snapshot),
        storage_mode="huggingface_cache_symlinks",
        content_cache_root=str(cache_root),
    )

    member = bind_trusted_model_snapshot_member(spec)

    assert member.files[0].storage_kind == "symlinked_blob"
    assert member.files[0].symlink_target == f"../../blobs/{first_blob.name}"
    assert member.files[0].resolved_relative_path == f"blobs/{first_blob.name}"
    assert revalidate_trusted_model_snapshot_member(member) == member

    second_blob = blobs / ("b" * 64)
    second_blob.write_bytes(b"different-model-bytes")
    config_link.unlink()
    config_link.symlink_to(f"../../blobs/{second_blob.name}")
    with pytest.raises(RuntimeError, match="snapshot member changed"):
        revalidate_trusted_model_snapshot_member(member)


def test_huggingface_snapshot_rejects_escaping_symlink(tmp_path: Path) -> None:
    cache_root = (tmp_path / "models--fixture--target").resolve()
    revision = "2" * 40
    snapshot = cache_root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    outside = (tmp_path / "outside.bin").resolve()
    outside.write_bytes(b"outside")
    (snapshot / "weights.bin").symlink_to(outside)
    spec = TrustedModelSnapshotSpec(
        model_id="fixture/target",
        revision=revision,
        role="target",
        stages=("E0",),
        local_snapshot_path=str(snapshot),
        storage_mode="huggingface_cache_symlinks",
        content_cache_root=str(cache_root),
    )

    with pytest.raises(ValueError, match="leaves its bound cache root"):
        bind_trusted_model_snapshot_member(spec)


def test_content_replay_deep_scans_19_unique_roots_then_reads_zero_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, specs, roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    deep_paths: list[str] = []
    payload_paths: list[str] = []
    original_deep = content_module._deep_snapshot_replay_closure
    original_digest = content_module._stable_file_digest

    def deep(spec: TrustedModelSnapshotSpec):
        deep_paths.append(spec.local_snapshot_path)
        return original_deep(spec)

    def digest(path: Path, *, label: str):
        if any(path.is_relative_to(root) for root in roots):
            payload_paths.append(str(path))
        return original_digest(path, label=label)

    monkeypatch.setattr(content_module, "_deep_snapshot_replay_closure", deep)
    monkeypatch.setattr(content_module, "_stable_file_digest", digest)
    binding = publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    assert len(specs) == 25
    assert len(deep_paths) == len(set(deep_paths)) == 19
    assert len(payload_paths) == len(set(payload_paths)) == 38
    authority = binding.reopen()
    assert len(authority.snapshot_closures) == 19
    assert len(authority.model_members) == 25
    assert all(
        Path(row.snapshot_mount.mount_point).is_absolute()
        for row in authority.snapshot_closures
    )
    assert authority.member(model_id="fixture/model-00", role="target").files == (
        authority.member(model_id="fixture/model-00", role="tokenizer").files
    )

    def forbid_model_payload(path: Path, *, label: str):
        if any(path.is_relative_to(root) for root in roots):
            raise AssertionError(f"model payload reopened: {path}")
        return original_digest(path, label=label)

    monkeypatch.setattr(content_module, "_stable_file_digest", forbid_model_payload)
    rebound = TrustedSingleOperatorContentReplayAuthorityBinding.bind(
        receipt_path,
        expected_content_path_spec_path=spec_path,
    )
    reopened = rebound.reopen()
    assert (
        revalidate_trusted_single_operator_content_replay_authority(reopened)
        == reopened
    )
    assert reopened.closure_for_member(
        model_id="fixture/model-00", role="tokenizer"
    ).local_snapshot_path == str(roots[0])


def test_content_replay_protocol_identity_freezes_mount_replay_semantics() -> None:
    assert TRUSTED_SINGLE_OPERATOR_CONTENT_REPLAY_PROTOCOL_SHA256 == (
        "c96caede80e3a88358f643b2068bbfb1dc48f8b82f39f37ac5aeccb54d9cd0d5"
    )


def test_content_bundle_binding_projects_replay_without_model_payload_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, _specs, roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    spec = load_trusted_single_operator_content_path_spec(spec_path)
    workloads = {
        workload_id: _fake_locked_workload(tmp_path, workload_id)
        for workload_id in ("livecodebench_v6_hard", "math500_level5")
    }
    monkeypatch.setattr(
        content_module,
        "bind_trusted_locked_workload",
        lambda workload_id, _path: workloads[workload_id],
    )
    original_digest = content_module._stable_file_digest

    def forbid_model_payload(path: Path, *, label: str):
        if any(path.is_relative_to(root) for root in roots):
            raise AssertionError(f"model payload reopened: {path}")
        return original_digest(path, label=label)

    monkeypatch.setattr(content_module, "_stable_file_digest", forbid_model_payload)
    bundle = build_trusted_single_operator_content_bundle(
        repository_root=spec.repository_root,
        model_specs=spec.model_specs,
        livecodebench_raw_path=spec.livecodebench_raw_path,
        math500_raw_path=spec.math500_raw_path,
        burstgpt_asset_paths={
            row.name: row.absolute_path for row in spec.burstgpt_asset_paths
        },
        content_path_spec_path=spec_path,
        content_replay_authority_path=receipt_path,
    )
    assert bundle.schema_version == 2
    assert bundle.content_replay_authority is not None
    assert revalidate_trusted_single_operator_content_bundle(bundle) == bundle
    output = (tmp_path / "replay-content-bundle.json").resolve()
    publish_trusted_single_operator_content_bundle(bundle, output)
    binding = TrustedSingleOperatorContentBundleBinding.bind(output)
    assert binding.reopen() == bundle


@pytest.mark.parametrize(
    "mutation",
    ("in_place_mtime_restore", "replace_inode", "add", "remove", "chmod"),
)
def test_content_replay_blocks_every_snapshot_identity_or_namespace_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    spec_path, receipt_path, _specs, roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    binding = publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    target = roots[0] / "weights.bin"
    before = target.stat(follow_symlinks=False)
    if mutation == "in_place_mtime_restore":
        target.write_bytes(b"changed-0")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert target.stat(follow_symlinks=False).st_mtime_ns == before.st_mtime_ns
        assert target.stat(follow_symlinks=False).st_ctime_ns != before.st_ctime_ns
    elif mutation == "replace_inode":
        replacement = roots[0] / "replacement.bin"
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
        assert target.stat(follow_symlinks=False).st_ino != before.st_ino
    elif mutation == "add":
        (roots[0] / "foreign.bin").write_bytes(b"foreign")
    elif mutation == "remove":
        target.unlink()
    else:
        target.chmod(before.st_mode ^ stat.S_IXUSR)
        assert target.stat(follow_symlinks=False).st_mode != before.st_mode
    with pytest.raises(TrustedSingleOperatorContentReplayBlocked):
        binding.reopen()


def test_content_replay_blocks_symlink_retarget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, _first, second, link = _symlink_content_replay_fixture(
        tmp_path, monkeypatch
    )
    binding = publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    link.unlink()
    link.symlink_to(f"../../blobs/{second.name}")
    with pytest.raises(TrustedSingleOperatorContentReplayBlocked):
        binding.reopen()


def test_content_replay_blocks_in_place_resolved_blob_change_with_restored_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, first, _second, _link = _symlink_content_replay_fixture(
        tmp_path, monkeypatch
    )
    binding = publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    before = first.stat(follow_symlinks=False)
    first.write_bytes(b"other")
    os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = first.stat(follow_symlinks=False)
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns

    with pytest.raises(TrustedSingleOperatorContentReplayBlocked):
        binding.reopen()


def test_content_replay_blocks_canonical_path_spec_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, _specs, _roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    binding = publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    raw["doctor_path"] = str((tmp_path / "foreign-doctor.json").resolve())
    replacement = (tmp_path / "replacement-content-path-spec.json").resolve()
    replacement.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(replacement, spec_path)

    with pytest.raises(ValueError, match="identity changed"):
        binding.reopen()


def test_content_replay_blocks_mount_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, _specs, _roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    binding = publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    original = content_module._content_replay_mount_identity

    def drift(path: Path):
        identity = original(path)
        return replace(identity, flags=identity.flags ^ 1)

    monkeypatch.setattr(content_module, "_content_replay_mount_identity", drift)
    with pytest.raises(TrustedSingleOperatorContentReplayBlocked):
        binding.reopen()


def test_content_replay_rejects_receipt_tamper_and_path_splice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, _specs, _roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=spec_path,
        output_path=receipt_path,
    )
    foreign = (tmp_path / "foreign-replay.json").resolve()
    foreign.write_bytes(receipt_path.read_bytes())
    with pytest.raises(ValueError, match="path differs"):
        TrustedSingleOperatorContentReplayAuthorityBinding.bind(foreign)

    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw["source_git_tree"] = "0" * 40
    receipt_path.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest differs"):
        TrustedSingleOperatorContentReplayAuthorityBinding.bind(receipt_path)


def test_content_replay_publication_is_no_replace_partial_safe_and_concurrent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, receipt_path, _specs, _roots = _content_replay_fixture(
        tmp_path, monkeypatch
    )
    original_deep = content_module._deep_snapshot_replay_closure
    calls = 0

    def interrupted(spec: TrustedModelSnapshotSpec):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected replay interruption")
        return original_deep(spec)

    monkeypatch.setattr(content_module, "_deep_snapshot_replay_closure", interrupted)
    with pytest.raises(RuntimeError, match="injected replay interruption"):
        publish_trusted_single_operator_content_replay_authority_from_spec(
            spec_path=spec_path,
            output_path=receipt_path,
        )
    assert not receipt_path.exists()
    monkeypatch.setattr(content_module, "_deep_snapshot_replay_closure", original_deep)

    original_load = content_module.load_trusted_single_operator_content_replay_authority

    def fail_post_publication(_path: str | Path):
        raise RuntimeError("injected replay post-publication failure")

    monkeypatch.setattr(
        content_module,
        "load_trusted_single_operator_content_replay_authority",
        fail_post_publication,
    )
    with pytest.raises(RuntimeError, match="post-publication failure"):
        publish_trusted_single_operator_content_replay_authority_from_spec(
            spec_path=spec_path,
            output_path=receipt_path,
        )
    assert not receipt_path.exists()
    assert not tuple(receipt_path.parent.glob(f".{receipt_path.name}.*.partial"))
    monkeypatch.setattr(
        content_module,
        "load_trusted_single_operator_content_replay_authority",
        original_load,
    )

    def publish():
        try:
            return publish_trusted_single_operator_content_replay_authority_from_spec(
                spec_path=spec_path,
                output_path=receipt_path,
            )
        except FileExistsError as error:  # returned for exact race assertions
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _index: publish(), range(2)))
    assert (
        sum(
            isinstance(row, TrustedSingleOperatorContentReplayAuthorityBinding)
            for row in results
        )
        == 1
    )
    assert sum(isinstance(row, FileExistsError) for row in results) == 1
    body = receipt_path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_trusted_single_operator_content_replay_authority_from_spec(
            spec_path=spec_path,
            output_path=receipt_path,
        )
    assert receipt_path.read_bytes() == body


def test_source_snapshot_binds_clean_git_and_eight_patch_bytes(tmp_path: Path) -> None:
    root = _source_repository(tmp_path)
    snapshot = bind_trusted_source_snapshot(root)

    assert snapshot.repository_root == str(root)
    assert snapshot.git_head == _git(root, "rev-parse", "HEAD")
    assert snapshot.git_tree == _git(root, "rev-parse", "HEAD^{tree}")
    assert len(snapshot.patches) == 8
    assert TrustedSourceSnapshot.from_dict(snapshot.to_dict()) == snapshot

    (root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git checkout"):
        bind_trusted_source_snapshot(root)


def test_json_artifact_is_path_bound_and_deep_revalidated(tmp_path: Path) -> None:
    path = (tmp_path / "inventory.json").resolve()
    path.write_text('{"gpu":["GPU-1"]}\n', encoding="utf-8")
    binding = bind_trusted_json_artifact("inventory", path)

    assert revalidate_trusted_json_artifact(binding) == binding
    path.write_text('{"gpu":["GPU-2"]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        revalidate_trusted_json_artifact(binding)

    duplicate = (tmp_path / "duplicate.json").resolve()
    duplicate.write_text('{"gpu":1,"gpu":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bind_trusted_json_artifact("duplicate", duplicate)


def test_release_workloads_replay_when_explicit_cache_is_available() -> None:
    cache_value = os.environ.get("LIGHTCONE_CONTENT_SOURCE_CACHE")
    if cache_value is None:
        pytest.skip("set LIGHTCONE_CONTENT_SOURCE_CACHE for release replay")
    cache = Path(cache_value).resolve(strict=True)

    livecodebench = bind_trusted_locked_workload(
        "livecodebench_v6_hard",
        cache / "livecodebench-code_generation_lite-0fe84c3" / "test6.jsonl",
    )
    math = bind_trusted_locked_workload(
        "math500_level5",
        cache / "math-500-6e4ed1a" / "test.jsonl",
    )

    assert livecodebench.selected_row_count == 80
    assert math.selected_row_count == 134
    assert math.verification_metadata.filter_value == 5


def test_bundle_codec_runtime_binding_no_replace_and_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _source_repository(tmp_path)
    specs = tuple(
        _model_spec(tmp_path, role) for role in ("target", "drafter", "tokenizer")
    )
    workloads = {
        workload_id: _fake_locked_workload(tmp_path, workload_id)
        for workload_id in ("livecodebench_v6_hard", "math500_level5")
    }
    monkeypatch.setattr(
        content_module,
        "bind_trusted_locked_workload",
        lambda workload_id, _path: workloads[workload_id],
    )
    burst_paths = _tiny_burstgpt_assets(tmp_path, monkeypatch)

    bundle = build_trusted_single_operator_content_bundle(
        repository_root=root,
        model_specs=specs,
        livecodebench_raw_path=workloads["livecodebench_v6_hard"].raw_source_path,
        math500_raw_path=workloads["math500_level5"].raw_source_path,
        burstgpt_asset_paths=burst_paths,
    )
    assert bundle.signature is None
    assert bundle.formal_measured_authorization is False
    assert bundle.runtime_binding_status == "PENDING_REMOTE_BINDING"
    assert TrustedSingleOperatorContentBundle.from_dict(bundle.to_dict()) == bundle
    assert bundle.protocol_lock_content_sha256 == bundle.semantic_sha256

    inventory = (tmp_path / "inventory.json").resolve()
    doctor = (tmp_path / "doctor.json").resolve()
    inventory.write_text('{"gpu_uuid":"GPU-test"}\n', encoding="utf-8")
    doctor.write_text('{"driver":"test"}\n', encoding="utf-8")
    bound = bind_trusted_single_operator_runtime_observations(
        bundle,
        inventory_path=inventory,
        doctor_path=doctor,
    )
    assert bound.runtime_binding_status == "BOUND"
    assert type(bound.runtime_observations) is TrustedRuntimeObservations
    assert bound.semantic_sha256 != bundle.semantic_sha256

    output = (tmp_path.parent / f"{tmp_path.name}-content.json").resolve()
    publish_trusted_single_operator_content_bundle(bound, output)
    assert load_trusted_single_operator_content_bundle(output) == bound
    reference = TrustedSingleOperatorContentBundleBinding.bind(output)
    assert reference.semantic_sha256 == bound.semantic_sha256
    assert reference.reopen() == bound

    with pytest.raises(FileExistsError):
        publish_trusted_single_operator_content_bundle(bound, output)

    doctor.write_text('{"driver":"changed"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        reference.reopen()


def test_runtime_bound_publisher_rejects_incomplete_v03_spec_before_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _source_repository(tmp_path)
    specs = tuple(
        sorted(
            (
                _model_spec(tmp_path, role)
                for role in ("target", "drafter", "tokenizer")
            ),
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )
    workloads = {
        workload_id: _fake_locked_workload(tmp_path, workload_id)
        for workload_id in ("livecodebench_v6_hard", "math500_level5")
    }
    monkeypatch.setattr(
        content_module,
        "bind_trusted_locked_workload",
        lambda workload_id, _path: workloads[workload_id],
    )
    burst_paths = _tiny_burstgpt_assets(tmp_path, monkeypatch)
    inventory = (tmp_path / "remote-inventory.json").resolve()
    doctor = (tmp_path / "remote-doctor.json").resolve()
    inventory.write_text('{"gpu_uuid":"GPU-test"}\n', encoding="utf-8")
    doctor.write_text('{"status":"PASS"}\n', encoding="utf-8")
    spec = TrustedSingleOperatorContentPathSpec(
        schema_version=1,
        kind="trusted_single_operator_content_path_spec",
        repository_root=str(root),
        model_specs=specs,
        livecodebench_raw_path=workloads["livecodebench_v6_hard"].raw_source_path,
        math500_raw_path=workloads["math500_level5"].raw_source_path,
        burstgpt_asset_paths=tuple(
            TrustedNamedInputPath(name=name, absolute_path=str(path))
            for name, path in sorted(burst_paths.items())
        ),
        e0_task_native_specs=(),
        inventory_path=str(inventory),
        doctor_path=str(doctor),
    )
    spec_path = (tmp_path / "content-path-spec.json").resolve()
    spec_path.write_text(
        json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output = (tmp_path / "runtime-bound-content.json").resolve()

    assert load_trusted_single_operator_content_path_spec(spec_path) == spec
    assert "sha256" not in spec.to_dict()
    with pytest.raises(ValueError, match="model member coverage differs"):
        publish_runtime_bound_trusted_single_operator_content_from_spec(
            spec_path=spec_path,
            output_path=output,
        )
    assert not output.exists()

    mutated = dict(spec.to_dict())
    mutated["inventory_path"] = str(doctor)
    spec_path.write_text(
        json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must differ"):
        load_trusted_single_operator_content_path_spec(spec_path)


def test_trusted_preflight_workload_publisher_has_no_scientific_inputs() -> None:
    assert tuple(
        inspect.signature(
            publish_trusted_preflight_workload_authority_from_content
        ).parameters
    ) == ("trusted_content_bundle_path", "output_path")
