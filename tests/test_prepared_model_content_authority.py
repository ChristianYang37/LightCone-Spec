from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest
from test_formal_single_operator_content import (
    _fake_locked_workload,
    _source_repository,
    _tiny_burstgpt_assets,
)

import lightcone_spec.experiments.formal_single_operator_content as content_module
import lightcone_spec.locking.prepared_models as prepared_module
from lightcone_spec.experiments.formal_single_operator_content import (
    TrustedModelSnapshotSpec,
    TrustedNamedInputPath,
    TrustedSingleOperatorContentBundleBinding,
    TrustedSingleOperatorContentPathSpec,
    TrustedSingleOperatorContentReplayBlocked,
    bind_trusted_single_operator_runtime_observations,
    build_trusted_single_operator_content_bundle,
    publish_trusted_single_operator_content_bundle,
    publish_trusted_single_operator_content_replay_authority_from_spec,
)
from lightcone_spec.locking import (
    ModelLock,
    PreparedModelContentAuthorityBinding,
    PreparedModelContentAuthorityBlocked,
    bind_prepared_model_content_authority,
    bind_prepared_models,
    materialize_prepared_model_content_manifest,
    materialize_trusted_prepared_model_content_manifest,
    prepared_model_content_authority_from_dict,
    prepared_model_content_authority_to_dict,
    revalidate_prepared_model_content_authority,
)
from lightcone_spec.locking.models import LockedModel

TARGET_ID = "Qwen/Qwen3-8B"
DRAFTER_ID = "z-lab/Qwen3-8B-DFlash-b16"
TARGET_REVISION = "1" * 40
DRAFTER_REVISION = "2" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_manifest(path: Path, value: object) -> str:
    body = _canonical(value)
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    Path(f"{path}.sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
) -> None:
    sizes = {"BF16": 2, "F32": 4}
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        count = 1
        for dimension in shape:
            count *= dimension
        end = offset + count * sizes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end
    encoded = _canonical(header)
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _fixture(tmp_path: Path) -> tuple[ModelLock, object, dict[str, Path]]:
    lock = ModelLock(
        schema_version=2,
        models=(
            LockedModel(TARGET_ID, TARGET_REVISION),
            LockedModel(DRAFTER_ID, DRAFTER_REVISION),
        ),
    )
    target = (tmp_path / "target" / "snapshots" / TARGET_REVISION).resolve()
    drafter = (tmp_path / "drafter" / "snapshots" / DRAFTER_REVISION).resolve()
    target.mkdir(parents=True)
    drafter.mkdir(parents=True)
    target_files = {
        "config.json": b'{"model_type":"qwen3"}',
        "generation_config.json": b'{"do_sample":false}',
        "merges.txt": b"#version: 0.2\na b\n",
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b'{"model_max_length":40960}',
        "vocab.json": b'{"a":0,"b":1}',
    }
    for name, body in target_files.items():
        (target / name).write_bytes(body)
    first = {
        "model.embed_tokens.weight": ("BF16", (4, 2)),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", (2, 2)),
    }
    second = {"lm_head.weight": ("BF16", (4, 2))}
    _write_safetensors(target / "model-00001-of-00002.safetensors", first)
    _write_safetensors(target / "model-00002-of-00002.safetensors", second)
    total_size = sum(
        2 * dimension
        for _, shape in (*first.values(), *second.values())
        for dimension in (int(__import__("math").prod(shape)),)
    )
    weight_map = {name: "model-00001-of-00002.safetensors" for name in first} | {
        name: "model-00002-of-00002.safetensors" for name in second
    }
    (target / "model.safetensors.index.json").write_bytes(
        _canonical({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )

    for name, body in {
        "config.json": b'{"model_type":"dflash"}',
        "dflash.py": b"class DFlash: pass\n",
        "modeling_dflash.py": b"class DFlashModel: pass\n",
        "utils.py": b"BLOCK_SIZE = 16\n",
    }.items():
        (drafter / name).write_bytes(body)
    _write_safetensors(
        drafter / "model.safetensors",
        {
            "layers.0.input_layernorm.weight": ("F32", (4,)),
            "layers.0.self_attn.q_proj.weight": ("BF16", (8, 4)),
            "lm_head.weight": ("BF16", (32, 4)),
            "target_model.layers.0.weight": ("BF16", (4, 4)),
        },
    )
    prepared = bind_prepared_models(
        lock,
        {TARGET_ID: target, DRAFTER_ID: drafter},
    )
    return lock, prepared, {"target": target, "drafter": drafter}


def _authority(tmp_path: Path) -> tuple[ModelLock, object, object, dict[str, Path]]:
    lock, prepared, roots = _fixture(tmp_path)
    manifest = materialize_prepared_model_content_manifest(lock, prepared)
    manifest_path = (tmp_path / "prepared-content.json").resolve()
    digest = _write_manifest(manifest_path, manifest)
    authority = bind_prepared_model_content_authority(
        lock,
        prepared,
        manifest_path,
        expected_release_manifest_sha256=digest,
    )
    return lock, prepared, authority, {**roots, "manifest": manifest_path}


def _trusted_bundle_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ModelLock,
    object,
    TrustedSingleOperatorContentBundleBinding,
    dict[str, Path],
]:
    lock, prepared, roots = _fixture(tmp_path / "prepared")
    repository = _source_repository(tmp_path)
    livecodebench = (tmp_path / "livecodebench.jsonl").resolve()
    math500 = (tmp_path / "math500.jsonl").resolve()
    inventory = (tmp_path / "inventory.json").resolve()
    runtime_doctor = (tmp_path / "runtime-doctor.json").resolve()
    for path in (livecodebench, math500, inventory, runtime_doctor):
        path.write_text("{}\n", encoding="utf-8")
    burst = _tiny_burstgpt_assets(tmp_path, monkeypatch)
    stages = ("preflight", "TTS-Cal", "E1")
    specs = tuple(
        sorted(
            (
                TrustedModelSnapshotSpec(
                    model_id=TARGET_ID,
                    revision=TARGET_REVISION,
                    role="target",
                    stages=stages,
                    local_snapshot_path=str(roots["target"]),
                ),
                TrustedModelSnapshotSpec(
                    model_id=TARGET_ID,
                    revision=TARGET_REVISION,
                    role="tokenizer",
                    stages=stages,
                    local_snapshot_path=str(roots["target"]),
                ),
                TrustedModelSnapshotSpec(
                    model_id=DRAFTER_ID,
                    revision=DRAFTER_REVISION,
                    role="drafter",
                    stages=stages,
                    local_snapshot_path=str(roots["drafter"]),
                ),
            ),
            key=lambda row: (
                row.role,
                row.model_id,
                row.revision,
                row.local_snapshot_path,
            ),
        )
    )
    replay_path = (tmp_path / "content-replay.json").resolve()
    path_spec = TrustedSingleOperatorContentPathSpec(
        schema_version=2,
        kind="trusted_single_operator_content_path_spec",
        repository_root=str(repository),
        model_specs=specs,
        livecodebench_raw_path=str(livecodebench),
        math500_raw_path=str(math500),
        burstgpt_asset_paths=tuple(
            TrustedNamedInputPath(name=name, absolute_path=str(path))
            for name, path in sorted(burst.items())
        ),
        e0_task_native_specs=(),
        inventory_path=str(inventory),
        doctor_path=str((tmp_path / "future-doctor.json").resolve()),
        content_replay_authority_path=str(replay_path),
    )
    path_spec_path = (tmp_path / "content-path-spec.json").resolve()
    path_spec_path.write_text(
        json.dumps(path_spec.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    publish_trusted_single_operator_content_replay_authority_from_spec(
        spec_path=path_spec_path,
        output_path=replay_path,
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
    bundle = build_trusted_single_operator_content_bundle(
        repository_root=repository,
        model_specs=specs,
        livecodebench_raw_path=livecodebench,
        math500_raw_path=math500,
        burstgpt_asset_paths=burst,
        content_path_spec_path=path_spec_path,
        content_replay_authority_path=replay_path,
    )
    bound = bind_trusted_single_operator_runtime_observations(
        bundle,
        inventory_path=inventory,
        doctor_path=runtime_doctor,
    )
    bundle_path = (tmp_path / "content-bundle.json").resolve()
    publish_trusted_single_operator_content_bundle(bound, bundle_path)
    return (
        lock,
        prepared,
        TrustedSingleOperatorContentBundleBinding.bind(bundle_path),
        roots,
    )


def test_content_authority_replays_critical_files_and_safetensors_headers(
    tmp_path: Path,
) -> None:
    lock, _, authority, _ = _authority(tmp_path)
    assert isinstance(authority, PreparedModelContentAuthorityBinding)
    encoded = prepared_model_content_authority_to_dict(authority)
    decoded = prepared_model_content_authority_from_dict(encoded)
    assert decoded == authority
    with pytest.raises(ValueError, match="fields differ"):
        prepared_model_content_authority_from_dict({**encoded, "summary": {}})
    replay = revalidate_prepared_model_content_authority(
        lock,
        decoded,
        expected_release_manifest_sha256=authority.release_manifest_sha256,
    )
    drafter = replay.snapshot(DRAFTER_ID)
    assert tuple(tensor.name for tensor in drafter.tensors) == (
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
        "target_model.layers.0.weight",
    )
    assert drafter.tensors[1].shape == (8, 4)
    assert drafter.tensors[1].dtype == "torch.bfloat16"
    assert (
        authority.revalidate(
            lock,
            expected_release_manifest_sha256=authority.release_manifest_sha256,
        )
        == replay
    )


def test_generic_profile_and_missing_required_file_are_fail_closed(
    tmp_path: Path,
) -> None:
    lock, prepared, roots = _fixture(tmp_path)
    (roots["drafter"] / "utils.py").unlink()
    with pytest.raises(
        PreparedModelContentAuthorityBlocked,
        match="prepared_model_content_required_files_unavailable",
    ):
        materialize_prepared_model_content_manifest(lock, prepared)

    unknown_lock = ModelLock(2, (LockedModel("unknown/model", "3" * 40),))
    unknown_root = (tmp_path / "unknown" / "snapshots" / ("3" * 40)).resolve()
    unknown_root.mkdir(parents=True)
    unknown = bind_prepared_models(unknown_lock, {"unknown/model": unknown_root})
    with pytest.raises(
        PreparedModelContentAuthorityBlocked,
        match="prepared_model_content_required_files_unavailable",
    ):
        materialize_prepared_model_content_manifest(unknown_lock, unknown)

    (unknown_root / "config.json").write_bytes(b'{"model_type":"unknown"}')
    (unknown_root / "source.py").write_bytes(b"MODEL = 'unknown'\n")
    _write_safetensors(
        unknown_root / "model.safetensors",
        {"model.layers.0.weight": ("BF16", (2, 2))},
    )
    manifest = materialize_prepared_model_content_manifest(unknown_lock, unknown)
    snapshot = manifest["snapshots"][0]
    assert snapshot["profile"] == "generic_complete_lightweight_safetensors_v1"
    assert [row["relative_path"] for row in snapshot["critical_files"]] == [
        "config.json",
        "source.py",
    ]


def test_snapshot_sources_reject_symlink_hardlink_and_manifest_path_escape(
    tmp_path: Path,
) -> None:
    lock, prepared, roots = _fixture(tmp_path)
    config = roots["drafter"] / "config.json"
    original = roots["drafter"] / "config-real.json"
    config.rename(original)
    config.symlink_to(original)
    with pytest.raises(ValueError, match="unsafe cache link|hardlink"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, roots = _fixture(tmp_path / "hardlink")
    config = roots["drafter"] / "config.json"
    alias = roots["drafter"] / "config-hardlink.json"
    os.link(config, alias)
    with pytest.raises(ValueError, match="hardlink"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, _, paths = _authority(tmp_path / "escape")
    manifest = json.loads(paths["manifest"].read_text())
    manifest["snapshots"][0]["critical_files"][0]["relative_path"] = "../escape"
    digest = _write_manifest(paths["manifest"], manifest)
    with pytest.raises(ValueError, match="snapshot-relative"):
        bind_prepared_model_content_authority(
            lock,
            prepared,
            paths["manifest"],
            expected_release_manifest_sha256=digest,
        )

    duplicate = (tmp_path / "duplicate.json").resolve()
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    Path(f"{duplicate}.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        bind_prepared_model_content_authority(
            lock,
            prepared,
            duplicate,
            expected_release_manifest_sha256="0" * 64,
        )


def _hf_cache_generic_fixture(
    tmp_path: Path,
) -> tuple[ModelLock, object, Path, dict[str, Path]]:
    model_id = "example/Generic-Model"
    revision = "4" * 40
    repository = (tmp_path / "models--example--Generic-Model").resolve()
    snapshot = repository / "snapshots" / revision
    blobs = repository / "blobs"
    (snapshot / "code").mkdir(parents=True)
    blobs.mkdir()

    def linked(relative: str, body: bytes) -> Path:
        digest = hashlib.sha256(body).hexdigest()
        blob = blobs / digest
        blob.write_bytes(body)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        depth = len(Path(relative).parts) - 1
        destination.symlink_to(Path(*([".."] * (2 + depth))) / "blobs" / digest)
        return blob

    config_blob = linked("config.json", b'{"model_type":"generic"}')
    source_blob = linked("code/source.py", b"MODEL = 'generic'\n")
    temporary_weight = tmp_path / "weight.tmp"
    _write_safetensors(
        temporary_weight,
        {"model.layers.0.weight": ("BF16", (2, 2))},
    )
    weight_blob = linked("model.safetensors", temporary_weight.read_bytes())
    temporary_weight.unlink()
    lock = ModelLock(2, (LockedModel(model_id, revision),))
    prepared = bind_prepared_models(lock, {model_id: snapshot})
    return (
        lock,
        prepared,
        snapshot,
        {
            "config": config_blob,
            "source": source_blob,
            "weight": weight_blob,
        },
    )


def test_hf_cache_links_are_repo_local_content_addressed_and_toctou_bound(
    tmp_path: Path,
) -> None:
    lock, prepared, snapshot, blobs = _hf_cache_generic_fixture(tmp_path)
    manifest = materialize_prepared_model_content_manifest(lock, prepared)
    content = manifest["snapshots"][0]
    assert [row["relative_path"] for row in content["critical_files"]] == [
        "code/source.py",
        "config.json",
    ]
    assert content["weight_headers"][0]["raw_sha256"] == blobs["weight"].name

    blobs["source"].write_bytes(blobs["source"].read_bytes() + b"# changed\n")
    with pytest.raises(ValueError, match="HF blob SHA-256"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, snapshot, _ = _hf_cache_generic_fixture(tmp_path / "foreign")
    foreign = (tmp_path / "outside").resolve()
    foreign.write_bytes(b'{"model_type":"foreign"}')
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(foreign)
    with pytest.raises(ValueError, match="unsafe cache link|canonical HF blobs"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, snapshot, blobs = _hf_cache_generic_fixture(tmp_path / "bad-hash")
    bad = blobs["config"].with_name("0" * 64)
    bad.write_bytes(blobs["config"].read_bytes())
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(Path("../../blobs") / bad.name)
    with pytest.raises(ValueError, match="HF blob SHA-256"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, snapshot, blobs = _hf_cache_generic_fixture(tmp_path / "chained")
    chained = blobs["config"].with_name("1" * 64)
    chained.symlink_to(blobs["config"])
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(Path("../../blobs") / chained.name)
    with pytest.raises(ValueError, match="chained"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, snapshot, _ = _hf_cache_generic_fixture(
        tmp_path / "blobs-root-symlink"
    )
    repository = snapshot.parent.parent
    real_blobs = repository / "blobs"
    foreign_blobs = (tmp_path / "foreign-blobs").resolve()
    real_blobs.rename(foreign_blobs)
    real_blobs.symlink_to(foreign_blobs, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe HF blobs directory"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, snapshot, blobs = _hf_cache_generic_fixture(
        tmp_path / "noncanonical-link"
    )
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(
        Path("../../blobs/../blobs") / blobs["config"].name
    )
    with pytest.raises(ValueError, match="non-canonical HF blob link"):
        materialize_prepared_model_content_manifest(lock, prepared)


def test_external_release_digest_rejects_coordinated_rehash_and_model_swap(
    tmp_path: Path,
) -> None:
    lock, prepared, authority, paths = _authority(tmp_path)
    original_release_digest = authority.release_manifest_sha256
    (paths["drafter"] / "config.json").write_bytes(b'{"model_type":"forged"}')
    forged_manifest = materialize_prepared_model_content_manifest(lock, prepared)
    _write_manifest(paths["manifest"], forged_manifest)
    with pytest.raises(ValueError, match="release digest"):
        bind_prepared_model_content_authority(
            lock,
            prepared,
            paths["manifest"],
            expected_release_manifest_sha256=original_release_digest,
        )
    with pytest.raises(RuntimeError, match="manifest or sidecar changed"):
        revalidate_prepared_model_content_authority(
            lock,
            authority,
            expected_release_manifest_sha256=original_release_digest,
        )

    forged_lock = ModelLock(
        2,
        (
            LockedModel(TARGET_ID, TARGET_REVISION),
            LockedModel(DRAFTER_ID, "9" * 40),
        ),
    )
    with pytest.raises(ValueError, match="differs from release/model lock"):
        revalidate_prepared_model_content_authority(
            forged_lock,
            authority,
            expected_release_manifest_sha256=original_release_digest,
        )

    lock, _, authority, paths = _authority(tmp_path / "payload-mutation")
    weight = paths["drafter"] / "model.safetensors"
    body = bytearray(weight.read_bytes())
    body[-1] ^= 1
    weight.write_bytes(body)
    with pytest.raises(ValueError, match="live snapshot replay"):
        revalidate_prepared_model_content_authority(
            lock,
            authority,
            expected_release_manifest_sha256=(authority.release_manifest_sha256),
        )


def test_trusted_replay_projects_exact_members_without_weight_payload_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, prepared, bundle_binding, roots = _trusted_bundle_fixture(
        tmp_path,
        monkeypatch,
    )
    original_digest = content_module._stable_file_digest

    def forbid_replay_payload(path: Path, *, label: str):
        if any(path.is_relative_to(root) for root in roots.values()):
            raise AssertionError(f"trusted replay reopened model payload: {path}")
        return original_digest(path, label=label)

    monkeypatch.setattr(
        content_module,
        "_stable_file_digest",
        forbid_replay_payload,
    )
    original_header = prepared_module._read_safetensors_header
    read_sizes: list[int] = []

    def bounded_header(*args, **kwargs):
        original_read = prepared_module.os.read

        def checked_read(descriptor: int, size: int) -> bytes:
            read_sizes.append(size)
            if size == 8 * 1024 * 1024:
                raise AssertionError("trusted prepared replay read weight payload")
            return original_read(descriptor, size)

        with monkeypatch.context() as nested:
            nested.setattr(prepared_module.os, "read", checked_read)
            return original_header(*args, **kwargs)

    monkeypatch.setattr(
        prepared_module,
        "_read_safetensors_header",
        bounded_header,
    )
    manifest = materialize_trusted_prepared_model_content_manifest(
        lock,
        prepared,
        trusted_content_bundle_binding=bundle_binding,
        target_model_id=TARGET_ID,
        drafter_model_id=DRAFTER_ID,
    )
    assert manifest["schema_version"] == 2
    snapshots = {
        row["model_id"]: row
        for row in manifest["snapshots"]  # type: ignore[index]
    }
    assert snapshots[TARGET_ID]["trusted_content_member"]["role"] == "target"
    assert snapshots[DRAFTER_ID]["trusted_content_member"]["role"] == "drafter"
    for snapshot in snapshots.values():
        member_files = {
            row["relative_path"]: row
            for row in snapshot["trusted_content_member"]["files"]
        }
        for header in snapshot["weight_headers"]:
            assert (
                header["raw_sha256"] == member_files[header["relative_path"]]["sha256"]
            )

    manifest_path = (tmp_path / "prepared-content.json").resolve()
    digest = _write_manifest(manifest_path, manifest)
    authority = bind_prepared_model_content_authority(
        lock,
        prepared,
        manifest_path,
        expected_release_manifest_sha256=digest,
    )
    assert authority.schema_version == 2
    first = revalidate_prepared_model_content_authority(
        lock,
        authority,
        expected_release_manifest_sha256=digest,
    )
    second = revalidate_prepared_model_content_authority(
        lock,
        authority,
        expected_release_manifest_sha256=digest,
    )
    assert first == second
    assert read_sizes and 8 * 1024 * 1024 not in read_sizes


def test_trusted_replay_blocks_metadata_role_and_member_projection_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, prepared, bundle_binding, roots = _trusted_bundle_fixture(
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(
        PreparedModelContentAuthorityBlocked,
        match="lacks its exact",
    ):
        materialize_trusted_prepared_model_content_manifest(
            lock,
            prepared,
            trusted_content_bundle_binding=bundle_binding,
            target_model_id=DRAFTER_ID,
            drafter_model_id=TARGET_ID,
        )

    manifest = materialize_trusted_prepared_model_content_manifest(
        lock,
        prepared,
        trusted_content_bundle_binding=bundle_binding,
        target_model_id=TARGET_ID,
        drafter_model_id=DRAFTER_ID,
    )
    swapped = json.loads(json.dumps(manifest))
    for snapshot in swapped["snapshots"]:
        member = snapshot["trusted_content_member"]
        member["role"] = "drafter" if member["role"] == "target" else "target"
    swapped_path = (tmp_path / "swapped-role.json").resolve()
    swapped_digest = _write_manifest(swapped_path, swapped)
    with pytest.raises(
        PreparedModelContentAuthorityBlocked,
        match="lacks its exact",
    ):
        bind_prepared_model_content_authority(
            lock,
            prepared,
            swapped_path,
            expected_release_manifest_sha256=swapped_digest,
        )

    spliced = json.loads(json.dumps(manifest))
    spliced["snapshots"][0]["trusted_content_member"]["member_sha256"] = "0" * 64
    spliced_path = (tmp_path / "spliced-member.json").resolve()
    spliced_digest = _write_manifest(spliced_path, spliced)
    with pytest.raises(ValueError, match="differs from live snapshot content"):
        bind_prepared_model_content_authority(
            lock,
            prepared,
            spliced_path,
            expected_release_manifest_sha256=spliced_digest,
        )

    manifest_path = (tmp_path / "prepared-content.json").resolve()
    digest = _write_manifest(manifest_path, manifest)
    authority = bind_prepared_model_content_authority(
        lock,
        prepared,
        manifest_path,
        expected_release_manifest_sha256=digest,
    )
    weight = roots["drafter"] / "model.safetensors"
    before = weight.stat(follow_symlinks=False)
    body = bytearray(weight.read_bytes())
    body[-1] ^= 1
    weight.write_bytes(body)
    os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = weight.stat(follow_symlinks=False)
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns
    with pytest.raises(TrustedSingleOperatorContentReplayBlocked, match="metadata"):
        revalidate_prepared_model_content_authority(
            lock,
            authority,
            expected_release_manifest_sha256=digest,
        )
