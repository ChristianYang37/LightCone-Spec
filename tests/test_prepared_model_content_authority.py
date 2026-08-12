from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from lightcone_spec.locking import (
    ModelLock,
    PreparedModelContentAuthorityBinding,
    PreparedModelContentAuthorityBlocked,
    bind_prepared_model_content_authority,
    bind_prepared_models,
    materialize_prepared_model_content_manifest,
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


def test_unregistered_profile_and_missing_required_file_are_named_blocks(
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
        match="prepared_model_content_profile_unregistered",
    ):
        materialize_prepared_model_content_manifest(unknown_lock, unknown)


def test_snapshot_sources_reject_symlink_hardlink_and_manifest_path_escape(
    tmp_path: Path,
) -> None:
    lock, prepared, roots = _fixture(tmp_path)
    config = roots["drafter"] / "config.json"
    original = roots["drafter"] / "config-real.json"
    config.rename(original)
    config.symlink_to(original)
    with pytest.raises(ValueError, match="symlink or hardlink"):
        materialize_prepared_model_content_manifest(lock, prepared)

    lock, prepared, roots = _fixture(tmp_path / "hardlink")
    config = roots["drafter"] / "config.json"
    alias = roots["drafter"] / "config-hardlink.json"
    os.link(config, alias)
    with pytest.raises(ValueError, match="symlink or hardlink"):
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
