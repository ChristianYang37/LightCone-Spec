from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _semantic(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_binding_derives_semantic_identity_and_rejects_caller_override(
    tmp_path: Path,
) -> None:
    value = {"kind": "unsigned_gpu_proof", "schema_version": 1, "rows": [1, 2]}
    path = (tmp_path / "proof.json").resolve()
    publish_canonical_json_no_replace(path, value)
    expected = _semantic(value)
    binding = CanonicalJsonProofBinding.bind(
        path,
        semantic_sha256=expected,
    )
    assert binding.semantic_sha256 == expected
    assert binding.reopen() == value
    with pytest.raises(ValueError, match="expected semantic digest differs"):
        CanonicalJsonProofBinding.bind(path, semantic_sha256="0" * 64)
    with pytest.raises(ValueError, match="raw or semantic"):
        replace(binding, semantic_sha256="0" * 64).reopen()


def test_binding_rejects_symlinked_ancestor_and_unsafe_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    value = {"kind": "unsigned_gpu_proof", "schema_version": 1}
    proof = (real / "proof.json").resolve()
    publish_canonical_json_no_replace(proof, value)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink-free"):
        CanonicalJsonProofBinding.bind(alias / "proof.json")

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    unsafe_proof = unsafe / "proof.json"
    publish_canonical_json_no_replace(unsafe_proof.resolve(), value)
    os.chmod(unsafe, 0o777)
    try:
        with pytest.raises(ValueError, match="non-writable directory"):
            CanonicalJsonProofBinding.bind(unsafe_proof.resolve())
    finally:
        os.chmod(unsafe, 0o700)


def test_binding_rejects_final_symlink_and_no_replace(tmp_path: Path) -> None:
    value = {"kind": "unsigned_gpu_proof", "schema_version": 1}
    target = (tmp_path / "target.json").resolve()
    publish_canonical_json_no_replace(target, value)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink-free regular file"):
        CanonicalJsonProofBinding.bind(link.resolve(strict=False).parent / link.name)
    with pytest.raises(RuntimeError, match="already exists"):
        publish_canonical_json_no_replace(target, value)
