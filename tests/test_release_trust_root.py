from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import lightcone_spec.runtime.release_trust_root as root_module
from lightcone_spec.runtime.release_trust_root import (
    SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256,
    SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256,
    SOURCE_RELEASE_ROOT_PUBLIC_KEY_SHA256,
    SOURCE_RELEASE_ROOT_SPKI_SHA256,
    SourceReleaseRootDescriptor,
    load_source_release_ed25519_root,
)


def test_packaged_public_root_is_canonical_public_only_and_matches_source() -> None:
    binding = load_source_release_ed25519_root()
    package_root = importlib.resources.files("lightcone_spec.runtime").joinpath(
        "trust/release_ed25519_root_v1.json"
    )
    package_sidecar = importlib.resources.files("lightcone_spec.runtime").joinpath(
        "trust/release_ed25519_root_v1.json.sha256"
    )
    body = package_root.read_bytes()
    sidecar = package_sidecar.read_bytes()
    source_root = (
        Path(__file__).resolve().parents[1]
        / "manifests/runtime/release_ed25519_root_v1.json"
    )
    source_sidecar = Path(f"{source_root}.sha256")
    assert body == source_root.read_bytes()
    assert sidecar == source_sidecar.read_bytes()
    assert hashlib.sha256(body).hexdigest() == SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256
    assert sidecar == f"{SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256}\n".encode("ascii")
    assert binding.semantic_sha256 == SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256
    assert binding.root.public_key_sha256 == SOURCE_RELEASE_ROOT_PUBLIC_KEY_SHA256
    assert binding.root.spki_sha256 == SOURCE_RELEASE_ROOT_SPKI_SHA256
    value = json.loads(body)
    assert body == (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    lowered = body.lower()
    for forbidden in (
        b"private_key",
        b"signing_key",
        b"signing_seed",
        b"secret_key",
        b"seed_phrase",
    ):
        assert forbidden not in lowered


def test_public_root_identity_is_install_prefix_independent() -> None:
    binding = load_source_release_ed25519_root()
    moved = replace(
        binding,
        path="/another/install/lightcone_spec/runtime/trust/root.json",
        sidecar_path=("/another/install/lightcone_spec/runtime/trust/root.json.sha256"),
    )
    assert moved.sha256 == binding.sha256


def test_public_root_rejects_tamper_and_cross_read_toctou(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = Path(root_module.SOURCE_RELEASE_ED25519_ROOT.manifest_path)
    body = original.read_bytes()
    copied = (tmp_path / original.name).resolve()
    copied.write_bytes(body[:-2] + b"0\n")
    Path(f"{copied}.sha256").write_text(
        hashlib.sha256(copied.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        root_module,
        "SOURCE_RELEASE_ED25519_ROOT",
        SourceReleaseRootDescriptor(
            manifest_path=str(copied),
            semantic_sha256=SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256,
            file_sha256=SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256,
        ),
    )
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_source_release_ed25519_root()

    copied.write_bytes(body)
    Path(f"{copied}.sha256").write_text(
        SOURCE_RELEASE_ROOT_MANIFEST_FILE_SHA256 + "\n",
        encoding="ascii",
    )
    real_reader = root_module._stable_regular_bytes
    reads = 0

    def changing_reader(path: Path, *, maximum_size: int) -> bytes:
        nonlocal reads
        value = real_reader(path, maximum_size=maximum_size)
        if path == copied:
            reads += 1
            if reads == 2:
                return value[:-1] + b" "
        return value

    monkeypatch.setattr(root_module, "_stable_regular_bytes", changing_reader)
    with pytest.raises(RuntimeError, match="changed while loaded"):
        load_source_release_ed25519_root()


def test_wheel_and_sdist_preserve_installed_public_root_lookup(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    distribution = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--skip-dependency-check",
            "--wheel",
            "--sdist",
            "--outdir",
            str(distribution),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(distribution.glob("*.whl"))
    sdist = next(distribution.glob("*.tar.gz"))
    expected_suffixes = {
        "lightcone_spec/runtime/trust/release_ed25519_root_v1.json",
        "lightcone_spec/runtime/trust/release_ed25519_root_v1.json.sha256",
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert expected_suffixes <= names
        installed = tmp_path / "installed"
        archive.extractall(installed)
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        assert all(
            any(name.endswith(suffix) for name in names) for suffix in expected_suffixes
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(installed)!r}); "
                "from lightcone_spec.runtime.release_trust_root import "
                "load_source_release_ed25519_root; "
                "print(load_source_release_ed25519_root().semantic_sha256)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == SOURCE_RELEASE_ROOT_MANIFEST_SEMANTIC_SHA256
