from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_NORMALIZER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "quality" / "normalize_sdist.py"
)
_NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "lightcone_quality_normalize_sdist", _NORMALIZER_PATH
)
if _NORMALIZER_SPEC is None or _NORMALIZER_SPEC.loader is None:
    raise RuntimeError(f"cannot load sdist normalizer from {_NORMALIZER_PATH}")
_NORMALIZER = importlib.util.module_from_spec(_NORMALIZER_SPEC)
_NORMALIZER_SPEC.loader.exec_module(_NORMALIZER)
normalize_sdist = _NORMALIZER.normalize_sdist


def _sdist(path: Path, *, timestamp: int, reverse: bool = False) -> None:
    rows = [
        ("package-1.0/", None, 0o755),
        ("package-1.0/README.md", b"same bytes\n", 0o644),
        ("package-1.0/src/module.py", b"VALUE = 1\n", 0o640),
    ]
    if reverse:
        rows.reverse()
    with (
        path.open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", mtime=timestamp) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, payload, mode in rows:
            member = tarfile.TarInfo(name)
            member.mtime = timestamp
            member.uid = 501
            member.gid = 20
            member.uname = "builder"
            member.gname = "staff"
            member.mode = mode
            if payload is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def _contents(path: Path) -> dict[str, bytes | None]:
    with tarfile.open(path, mode="r:gz") as archive:
        return {
            member.name: (
                archive.extractfile(member).read() if member.isfile() else None
            )
            for member in archive.getmembers()
        }


def test_normalizer_preserves_contents_and_canonicalizes_metadata(tmp_path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _sdist(first, timestamp=1_700_000_001)
    _sdist(second, timestamp=1_800_000_002, reverse=True)
    first.chmod(0o640)
    second.chmod(0o640)
    expected = _contents(first)

    first_digest = normalize_sdist(first, epoch=1_600_000_000)
    second_digest = normalize_sdist(second, epoch=1_600_000_000)

    assert first_digest == second_digest
    assert first_digest == hashlib.sha256(first.read_bytes()).hexdigest()
    assert _contents(first) == _contents(second) == expected
    assert first.stat().st_mode & 0o777 == second.stat().st_mode & 0o777 == 0o640
    with tarfile.open(first, mode="r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == sorted(expected)
        assert {member.mtime for member in members} == {1_600_000_000}
        assert {member.uid for member in members} == {0}
        assert {member.gid for member in members} == {0}
        assert {member.uname for member in members} == {""}
        assert {member.gname for member in members} == {""}
        assert {member.mode for member in members} == {0o640, 0o644, 0o755}


def test_normalizer_rejects_unsafe_archive_paths(tmp_path) -> None:
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, mode="w:gz") as archive:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="safe relative path"):
        normalize_sdist(path, epoch=1_600_000_000)
