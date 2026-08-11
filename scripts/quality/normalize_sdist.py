"""Canonicalize sdist archive metadata under ``SOURCE_DATE_EPOCH``.

Setuptools currently emits wall-clock gzip and tar member timestamps for
sdists.  This helper deliberately preserves archive paths, member types,
permissions, link targets, and file bytes while normalizing only ownership and
time metadata.  It never extracts archive members to the filesystem.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import os
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _source_date_epoch(value: str | None) -> int:
    if value is None or not value.isascii() or not value.isdecimal():
        raise ValueError("SOURCE_DATE_EPOCH must be a non-negative integer")
    epoch = int(value)
    if epoch > 0xFFFFFFFF:
        raise ValueError("SOURCE_DATE_EPOCH exceeds the reproducible gzip range")
    return epoch


def _safe_archive_path(value: str, *, field: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"sdist {field} is not a safe relative path: {value!r}")


def _read_members(path: Path) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    rows: list[tuple[tarfile.TarInfo, bytes | None]] = []
    names: set[str] = set()
    with tarfile.open(path, mode="r:gz", errorlevel=2) as archive:
        for member in archive.getmembers():
            _safe_archive_path(member.name, field="member name")
            if member.name in names:
                raise ValueError(f"sdist contains duplicate member {member.name!r}")
            names.add(member.name)
            if member.islnk() or member.issym():
                _safe_archive_path(member.linkname, field="link target")
            elif not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"sdist contains unsupported member type for {member.name!r}"
                )
            payload = None
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"sdist member {member.name!r} has no payload")
                payload = extracted.read()
                if len(payload) != member.size:
                    raise ValueError(f"sdist member {member.name!r} is truncated")
            rows.append((member, payload))
    return rows


def _normalized_member(member: tarfile.TarInfo, *, epoch: int) -> tarfile.TarInfo:
    normalized = copy.copy(member)
    normalized.mtime = epoch
    normalized.uid = 0
    normalized.gid = 0
    normalized.uname = ""
    normalized.gname = ""
    # Tarfile recreates path/linkpath PAX keys when a value exceeds ustar
    # limits.  Dropping input PAX metadata prevents fractional wall-clock
    # timestamps and host-specific extended attributes from surviving.
    normalized.pax_headers = {}
    return normalized


def normalize_sdist(path: str | Path, *, epoch: int) -> str:
    """Rewrite one ``.tar.gz`` deterministically and return its SHA-256."""

    source = Path(path)
    if (
        source.suffixes[-2:] != [".tar", ".gz"]
        or source.is_symlink()
        or not source.is_file()
    ):
        raise ValueError("sdist must be a regular non-symlinked .tar.gz file")
    if epoch < 0 or epoch > 0xFFFFFFFF:
        raise ValueError("epoch is outside the reproducible gzip range")
    source_mode = stat.S_IMODE(source.stat().st_mode)
    rows = _read_members(source)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{source.name}.", suffix=".tmp", dir=source.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            with (
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=9,
                    fileobj=temporary,
                    mtime=epoch,
                ) as compressed,
                tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as output,
            ):
                for member, payload in sorted(rows, key=lambda row: row[0].name):
                    normalized = _normalized_member(member, epoch=epoch)
                    output.addfile(
                        normalized,
                        io.BytesIO(payload) if payload is not None else None,
                    )
        os.chmod(temporary_name, source_mode)
        os.replace(temporary_name, source)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sdist", nargs="+", type=Path)
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="override SOURCE_DATE_EPOCH (environment is required by default)",
    )
    args = parser.parse_args()
    epoch = (
        args.epoch
        if args.epoch is not None
        else _source_date_epoch(os.environ.get("SOURCE_DATE_EPOCH"))
    )
    for path in args.sdist:
        print(f"{normalize_sdist(path, epoch=epoch)}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
