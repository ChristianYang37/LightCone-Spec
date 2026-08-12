from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lightcone_spec.cli.main import _load_bound_json, _write_json


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_bound_json_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value":1,"value":2}\n', encoding="utf-8")
    Path(f"{duplicate}.sha256").write_text(
        _canonical_sha256({"value": 2}) + "\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_bound_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    Path(f"{nonfinite}.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _load_bound_json(nonfinite)


def test_bound_json_rejects_symlink_sources(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_json(target, {"value": 1})
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    Path(f"{alias}.sha256").write_text(
        _canonical_sha256({"value": 1}) + "\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="readable regular file"):
        _load_bound_json(alias)


def test_immutable_json_publish_recovers_identical_missing_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifact.json"
    value = {"schema_version": 1, "rows": [1, 2, 3]}
    _write_json(output, value)
    body = output.read_bytes()
    Path(f"{output}.sha256").unlink()

    _write_json(output, value)

    assert output.read_bytes() == body
    assert _load_bound_json(output) == value
    assert output.stat().st_nlink == 1
    assert Path(f"{output}.sha256").stat().st_nlink == 1


def test_immutable_json_publish_never_follows_output_symlink(tmp_path: Path) -> None:
    protected = tmp_path / "protected.json"
    protected.write_text("do-not-change\n", encoding="utf-8")
    output = tmp_path / "artifact.json"
    output.symlink_to(protected)

    with pytest.raises(ValueError, match="readable regular file"):
        _write_json(output, {"value": "forged"})

    assert protected.read_text(encoding="utf-8") == "do-not-change\n"
