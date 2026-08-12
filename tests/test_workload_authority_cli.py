from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lightcone_spec.cli.main import main
from lightcone_spec.experiments import (
    FORMAL_WORKLOAD_PROTOCOLS,
    FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON,
    FormalWorkloadAuthority,
    FormalWorkloadAuthorityBlocked,
    ReleaseWorkloadSourceLock,
    bind_formal_workload_authority,
    formal_workload_samples_sha256,
    revalidate_formal_workload_authority,
    workload_authority,
)

REVISION = "a" * 40


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _raw_math_source(path: Path) -> tuple[dict[str, object], bytes]:
    rows = [
        {
            "unique_id": f"math-{index:03d}",
            "problem": f"Solve formal problem {index}.",
            "level": "Level 5" if index < 40 else "Level 4",
        }
        for index in range(47)
    ]
    protocol = FORMAL_WORKLOAD_PROTOCOLS["math500_level5"]
    value = {
        "schema_version": 1,
        "repository": protocol.repository,
        "repository_revision": REVISION,
        "dataset_config": protocol.dataset_config,
        "split": protocol.split,
        "rows": rows,
    }
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return value, raw


def _install_test_source_lock(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
) -> None:
    value, raw = _raw_math_source(path)
    protocol = FORMAL_WORKLOAD_PROTOCOLS["math500_level5"]
    samples = workload_authority._select_all_rows(protocol, value["rows"])
    lock = ReleaseWorkloadSourceLock(
        workload_id="math500_level5",
        repository_revision=REVISION,
        raw_file_sha256=hashlib.sha256(raw).hexdigest(),
        raw_row_count=len(value["rows"]),
        selected_row_count=len(samples),
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        protocol_sha256=protocol.sha256,
    )
    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (lock,),
    )


def test_public_workload_authority_exports_are_available() -> None:
    assert FormalWorkloadAuthority.__module__.endswith("workload_authority")
    assert FormalWorkloadAuthorityBlocked.__module__.endswith("workload_authority")
    assert callable(bind_formal_workload_authority)
    assert callable(revalidate_formal_workload_authority)


def test_cli_blocks_before_missing_source_read_or_output_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "missing-source" / "math500.json"
    output = tmp_path / "must-not-exist" / "authority.json"

    status = main(
        [
            "bind-formal-workload-authority",
            "--workload",
            "math500_level5",
            "--source",
            str(source),
            "--output",
            str(output),
        ]
    )

    decision = json.loads(capsys.readouterr().out)
    assert status == 42
    assert decision == {
        "schema_version": 1,
        "kind": "formal_workload_authority_decision",
        "status": "BLOCKED",
        "reason_code": FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON,
        "workload_id": "math500_level5",
        "authority_sha256": None,
        "formal_execution_authorized": False,
    }
    assert not source.parent.exists()
    assert not output.parent.exists()


def test_diagnostic_bind_and_revalidate_never_authorize_formal_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = (tmp_path / "math500.json").resolve()
    output = tmp_path / "authority.json"
    _install_test_source_lock(monkeypatch, source)

    bind_status = main(
        [
            "bind-formal-workload-authority",
            "--workload",
            "math500_level5",
            "--source",
            str(source),
            "--output",
            str(output),
        ]
    )
    bind_decision = json.loads(capsys.readouterr().out)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert bind_status == 0
    assert bind_decision["status"] == "BOUND_DIAGNOSTIC"
    assert bind_decision["formal_execution_authorized"] is False
    assert artifact["formal_execution_authorized"] is False
    assert artifact["authority"]["selected_row_count"] == 40
    assert output.with_suffix(".json.sha256").read_text().strip() == (
        _canonical_sha256(artifact)
    )

    revalidate_status = main(
        [
            "revalidate-formal-workload-authority",
            "--authority",
            str(output),
        ]
    )
    revalidate_decision = json.loads(capsys.readouterr().out)
    assert revalidate_status == 0
    assert revalidate_decision == bind_decision

    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (),
    )
    production_status = main(
        [
            "revalidate-formal-workload-authority",
            "--authority",
            str(output),
        ]
    )
    production_decision = json.loads(capsys.readouterr().out)
    assert production_status == 42
    assert production_decision["status"] == "BLOCKED"
    assert (
        production_decision["reason_code"]
        == FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON
    )
    assert production_decision["formal_execution_authorized"] is False
