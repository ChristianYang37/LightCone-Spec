from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lightcone_spec.experiments import workload_authority
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_FILTER_EMPTY_REASON,
    FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON,
    FORMAL_WORKLOAD_PROTOCOLS,
    FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON,
    FormalWorkloadAuthorityBlocked,
    ReleaseWorkloadSourceLock,
    bind_formal_workload_authority,
    formal_workload_samples_sha256,
    require_formal_workload_authority,
    revalidate_formal_workload_authority,
)

REVISION = "a" * 40


def _envelope(workload_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    return {
        "schema_version": 1,
        "repository": protocol.repository,
        "repository_revision": REVISION,
        "dataset_config": protocol.dataset_config,
        "split": protocol.split,
        "rows": rows,
    }


def _write_raw(path: Path, envelope: dict[str, object]) -> bytes:
    raw = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _register(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workload_id: str,
    path: Path,
    envelope: dict[str, object],
) -> ReleaseWorkloadSourceLock:
    raw = _write_raw(path, envelope)
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    samples = workload_authority._select_all_rows(protocol, envelope["rows"])
    lock = ReleaseWorkloadSourceLock(
        workload_id=workload_id,
        repository_revision=REVISION,
        raw_file_sha256=hashlib.sha256(raw).hexdigest(),
        raw_row_count=len(envelope["rows"]),
        selected_row_count=len(samples),
        selected_rows_sha256=formal_workload_samples_sha256(samples),
        protocol_sha256=protocol.sha256,
    )
    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (lock,),
    )
    return lock


def _register_unreduced(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workload_id: str,
    path: Path,
    envelope: dict[str, object],
) -> None:
    raw = _write_raw(path, envelope)
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    lock = ReleaseWorkloadSourceLock(
        workload_id=workload_id,
        repository_revision=REVISION,
        raw_file_sha256=hashlib.sha256(raw).hexdigest(),
        raw_row_count=len(envelope["rows"]),
        selected_row_count=1,
        selected_rows_sha256="b" * 64,
        protocol_sha256=protocol.sha256,
    )
    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (lock,),
    )


@pytest.mark.parametrize(
    "workload_id",
    ("livecodebench_v6_hard", "math500_level5"),
)
def test_empty_release_allowlist_blocks_before_path_or_network_side_effect(
    tmp_path: Path,
    workload_id: str,
) -> None:
    missing = tmp_path / "must-not-be-created" / "source.json"

    with pytest.raises(FormalWorkloadAuthorityBlocked) as error:
        bind_formal_workload_authority(workload_id, missing)

    assert error.value.reason == FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON
    assert not missing.parent.exists()


def test_math_level5_selects_every_match_instead_of_first_32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "unique_id": f"math-{index:03d}",
            "problem": f"Solve formal problem {index}.",
            "level": "Level 5" if index < 40 else "Level 4",
        }
        for index in range(47)
    ]
    path = (tmp_path / "math500.json").resolve()
    lock = _register(
        monkeypatch,
        workload_id="math500_level5",
        path=path,
        envelope=_envelope("math500_level5", rows),
    )

    authority = bind_formal_workload_authority("math500_level5", path)

    assert authority.repository_revision == REVISION
    assert authority.raw_file_sha256 == lock.raw_file_sha256
    assert authority.raw_row_count == 47
    assert authority.selected_row_count == 40
    assert [sample.source_row_id for sample in authority.samples] == [
        f"math-{index:03d}" for index in range(40)
    ]
    assert len(require_formal_workload_authority(authority)) == 40


def test_livecodebench_v6_hard_filter_is_exact_and_case_sensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "question_id": f"lcb-{index:03d}",
            "question_content": f"Implement hard task {index}.",
            "difficulty": "hard",
        }
        for index in range(35)
    ]
    rows.extend(
        (
            {
                "question_id": "lcb-case-mismatch",
                "question_content": "This label must not be normalized.",
                "difficulty": "Hard",
            },
            {
                "question_id": "lcb-easy",
                "question_content": "An easy task.",
                "difficulty": "easy",
            },
        )
    )
    path = (tmp_path / "livecodebench.json").resolve()
    _register(
        monkeypatch,
        workload_id="livecodebench_v6_hard",
        path=path,
        envelope=_envelope("livecodebench_v6_hard", rows),
    )

    authority = bind_formal_workload_authority("livecodebench_v6_hard", path)

    assert authority.raw_row_count == 37
    assert authority.selected_row_count == 35
    assert all(sample.source_row_id.startswith("lcb-0") for sample in authority.samples)


def test_registered_missing_source_is_named_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "math500.json").resolve()
    rows = [{"unique_id": "math-1", "problem": "P", "level": "Level 5"}]
    _register(
        monkeypatch,
        workload_id="math500_level5",
        path=path,
        envelope=_envelope("math500_level5", rows),
    )
    path.unlink()

    with pytest.raises(FormalWorkloadAuthorityBlocked) as error:
        bind_formal_workload_authority("math500_level5", path)

    assert error.value.reason == FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON


def test_empty_strict_filter_is_blocked_not_an_empty_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "math500.json").resolve()
    rows = [{"unique_id": "math-1", "problem": "P", "level": "Level 4"}]
    envelope = _envelope("math500_level5", rows)
    _register_unreduced(
        monkeypatch,
        workload_id="math500_level5",
        path=path,
        envelope=envelope,
    )

    with pytest.raises(FormalWorkloadAuthorityBlocked) as error:
        bind_formal_workload_authority("math500_level5", path)

    assert error.value.reason == FORMAL_WORKLOAD_FILTER_EMPTY_REASON


def test_path_symlink_raw_tamper_and_coordinated_relock_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "math500.json").resolve()
    rows = [{"unique_id": "math-1", "problem": "P", "level": "Level 5"}]
    envelope = _envelope("math500_level5", rows)
    _register(
        monkeypatch,
        workload_id="math500_level5",
        path=path,
        envelope=envelope,
    )
    link = tmp_path / "math500-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="resolved and non-symlink"):
        bind_formal_workload_authority("math500_level5", link)

    authority = bind_formal_workload_authority("math500_level5", path)
    changed = _envelope(
        "math500_level5",
        [{"unique_id": "math-2", "problem": "Changed", "level": "Level 5"}],
    )
    _write_raw(path, changed)
    with pytest.raises(ValueError, match="raw bytes"):
        revalidate_formal_workload_authority(authority)

    _register(
        monkeypatch,
        workload_id="math500_level5",
        path=path,
        envelope=changed,
    )
    with pytest.raises(ValueError, match="changed during revalidation"):
        revalidate_formal_workload_authority(authority)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_config", "v5", "metadata differs"),
        ("split", "train", "metadata differs"),
        ("repository_revision", "c" * 40, "metadata differs"),
    ),
)
def test_source_metadata_cannot_self_describe_another_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    rows = [
        {
            "question_id": "lcb-1",
            "question_content": "Implement task.",
            "difficulty": "hard",
        }
    ]
    envelope = _envelope("livecodebench_v6_hard", rows)
    envelope[field] = value
    path = (tmp_path / f"bad-{field}.json").resolve()
    _register_unreduced(
        monkeypatch,
        workload_id="livecodebench_v6_hard",
        path=path,
        envelope=envelope,
    )

    with pytest.raises(ValueError, match=message):
        bind_formal_workload_authority("livecodebench_v6_hard", path)


def test_duplicate_ids_and_unregistered_aliases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "duplicate.json").resolve()
    rows = [
        {"unique_id": "same", "problem": "P1", "level": "Level 5"},
        {"unique_id": "same", "problem": "P2", "level": "Level 5"},
    ]
    _register_unreduced(
        monkeypatch,
        workload_id="math500_level5",
        path=path,
        envelope=_envelope("math500_level5", rows),
    )
    with pytest.raises(ValueError, match="identities must be unique"):
        bind_formal_workload_authority("math500_level5", path)

    with pytest.raises(ValueError, match="not registered"):
        bind_formal_workload_authority("math500", path)  # type: ignore[arg-type]
