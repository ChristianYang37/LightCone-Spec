from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lightcone_spec.experiments import workload_authority
from lightcone_spec.experiments.workload_authority import (
    FORMAL_WORKLOAD_FILTER_EMPTY_REASON,
    FORMAL_WORKLOAD_LOCAL_SOURCE_MISSING_REASON,
    FORMAL_WORKLOAD_PROTOCOLS,
    FORMAL_WORKLOAD_SOURCE_ALLOWLIST_EMPTY_REASON,
    LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256,
    LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256,
    LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION,
    LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS,
    LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS_SHA256,
    LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256,
    LIVECODEBENCH_V6_HARD_SOURCE_LOCK,
    MATH500_LEVEL5_FORMAL_SAMPLES_SHA256,
    MATH500_LEVEL5_RAW_FILE_SHA256,
    MATH500_LEVEL5_REPOSITORY_REVISION,
    MATH500_LEVEL5_SELECTED_RAW_ROWS_SHA256,
    MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS,
    MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS_SHA256,
    MATH500_LEVEL5_SOURCE_LOCK,
    RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA,
    RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA,
    FormalWorkloadAuthorityBlocked,
    LiveCodeBenchV6HardVerificationMetadata,
    Math500Level5VerificationMetadata,
    ReleaseWorkloadSourceLock,
    bind_formal_workload_authority,
    build_math500_level5_verification_metadata,
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    raw = b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
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


def _register_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workload_id: str,
    path: Path,
    rows: list[dict[str, object]],
) -> ReleaseWorkloadSourceLock:
    raw = _write_jsonl(path, rows)
    protocol = FORMAL_WORKLOAD_PROTOCOLS[workload_id]
    samples = workload_authority._select_all_rows(protocol, rows)
    lock = ReleaseWorkloadSourceLock(
        workload_id=workload_id,
        repository_revision=REVISION,
        raw_file_sha256=hashlib.sha256(raw).hexdigest(),
        raw_row_count=len(rows),
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
    monkeypatch: pytest.MonkeyPatch,
    workload_id: str,
) -> None:
    monkeypatch.setattr(workload_authority, "RELEASE_FORMAL_WORKLOAD_SOURCES", ())
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
            "level": 5 if index < 40 else 4,
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


def test_math_integer_filter_is_exact_and_never_coerces_a_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "math500.jsonl").resolve()
    rows = [
        {"unique_id": "integer", "problem": "Integer row.", "level": 5},
        {"unique_id": "string", "problem": "String row.", "level": "5"},
    ]
    raw = _write_jsonl(path, rows)
    protocol = FORMAL_WORKLOAD_PROTOCOLS["math500_level5"]
    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (
            ReleaseWorkloadSourceLock(
                workload_id="math500_level5",
                repository_revision=REVISION,
                raw_file_sha256=hashlib.sha256(raw).hexdigest(),
                raw_row_count=2,
                selected_row_count=1,
                selected_rows_sha256="b" * 64,
                protocol_sha256=protocol.sha256,
            ),
        ),
    )

    with pytest.raises(TypeError, match="exact JSON type"):
        bind_formal_workload_authority("math500_level5", path)


def test_math_release_lock_and_verification_codec_are_exact() -> None:
    metadata = RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA

    assert FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].filter_value == 5
    assert type(FORMAL_WORKLOAD_PROTOCOLS["math500_level5"].filter_value) is int
    assert MATH500_LEVEL5_SOURCE_LOCK.repository_revision == (
        MATH500_LEVEL5_REPOSITORY_REVISION
    )
    assert MATH500_LEVEL5_SOURCE_LOCK.raw_file_sha256 == (
        MATH500_LEVEL5_RAW_FILE_SHA256
    )
    assert MATH500_LEVEL5_SOURCE_LOCK.raw_row_count == 500
    assert MATH500_LEVEL5_SOURCE_LOCK.selected_row_count == 134
    assert MATH500_LEVEL5_SOURCE_LOCK.selected_rows_sha256 == (
        MATH500_LEVEL5_FORMAL_SAMPLES_SHA256
    )
    assert len(MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS) == 134
    assert len(set(MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS)) == 134
    assert (
        workload_authority.content_sha256(list(MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS))
        == MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS_SHA256
    )
    assert metadata.selected_source_row_ids == MATH500_LEVEL5_SELECTED_SOURCE_ROW_IDS
    assert metadata.selected_raw_rows_sha256 == (
        MATH500_LEVEL5_SELECTED_RAW_ROWS_SHA256
    )
    assert metadata.formal_samples_sha256 == MATH500_LEVEL5_FORMAL_SAMPLES_SHA256
    assert metadata.filter_value == 5
    assert metadata.tokenizer_statistics_status == ("PENDING_TOKENIZER_REVISION_LOCK")
    assert Math500Level5VerificationMetadata.from_dict(metadata.to_dict()) == metadata


def test_math_release_jsonl_replays_when_explicit_cache_is_available() -> None:
    source = os.environ.get("LIGHTCONE_MATH500_TEST_JSONL")
    if source is None:
        pytest.skip("set LIGHTCONE_MATH500_TEST_JSONL for upstream release replay")
    path = Path(source).resolve(strict=True)

    authority = bind_formal_workload_authority("math500_level5", path)
    metadata = build_math500_level5_verification_metadata(authority)

    assert metadata == RELEASE_MATH500_LEVEL5_VERIFICATION_METADATA


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


def test_locked_jsonl_preserves_all_80_exact_hard_matches_in_raw_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "question_id": f"question-{index:03d}",
            "question_content": f"Implement NFC task café {index}.",
            "difficulty": ("hard" if index < 160 and index % 2 == 0 else "easy"),
            "uncompiled_upstream_field": {"source_index": index},
        }
        for index in range(175)
    ]
    path = (tmp_path / "test6.jsonl").resolve()
    lock = _register_jsonl(
        monkeypatch,
        workload_id="livecodebench_v6_hard",
        path=path,
        rows=rows,
    )

    authority = bind_formal_workload_authority("livecodebench_v6_hard", path)

    expected_ids = tuple(f"question-{index:03d}" for index in range(0, 160, 2))
    assert authority.raw_row_count == 175
    assert authority.selected_row_count == 80
    assert tuple(sample.source_row_id for sample in authority.samples) == expected_ids
    assert authority.selected_rows_sha256 == lock.selected_rows_sha256
    assert authority.samples[-1].source_row_id == "question-158"
    assert require_formal_workload_authority(authority) == authority.prompts


def test_livecodebench_release_lock_and_verification_codec_are_exact() -> None:
    metadata = RELEASE_LIVECODEBENCH_V6_HARD_VERIFICATION_METADATA

    assert LIVECODEBENCH_V6_HARD_SOURCE_LOCK.repository_revision == (
        LIVECODEBENCH_V6_HARD_REPOSITORY_REVISION
    )
    assert LIVECODEBENCH_V6_HARD_SOURCE_LOCK.raw_file_sha256 == (
        LIVECODEBENCH_V6_HARD_RAW_FILE_SHA256
    )
    assert LIVECODEBENCH_V6_HARD_SOURCE_LOCK.raw_row_count == 175
    assert LIVECODEBENCH_V6_HARD_SOURCE_LOCK.selected_row_count == 80
    assert LIVECODEBENCH_V6_HARD_SOURCE_LOCK.selected_rows_sha256 == (
        LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256
    )
    assert len(LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS) == 80
    assert len(set(LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS)) == 80
    assert (
        workload_authority.content_sha256(
            list(LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS)
        )
        == LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS_SHA256
    )
    assert metadata.selected_question_ids == (
        LIVECODEBENCH_V6_HARD_SELECTED_QUESTION_IDS
    )
    assert metadata.selected_raw_rows_sha256 == (
        LIVECODEBENCH_V6_HARD_SELECTED_RAW_ROWS_SHA256
    )
    assert metadata.formal_samples_sha256 == (
        LIVECODEBENCH_V6_HARD_FORMAL_SAMPLES_SHA256
    )
    assert metadata.tokenizer_statistics_status == ("PENDING_TOKENIZER_REVISION_LOCK")
    assert metadata.tokenizer_revision is None
    assert metadata.prompt_token_statistics is None
    assert (
        LiveCodeBenchV6HardVerificationMetadata.from_dict(metadata.to_dict())
        == metadata
    )

    changed = metadata.to_dict()
    changed_ids = list(changed["selected_question_ids"])
    changed_ids[0], changed_ids[1] = changed_ids[1], changed_ids[0]
    changed["selected_question_ids"] = changed_ids
    with pytest.raises(ValueError, match="release identity differs|digest differs"):
        LiveCodeBenchV6HardVerificationMetadata.from_dict(changed)


@pytest.mark.parametrize(
    ("rows", "message"),
    (
        (
            [
                {
                    "question_id": "same",
                    "question_content": "First prompt.",
                    "difficulty": "hard",
                },
                {
                    "question_id": "same",
                    "question_content": "Second prompt.",
                    "difficulty": "hard",
                },
            ],
            "identities must be unique",
        ),
        (
            [
                {
                    "question_id": "question-e\u0301",
                    "question_content": "Prompt.",
                    "difficulty": "hard",
                }
            ],
            "identities must be NFC normalized",
        ),
        (
            [
                {
                    "question_id": "question-nfd-prompt",
                    "question_content": "Cafe\u0301 task.",
                    "difficulty": "hard",
                }
            ],
            "prompts must already be NFC normalized",
        ),
    ),
)
def test_jsonl_duplicate_identity_and_nfd_text_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    path = (tmp_path / "invalid.jsonl").resolve()
    raw = _write_jsonl(path, rows)
    protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (
            ReleaseWorkloadSourceLock(
                workload_id="livecodebench_v6_hard",
                repository_revision=REVISION,
                raw_file_sha256=hashlib.sha256(raw).hexdigest(),
                raw_row_count=len(rows),
                selected_row_count=1,
                selected_rows_sha256="b" * 64,
                protocol_sha256=protocol.sha256,
            ),
        ),
    )
    with pytest.raises(ValueError, match=message):
        bind_formal_workload_authority("livecodebench_v6_hard", path)


def test_jsonl_duplicate_object_key_is_rejected_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "duplicate-key.jsonl").resolve()
    raw = (
        b'{"question_id":"first","question_id":"second",'
        b'"question_content":"Prompt.","difficulty":"hard"}\n'
    )
    path.write_bytes(raw)
    protocol = FORMAL_WORKLOAD_PROTOCOLS["livecodebench_v6_hard"]
    monkeypatch.setattr(
        workload_authority,
        "RELEASE_FORMAL_WORKLOAD_SOURCES",
        (
            ReleaseWorkloadSourceLock(
                workload_id="livecodebench_v6_hard",
                repository_revision=REVISION,
                raw_file_sha256=hashlib.sha256(raw).hexdigest(),
                raw_row_count=1,
                selected_row_count=1,
                selected_rows_sha256="b" * 64,
                protocol_sha256=protocol.sha256,
            ),
        ),
    )
    with pytest.raises(ValueError, match="duplicate key 'question_id'"):
        bind_formal_workload_authority("livecodebench_v6_hard", path)


def test_registered_missing_source_is_named_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "math500.json").resolve()
    rows = [{"unique_id": "math-1", "problem": "P", "level": 5}]
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
    rows = [{"unique_id": "math-1", "problem": "P", "level": 4}]
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
    rows = [{"unique_id": "math-1", "problem": "P", "level": 5}]
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
        [{"unique_id": "math-2", "problem": "Changed", "level": 5}],
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
        {"unique_id": "same", "problem": "P1", "level": 5},
        {"unique_id": "same", "problem": "P2", "level": 5},
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
