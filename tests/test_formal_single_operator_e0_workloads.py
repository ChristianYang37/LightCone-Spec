from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from pathlib import Path
from types import MappingProxyType

import pytest

from lightcone_spec.experiments import (
    formal_single_operator_e0_workloads as workload_module,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_e0_workloads import (
    E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S,
    E0_TASK_NATIVE_SOURCE_PINS,
    E0TaskNativeSourcePin,
    bind_e0_task_native_workload_authority,
    load_e0_task_native_source_authority,
    publish_e0_task_native_source_authority,
    scan_e0_task_native_source,
)


def _jsonl(rows: list[object]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
        + b"\n"
        for row in rows
    )


def _install_pin(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task: str,
    raw: bytes,
    row_count: int,
    source_format: str = "jsonl",
) -> tuple[Path, E0TaskNativeSourcePin]:
    original = E0_TASK_NATIVE_SOURCE_PINS[task]
    path = (tmp_path / original.source_file_name).resolve()
    path.write_bytes(raw)
    pin = E0TaskNativeSourcePin(
        task=task,  # type: ignore[arg-type]
        repository=original.repository,
        repository_revision="1" * 40,
        source_file_name=original.source_file_name,
        source_format=source_format,  # type: ignore[arg-type]
        raw_file_size=len(raw),
        raw_file_sha256=hashlib.sha256(raw).hexdigest(),
        raw_row_count=row_count,
    )
    monkeypatch.setattr(
        workload_module,
        "E0_TASK_NATIVE_SOURCE_PINS",
        MappingProxyType({task: pin}),
    )
    monkeypatch.setattr(
        workload_module,
        "E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S",
        MappingProxyType({task: ("a" * 64, "b" * 64)}),
    )
    return path, pin


def _install_valid_gsm8k(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, object]:
    rows = [
        {"question": "Cafe\u0301 costs?", "answer": "5"},
        {"question": "Already NFC", "answer": "7"},
    ]
    raw = _jsonl(rows)
    path, pin = _install_pin(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        task="GSM8K",
        raw=raw,
        row_count=2,
    )
    decoded = workload_module._decode_rows(raw, pin)
    requests = workload_module._compile_request_rows("GSM8K", decoded)
    expected = (
        content_sha256(decoded),
        content_sha256([row.to_dict() for row in requests]),
    )
    monkeypatch.setattr(
        workload_module,
        "E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S",
        MappingProxyType({"GSM8K": expected}),
    )
    authority = scan_e0_task_native_source(
        task="GSM8K",
        raw_source_path=path,
    )
    return path, authority


def test_source_scan_separates_raw_rows_from_nfc_request_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, authority = _install_valid_gsm8k(monkeypatch, tmp_path)

    assert authority.request_row_count == 2
    assert authority.support_status == "READY"
    assert authority.reason_code == "TASK_WORKLOAD_READY"
    assert authority.request_rows[0].turns == ("Caf\u00e9 costs?",)
    assert unicodedata.normalize("NFC", "Cafe\u0301 costs?") != "Cafe\u0301 costs?"
    assert authority.decoded_raw_rows_sha256 != authority.request_rows_sha256
    assert authority.request_rows[0].source_id == "row:000000"
    assert authority.request_rows[0].seed == authority.revalidate().request_rows[0].seed


def test_model_tokenizer_binding_uses_compatible_source_revision_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, source = _install_valid_gsm8k(monkeypatch, tmp_path)
    authority = bind_e0_task_native_workload_authority(
        source=source,
        protocol_lock_sha256="2" * 64,
        upstream_e6_confirmation_sha256="3" * 64,
        model="Qwen/Qwen3-4B",
        tokenizer_sha256="4" * 64,
    )

    assert authority.task_native_workload_sha256 == source.task_native_workload_sha256
    assert authority.source_revision_sha256 == source.source_revision_sha256
    assert len(authority.source_revision_sha256) == 64
    assert authority.evidence_sha256 == source.sha256


def test_publication_roundtrip_is_deep_and_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path, source = _install_valid_gsm8k(monkeypatch, tmp_path)
    output = (tmp_path / "published.json").resolve()
    binding = publish_e0_task_native_source_authority(
        source,
        output_path=output,
    )

    assert binding.reopen()["authority_sha256"] == source.sha256
    assert load_e0_task_native_source_authority(output).sha256 == source.sha256
    with pytest.raises(RuntimeError, match="already exists"):
        publish_e0_task_native_source_authority(source, output_path=output)

    raw_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned bounded regular file"):
        load_e0_task_native_source_authority(output)


def test_duplicate_source_id_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _jsonl(
        [
            {"problem": "p1", "answer": 1, "id": "same"},
            {"problem": "p2", "answer": 2, "id": "same"},
        ]
    )
    path, _pin = _install_pin(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        task="AIME-2025",
        raw=raw,
        row_count=2,
    )

    with pytest.raises(ValueError, match="duplicate source ID"):
        scan_e0_task_native_source(task="AIME-2025", raw_source_path=path)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            b'{"question":"one","question":"two","answer":"3"}\n',
            "duplicate object key",
        ),
        (b'{"question":"bad\\u0000prompt","answer":"3"}\n', "contains NUL"),
        (b'{"question":"q","answer":NaN}\n', "non-finite constant"),
        (b'{"question":"q"}\n', "fields differ"),
    ],
)
def test_malformed_json_and_schema_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    message: str,
) -> None:
    path, _pin = _install_pin(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        task="GSM8K",
        raw=raw,
        row_count=1,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        scan_e0_task_native_source(task="GSM8K", raw_source_path=path)


def test_wrong_bytes_symlink_and_relative_path_fail_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _authority = _install_valid_gsm8k(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="absolute and normalized"):
        scan_e0_task_native_source(
            task="GSM8K",
            raw_source_path=Path(path.name),
        )
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="file name differs"):
        scan_e0_task_native_source(task="GSM8K", raw_source_path=alias)

    path.write_bytes(path.read_bytes()[:-1] + b" ")
    with pytest.raises(ValueError, match="bytes differ"):
        scan_e0_task_native_source(task="GSM8K", raw_source_path=path)


def test_exact_local_cache_matches_all_code_owned_outputs() -> None:
    cache_value = os.environ.get("LIGHTCONE_E0_TASK_NATIVE_CACHE")
    if cache_value is None:
        pytest.skip("set LIGHTCONE_E0_TASK_NATIVE_CACHE for exact source scan")
    cache = Path(cache_value).resolve(strict=True)

    authorities = {
        task: scan_e0_task_native_source(
            task=task,
            raw_source_path=(cache / pin.source_file_name).resolve(strict=True),
        )
        for task, pin in E0_TASK_NATIVE_SOURCE_PINS.items()
    }

    assert {
        task: (
            authority.decoded_raw_rows_sha256,
            authority.request_rows_sha256,
        )
        for task, authority in authorities.items()
    } == dict(E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S)
    assert authorities["MT-Bench"].support_status == "UNSUPPORTED"
    assert all(len(row.turns) == 2 for row in authorities["MT-Bench"].request_rows)
    assert all(
        authority.support_status == "READY"
        for task, authority in authorities.items()
        if task != "MT-Bench"
    )
