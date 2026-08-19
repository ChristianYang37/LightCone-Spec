"""Offline, code-owned E0 task-native workload authorities.

LiveCodeBench and MATH-500 have their own formal workload loaders.  This
module owns the remaining seven E0 sources.  It never downloads data: callers
provide one local, absolute file whose revision, raw bytes, size, row count,
decoded rows, and compiled request rows are checked against release constants.

Raw rows and request rows deliberately have different identities.  Raw rows
are hashed before any Unicode normalization.  Request turns are NFC-normalized
and contain only a stable source ID, the native turn tuple, and a deterministic
seed.  MT-Bench is scanned in full but remains explicitly unsupported because
the current serving transport cannot preserve its native two-turn interaction.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
    E0TaskNativeWorkloadAuthority,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

type E0StandaloneTask = Literal[
    "GSM8K",
    "AIME-2025",
    "MBPP",
    "HumanEval",
    "MT-Bench",
    "Alpaca",
    "Arena-Hard",
]
type E0SourceFormat = Literal[
    "jsonl",
    "json_array",
    "gzip_jsonl",
    "json_stream",
]

_SHA256_CHARS = frozenset("0123456789abcdef")
_MAX_RAW_BYTES = 4 * 1024 * 1024
_MAX_DECODED_BYTES = 16 * 1024 * 1024
_SUPPORTED_REASON = "TASK_WORKLOAD_READY"
_UNSUPPORTED_REASON = "TOKENIZER_TASK_WORKLOAD_UNSUPPORTED"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_revision(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a full lowercase Git revision")
    return value


def _require_positive_int(label: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_mapping(label: str, value: object) -> Mapping[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{label} must be a string-keyed JSON object")
    return value


def _strict_sequence(label: str, value: object) -> Sequence[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("E0 workload JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"E0 workload JSON contains non-finite constant {value}")


_STRICT_DECODER = json.JSONDecoder(
    object_pairs_hook=_reject_duplicate_pairs,
    parse_constant=_reject_nonfinite_constant,
)


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("E0 workload JSON contains a non-finite number")
        return
    if type(value) is str:
        if "\x00" in value:
            raise ValueError("E0 workload JSON contains NUL")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        if not all(type(key) is str and "\x00" not in key for key in value):
            raise ValueError("E0 workload JSON has an invalid object key")
        for item in value.values():
            _validate_json_value(item)
        return
    raise TypeError("E0 workload JSON contains a non-JSON value")


@dataclass(frozen=True)
class E0TaskNativeSourcePin:
    task: E0StandaloneTask
    repository: str
    repository_revision: str
    source_file_name: str
    source_format: E0SourceFormat
    raw_file_size: int
    raw_file_sha256: str
    raw_row_count: int

    def __post_init__(self) -> None:
        if self.task not in {
            "GSM8K",
            "AIME-2025",
            "MBPP",
            "HumanEval",
            "MT-Bench",
            "Alpaca",
            "Arena-Hard",
        }:
            raise ValueError("E0 source pin task is unsupported")
        if type(self.repository) is not str or not self.repository:
            raise ValueError("E0 source repository must be nonempty")
        _require_git_revision("E0 source revision", self.repository_revision)
        if (
            type(self.source_file_name) is not str
            or not self.source_file_name
            or Path(self.source_file_name).name != self.source_file_name
        ):
            raise ValueError("E0 source file name must be one basename")
        if self.source_format not in {
            "jsonl",
            "json_array",
            "gzip_jsonl",
            "json_stream",
        }:
            raise ValueError("E0 source serialization is unsupported")
        _require_positive_int("E0 source byte size", self.raw_file_size)
        _require_sha256("E0 raw source", self.raw_file_sha256)
        _require_positive_int("E0 raw row count", self.raw_row_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "repository": self.repository,
            "repository_revision": self.repository_revision,
            "source_file_name": self.source_file_name,
            "source_format": self.source_format,
            "raw_file_size": self.raw_file_size,
            "raw_file_sha256": self.raw_file_sha256,
            "raw_row_count": self.raw_row_count,
        }

    @cached_property
    def source_revision_sha256(self) -> str:
        """Path-free digest accepted by E0TaskNativeWorkloadAuthority."""

        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_e0_source_revision",
                "task": self.task,
                "repository": self.repository,
                "repository_revision": self.repository_revision,
                "source_file_name": self.source_file_name,
            }
        )


E0_TASK_NATIVE_SOURCE_PINS: Mapping[E0StandaloneTask, E0TaskNativeSourcePin] = (
    MappingProxyType(
        {
            "GSM8K": E0TaskNativeSourcePin(
                task="GSM8K",
                repository="openai/grade-school-math",
                repository_revision="3101c7d5072418e28b9008a6636bde82a006892c",
                source_file_name="gsm8k-test.jsonl",
                source_format="jsonl",
                raw_file_size=749_738,
                raw_file_sha256=(
                    "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
                ),
                raw_row_count=1_319,
            ),
            "AIME-2025": E0TaskNativeSourcePin(
                task="AIME-2025",
                repository="math-ai/aime25",
                repository_revision="563bb8404243c5f09de6ec262f2db674fe5bce9b",
                source_file_name="aime25-test.jsonl",
                source_format="jsonl",
                raw_file_size=15_813,
                raw_file_sha256=(
                    "b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331"
                ),
                raw_row_count=30,
            ),
            "MBPP": E0TaskNativeSourcePin(
                task="MBPP",
                repository="google-research/google-research",
                repository_revision="1eb8bb0cbe5fd9072311ae3fd760e3644fee690b",
                source_file_name="mbpp-sanitized.json",
                source_format="json_array",
                raw_file_size=255_053,
                raw_file_sha256=(
                    "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
                ),
                raw_row_count=427,
            ),
            "HumanEval": E0TaskNativeSourcePin(
                task="HumanEval",
                repository="openai/human-eval",
                repository_revision="6d43fb980f9fee3c892a914eda09951f772ad10d",
                source_file_name="humaneval.jsonl.gz",
                source_format="gzip_jsonl",
                raw_file_size=44_877,
                raw_file_sha256=(
                    "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
                ),
                raw_row_count=164,
            ),
            "MT-Bench": E0TaskNativeSourcePin(
                task="MT-Bench",
                repository="lm-sys/FastChat",
                repository_revision="587d5cfa1609a43d192cedb8441cac3c17db105d",
                source_file_name="mt-bench-question.jsonl",
                source_format="jsonl",
                raw_file_size=48_929,
                raw_file_sha256=(
                    "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7"
                ),
                raw_row_count=80,
            ),
            "Alpaca": E0TaskNativeSourcePin(
                task="Alpaca",
                repository="tatsu-lab/alpaca_eval",
                repository_revision="2edc6fad8be6b14ea7230aabfd08188da6b8b814",
                source_file_name="alpaca-eval.json",
                source_format="json_array",
                raw_file_size=620_778,
                raw_file_sha256=(
                    "d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51"
                ),
                raw_row_count=805,
            ),
            "Arena-Hard": E0TaskNativeSourcePin(
                task="Arena-Hard",
                repository="lmarena/arena-hard-auto",
                repository_revision="196f6b826783b3da7310e361a805fa36f0be83f3",
                source_file_name="arena-hard-v2.0-question.jsonl",
                source_format="json_stream",
                raw_file_size=947_596,
                raw_file_sha256=(
                    "a75cef6623db6b27aec810497b059f0659705d7ff48dcbccc5bd5b130728ac73"
                ),
                raw_row_count=750,
            ),
        }
    )
)

E0_TASK_NATIVE_REQUEST_COMPILER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_task_native_request_compiler",
        "selection": "all_rows_in_raw_order",
        "request_fields": ["source_id", "turns", "seed"],
        "turn_compiler": "verbatim_then_unicode_nfc_no_trim_v1",
        "seed": (
            "first_63_bits_of_sha256(canonical_json(protocol_sha256,task,source_id))"
        ),
        "source_ids": {
            "GSM8K": "zero_based_raw_row_ordinal",
            "AIME-2025": "id",
            "MBPP": "task_id",
            "HumanEval": "task_id",
            "MT-Bench": "question_id",
            "Alpaca": "zero_based_raw_row_ordinal",
            "Arena-Hard": "uid",
        },
        "turn_fields": {
            "GSM8K": ["question"],
            "AIME-2025": ["problem"],
            "MBPP": ["prompt"],
            "HumanEval": ["prompt"],
            "MT-Bench": ["turns[0]", "turns[1]"],
            "Alpaca": ["instruction"],
            "Arena-Hard": ["prompt"],
        },
        "transport": {
            "single_turn": "READY",
            "MT-Bench": "UNSUPPORTED_native_two_turn_state_not_preserved",
        },
        "arena_v2_categories": {
            "hard_prompt": 500,
            "creative_writing": 250,
        },
    }
)

E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S: Mapping[E0StandaloneTask, tuple[str, str]] = (
    MappingProxyType(
        {
            "GSM8K": (
                "13ee343c128dbc3cee820f744d547416974a23f2ea356a454d4aa3376a35bd37",
                "ffd5fa25111d53ee638779abbff427e5d3fa9be8763a158f54ac175057a5befa",
            ),
            "AIME-2025": (
                "b67468f6ff2702fda719058e6fb13dcfcba111b1030ef0884241cd63520f244f",
                "153b22aab1c925def1c60a7e682b85879c02025404f603c0fb93abca6bf12eb3",
            ),
            "MBPP": (
                "0a699546e9e40b657bfd6bdaf9ecd330bd590ecc8e0602e9cb0ef85cd41d6e16",
                "319ab84bcefcab5fc659a17ceebe6c0a652eb8fe575dd7803e16f1690699ba06",
            ),
            "HumanEval": (
                "62eeaa082ca375405caa318234d2eadb3536c69ab5a5cbf7a70138aac686645c",
                "9ef4c7e7a00cade59b44aea2f36d1e657003ba42161fcbe3b7649eeb62083ed6",
            ),
            "MT-Bench": (
                "f10786f04ba58a0cbd64143e3b9bdcf2cb6e0a83dab064376eb1f140b5c22128",
                "8cb070770cbd5608af8758a430106cb834e30384b481fa3f28e2d7e82c4cf5f5",
            ),
            "Alpaca": (
                "5a23d65d41e0eaa77b71eefa7d25a9e9623f8549670060053c7b1f48e2ab5513",
                "817e6f60e1996fab8e1ad9e1e81fab9403e4160397022ac19781c3924beb2b1b",
            ),
            "Arena-Hard": (
                "4fbd7603bda89e13406d6c0d19a455e2e7661d00f6958c4c3e55edb7d2fb7595",
                "39213ddff79abafd50ea98bafa82aa5835043fcbca66340972b33b2516b0895e",
            ),
        }
    )
)

FORMAL_SINGLE_OPERATOR_E0_TASK_NATIVE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_task_native_sources",
        "sources": {
            task: pin.to_dict()
            for task, pin in sorted(E0_TASK_NATIVE_SOURCE_PINS.items())
        },
        "request_compiler_sha256": (E0_TASK_NATIVE_REQUEST_COMPILER_PROTOCOL_SHA256),
        "expected_outputs": {
            task: {
                "decoded_raw_rows_sha256": values[0],
                "request_rows_sha256": values[1],
            }
            for task, values in sorted(E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S.items())
        },
        "raw_rows": "full_decoded_rows_in_raw_order_before_normalization",
        "network": "forbidden",
        "missing_or_mismatch": "fail_closed",
    }
)


@dataclass(frozen=True)
class E0TaskNativeRequestRow:
    source_id: str
    turns: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if (
            type(self.source_id) is not str
            or not self.source_id
            or "\x00" in self.source_id
        ):
            raise ValueError("E0 request source ID must be nonempty and NUL-free")
        if (
            type(self.turns) is not tuple
            or not self.turns
            or any(type(turn) is not str or not turn for turn in self.turns)
        ):
            raise ValueError("E0 request turns must be a nonempty string tuple")
        for turn in self.turns:
            if "\x00" in turn or unicodedata.normalize("NFC", turn) != turn:
                raise ValueError("E0 request turns must be NFC and NUL-free")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("E0 request seed must be a signed-64-bit-safe integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "turns": list(self.turns),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_mapping("E0 request row", value)
        if set(row) != {"source_id", "turns", "seed"}:
            raise ValueError("E0 request row fields differ")
        turns = _strict_sequence("E0 request turns", row["turns"])
        return cls(
            source_id=row["source_id"],  # type: ignore[arg-type]
            turns=tuple(turns),  # type: ignore[arg-type]
            seed=row["seed"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class E0TaskNativeSourceAuthority:
    schema_version: Literal[1]
    kind: Literal["formal_single_operator_e0_task_native_source_authority"]
    task: E0StandaloneTask
    repository: str
    repository_revision: str
    source_file_name: str
    source_format: E0SourceFormat
    raw_source_path: str
    raw_file_size: int
    raw_file_sha256: str
    raw_row_count: int
    decoded_raw_rows_sha256: str
    request_row_count: int
    request_rows_sha256: str
    source_revision_sha256: str
    protocol_sha256: str
    support_status: Literal["READY", "UNSUPPORTED"]
    reason_code: str
    request_rows: tuple[E0TaskNativeRequestRow, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_single_operator_e0_task_native_source_authority"
            or self.task not in E0_TASK_NATIVE_SOURCE_PINS
        ):
            raise ValueError("E0 task-native authority schema differs")
        pin = E0_TASK_NATIVE_SOURCE_PINS[self.task]
        path = Path(self.raw_source_path)
        if (
            not path.is_absolute()
            or Path(os.path.abspath(path)) != path
            or path.name != self.source_file_name
        ):
            raise ValueError("E0 raw source path must be absolute and normalized")
        for label, digest in (
            ("raw source", self.raw_file_sha256),
            ("decoded raw rows", self.decoded_raw_rows_sha256),
            ("request rows", self.request_rows_sha256),
            ("source revision", self.source_revision_sha256),
            ("protocol", self.protocol_sha256),
        ):
            _require_sha256(f"E0 {label}", digest)
        _require_positive_int("E0 raw source size", self.raw_file_size)
        _require_positive_int("E0 raw row count", self.raw_row_count)
        _require_positive_int("E0 request row count", self.request_row_count)
        if (
            self.repository != pin.repository
            or self.repository_revision != pin.repository_revision
            or self.source_file_name != pin.source_file_name
            or self.source_format != pin.source_format
            or self.raw_file_size != pin.raw_file_size
            or self.raw_file_sha256 != pin.raw_file_sha256
            or self.raw_row_count != pin.raw_row_count
            or self.request_row_count != pin.raw_row_count
            or self.source_revision_sha256 != pin.source_revision_sha256
            or self.protocol_sha256
            != FORMAL_SINGLE_OPERATOR_E0_TASK_NATIVE_PROTOCOL_SHA256
        ):
            raise ValueError("E0 task-native authority differs from its source pin")
        if (
            type(self.request_rows) is not tuple
            or len(self.request_rows) != self.request_row_count
            or any(type(row) is not E0TaskNativeRequestRow for row in self.request_rows)
            or len({row.source_id for row in self.request_rows})
            != self.request_row_count
            or self.request_rows_sha256
            != content_sha256([row.to_dict() for row in self.request_rows])
        ):
            raise ValueError("E0 task-native request coverage differs")
        if self.task == "MT-Bench":
            expected_status = "UNSUPPORTED"
            expected_reason = _UNSUPPORTED_REASON
            if any(len(row.turns) != 2 for row in self.request_rows):
                raise ValueError("MT-Bench authority must preserve both native turns")
        else:
            expected_status = "READY"
            expected_reason = _SUPPORTED_REASON
            if any(len(row.turns) != 1 for row in self.request_rows):
                raise ValueError("single-turn E0 authority has another turn shape")
        if (
            self.support_status != expected_status
            or self.reason_code != expected_reason
        ):
            raise ValueError("E0 task-native support disposition differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "task": self.task,
            "repository": self.repository,
            "repository_revision": self.repository_revision,
            "source_file_name": self.source_file_name,
            "source_format": self.source_format,
            "raw_source_path": self.raw_source_path,
            "raw_file_size": self.raw_file_size,
            "raw_file_sha256": self.raw_file_sha256,
            "raw_row_count": self.raw_row_count,
            "decoded_raw_rows_sha256": self.decoded_raw_rows_sha256,
            "request_row_count": self.request_row_count,
            "request_rows_sha256": self.request_rows_sha256,
            "source_revision_sha256": self.source_revision_sha256,
            "protocol_sha256": self.protocol_sha256,
            "support_status": self.support_status,
            "reason_code": self.reason_code,
            "request_rows": [row.to_dict() for row in self.request_rows],
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_mapping("E0 task-native authority", value)
        expected = {
            "schema_version",
            "kind",
            "task",
            "repository",
            "repository_revision",
            "source_file_name",
            "source_format",
            "raw_source_path",
            "raw_file_size",
            "raw_file_sha256",
            "raw_row_count",
            "decoded_raw_rows_sha256",
            "request_row_count",
            "request_rows_sha256",
            "source_revision_sha256",
            "protocol_sha256",
            "support_status",
            "reason_code",
            "request_rows",
        }
        if set(row) != expected:
            raise ValueError("E0 task-native authority fields differ")
        request_rows = _strict_sequence(
            "E0 task-native authority request rows", row["request_rows"]
        )
        fields = dict(row)
        fields.pop("request_rows")
        return cls(
            **fields,
            request_rows=tuple(
                E0TaskNativeRequestRow.from_dict(item) for item in request_rows
            ),
        )  # type: ignore[arg-type]

    @cached_property
    def task_native_workload_sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "formal_single_operator_e0_task_native_workload",
                "task": self.task,
                "source_revision_sha256": self.source_revision_sha256,
                "raw_file_sha256": self.raw_file_sha256,
                "decoded_raw_rows_sha256": self.decoded_raw_rows_sha256,
                "request_rows_sha256": self.request_rows_sha256,
                "request_row_count": self.request_row_count,
                "protocol_sha256": self.protocol_sha256,
                "support_status": self.support_status,
                "reason_code": self.reason_code,
            }
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def revalidate(self) -> Self:
        rebuilt = scan_e0_task_native_source(
            task=self.task,
            raw_source_path=self.raw_source_path,
        )
        if rebuilt.to_dict() != self.to_dict() or rebuilt.sha256 != self.sha256:
            raise ValueError("E0 task-native authority changed on deep reopen")
        return self


def _read_pinned_raw(
    path_value: str | Path, pin: E0TaskNativeSourcePin
) -> tuple[Path, bytes]:
    path = Path(path_value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ValueError("E0 raw source path must be absolute and normalized")
    if path.name != pin.source_file_name or path.is_symlink():
        raise ValueError("E0 raw source path/file name differs from its pin")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("E0 raw source path must not traverse symlinks")
    status = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(status.st_mode)
        or status.st_size != pin.raw_file_size
        or status.st_size > _MAX_RAW_BYTES
    ):
        raise ValueError("E0 raw source is not the pinned bounded regular file")
    raw = path.read_bytes()
    if (
        len(raw) != status.st_size
        or hashlib.sha256(raw).hexdigest() != pin.raw_file_sha256
    ):
        raise ValueError("E0 raw source bytes differ from the code-owned pin")
    after = path.stat(follow_symlinks=False)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
    ):
        raise RuntimeError("E0 raw source changed while it was read")
    return path, raw


def _decode_jsonl(text: str) -> list[object]:
    rows: list[object] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"E0 JSONL has blank row {line_number}")
        try:
            rows.append(_STRICT_DECODER.decode(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"E0 JSONL row {line_number} is invalid") from error
    return rows


def _decode_json_stream(text: str) -> list[object]:
    rows: list[object] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        try:
            value, cursor = _STRICT_DECODER.raw_decode(text, cursor)
        except json.JSONDecodeError as error:
            raise ValueError("E0 JSON stream is invalid") from error
        rows.append(value)
    return rows


def _decode_rows(raw: bytes, pin: E0TaskNativeSourcePin) -> list[dict[str, object]]:
    payload = raw
    if pin.source_format == "gzip_jsonl":
        try:
            payload = gzip.decompress(raw)
        except (EOFError, OSError) as error:
            raise ValueError("E0 gzip JSONL is invalid") from error
    if len(payload) > _MAX_DECODED_BYTES:
        raise ValueError("E0 decoded workload exceeds the registered bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("E0 workload source is not strict UTF-8") from error
    if pin.source_format in {"jsonl", "gzip_jsonl"}:
        raw_rows: object = _decode_jsonl(text)
    elif pin.source_format == "json_stream":
        raw_rows = _decode_json_stream(text)
    else:
        try:
            raw_rows = _STRICT_DECODER.decode(text)
        except json.JSONDecodeError as error:
            raise ValueError("E0 JSON array is invalid") from error
        raw_rows = list(_strict_sequence("E0 source root", raw_rows))
    if type(raw_rows) is not list or len(raw_rows) != pin.raw_row_count:
        raise ValueError("E0 raw source row count differs from its pin")
    rows: list[dict[str, object]] = []
    for ordinal, value in enumerate(raw_rows):
        row = dict(_strict_mapping(f"E0 raw row {ordinal}", value))
        _validate_json_value(row)
        rows.append(row)
    return rows


def _exact_fields(*, row: Mapping[str, object], expected: set[str], task: str) -> None:
    if set(row) != expected:
        raise ValueError(f"{task} raw row fields differ from the pinned schema")


def _nonempty_string(label: str, value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty NUL-free string")
    return value


def _string_array(label: str, value: object, *, nonempty: bool = False) -> list[str]:
    rows = _strict_sequence(label, value)
    if nonempty and not rows:
        raise ValueError(f"{label} must be nonempty")
    if any(type(item) is not str or "\x00" in item for item in rows):
        raise ValueError(f"{label} must contain only NUL-free strings")
    return list(rows)  # type: ignore[return-value]


def _source_id(task: E0StandaloneTask, row: Mapping[str, object], ordinal: int) -> str:
    if task in {"GSM8K", "Alpaca"}:
        return f"row:{ordinal:06d}"
    field = {
        "AIME-2025": "id",
        "MBPP": "task_id",
        "HumanEval": "task_id",
        "MT-Bench": "question_id",
        "Arena-Hard": "uid",
    }[task]
    value = row[field]
    if type(value) is int and value >= 0:
        return str(value)
    return _nonempty_string(f"{task} source ID", value)


def _compile_turns(
    task: E0StandaloneTask, row: Mapping[str, object]
) -> tuple[str, ...]:
    if task == "GSM8K":
        _exact_fields(row=row, expected={"question", "answer"}, task=task)
        _nonempty_string("GSM8K answer", row["answer"])
        turns = (_nonempty_string("GSM8K question", row["question"]),)
    elif task == "AIME-2025":
        _exact_fields(row=row, expected={"problem", "answer", "id"}, task=task)
        if type(row["answer"]) is not int:
            raise TypeError("AIME-2025 answer must remain an integer")
        turns = (_nonempty_string("AIME-2025 problem", row["problem"]),)
    elif task == "MBPP":
        _exact_fields(
            row=row,
            expected={
                "source_file",
                "task_id",
                "prompt",
                "code",
                "test_imports",
                "test_list",
            },
            task=task,
        )
        if type(row["task_id"]) is not int or row["task_id"] < 0:
            raise TypeError("MBPP task_id must be a non-negative integer")
        _nonempty_string("MBPP source_file", row["source_file"])
        _nonempty_string("MBPP code", row["code"])
        _string_array("MBPP test imports", row["test_imports"])
        _string_array("MBPP tests", row["test_list"], nonempty=True)
        turns = (_nonempty_string("MBPP prompt", row["prompt"]),)
    elif task == "HumanEval":
        _exact_fields(
            row=row,
            expected={
                "task_id",
                "prompt",
                "entry_point",
                "canonical_solution",
                "test",
            },
            task=task,
        )
        for field in ("entry_point", "canonical_solution", "test"):
            _nonempty_string(f"HumanEval {field}", row[field])
        turns = (_nonempty_string("HumanEval prompt", row["prompt"]),)
    elif task == "MT-Bench":
        fields = set(row)
        if fields not in (
            {"question_id", "category", "turns"},
            {"question_id", "category", "turns", "reference"},
        ):
            raise ValueError("MT-Bench raw row fields differ from the pinned schema")
        if type(row["question_id"]) is not int or row["question_id"] < 0:
            raise TypeError("MT-Bench question_id must be a non-negative integer")
        _nonempty_string("MT-Bench category", row["category"])
        native_turns = _string_array("MT-Bench turns", row["turns"], nonempty=True)
        if len(native_turns) != 2 or any(not turn for turn in native_turns):
            raise ValueError("MT-Bench must preserve exactly two nonempty turns")
        if "reference" in row:
            reference = row["reference"]
            if type(reference) is str:
                _nonempty_string("MT-Bench reference", reference)
            else:
                _string_array("MT-Bench reference", reference, nonempty=True)
        turns = tuple(native_turns)
    elif task == "Alpaca":
        _exact_fields(
            row=row,
            expected={"dataset", "instruction", "output", "generator"},
            task=task,
        )
        for field in ("dataset", "output", "generator"):
            _nonempty_string(f"Alpaca {field}", row[field])
        turns = (_nonempty_string("Alpaca instruction", row["instruction"]),)
    else:
        _exact_fields(
            row=row,
            expected={"uid", "category", "subcategory", "prompt"},
            task=task,
        )
        category = _nonempty_string("Arena-Hard category", row["category"])
        if category not in {"hard_prompt", "creative_writing"}:
            raise ValueError("Arena-Hard category lies outside v2.0")
        _nonempty_string("Arena-Hard subcategory", row["subcategory"])
        turns = (_nonempty_string("Arena-Hard prompt", row["prompt"]),)
    return tuple(unicodedata.normalize("NFC", turn) for turn in turns)


def _request_seed(task: E0StandaloneTask, source_id: str) -> int:
    digest = hashlib.sha256(
        _canonical_bytes(
            {
                "protocol_sha256": E0_TASK_NATIVE_REQUEST_COMPILER_PROTOCOL_SHA256,
                "task": task,
                "source_id": source_id,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _compile_request_rows(
    task: E0StandaloneTask, rows: Sequence[Mapping[str, object]]
) -> tuple[E0TaskNativeRequestRow, ...]:
    result: list[E0TaskNativeRequestRow] = []
    source_ids: set[str] = set()
    arena_categories: Counter[str] = Counter()
    for ordinal, row in enumerate(rows):
        source_id = _source_id(task, row, ordinal)
        if source_id in source_ids:
            raise ValueError(f"{task} contains a duplicate source ID")
        source_ids.add(source_id)
        turns = _compile_turns(task, row)
        result.append(
            E0TaskNativeRequestRow(
                source_id=source_id,
                turns=turns,
                seed=_request_seed(task, source_id),
            )
        )
        if task == "Arena-Hard":
            arena_categories[str(row["category"])] += 1
    if task == "Arena-Hard" and arena_categories != Counter(
        {"hard_prompt": 500, "creative_writing": 250}
    ):
        raise ValueError("Arena-Hard v2.0 category coverage is not 500/250")
    return tuple(result)


def scan_e0_task_native_source(
    *, task: E0StandaloneTask, raw_source_path: str | Path
) -> E0TaskNativeSourceAuthority:
    """Scan one exact local source without any network or caller-supplied pin."""

    if task not in E0_TASK_NATIVE_SOURCE_PINS:
        raise ValueError("E0 standalone workload task is unsupported")
    pin = E0_TASK_NATIVE_SOURCE_PINS[task]
    path, raw = _read_pinned_raw(raw_source_path, pin)
    rows = _decode_rows(raw, pin)
    request_rows = _compile_request_rows(task, rows)
    decoded_raw_rows_sha256 = content_sha256(rows)
    request_rows_sha256 = content_sha256([row.to_dict() for row in request_rows])
    if (
        decoded_raw_rows_sha256,
        request_rows_sha256,
    ) != E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S[task]:
        raise ValueError("E0 decoded/request rows differ from code-owned outputs")
    support_status: Literal["READY", "UNSUPPORTED"] = (
        "UNSUPPORTED" if task == "MT-Bench" else "READY"
    )
    reason_code = _UNSUPPORTED_REASON if task == "MT-Bench" else _SUPPORTED_REASON
    return E0TaskNativeSourceAuthority(
        schema_version=1,
        kind="formal_single_operator_e0_task_native_source_authority",
        task=task,
        repository=pin.repository,
        repository_revision=pin.repository_revision,
        source_file_name=pin.source_file_name,
        source_format=pin.source_format,
        raw_source_path=str(path),
        raw_file_size=pin.raw_file_size,
        raw_file_sha256=pin.raw_file_sha256,
        raw_row_count=pin.raw_row_count,
        decoded_raw_rows_sha256=decoded_raw_rows_sha256,
        request_row_count=len(request_rows),
        request_rows_sha256=request_rows_sha256,
        source_revision_sha256=pin.source_revision_sha256,
        protocol_sha256=FORMAL_SINGLE_OPERATOR_E0_TASK_NATIVE_PROTOCOL_SHA256,
        support_status=support_status,
        reason_code=reason_code,
        request_rows=request_rows,
    )


def bind_e0_task_native_workload_authority(
    *,
    source: E0TaskNativeSourceAuthority,
    protocol_lock_sha256: str,
    upstream_e6_confirmation_sha256: str,
    model: str,
    tokenizer_sha256: str,
) -> E0TaskNativeWorkloadAuthority:
    """Bind one verified task source to an existing model/tokenizer identity."""

    if type(source) is not E0TaskNativeSourceAuthority:
        raise TypeError("E0 workload binding requires an exact source authority")
    source.revalidate()
    return E0TaskNativeWorkloadAuthority(
        schema_version=1,
        protocol_lock_sha256=_require_sha256("E0 protocol lock", protocol_lock_sha256),
        upstream_e6_confirmation_sha256=_require_sha256(
            "E0 E6 confirmation", upstream_e6_confirmation_sha256
        ),
        model=model,
        task=source.task,
        tokenizer_sha256=_require_sha256("E0 tokenizer", tokenizer_sha256),
        task_native_workload_sha256=source.task_native_workload_sha256,
        source_revision_sha256=source.source_revision_sha256,
        support_status=source.support_status,
        reason_code=source.reason_code,
        evidence_sha256=source.sha256,
    )


def publish_e0_task_native_source_authority(
    authority: E0TaskNativeSourceAuthority,
    *,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Deep-revalidate and atomically publish one no-replace authority."""

    if type(authority) is not E0TaskNativeSourceAuthority:
        raise TypeError("E0 publisher requires an exact source authority")
    authority.revalidate()
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_task_native_source_publication",
        "authority": authority.to_dict(),
        "authority_sha256": authority.sha256,
    }
    payload["publication_sha256"] = content_sha256(payload)
    destination = Path(output_path)
    if (
        not destination.is_absolute()
        or Path(os.path.abspath(destination)) != destination
    ):
        raise ValueError("E0 publication path must be absolute and normalized")
    publish_canonical_json_no_replace(destination, payload)
    return CanonicalJsonProofBinding.bind(destination)


def load_e0_task_native_source_authority(
    publication_path: str | Path,
) -> E0TaskNativeSourceAuthority:
    """Load canonical publication bytes and deep-reopen their raw source."""

    binding = CanonicalJsonProofBinding.bind(publication_path)
    payload = binding.reopen()
    if set(payload) != {
        "schema_version",
        "kind",
        "authority",
        "authority_sha256",
        "publication_sha256",
    }:
        raise ValueError("E0 source publication fields differ")
    if (
        payload["schema_version"] != 1
        or payload["kind"] != "formal_single_operator_e0_task_native_source_publication"
    ):
        raise ValueError("E0 source publication schema differs")
    declared_publication_sha256 = payload["publication_sha256"]
    unsigned = dict(payload)
    unsigned.pop("publication_sha256")
    if declared_publication_sha256 != content_sha256(unsigned):
        raise ValueError("E0 source publication digest differs")
    authority = E0TaskNativeSourceAuthority.from_dict(payload["authority"])
    if payload["authority_sha256"] != authority.sha256:
        raise ValueError("E0 source authority digest differs")
    return authority.revalidate()


__all__ = [
    "E0_TASK_NATIVE_EXPECTED_OUTPUT_SHA256S",
    "E0_TASK_NATIVE_REQUEST_COMPILER_PROTOCOL_SHA256",
    "E0_TASK_NATIVE_SOURCE_PINS",
    "FORMAL_SINGLE_OPERATOR_E0_TASK_NATIVE_PROTOCOL_SHA256",
    "E0StandaloneTask",
    "E0TaskNativeRequestRow",
    "E0TaskNativeSourceAuthority",
    "E0TaskNativeSourcePin",
    "bind_e0_task_native_workload_authority",
    "load_e0_task_native_source_authority",
    "publish_e0_task_native_source_authority",
    "scan_e0_task_native_source",
]
