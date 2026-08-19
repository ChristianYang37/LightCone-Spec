"""Deterministic bounded JSON sequence shards for large formal evidence.

The index never embeds payload rows.  Each immutable shard carries one
contiguous ordinal interval and is independently path/raw/semantic bound.
Consumers may reopen one row/shard lazily or deep-replay the complete sequence.
This module is trust-mode neutral and makes no scientific interpretation of a
row; typed consumers remain responsible for decoding and validating rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_SHA256_CHARS = frozenset("0123456789abcdef")
_DEFAULT_SHARD_MAX_BYTES = 1_500_000
_DEFAULT_SHARD_MAX_ROWS = 512


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        # Match CanonicalJsonProofBinding/publish_canonical_json_no_replace.
        # Escaping non-ASCII here creates a different semantic digest even
        # though the published canonical UTF-8 JSON is otherwise identical.
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_text(label: str, value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256 = _sha256(
    {
        "schema_version": 1,
        "kind": "formal_canonical_sequence_shard_protocol",
        "partition": "contiguous_zero_based_ordinals",
        "rows": "canonical_JSON_values_preserved_byte_semantically",
        "identity": "artifact_kind_id_interval_rows_sha256",
        "publication": "each_shard_then_index_atomic_no_replace",
        "revalidation": "lazy_one_shard_or_complete_ordered_replay",
    }
)


@dataclass(frozen=True)
class FormalCanonicalSequenceShard:
    schema_version: Literal[1]
    kind: Literal["formal_canonical_sequence_shard"]
    protocol_sha256: str
    artifact_kind: str
    artifact_id: str
    shard_ordinal: int
    start_ordinal: int
    end_ordinal_exclusive: int
    rows_sha256: str
    rows: tuple[object, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_canonical_sequence_shard"
            or self.protocol_sha256 != FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256
        ):
            raise ValueError("formal sequence shard schema differs")
        _require_text("formal sequence artifact kind", self.artifact_kind)
        _require_sha256("formal sequence artifact ID", self.artifact_id)
        if (
            type(self.shard_ordinal) is not int
            or self.shard_ordinal < 0
            or type(self.start_ordinal) is not int
            or self.start_ordinal < 0
            or type(self.end_ordinal_exclusive) is not int
            or self.end_ordinal_exclusive <= self.start_ordinal
            or type(self.rows) is not tuple
            or len(self.rows) != self.end_ordinal_exclusive - self.start_ordinal
        ):
            raise ValueError("formal sequence shard interval differs")
        _require_sha256("formal sequence shard rows", self.rows_sha256)
        if self.rows_sha256 != _sha256(list(self.rows)):
            raise ValueError("formal sequence shard row digest differs")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "artifact_kind": self.artifact_kind,
            "artifact_id": self.artifact_id,
            "shard_ordinal": self.shard_ordinal,
            "start_ordinal": self.start_ordinal,
            "end_ordinal_exclusive": self.end_ordinal_exclusive,
            "rows_sha256": self.rows_sha256,
            "rows": list(self.rows),
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal sequence shard",
            value,
            {
                "schema_version",
                "kind",
                "protocol_sha256",
                "artifact_kind",
                "artifact_id",
                "shard_ordinal",
                "start_ordinal",
                "end_ordinal_exclusive",
                "rows_sha256",
                "rows",
                "shard_sha256",
            },
        )
        declared = _require_sha256("formal sequence shard", row.pop("shard_sha256"))
        raw_rows = row.pop("rows")
        if type(raw_rows) is not list:
            raise TypeError("formal sequence shard rows must be an array")
        shard = cls(**row, rows=tuple(raw_rows))  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("formal sequence shard declared digest differs")
        return shard


@dataclass(frozen=True)
class FormalCanonicalSequenceShardReference:
    shard_ordinal: int
    start_ordinal: int
    end_ordinal_exclusive: int
    rows_sha256: str
    shard_sha256: str
    binding: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            type(self.shard_ordinal) is not int
            or self.shard_ordinal < 0
            or type(self.start_ordinal) is not int
            or self.start_ordinal < 0
            or type(self.end_ordinal_exclusive) is not int
            or self.end_ordinal_exclusive <= self.start_ordinal
        ):
            raise ValueError("formal sequence shard reference interval differs")
        _require_sha256("formal sequence reference rows", self.rows_sha256)
        _require_sha256("formal sequence reference shard", self.shard_sha256)
        if type(self.binding) is not CanonicalJsonProofBinding:
            raise TypeError("formal sequence shard reference is not path-bound")

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_ordinal": self.shard_ordinal,
            "start_ordinal": self.start_ordinal,
            "end_ordinal_exclusive": self.end_ordinal_exclusive,
            "rows_sha256": self.rows_sha256,
            "shard_sha256": self.shard_sha256,
            "binding": self.binding.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal sequence shard reference",
            value,
            {
                "shard_ordinal",
                "start_ordinal",
                "end_ordinal_exclusive",
                "rows_sha256",
                "shard_sha256",
                "binding",
            },
        )
        row["binding"] = CanonicalJsonProofBinding.from_dict(row["binding"])
        return cls(**row)  # type: ignore[arg-type]

    def reopen(
        self, *, artifact_kind: str, artifact_id: str
    ) -> FormalCanonicalSequenceShard:
        shard = FormalCanonicalSequenceShard.from_dict(self.binding.reopen())
        if (
            shard.artifact_kind != artifact_kind
            or shard.artifact_id != artifact_id
            or shard.shard_ordinal != self.shard_ordinal
            or shard.start_ordinal != self.start_ordinal
            or shard.end_ordinal_exclusive != self.end_ordinal_exclusive
            or shard.rows_sha256 != self.rows_sha256
            or shard.sha256 != self.shard_sha256
            or self.binding.semantic_sha256 != _sha256(shard.to_dict())
        ):
            raise ValueError("formal sequence shard reference changed")
        return shard


@dataclass(frozen=True)
class FormalCanonicalSequenceShardIndex:
    schema_version: Literal[1]
    kind: Literal["formal_canonical_sequence_shard_index"]
    protocol_sha256: str
    artifact_kind: str
    artifact_id: str
    total_rows: int
    shard_count: int
    ordered_shard_sha256s_sha256: str
    shards: tuple[FormalCanonicalSequenceShardReference, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_canonical_sequence_shard_index"
            or self.protocol_sha256 != FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256
        ):
            raise ValueError("formal sequence index schema differs")
        _require_text("formal sequence artifact kind", self.artifact_kind)
        _require_sha256("formal sequence artifact ID", self.artifact_id)
        if (
            type(self.total_rows) is not int
            or self.total_rows < 1
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or type(self.shards) is not tuple
            or len(self.shards) != self.shard_count
            or any(
                type(row) is not FormalCanonicalSequenceShardReference
                for row in self.shards
            )
        ):
            raise ValueError("formal sequence index coverage differs")
        cursor = 0
        for ordinal, reference in enumerate(self.shards):
            if reference.shard_ordinal != ordinal or reference.start_ordinal != cursor:
                raise ValueError("formal sequence shard intervals are not contiguous")
            cursor = reference.end_ordinal_exclusive
        if cursor != self.total_rows:
            raise ValueError("formal sequence index total differs from shards")
        _require_sha256(
            "formal sequence ordered shard identities",
            self.ordered_shard_sha256s_sha256,
        )
        if self.ordered_shard_sha256s_sha256 != _sha256(
            [row.shard_sha256 for row in self.shards]
        ):
            raise ValueError("formal sequence shard identity order differs")

    @cached_property
    def sha256(self) -> str:
        return _sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "artifact_kind": self.artifact_kind,
            "artifact_id": self.artifact_id,
            "total_rows": self.total_rows,
            "shard_count": self.shard_count,
            "ordered_shard_sha256s_sha256": (self.ordered_shard_sha256s_sha256),
            "shards": [row.to_dict() for row in self.shards],
        }
        if include_sha256:
            value["index_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal sequence shard index",
            value,
            {
                "schema_version",
                "kind",
                "protocol_sha256",
                "artifact_kind",
                "artifact_id",
                "total_rows",
                "shard_count",
                "ordered_shard_sha256s_sha256",
                "shards",
                "index_sha256",
            },
        )
        declared = _require_sha256(
            "formal sequence shard index", row.pop("index_sha256")
        )
        raw_shards = row.pop("shards")
        if type(raw_shards) is not list:
            raise TypeError("formal sequence shard references must be an array")
        index = cls(
            **row,
            shards=tuple(
                FormalCanonicalSequenceShardReference.from_dict(item)
                for item in raw_shards
            ),
        )  # type: ignore[arg-type]
        if index.sha256 != declared:
            raise ValueError("formal sequence index declared digest differs")
        return index

    def reopen_shard(self, shard_ordinal: int) -> FormalCanonicalSequenceShard:
        if type(shard_ordinal) is not int or not 0 <= shard_ordinal < self.shard_count:
            raise IndexError("formal sequence shard ordinal is outside the index")
        return self.shards[shard_ordinal].reopen(
            artifact_kind=self.artifact_kind,
            artifact_id=self.artifact_id,
        )

    def row_at(self, ordinal: int) -> object:
        if type(ordinal) is not int or not 0 <= ordinal < self.total_rows:
            raise IndexError("formal sequence row ordinal is outside the index")
        low = 0
        high = self.shard_count
        while low < high:
            middle = (low + high) // 2
            reference = self.shards[middle]
            if ordinal < reference.start_ordinal:
                high = middle
            elif ordinal >= reference.end_ordinal_exclusive:
                low = middle + 1
            else:
                shard = reference.reopen(
                    artifact_kind=self.artifact_kind,
                    artifact_id=self.artifact_id,
                )
                return shard.rows[ordinal - reference.start_ordinal]
        raise RuntimeError("formal sequence index interval search failed")

    def iter_rows(self) -> Iterator[object]:
        for ordinal in range(self.shard_count):
            yield from self.reopen_shard(ordinal).rows

    def revalidate(self) -> Self:
        count = 0
        for shard_ordinal in range(self.shard_count):
            count += len(self.reopen_shard(shard_ordinal).rows)
        if count != self.total_rows:
            raise ValueError("formal sequence deep replay count differs")
        return self


def _partition_rows(
    *,
    rows: Sequence[object],
    artifact_kind: str,
    artifact_id: str,
    maximum_shard_bytes: int,
    maximum_shard_rows: int,
) -> tuple[tuple[object, ...], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("formal sequence publisher requires non-empty rows")
    if (
        type(maximum_shard_bytes) is not int
        or not 1024 <= maximum_shard_bytes <= _DEFAULT_SHARD_MAX_BYTES
        or type(maximum_shard_rows) is not int
        or maximum_shard_rows < 1
    ):
        raise ValueError("formal sequence shard bounds are invalid")
    groups: list[tuple[object, ...]] = []
    current: list[object] = []
    start = 0
    for row in rows:
        candidate = (*current, row)
        probe = FormalCanonicalSequenceShard(
            schema_version=1,
            kind="formal_canonical_sequence_shard",
            protocol_sha256=FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            shard_ordinal=len(groups),
            start_ordinal=start,
            end_ordinal_exclusive=start + len(candidate),
            rows_sha256=_sha256(list(candidate)),
            rows=candidate,
        )
        encoded_size = len(_canonical_bytes(probe.to_dict())) + 1
        if not current and encoded_size > maximum_shard_bytes:
            raise ValueError("one formal sequence row exceeds shard byte bound")
        if current and (
            len(candidate) > maximum_shard_rows or encoded_size > maximum_shard_bytes
        ):
            groups.append(tuple(current))
            start += len(current)
            current = [row]
            single = FormalCanonicalSequenceShard(
                schema_version=1,
                kind="formal_canonical_sequence_shard",
                protocol_sha256=FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256,
                artifact_kind=artifact_kind,
                artifact_id=artifact_id,
                shard_ordinal=len(groups),
                start_ordinal=start,
                end_ordinal_exclusive=start + 1,
                rows_sha256=_sha256([row]),
                rows=(row,),
            )
            if len(_canonical_bytes(single.to_dict())) + 1 > maximum_shard_bytes:
                raise ValueError("one formal sequence row exceeds shard byte bound")
        else:
            current.append(row)
    groups.append(tuple(current))
    return tuple(groups)


def publish_formal_canonical_sequence_shards(
    *,
    artifact_kind: str,
    artifact_id: str,
    rows: Sequence[object],
    output_directory: str | Path,
    index_file_name: str = "index.json",
    maximum_shard_bytes: int = _DEFAULT_SHARD_MAX_BYTES,
    maximum_shard_rows: int = _DEFAULT_SHARD_MAX_ROWS,
) -> tuple[CanonicalJsonProofBinding, FormalCanonicalSequenceShardIndex]:
    _require_text("formal sequence artifact kind", artifact_kind)
    _require_sha256("formal sequence artifact ID", artifact_id)
    root = Path(output_directory)
    if (
        not root.is_absolute()
        or root != root.resolve(strict=False)
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise ValueError("formal sequence output directory must already exist")
    if Path(index_file_name).name != index_file_name or not index_file_name:
        raise ValueError("formal sequence index file name must be one basename")
    groups = _partition_rows(
        rows=rows,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        maximum_shard_bytes=maximum_shard_bytes,
        maximum_shard_rows=maximum_shard_rows,
    )
    references: list[FormalCanonicalSequenceShardReference] = []
    cursor = 0
    for shard_ordinal, group in enumerate(groups):
        shard = FormalCanonicalSequenceShard(
            schema_version=1,
            kind="formal_canonical_sequence_shard",
            protocol_sha256=FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            shard_ordinal=shard_ordinal,
            start_ordinal=cursor,
            end_ordinal_exclusive=cursor + len(group),
            rows_sha256=_sha256(list(group)),
            rows=group,
        )
        shard_path = root / f"shard-{shard_ordinal:06d}.json"
        publish_canonical_json_no_replace(shard_path, shard.to_dict())
        binding = CanonicalJsonProofBinding.bind(shard_path)
        if binding.semantic_sha256 != _sha256(shard.to_dict()):
            raise RuntimeError("formal sequence shard publication changed")
        references.append(
            FormalCanonicalSequenceShardReference(
                shard_ordinal=shard_ordinal,
                start_ordinal=shard.start_ordinal,
                end_ordinal_exclusive=shard.end_ordinal_exclusive,
                rows_sha256=shard.rows_sha256,
                shard_sha256=shard.sha256,
                binding=binding,
            )
        )
        cursor += len(group)
    index = FormalCanonicalSequenceShardIndex(
        schema_version=1,
        kind="formal_canonical_sequence_shard_index",
        protocol_sha256=FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        total_rows=len(rows),
        shard_count=len(references),
        ordered_shard_sha256s_sha256=_sha256([row.shard_sha256 for row in references]),
        shards=tuple(references),
    )
    index_path = root / index_file_name
    publish_canonical_json_no_replace(index_path, index.to_dict())
    binding = CanonicalJsonProofBinding.bind(index_path)
    if binding.semantic_sha256 != _sha256(index.to_dict()):
        raise RuntimeError("formal sequence index publication changed")
    return binding, index


def load_formal_canonical_sequence_shard_index(
    path: str | Path,
    *,
    deep: bool = False,
) -> FormalCanonicalSequenceShardIndex:
    binding = CanonicalJsonProofBinding.bind(path)
    index = FormalCanonicalSequenceShardIndex.from_dict(binding.reopen())
    if binding.semantic_sha256 != _sha256(index.to_dict()):
        raise ValueError("formal sequence index path identity differs")
    return index.revalidate() if deep else index


__all__ = [
    "FORMAL_SEQUENCE_SHARD_PROTOCOL_SHA256",
    "FormalCanonicalSequenceShard",
    "FormalCanonicalSequenceShardIndex",
    "FormalCanonicalSequenceShardReference",
    "load_formal_canonical_sequence_shard_index",
    "publish_formal_canonical_sequence_shards",
]
