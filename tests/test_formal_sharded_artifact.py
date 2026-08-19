from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lightcone_spec.runtime.formal_sharded_artifact import (
    FormalCanonicalSequenceShardIndex,
    load_formal_canonical_sequence_shard_index,
    publish_formal_canonical_sequence_shards,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def test_formal_sequence_shards_roundtrip_and_scale_to_ten_thousand_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sequence"
    root.mkdir()
    rows = tuple(
        {"ordinal": ordinal, "request_id": _sha(f"request:{ordinal}")}
        for ordinal in range(10_000)
    )

    binding, index = publish_formal_canonical_sequence_shards(
        artifact_kind="test_request_rows",
        artifact_id=_sha("test request sequence"),
        rows=rows,
        output_directory=root,
        maximum_shard_rows=128,
    )

    assert index.total_rows == 10_000
    assert index.shard_count == 79
    assert index.row_at(0) == rows[0]
    assert index.row_at(5_123) == rows[5_123]
    assert index.row_at(9_999) == rows[-1]
    rebound = load_formal_canonical_sequence_shard_index(
        binding.absolute_path,
        deep=True,
    )
    assert rebound == index
    assert tuple(rebound.iter_rows()) == rows
    assert FormalCanonicalSequenceShardIndex.from_dict(index.to_dict()) == index


def test_formal_sequence_shard_mutation_is_detected_by_lazy_lookup(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sequence"
    root.mkdir()
    _binding, index = publish_formal_canonical_sequence_shards(
        artifact_kind="test_request_rows",
        artifact_id=_sha("test request sequence"),
        rows=tuple({"ordinal": ordinal} for ordinal in range(4)),
        output_directory=root,
        maximum_shard_rows=2,
    )
    second_path = Path(index.shards[1].binding.absolute_path)
    value = json.loads(second_path.read_text(encoding="utf-8"))
    value["rows"][0]["ordinal"] = 999
    second_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    assert index.row_at(0) == {"ordinal": 0}
    with pytest.raises(ValueError, match="changed"):
        index.row_at(2)


def test_formal_sequence_shards_reject_one_unbounded_row(tmp_path: Path) -> None:
    root = tmp_path / "sequence"
    root.mkdir()
    with pytest.raises(ValueError, match="one formal sequence row exceeds"):
        publish_formal_canonical_sequence_shards(
            artifact_kind="test_request_rows",
            artifact_id=_sha("test request sequence"),
            rows=({"payload": "x" * 2_000},),
            output_directory=root,
            maximum_shard_bytes=1_024,
        )
