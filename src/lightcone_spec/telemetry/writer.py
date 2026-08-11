"""Buffered, process-unique Parquet evidence writer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Self, TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq

from .records import (
    OUTPUT_HASH_FORMAT,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
)

EvidenceRecord: TypeAlias = (
    RunRecord | RequestRecord | RoundRecord | UpdateRecord | PerformanceRecord
)
_TABLE = {
    RunRecord: "run",
    RequestRecord: "request",
    RoundRecord: "round",
    UpdateRecord: "update",
    PerformanceRecord: "performance",
}
_EVIDENCE_METHODS = {
    "static",
    "tts",
    "naive_async",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_completed_evidence(
    root: str | Path,
    *,
    run_id: str,
    rank: int,
) -> dict[str, Path] | None:
    """Return the one hash-bound completed attempt for ``run_id``.

    Parquet shards without a terminal receipt are intentionally ignored. They
    are interrupted attempts, not evidence. More than one valid terminal is a
    protocol error because formal analysis must never double count a run.
    """
    directory = Path(root)
    receipts = sorted(directory.glob(f"{run_id}.rank{rank}.pid*.complete.json"))
    completed: list[dict[str, Path]] = []
    for receipt in receipts:
        value = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != 2
            or value.get("run_id") != run_id
            or value.get("rank") != rank
        ):
            raise RuntimeError(f"invalid completion receipt {receipt}")
        files = value.get("files")
        if not isinstance(files, dict) or not {"run", "performance"} <= set(files):
            raise RuntimeError(f"incomplete completion receipt {receipt}")
        resolved: dict[str, Path] = {}
        for table, entry in files.items():
            if table not in _TABLE.values() or not isinstance(entry, dict):
                raise RuntimeError(f"invalid evidence entry in {receipt}")
            name = entry.get("name")
            if not isinstance(name, str) or Path(name).name != name:
                raise RuntimeError(f"unsafe evidence path in {receipt}")
            path = directory / name
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != entry.get("size")
                or _sha256(path) != entry.get("sha256")
            ):
                raise RuntimeError(f"completion receipt does not bind {path}")
            resolved[table] = path
        run_table = pq.read_table(resolved["run"]).to_pylist()
        performance = pq.read_table(resolved["performance"]).to_pylist()
        method = run_table[0].get("method") if len(run_table) == 1 else None
        expected_tables = (
            {"run", "request", "performance"}
            if method == "static"
            else {"run", "request", "round", "update", "performance"}
        )
        table_rows = {
            table: pq.read_table(path).to_pylist()
            for table, path in resolved.items()
        }
        if (
            len(run_table) != 1
            or run_table[0].get("run_id") != run_id
            or run_table[0].get("status") != "complete"
            or method not in _EVIDENCE_METHODS
            or set(resolved) != expected_tables
            or not performance
            or any(row.get("run_id") != run_id for row in performance)
            or any(
                not rows or any(row.get("run_id") != run_id for row in rows)
                for rows in table_rows.values()
            )
            or any(
                row.get("method") != method
                for table in ("request", "performance")
                for row in table_rows[table]
            )
            or any(
                row.get("output_hash_format") != OUTPUT_HASH_FORMAT
                for row in table_rows["request"]
            )
        ):
            raise RuntimeError(f"receipt binds invalid run evidence {receipt}")
        completed.append(resolved)
    if len(completed) > 1:
        raise RuntimeError(f"multiple completed attempts exist for run {run_id}")
    return completed[0] if completed else None


class EvidenceWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        rank: int,
        process_id: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.rank = rank
        self.process_id = os.getpid() if process_id is None else process_id
        if load_completed_evidence(self.root, run_id=run_id, rank=rank) is not None:
            raise RuntimeError(f"completed evidence already exists for run {run_id}")
        base_prefix = f"{run_id}.rank{rank}.pid{self.process_id}"
        self.prefix = base_prefix
        if any(self.root.glob(f"{base_prefix}.*")):
            self.prefix = f"{base_prefix}.attempt{time.time_ns()}"
        self._rows: dict[str, list[dict]] = {
            name: [] for name in _TABLE.values()
        }
        self._closed = False

    def write(self, record: EvidenceRecord) -> None:
        if self._closed:
            raise RuntimeError("evidence writer is closed")
        try:
            table = _TABLE[type(record)]
        except KeyError as exc:
            raise TypeError(f"unsupported evidence record {type(record)!r}") from exc
        if record.run_id != self.run_id:
            raise ValueError("evidence record belongs to another run")
        self._rows[table].append(asdict(record))

    def close(self) -> dict[str, Path]:
        if self._closed:
            raise RuntimeError("evidence writer already closed")
        if len(self._rows["run"]) != 1 or not self._rows["performance"]:
            raise RuntimeError(
                "a complete attempt requires exactly one run and performance evidence"
            )
        run = self._rows["run"][0]
        method = run.get("method")
        expected_tables = (
            {"run", "request", "performance"}
            if method == "static"
            else {"run", "request", "round", "update", "performance"}
        )
        populated = {name for name, rows in self._rows.items() if rows}
        if (
            run.get("run_id") != self.run_id
            or run.get("status") != "complete"
            or method not in _EVIDENCE_METHODS
            or populated != expected_tables
            or any(
                row.get("method") != method
                for table in ("request", "performance")
                for row in self._rows[table]
            )
            or any(
                row.get("output_hash_format") != OUTPUT_HASH_FORMAT
                for row in self._rows["request"]
            )
        ):
            raise RuntimeError(
                "evidence tables do not satisfy the method completion contract"
            )
        written: dict[str, Path] = {}
        receipt_path = self.root / f"{self.prefix}.complete.json"
        if receipt_path.exists():
            raise RuntimeError(f"refusing to overwrite evidence receipt {receipt_path}")
        for name, rows in self._rows.items():
            if not rows:
                continue
            output = self.root / f"{self.prefix}.{name}.parquet"
            if output.exists():
                raise RuntimeError(f"refusing to overwrite evidence shard {output}")
            temporary = output.with_suffix(output.suffix + ".tmp")
            pq.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, output)
            written[name] = output
        receipt = {
            "schema_version": 2,
            "run_id": self.run_id,
            "rank": self.rank,
            "process_id": self.process_id,
            "files": {
                name: {
                    "name": path.name,
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
                for name, path in sorted(written.items())
            },
        }
        temporary_receipt = receipt_path.with_suffix(".json.tmp")
        temporary_receipt.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_receipt, receipt_path)
        self._closed = True
        return written

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None and not self._closed:
            self.close()
