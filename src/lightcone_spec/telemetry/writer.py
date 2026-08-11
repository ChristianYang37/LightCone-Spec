"""Bounded, durable, process-unique Parquet evidence writer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal, Self, get_args, get_type_hints

import pyarrow as pa
import pyarrow.parquet as pq

from .records import (
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
)

type EvidenceRecord = (
    RunRecord | RequestRecord | RoundRecord | UpdateRecord | PerformanceRecord
)
type OverflowPolicy = Literal["backpressure", "drop"]

_TABLE = {
    RunRecord: "run",
    RequestRecord: "request",
    RoundRecord: "round",
    UpdateRecord: "update",
    PerformanceRecord: "performance",
}
_RECORD = {table: record for record, table in _TABLE.items()}
_EVIDENCE_METHODS = {
    "target_only",
    "static",
    "tts",
    "l0",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
}
_ALLOCATION_FREE_METHODS = {"target_only", "static"}
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOWER_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_INDUSTRIAL_WORKLOAD_CONTRACTS = {
    "industrial_target_only",
    "industrial_static",
    "industrial_adapted",
    "industrial_preflight_target_only",
    "industrial_preflight_static",
}
_INDUSTRIAL_HASH_FIELDS = (
    "industrial_cell_id",
    "runtime_sha256",
    "split_sha256",
    "corpus_sha256",
    "arrival_trace_sha256",
    "request_ids_sha256",
    "sampling_profile_sha256",
    "model_lock_sha256",
    "run_nonce_sha256",
    "topology_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
    body = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _publish_receipt_exclusive(path: Path, value: object) -> None:
    """Publish one canonical terminal receipt without an overwrite race."""
    temporary = path.with_name(f"{path.name}.candidate.{uuid.uuid4().hex}")
    body = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        temporary.unlink()
        raise RuntimeError(
            f"completed evidence already exists for run {value['run_id']}"
        ) from exc
    _fsync_directory(path.parent)
    # The candidate is deliberately retained. Removing it after the canonical
    # receipt would make receipt publication no longer the final filesystem
    # mutation, and it remains outside every completion glob.


def _arrow_type(annotation: object) -> pa.DataType:
    optional = [
        argument for argument in get_args(annotation) if argument is not type(None)
    ]
    if optional:
        if len(optional) != 1:
            raise TypeError(f"unsupported evidence annotation {annotation!r}")
        annotation = optional[0]
    if annotation is bool:
        return pa.bool_()
    if annotation is int:
        return pa.int64()
    if annotation is float:
        return pa.float64()
    if annotation is str:
        return pa.string()
    raise TypeError(f"unsupported evidence annotation {annotation!r}")


def _schema(record_type: type[EvidenceRecord]) -> pa.Schema:
    annotations = get_type_hints(record_type)
    return pa.schema(
        [
            pa.field(field.name, _arrow_type(annotations[field.name]), nullable=True)
            for field in fields(record_type)
        ]
    )


_SCHEMAS = {table: _schema(record) for table, record in _RECORD.items()}
_REQUIRED_FIELDS = {
    table: {
        name
        for name, annotation in get_type_hints(record).items()
        if type(None) not in get_args(annotation)
    }
    for table, record in _RECORD.items()
}


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validate_row(table: str, row: dict[str, object]) -> None:
    missing = [name for name in _REQUIRED_FIELDS[table] if row.get(name) is None]
    if missing:
        raise ValueError(f"required {table} evidence is missing {sorted(missing)}")
    for name, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{table}.{name} must be finite when measured")
    if table == "request":
        coverage = row.get("token_timing_coverage")
        coalesced = row.get("coalesced_intervals")
        retry = row.get("retry_attempt")
        if coverage is not None and not 0.0 <= float(coverage) <= 1.0:
            raise ValueError("request token_timing_coverage must be within [0, 1]")
        if coalesced is not None and int(coalesced) < 0:
            raise ValueError("request coalesced_intervals cannot be negative")
        if retry is not None and int(retry) < 0:
            raise ValueError("request retry_attempt cannot be negative")
    if table == "run":
        contract = row.get("workload_contract")
        industrial_values = tuple(row.get(name) for name in _INDUSTRIAL_HASH_FIELDS) + (
            row.get("rank_config_sha256"),
            row.get("patched_sglang_tree"),
            row.get("tensor_parallel_size"),
            row.get("data_parallel_size"),
            row.get("world_size"),
            row.get("rank"),
            row.get("expected_request_rows"),
            row.get("expected_round_rows"),
            row.get("expected_update_rows"),
            row.get("expected_performance_rows"),
            row.get("preflight_attestation_sha256"),
        )
        if contract is None:
            if any(value is not None for value in industrial_values):
                raise ValueError(
                    "industrial run identities require an explicit workload contract"
                )
            return
        if contract not in _INDUSTRIAL_WORKLOAD_CONTRACTS:
            raise ValueError("unknown industrial workload contract")
        method = str(row.get("method"))
        expected_contracts = {
            "target_only": {
                "industrial_target_only",
                "industrial_preflight_target_only",
            },
            "static": {"industrial_static", "industrial_preflight_static"},
        }
        if method in expected_contracts:
            if contract not in expected_contracts[method]:
                raise ValueError("industrial workload contract disagrees with method")
        elif contract != "industrial_adapted":
            raise ValueError("adapted methods require industrial_adapted")
        for name in _INDUSTRIAL_HASH_FIELDS:
            value = row.get(name)
            if not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None:
                raise ValueError(f"run.{name} must be a lowercase SHA-256")
        rank_config_sha256 = row.get("rank_config_sha256")
        if contract in {
            "industrial_preflight_target_only",
            "industrial_preflight_static",
        }:
            if rank_config_sha256 is not None:
                raise ValueError("preflight runs cannot claim a serving RunConfig")
        elif (
            not isinstance(rank_config_sha256, str)
            or _LOWER_SHA256.fullmatch(rank_config_sha256) is None
        ):
            raise ValueError("serving runs require a lowercase rank-config SHA-256")
        patched_tree = row.get("patched_sglang_tree")
        if (
            not isinstance(patched_tree, str)
            or _LOWER_GIT_OBJECT.fullmatch(patched_tree) is None
        ):
            raise ValueError("run.patched_sglang_tree must be a lowercase Git tree")
        counters = {
            name: row.get(name)
            for name in (
                "tensor_parallel_size",
                "data_parallel_size",
                "world_size",
                "rank",
                "expected_request_rows",
                "expected_round_rows",
                "expected_update_rows",
                "expected_performance_rows",
            )
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counters.values()
        ):
            raise ValueError("industrial run topology and row counts must be integers")
        if (
            counters["tensor_parallel_size"] < 1
            or counters["data_parallel_size"] < 1
            or counters["world_size"] < 1
            or counters["rank"] >= counters["world_size"]
            or counters["expected_request_rows"] < 1
            or counters["expected_performance_rows"] < 1
        ):
            raise ValueError(
                "industrial run topology or required row counts are invalid"
            )
        preflight = str(contract).startswith("industrial_preflight_")
        attestation = row.get("preflight_attestation_sha256")
        if preflight:
            if (
                not isinstance(attestation, str)
                or _LOWER_SHA256.fullmatch(attestation) is None
            ):
                raise ValueError("preflight runs require a bound attestation")
        elif attestation is not None:
            raise ValueError("only preflight runs may bind a preflight attestation")
        expected = _expected_tables(str(row["method"]), str(contract))
        if ("round" in expected) != (counters["expected_round_rows"] > 0):
            raise ValueError(
                "industrial round coverage disagrees with workload contract"
            )
        if ("update" in expected) != (counters["expected_update_rows"] > 0):
            raise ValueError(
                "industrial update coverage disagrees with workload contract"
            )


def _expected_tables(method: str, workload_contract: str | None = None) -> set[str]:
    if workload_contract is not None:
        if workload_contract not in _INDUSTRIAL_WORKLOAD_CONTRACTS:
            return set()
        if workload_contract in {
            "industrial_target_only",
            "industrial_preflight_target_only",
        }:
            return {"run", "request", "performance"}
        if workload_contract in {
            "industrial_static",
            "industrial_preflight_static",
        }:
            return {"run", "request", "performance"}
        return {"run", "request", "round", "update", "performance"}
    if method in _ALLOCATION_FREE_METHODS:
        return {"run", "request", "performance"}
    return {"run", "request", "round", "update", "performance"}


def _read_identity_columns(
    path: Path,
    *,
    table: str,
    run_id: str,
    method: str | None,
) -> None:
    columns = ["run_id"]
    if table in {"run", "request", "performance"}:
        columns.append("method")
    if table == "run":
        columns.append("status")
    try:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows < 1:
            raise RuntimeError(f"empty completed evidence table {path}")
        seen = 0
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            values = batch.to_pydict()
            count = batch.num_rows
            seen += count
            if any(value != run_id for value in values["run_id"]):
                raise RuntimeError(f"cross-run row in completed evidence {path}")
            if "method" in values and any(
                value != method for value in values["method"]
            ):
                raise RuntimeError(f"cross-method row in completed evidence {path}")
            if table == "run" and any(
                value != "complete" for value in values["status"]
            ):
                raise RuntimeError(f"nonterminal run row in completed evidence {path}")
        if seen != parquet.metadata.num_rows:
            raise RuntimeError(f"incomplete Parquet scan for {path}")
        if table == "run" and seen != 1:
            raise RuntimeError(f"completed evidence must contain one run row: {path}")
    except (KeyError, pa.ArrowException) as exc:
        raise RuntimeError(f"invalid completed evidence table {path}") from exc


def _load_receipt(path: Path, *, run_id: str, rank: int) -> dict[str, Path]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"invalid completion receipt {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completion receipt {path}") from exc
    schema_version = value.get("schema_version")
    if (
        schema_version not in {2, 3}
        or value.get("run_id") != run_id
        or value.get("rank") != rank
    ):
        raise RuntimeError(f"invalid completion receipt {path}")
    files = value.get("files")
    if not isinstance(files, dict) or not {"run", "performance"} <= set(files):
        raise RuntimeError(f"incomplete completion receipt {path}")
    prefix = value.get("prefix")
    if schema_version == 3 and (
        not isinstance(prefix, str) or not _SAFE_COMPONENT.fullmatch(prefix)
    ):
        raise RuntimeError(f"invalid attempt identity in {path}")
    resolved: dict[str, Path] = {}
    metadata: dict[str, pq.FileMetaData] = {}
    for table, entry in files.items():
        if table not in _TABLE.values() or not isinstance(entry, dict):
            raise RuntimeError(f"invalid evidence entry in {path}")
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or (schema_version == 3 and not name.startswith(f"{prefix}."))
        ):
            raise RuntimeError(f"unsafe or cross-attempt evidence path in {path}")
        shard = path.parent / name
        if (
            not shard.is_file()
            or shard.is_symlink()
            or shard.stat().st_size != entry.get("size")
            or _sha256(shard) != entry.get("sha256")
        ):
            raise RuntimeError(f"completion receipt does not bind {shard}")
        try:
            parquet = pq.ParquetFile(shard)
            parquet_metadata = parquet.metadata
        except pa.ArrowException as exc:
            raise RuntimeError(f"invalid completed evidence table {shard}") from exc
        if schema_version == 3 and (
            parquet_metadata.num_rows != entry.get("rows")
            or parquet_metadata.num_row_groups != entry.get("row_groups")
            or _schema_sha256(parquet.schema_arrow) != entry.get("schema_sha256")
        ):
            raise RuntimeError(f"completion receipt has wrong row coverage for {shard}")
        resolved[table] = shard
        metadata[table] = parquet_metadata
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError(f"duplicate evidence shard in {path}")

    run_parquet = pq.ParquetFile(resolved["run"])
    run_columns = ["method"]
    if "workload_contract" in run_parquet.schema_arrow.names:
        run_columns.append("workload_contract")
    run_rows = run_parquet.iter_batches(batch_size=1, columns=run_columns)
    try:
        run_batch = next(run_rows)
        method = run_batch.column(0)[0].as_py()
        workload_contract = (
            run_batch.column(1)[0].as_py() if len(run_columns) == 2 else None
        )
    except (StopIteration, KeyError, pa.ArrowException) as exc:
        raise RuntimeError(f"receipt binds invalid run evidence {path}") from exc
    expected = (
        _expected_tables(method, workload_contract)
        if method in _EVIDENCE_METHODS
        else set()
    )
    if set(resolved) != expected:
        raise RuntimeError(f"receipt binds invalid run evidence {path}")
    if schema_version == 3:
        coverage = value.get("coverage")
        counters = value.get("counters")
        checkpoint = value.get("checkpoint")
        if not isinstance(coverage, dict) or set(coverage) != expected:
            raise RuntimeError(f"receipt lacks exact table coverage {path}")
        if not isinstance(counters, dict):
            raise RuntimeError(f"receipt lacks evidence counters {path}")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"receipt lacks a checkpoint binding {path}")
        checkpoint_name = checkpoint.get("name")
        if (
            not isinstance(checkpoint_name, str)
            or checkpoint_name != f"{prefix}.checkpoint.json"
        ):
            raise RuntimeError(f"receipt has an invalid checkpoint path {path}")
        checkpoint_path = path.parent / checkpoint_name
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.is_symlink()
            or checkpoint_path.stat().st_size != checkpoint.get("size")
            or _sha256(checkpoint_path) != checkpoint.get("sha256")
        ):
            raise RuntimeError(f"completion receipt does not bind {checkpoint_path}")
        dropped = counters.get("dropped_by_table")
        if not isinstance(dropped, dict) or any(
            dropped.get(table) != 0 for table in expected
        ):
            raise RuntimeError(f"completed receipt reports evidence loss {path}")
        for table in expected:
            entry = coverage.get(table)
            if not isinstance(entry, dict) or (
                entry.get("rows") != metadata[table].num_rows
                or entry.get("row_groups") != metadata[table].num_row_groups
            ):
                raise RuntimeError(f"receipt has inconsistent coverage for {table}")
    for table, shard in resolved.items():
        _read_identity_columns(shard, table=table, run_id=run_id, method=method)
    return resolved


def load_completed_evidence(
    root: str | Path,
    *,
    run_id: str,
    rank: int,
) -> dict[str, Path] | None:
    """Return the one hash-bound completed attempt for ``run_id``.

    WAL shards, checkpoints, aborted attempts, and final Parquet shards without
    a terminal receipt are intentionally ignored. Legacy schema-v2 attempt
    receipts remain readable so a valid completed run stays immutable.
    """
    if not _SAFE_COMPONENT.fullmatch(run_id):
        raise ValueError("run_id must be a safe non-empty path component")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    directory = Path(root)
    canonical = directory / f"{run_id}.rank{rank}.complete.json"
    receipts = [canonical] if os.path.lexists(canonical) else []
    receipts.extend(sorted(directory.glob(f"{run_id}.rank{rank}.pid*.complete.json")))
    completed = [
        _load_receipt(receipt, run_id=run_id, rank=rank) for receipt in receipts
    ]
    if len(completed) > 1:
        raise RuntimeError(f"multiple completed attempts exist for run {run_id}")
    return completed[0] if completed else None


class EvidenceWriter:
    """Write one bounded attempt and publish a receipt only after durability."""

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        rank: int,
        process_id: int | None = None,
        attempt_id: str | None = None,
        max_queued_rows: int = 1024,
        row_group_rows: int = 256,
        checkpoint_interval_s: float | None = 5.0,
        overflow_policy: OverflowPolicy = "backpressure",
    ) -> None:
        if not _SAFE_COMPONENT.fullmatch(run_id):
            raise ValueError("run_id must be a safe non-empty path component")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        resolved_process_id = os.getpid() if process_id is None else process_id
        if (
            not isinstance(resolved_process_id, int)
            or isinstance(resolved_process_id, bool)
            or resolved_process_id < 0
        ):
            raise ValueError("process_id must be a non-negative integer")
        if (
            not isinstance(max_queued_rows, int)
            or isinstance(max_queued_rows, bool)
            or max_queued_rows < 1
        ):
            raise ValueError("max_queued_rows must be positive")
        if (
            not isinstance(row_group_rows, int)
            or isinstance(row_group_rows, bool)
            or row_group_rows < 1
        ):
            raise ValueError("row_group_rows must be positive")
        if checkpoint_interval_s is not None and (
            not math.isfinite(checkpoint_interval_s) or checkpoint_interval_s <= 0
        ):
            raise ValueError("checkpoint_interval_s must be positive or None")
        if overflow_policy not in {"backpressure", "drop"}:
            raise ValueError("overflow_policy must be backpressure or drop")

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.rank = rank
        self.process_id = resolved_process_id
        if load_completed_evidence(self.root, run_id=run_id, rank=rank) is not None:
            raise RuntimeError(f"completed evidence already exists for run {run_id}")
        if attempt_id is None:
            attempt_id = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        if not _SAFE_COMPONENT.fullmatch(attempt_id):
            raise ValueError("attempt_id must be a safe non-empty path component")
        self.attempt_id = attempt_id
        self.prefix = (
            f"{run_id}.rank{rank}.pid{self.process_id}.attempt{self.attempt_id}"
        )
        if any(self.root.glob(f"{self.prefix}.*")):
            raise RuntimeError(f"attempt path already exists for {self.prefix}")

        self.max_queued_rows = max_queued_rows
        self.row_group_rows = row_group_rows
        self.checkpoint_interval_s = checkpoint_interval_s
        self.overflow_policy = overflow_policy
        self._queues: dict[str, list[dict[str, object]]] = {
            table: [] for table in _TABLE.values()
        }
        self._queued_rows = 0
        self._row_counts = {table: 0 for table in _TABLE.values()}
        self._row_groups = {table: 0 for table in _TABLE.values()}
        self._dropped_by_table = {table: 0 for table in _TABLE.values()}
        self._methods_by_table = {
            table: set() for table in ("run", "request", "performance")
        }
        self._run_record: dict[str, object] | None = None
        self._flushes = 0
        self._backpressure_events = 0
        self._backpressured_rows = 0
        self._dropped_rows = 0
        self._max_observed_queued_rows = 0
        self._last_checkpoint = time.monotonic()
        self._closed = False
        self._state = "open"
        self._checkpoint_path = self.root / f"{self.prefix}.checkpoint.json"
        self._index_path = self.root / f"{self.prefix}.index.sqlite3"
        # The industrial executor drains one bounded async queue through a
        # single background worker, then joins it before close/abort. Disabling
        # SQLite's creator-thread check is safe under that single-writer
        # lifecycle and keeps filesystem work off the measured event loop.
        self._index = sqlite3.connect(self._index_path, check_same_thread=False)
        self._index.execute("PRAGMA journal_mode=WAL")
        self._index.execute("PRAGMA synchronous=FULL")
        self._index.execute(
            "CREATE TABLE seen (table_name TEXT NOT NULL, evidence_key TEXT NOT NULL, "
            "PRIMARY KEY (table_name, evidence_key)) WITHOUT ROWID"
        )
        self._write_checkpoint()

    @property
    def queued_rows(self) -> int:
        return self._queued_rows

    @property
    def dropped_rows(self) -> int:
        return self._dropped_rows

    @property
    def backpressure_events(self) -> int:
        return self._backpressure_events

    def register_external_backpressure_events(self, count: int) -> None:
        """Bind fail-closed producer-queue saturation to the durable attempt."""

        if self._closed:
            raise RuntimeError("evidence writer is closed")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("external backpressure count must be a positive integer")
        self._backpressure_events += count
        self._backpressured_rows += count
        self._write_checkpoint()

    @property
    def counters(self) -> dict[str, object]:
        return {
            "backpressure_events": self._backpressure_events,
            "backpressured_rows": self._backpressured_rows,
            "dropped_rows": self._dropped_rows,
            "dropped_by_table": dict(self._dropped_by_table),
            "flushes": self._flushes,
            "max_observed_queued_rows": self._max_observed_queued_rows,
        }

    def _identity(self, table: str, row: dict[str, object]) -> tuple[object, ...]:
        if table == "run":
            return ("run",)
        if table == "request":
            return (row["request_id"],)
        if table == "round":
            return (row["request_id"], row["round_index"])
        if table == "update":
            return (row["cohort_sha256"], row["update_index"])
        return (
            row["prompt_id"],
            row["method"],
            row["repetition_block"],
            row["region"],
            row["concurrency"],
            row["generated_bucket_start"],
            row["generated_bucket_end"],
        )

    def _register_identity(self, table: str, row: dict[str, object]) -> None:
        key = json.dumps(self._identity(table, row), separators=(",", ":"))
        try:
            self._index.execute(
                "INSERT INTO seen(table_name, evidence_key) VALUES (?, ?)",
                (table, key),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate {table} evidence identity") from exc

    def _checkpoint_value(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "run_id": self.run_id,
            "rank": self.rank,
            "process_id": self.process_id,
            "attempt_id": self.attempt_id,
            "prefix": self.prefix,
            "state": self._state,
            "durable_rows": dict(self._row_counts),
            "wal_row_groups": dict(self._row_groups),
            "queued_rows": self._queued_rows,
            "counters": self.counters,
        }

    def _write_checkpoint(self) -> None:
        _atomic_json(self._checkpoint_path, self._checkpoint_value())
        self._last_checkpoint = time.monotonic()

    def _wal_path(self, table: str) -> Path:
        sequence = self._row_groups[table]
        return self.root / f"{self.prefix}.{table}.wal.{sequence:08d}.parquet"

    def _write_wal_segment(self, table: str, rows: list[dict[str, object]]) -> None:
        output = self._wal_path(table)
        if output.exists():
            raise RuntimeError(f"refusing to overwrite evidence WAL {output}")
        temporary = output.with_name(f"{output.name}.tmp.{uuid.uuid4().hex}")
        evidence = pa.Table.from_pylist(rows, schema=_SCHEMAS[table])
        pq.write_table(evidence, temporary, row_group_size=len(rows))
        _fsync_file(temporary)
        os.replace(temporary, output)
        _fsync_directory(self.root)

    def write(self, record: EvidenceRecord) -> bool:
        """Queue one row, returning ``False`` only for explicit drop policy."""
        if self._closed:
            raise RuntimeError("evidence writer is closed")
        try:
            table = _TABLE[type(record)]
        except KeyError as exc:
            raise TypeError(f"unsupported evidence record {type(record)!r}") from exc
        if record.run_id != self.run_id:
            raise ValueError("evidence record belongs to another run")
        row = asdict(record)
        _validate_row(table, row)
        now = time.monotonic()
        if (
            self.checkpoint_interval_s is not None
            and self._queued_rows
            and now - self._last_checkpoint >= self.checkpoint_interval_s
        ):
            self.flush()
        if self._queued_rows >= self.max_queued_rows:
            if self.overflow_policy == "drop":
                self._dropped_rows += 1
                self._dropped_by_table[table] += 1
                self._write_checkpoint()
                return False
            self._backpressure_events += 1
            self._backpressured_rows += 1
            self.flush()

        self._register_identity(table, row)
        self._queues[table].append(row)
        self._queued_rows += 1
        self._max_observed_queued_rows = max(
            self._max_observed_queued_rows, self._queued_rows
        )
        if table in self._methods_by_table:
            self._methods_by_table[table].add(str(row["method"]))
        if table == "run":
            self._run_record = row
        if self._queued_rows >= self.row_group_rows:
            self.flush()
        return True

    def flush(self) -> dict[str, int]:
        """Durably checkpoint all queued rows as inspectable Parquet WALs."""
        if self._closed:
            raise RuntimeError("evidence writer is closed")
        flushed = {table: 0 for table in _TABLE.values()}
        for table, rows in self._queues.items():
            if not rows:
                continue
            self._write_wal_segment(table, rows)
            count = len(rows)
            flushed[table] = count
            self._row_counts[table] += count
            self._row_groups[table] += 1
            self._queued_rows -= count
            self._queues[table] = []
        if any(flushed.values()):
            self._flushes += 1
        self._index.commit()
        self._write_checkpoint()
        return flushed

    def _validate_completion(self) -> tuple[str, set[str]]:
        if self._row_counts["run"] != 1 or self._run_record is None:
            raise RuntimeError("a complete attempt requires exactly one run row")
        method = str(self._run_record.get("method"))
        workload_contract = self._run_record.get("workload_contract")
        expected = (
            _expected_tables(
                method,
                None if workload_contract is None else str(workload_contract),
            )
            if method in _EVIDENCE_METHODS
            else set()
        )
        populated = {
            table for table, row_count in self._row_counts.items() if row_count
        }
        if (
            self._run_record.get("run_id") != self.run_id
            or self._run_record.get("status") != "complete"
            or self._run_record.get("completed_ns") is None
            or not expected
            or populated != expected
            or any(self._dropped_by_table[table] for table in expected)
            or self._methods_by_table["run"] != {method}
            or self._methods_by_table["request"] != {method}
            or self._methods_by_table["performance"] != {method}
        ):
            raise RuntimeError(
                "evidence tables do not satisfy the method completion contract"
            )
        if workload_contract is not None:
            expected_counts = {
                "request": self._run_record.get("expected_request_rows"),
                "round": self._run_record.get("expected_round_rows"),
                "update": self._run_record.get("expected_update_rows"),
                "performance": self._run_record.get("expected_performance_rows"),
            }
            if any(
                self._row_counts[table] != expected_counts[table]
                for table in expected_counts
            ):
                raise RuntimeError(
                    "industrial evidence row counts do not match the locked contract"
                )
            if self._run_record.get("rank") != self.rank:
                raise RuntimeError(
                    "industrial run rank differs from its evidence writer"
                )
        return method, expected

    def _build_final_table(self, table: str) -> Path:
        segments = sorted(self.root.glob(f"{self.prefix}.{table}.wal.*.parquet"))
        if len(segments) != self._row_groups[table]:
            raise RuntimeError(f"evidence WAL coverage changed for {table}")
        output = self.root / f"{self.prefix}.{table}.parquet"
        if output.exists():
            raise RuntimeError(f"refusing to overwrite evidence shard {output}")
        temporary = output.with_name(f"{output.name}.tmp.{uuid.uuid4().hex}")
        writer = pq.ParquetWriter(temporary, _SCHEMAS[table])
        try:
            for segment in segments:
                if segment.is_symlink():
                    raise RuntimeError(f"evidence WAL cannot be a symlink: {segment}")
                parquet = pq.ParquetFile(segment)
                if parquet.metadata.num_rows > self.max_queued_rows:
                    raise RuntimeError(f"evidence WAL exceeded queue bound: {segment}")
                rows = parquet.read()
                writer.write_table(rows, row_group_size=rows.num_rows)
        finally:
            writer.close()
        _fsync_file(temporary)
        os.replace(temporary, output)
        _fsync_directory(self.root)
        return output

    def close(self) -> dict[str, Path]:
        if self._closed:
            raise RuntimeError("evidence writer already closed")
        self.flush()
        _, expected = self._validate_completion()
        if (
            load_completed_evidence(self.root, run_id=self.run_id, rank=self.rank)
            is not None
        ):
            raise RuntimeError(
                f"completed evidence already exists for run {self.run_id}"
            )
        self._state = "publishing"
        self._write_checkpoint()
        written = {table: self._build_final_table(table) for table in sorted(expected)}
        coverage = {
            table: {
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
            }
            for table, path in sorted(written.items())
        }
        receipt = {
            "schema_version": 3,
            "run_id": self.run_id,
            "rank": self.rank,
            "process_id": self.process_id,
            "attempt_id": self.attempt_id,
            "prefix": self.prefix,
            "checkpoint": {
                "name": self._checkpoint_path.name,
                "sha256": _sha256(self._checkpoint_path),
                "size": self._checkpoint_path.stat().st_size,
            },
            "coverage": coverage,
            "counters": self.counters,
            "files": {
                table: {
                    "name": path.name,
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                    "schema_sha256": _schema_sha256(pq.ParquetFile(path).schema_arrow),
                    **coverage[table],
                }
                for table, path in sorted(written.items())
            },
        }
        self._index.commit()
        self._index.close()
        receipt_path = self.root / f"{self.run_id}.rank{self.rank}.complete.json"
        _publish_receipt_exclusive(receipt_path, receipt)
        self._state = "complete"
        self._closed = True
        return written

    def abort(self, reason: str | None = None) -> None:
        """Durably retain an inspectable attempt without completing it."""
        if self._closed:
            raise RuntimeError("evidence writer is closed")
        self.flush()
        self._state = "aborted"
        aborted = {
            **self._checkpoint_value(),
            "reason": reason,
        }
        self._index.commit()
        self._index.close()
        _atomic_json(self.root / f"{self.prefix}.aborted.json", aborted)
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._closed:
            return
        if exc_type is None:
            self.close()
            return
        try:
            self.abort(reason=f"{getattr(exc_type, '__name__', 'exception')}: {exc}")
        except (OSError, RuntimeError, sqlite3.Error, pa.ArrowException):
            # Never hide the original workload failure. Any already-published
            # WAL/checkpoint remains inspectable and no completion receipt exists.
            return
