"""Bounded, durable, process-unique Parquet evidence writer."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Self, get_args, get_type_hints

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

type EvidenceRecord = (
    RunRecord | RequestRecord | RoundRecord | UpdateRecord | PerformanceRecord
)
type OverflowPolicy = Literal["backpressure", "drop"]


@dataclass(frozen=True)
class EvidenceWriterPolicy:
    """Registered batching, checkpoint, and durability policy for one run."""

    schema_version: int
    async_queue_rows: int
    async_batch_rows: int
    writer_queue_rows: int
    parquet_row_group_rows: int
    checkpoint_interval_ms: int
    overflow_policy: OverflowPolicy
    sqlite_journal_mode: str
    sqlite_synchronous: str
    wal_fsync: bool
    directory_fsync: bool
    checkpoint_fsync: bool

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only evidence-writer policy schema 1 is supported")
        for name in (
            "async_queue_rows",
            "async_batch_rows",
            "writer_queue_rows",
            "parquet_row_group_rows",
            "checkpoint_interval_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.async_batch_rows > self.async_queue_rows:
            raise ValueError("async evidence batch exceeds its queue bound")
        if self.parquet_row_group_rows > self.writer_queue_rows:
            raise ValueError("Parquet row group exceeds the writer queue bound")
        if self.overflow_policy != "backpressure":
            raise ValueError("claimable evidence requires backpressure overflow policy")
        if self.sqlite_journal_mode != "WAL":
            raise ValueError("evidence index requires SQLite WAL mode")
        if self.sqlite_synchronous != "FULL":
            raise ValueError("evidence index requires SQLite FULL synchronization")
        if not (self.wal_fsync and self.directory_fsync and self.checkpoint_fsync):
            raise ValueError("claimable evidence requires every registered fsync gate")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "kind": "industrial_evidence_writer_policy",
            "async_queue_rows": self.async_queue_rows,
            "async_batch_rows": self.async_batch_rows,
            "writer_queue_rows": self.writer_queue_rows,
            "parquet_row_group_rows": self.parquet_row_group_rows,
            "checkpoint_interval_ms": self.checkpoint_interval_ms,
            "overflow_policy": self.overflow_policy,
            "sqlite_journal_mode": self.sqlite_journal_mode,
            "sqlite_synchronous": self.sqlite_synchronous,
            "wal_fsync": self.wal_fsync,
            "directory_fsync": self.directory_fsync,
            "checkpoint_fsync": self.checkpoint_fsync,
        }

    @property
    def sha256(self) -> str:
        body = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {
            "schema_version",
            "kind",
            "async_queue_rows",
            "async_batch_rows",
            "writer_queue_rows",
            "parquet_row_group_rows",
            "checkpoint_interval_ms",
            "overflow_policy",
            "sqlite_journal_mode",
            "sqlite_synchronous",
            "wal_fsync",
            "directory_fsync",
            "checkpoint_fsync",
        }
        if type(value) is not dict or set(value) != fields:
            raise ValueError("evidence-writer policy fields differ")
        if value.get("kind") != "industrial_evidence_writer_policy":
            raise ValueError("evidence-writer policy kind differs")

        def integer(name: str) -> int:
            result = value[name]
            if isinstance(result, bool) or not isinstance(result, int):
                raise TypeError(f"evidence-writer {name} must be an integer")
            return result

        for name in ("wal_fsync", "directory_fsync", "checkpoint_fsync"):
            if type(value[name]) is not bool:
                raise TypeError(f"evidence-writer {name} must be boolean")
        result = cls(
            schema_version=integer("schema_version"),
            async_queue_rows=integer("async_queue_rows"),
            async_batch_rows=integer("async_batch_rows"),
            writer_queue_rows=integer("writer_queue_rows"),
            parquet_row_group_rows=integer("parquet_row_group_rows"),
            checkpoint_interval_ms=integer("checkpoint_interval_ms"),
            overflow_policy=value["overflow_policy"],
            sqlite_journal_mode=value["sqlite_journal_mode"],
            sqlite_synchronous=value["sqlite_synchronous"],
            wal_fsync=value["wal_fsync"],
            directory_fsync=value["directory_fsync"],
            checkpoint_fsync=value["checkpoint_fsync"],
        )
        result.validate()
        return result


DEFAULT_EVIDENCE_WRITER_POLICY = EvidenceWriterPolicy(
    schema_version=1,
    async_queue_rows=1024,
    async_batch_rows=128,
    writer_queue_rows=1024,
    parquet_row_group_rows=256,
    checkpoint_interval_ms=5000,
    overflow_policy="backpressure",
    sqlite_journal_mode="WAL",
    sqlite_synchronous="FULL",
    wal_fsync=True,
    directory_fsync=True,
    checkpoint_fsync=True,
)
DEFAULT_EVIDENCE_WRITER_POLICY.validate()


def evidence_writer_policy_from_receipt(
    path: str | Path,
) -> EvidenceWriterPolicy | None:
    """Load the exact registered writer policy from one terminal receipt."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise RuntimeError("evidence terminal receipt is not a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("evidence terminal receipt is not valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError("evidence terminal receipt is not an object")
    policy_value = value.get("writer_policy")
    policy_sha256 = value.get("writer_policy_sha256")
    if policy_value is None and policy_sha256 is None:
        return None
    if policy_value is None or policy_sha256 is None:
        raise RuntimeError("evidence terminal receipt has a partial writer policy")
    try:
        policy = EvidenceWriterPolicy.from_dict(policy_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "evidence terminal receipt writer policy is invalid"
        ) from error
    if policy_sha256 != policy.sha256:
        raise RuntimeError("evidence terminal receipt writer policy digest differs")
    return policy


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
_BUDGET_OBSERVED_WORKLOAD_CONTRACTS = {
    "industrial_target_only",
    "industrial_static",
    "industrial_adapted",
}
_NATIVE_TERMINAL_RUN_FIELDS = (
    "native_terminal_artifact_path",
    "native_terminal_artifact_size",
    "native_terminal_raw_sha256",
    "native_terminal_sha256",
    "trusted_attester_policy_sha256",
)
_NATIVE_TERMINAL_BINDING_FIELDS = {
    "path",
    "size",
    "raw_sha256",
    "terminal_sha256",
    "trusted_attester_policy_sha256",
}
_NATIVE_TERMINAL_ARTIFACT_FIELDS = {
    "schema_version",
    "artifact_kind",
    "run_id",
    "rank",
    "trusted_attester_policy_sha256",
    "begin_sha256",
    "reset_sha256",
    "terminal_sha256",
    "binding",
    "warmup_requests",
    "scored_requests",
    "begin",
    "reset",
    "terminal",
}
_BUDGET_OBSERVATION_KIND = "industrial_budget_observation_receipt_v1"
_BUDGET_OBSERVATION_COMPONENTS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)
_BUDGET_OBSERVATION_FIELDS = {
    "schema_version",
    "artifact_kind",
    "experiment_budget_sha256",
    "budget_observation_sha256",
    "budget",
    "observed_component_ms",
    "measured_gpu_ms",
    "fixed_instance_billed_gpu_ms",
    "terminal_evidence_sha256",
    "observed_wall_ms",
    "registered_wall_delta_ms",
    "registered_gpu_delta_ms",
    "registered_billed_delta_ms",
    "gpu_measurement_semantics",
    "fixed_instance_billing_semantics",
}
_BUDGET_OBSERVATION_BINDING_FIELDS = {
    "directory",
    "receipt_name",
    "receipt_sha256",
    "receipt_size",
    "sidecar_name",
    "sidecar_sha256",
    "sidecar_size",
    "budget_observation_sha256",
}
_RESERVED_GANG_MEASUREMENT = "exclusive_reserved_gang_wall_ms_x_gpu_count"
_WHOLE_INSTANCE_BILLING = "whole_inventory_wall_clock_v1"
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


def _canonical_content_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _validate_output_token_identity(row: dict[str, object]) -> None:
    """Require one canonical, complete ordered-token identity."""

    if row.get("output_hash_format") != OUTPUT_HASH_FORMAT:
        raise RuntimeError("request evidence has a wrong output hash format")
    serialized = row.get("output_token_ids")
    if not isinstance(serialized, str):
        raise TypeError("request evidence lacks ordered output token IDs")
    try:
        token_ids = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("request evidence has malformed output token IDs") from exc
    if (
        not isinstance(token_ids, list)
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in token_ids
        )
        or json.dumps(token_ids, separators=(",", ":")) != serialized
    ):
        raise RuntimeError("request evidence has non-canonical output token IDs")
    output_tokens = row.get("output_tokens")
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens != len(token_ids)
    ):
        raise RuntimeError("request evidence token IDs do not cover its output")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if (
        row.get("output_token_ids_sha256") != digest
        or row.get("output_sha256") != digest
    ):
        raise RuntimeError("request evidence token-ID digest is inconsistent")


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
    lock_path = path.with_name(f".{path.name}.publish.lock")
    body = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"invalid receipt publication lock {lock_path}") from exc
    published = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"invalid receipt publication lock {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if os.path.lexists(path):
            raise RuntimeError(
                f"completed evidence already exists for run {value['run_id']}"
            )
        # The persistent cooperative lock makes this rename no-replace among
        # every writer in this package.  Unlike the former link publication,
        # rename leaves no second writable name for the canonical inode.
        os.rename(temporary, path)
        published = True
        _fsync_directory(path.parent)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        if not published:
            temporary.unlink(missing_ok=True)


def _validate_native_terminal_artifact_binding(
    root: Path,
    *,
    binding: object,
    run_id: str,
    rank: int,
    expected_prefix: str | None = None,
) -> Path:
    """Resolve and hash-check one canonical release terminal artifact."""

    if type(binding) is not dict or set(binding) != _NATIVE_TERMINAL_BINDING_FIELDS:
        raise RuntimeError("native terminal artifact binding is incomplete")
    name = binding.get("path")
    size = binding.get("size")
    raw_sha256 = binding.get("raw_sha256")
    terminal_sha256 = binding.get("terminal_sha256")
    policy_sha256 = binding.get("trusted_attester_policy_sha256")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or _SAFE_COMPONENT.fullmatch(name) is None
        or (
            expected_prefix is not None
            and name != f"{expected_prefix}.native-terminal.json"
        )
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or not isinstance(raw_sha256, str)
        or _LOWER_SHA256.fullmatch(raw_sha256) is None
        or not isinstance(terminal_sha256, str)
        or _LOWER_SHA256.fullmatch(terminal_sha256) is None
        or not isinstance(policy_sha256, str)
        or _LOWER_SHA256.fullmatch(policy_sha256) is None
    ):
        raise RuntimeError("native terminal artifact binding is malformed")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("native terminal artifact path is not a regular file")
    try:
        body = path.read_bytes()
        value = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("native terminal artifact is unreadable") from exc
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if (
        len(body) != size
        or hashlib.sha256(body).hexdigest() != raw_sha256
        or body != canonical
        or type(value) is not dict
        or set(value) != _NATIVE_TERMINAL_ARTIFACT_FIELDS
        or (
            value.get("schema_version"),
            value.get("artifact_kind"),
        )
        not in {
            (1, "native_terminal_evidence_bundle_v1"),
            (2, "native_terminal_evidence_bundle_v2"),
        }
        or value.get("run_id") != run_id
        or value.get("rank") != rank
        or value.get("terminal_sha256") != terminal_sha256
        or value.get("trusted_attester_policy_sha256") != policy_sha256
    ):
        raise RuntimeError("native terminal artifact content binding is invalid")
    return path


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
            row.get("experiment_budget_sha256"),
            row.get("preflight_attestation_sha256"),
            row.get("session_plan_sha256"),
            row.get("session_open_receipt_sha256"),
            row.get("reset_receipt_sha256"),
            row.get("session_epoch"),
            *(row.get(name) for name in _NATIVE_TERMINAL_RUN_FIELDS),
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
        budget_sha256 = row.get("experiment_budget_sha256")
        if budget_sha256 is not None and (
            not isinstance(budget_sha256, str)
            or _LOWER_SHA256.fullmatch(budget_sha256) is None
        ):
            raise ValueError("run.experiment_budget_sha256 must be a lowercase SHA-256")
        session_values = (
            row.get("session_plan_sha256"),
            row.get("session_open_receipt_sha256"),
            row.get("reset_receipt_sha256"),
            row.get("session_epoch"),
        )
        if any(value is not None for value in session_values):
            if any(value is None for value in session_values):
                raise ValueError("industrial session identity must be complete")
            if any(
                not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None
                for value in session_values[:3]
            ) or (
                not isinstance(session_values[3], int)
                or isinstance(session_values[3], bool)
                or session_values[3] < 0
            ):
                raise ValueError("industrial session identity is invalid")
        native_values = tuple(row.get(name) for name in _NATIVE_TERMINAL_RUN_FIELDS)
        requires_native_terminal = contract in _BUDGET_OBSERVED_WORKLOAD_CONTRACTS
        if requires_native_terminal:
            if any(value is None for value in native_values):
                raise ValueError(
                    "serving runs require a complete native terminal artifact binding"
                )
            native_path, native_size, *native_hashes = native_values
            if (
                not isinstance(native_path, str)
                or Path(native_path).name != native_path
                or _SAFE_COMPONENT.fullmatch(native_path) is None
                or not native_path.endswith(".native-terminal.json")
                or not isinstance(native_size, int)
                or isinstance(native_size, bool)
                or native_size < 1
                or any(
                    not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None
                    for value in native_hashes
                )
            ):
                raise ValueError("native terminal artifact binding is invalid")
        elif any(value is not None for value in native_values):
            raise ValueError("only serving runs may bind a native terminal artifact")
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
    if not isinstance(value, dict):
        raise TypeError(f"invalid completion receipt {path}")
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
    if "experiment_budget_sha256" in run_parquet.schema_arrow.names:
        run_columns.append("experiment_budget_sha256")
    native_schema_fields = set(_NATIVE_TERMINAL_RUN_FIELDS) & set(
        run_parquet.schema_arrow.names
    )
    if native_schema_fields and native_schema_fields != set(
        _NATIVE_TERMINAL_RUN_FIELDS
    ):
        raise RuntimeError(f"run evidence has a partial native binding {path}")
    native_columns_present = bool(native_schema_fields)
    if native_columns_present:
        run_columns.extend(_NATIVE_TERMINAL_RUN_FIELDS)
    run_rows = run_parquet.iter_batches(batch_size=1, columns=run_columns)
    try:
        run_batch = next(run_rows)
        method = run_batch.column(0)[0].as_py()
        run_values = run_batch.to_pydict()
        workload_contract = run_values.get("workload_contract", [None])[0]
        experiment_budget_sha256 = run_values.get("experiment_budget_sha256", [None])[0]
    except (StopIteration, KeyError, pa.ArrowException) as exc:
        raise RuntimeError(f"receipt binds invalid run evidence {path}") from exc
    if schema_version == 2 and (
        workload_contract in _INDUSTRIAL_WORKLOAD_CONTRACTS
        or experiment_budget_sha256 is not None
        or "experiment_budget_sha256" in value
    ):
        raise RuntimeError(
            f"schema-v2 receipt cannot wrap schema-v3 industrial evidence {path}"
        )
    expected = (
        _expected_tables(method, workload_contract)
        if method in _EVIDENCE_METHODS
        else set()
    )
    if set(resolved) != expected:
        raise RuntimeError(f"receipt binds invalid run evidence {path}")
    if schema_version == 3:
        native_receipt_binding = value.get("native_terminal_artifact")
        requires_native_terminal = (
            workload_contract in _BUDGET_OBSERVED_WORKLOAD_CONTRACTS
        )
        if native_columns_present:
            native_run_binding = {
                "path": run_values["native_terminal_artifact_path"][0],
                "size": run_values["native_terminal_artifact_size"][0],
                "raw_sha256": run_values["native_terminal_raw_sha256"][0],
                "terminal_sha256": run_values["native_terminal_sha256"][0],
                "trusted_attester_policy_sha256": run_values[
                    "trusted_attester_policy_sha256"
                ][0],
            }
            if requires_native_terminal:
                if (
                    any(value is None for value in native_run_binding.values())
                    or native_receipt_binding != native_run_binding
                ):
                    raise RuntimeError(
                        f"receipt lacks its native terminal artifact binding {path}"
                    )
                _validate_native_terminal_artifact_binding(
                    path.parent,
                    binding=native_run_binding,
                    run_id=run_id,
                    rank=rank,
                    expected_prefix=str(prefix),
                )
            elif (
                any(value is not None for value in native_run_binding.values())
                or native_receipt_binding is not None
            ):
                raise RuntimeError(
                    f"non-serving evidence carries a native terminal binding {path}"
                )
        elif native_receipt_binding is not None:
            raise RuntimeError(
                f"legacy run evidence cannot bind a native terminal artifact {path}"
            )
        if experiment_budget_sha256 is not None:
            if (
                not isinstance(experiment_budget_sha256, str)
                or _LOWER_SHA256.fullmatch(experiment_budget_sha256) is None
                or value.get("experiment_budget_sha256") != experiment_budget_sha256
            ):
                raise RuntimeError(
                    f"completion receipt has a wrong experiment budget {path}"
                )
        elif "experiment_budget_sha256" in value:
            raise RuntimeError(
                f"completion receipt has an unbound experiment budget {path}"
            )
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
        receipt_policy_value = value.get("writer_policy")
        receipt_policy_sha256 = value.get("writer_policy_sha256")
        try:
            checkpoint_value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"completion checkpoint is not valid JSON {checkpoint_path}"
            ) from error
        if not isinstance(checkpoint_value, dict):
            raise RuntimeError(
                f"completion checkpoint is not an object {checkpoint_path}"
            )
        checkpoint_policy_value = checkpoint_value.get("writer_policy")
        checkpoint_policy_sha256 = checkpoint_value.get("writer_policy_sha256")
        if (receipt_policy_value is None) != (receipt_policy_sha256 is None):
            raise RuntimeError(f"completion receipt has a partial writer policy {path}")
        if receipt_policy_value is None:
            if (
                checkpoint_policy_value is not None
                or checkpoint_policy_sha256 is not None
            ):
                raise RuntimeError(
                    f"completion checkpoint changes the writer policy {path}"
                )
        else:
            try:
                registered_policy = EvidenceWriterPolicy.from_dict(receipt_policy_value)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"completion receipt has an invalid writer policy {path}"
                ) from error
            if (
                receipt_policy_sha256 != registered_policy.sha256
                or checkpoint_policy_value != receipt_policy_value
                or checkpoint_policy_sha256 != registered_policy.sha256
            ):
                raise RuntimeError(
                    f"completion receipt writer policy binding differs {path}"
                )
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
    if schema_version == 3:
        try:
            run_contracts = pq.read_table(
                resolved["run"], columns=["workload_contract"]
            ).column("workload_contract")
        except (KeyError, pa.ArrowException) as exc:
            raise RuntimeError(
                f"completed run evidence lacks its workload contract {path}"
            ) from exc
        workload_contracts = [value.as_py() for value in run_contracts]
        if len(workload_contracts) != 1:
            raise RuntimeError(f"completed run evidence has invalid coverage {path}")
        columns = ["output_hash_format"]
        industrial = workload_contracts[0] in _INDUSTRIAL_WORKLOAD_CONTRACTS
        if industrial:
            columns.extend(
                [
                    "output_tokens",
                    "output_sha256",
                    "output_token_ids",
                    "output_token_ids_sha256",
                ]
            )
        try:
            request_rows = pq.read_table(
                resolved["request"], columns=columns
            ).to_pylist()
        except (KeyError, pa.ArrowException) as exc:
            raise RuntimeError(
                f"completed request evidence lacks its output identity {path}"
            ) from exc
        if not request_rows:
            raise RuntimeError(f"completed request evidence is empty {path}")
        if industrial:
            for row in request_rows:
                _validate_output_token_identity(row)
        elif any(
            row["output_hash_format"] != OUTPUT_HASH_FORMAT for row in request_rows
        ):
            raise RuntimeError(
                f"completed request evidence has a wrong hash format {path}"
            )
    return resolved


def _run_completion_binding(
    resolved: dict[str, Path],
) -> tuple[str | None, str | None]:
    try:
        rows = pq.read_table(
            resolved["run"],
            columns=["workload_contract", "experiment_budget_sha256"],
        ).to_pylist()
    except (KeyError, pa.ArrowException) as exc:
        raise RuntimeError("completed evidence lacks its completion binding") from exc
    if len(rows) != 1:
        raise RuntimeError("completed evidence has invalid run coverage")
    workload_contract = rows[0]["workload_contract"]
    experiment_budget_sha256 = rows[0]["experiment_budget_sha256"]
    if workload_contract is not None and not isinstance(workload_contract, str):
        raise TypeError("completed evidence has a malformed workload contract")
    if experiment_budget_sha256 is not None and not isinstance(
        experiment_budget_sha256, str
    ):
        raise TypeError("completed evidence has a malformed ExperimentBudget binding")
    return workload_contract, experiment_budget_sha256


def _registered_budget_milliseconds(budget: dict[str, object], name: str) -> int:
    scenario = budget.get(name)
    if type(scenario) is not dict or type(scenario.get("registered")) is not int:
        raise RuntimeError("budget observation contains a malformed registered budget")
    registered = scenario["registered"]
    if registered < 0:
        raise RuntimeError("budget observation contains a negative registered budget")
    return registered


def _validate_budget_observation_binding(
    root: Path,
    *,
    run_id: str,
    rank: int,
    experiment_budget_sha256: str,
    terminal_evidence_sha256: str,
) -> dict[str, object]:
    """Validate the concrete observation that makes industrial completion legal."""

    if _LOWER_SHA256.fullmatch(experiment_budget_sha256) is None:
        raise RuntimeError("industrial evidence lacks its ExperimentBudget binding")
    directory = root / f"{run_id}.rank{rank}.budget-observation"
    receipt = directory / "observation.json"
    sidecar = directory / "observation.json.sha256"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or receipt.is_symlink()
        or not receipt.is_file()
        or sidecar.is_symlink()
        or not sidecar.is_file()
    ):
        raise RuntimeError("industrial evidence requires a durable budget observation")
    try:
        receipt_body = receipt.read_bytes()
        sidecar_body = sidecar.read_bytes()
        artifact = json.loads(receipt_body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("industrial budget observation is not durable JSON") from exc
    if type(artifact) is not dict or set(artifact) != _BUDGET_OBSERVATION_FIELDS:
        raise RuntimeError("industrial budget observation has the wrong schema")
    identity_fields = (
        "experiment_budget_sha256",
        "budget_observation_sha256",
        "terminal_evidence_sha256",
    )
    if (
        type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != 1
        or artifact["artifact_kind"] != _BUDGET_OBSERVATION_KIND
        or artifact["gpu_measurement_semantics"] != _RESERVED_GANG_MEASUREMENT
        or artifact["fixed_instance_billing_semantics"] != _WHOLE_INSTANCE_BILLING
        or any(
            type(artifact[name]) is not str
            or _LOWER_SHA256.fullmatch(artifact[name]) is None
            for name in identity_fields
        )
    ):
        raise RuntimeError("industrial budget observation has malformed identities")
    if (
        artifact["experiment_budget_sha256"] != experiment_budget_sha256
        or artifact["terminal_evidence_sha256"] != terminal_evidence_sha256
    ):
        raise RuntimeError(
            "industrial budget observation has the wrong content binding"
        )
    budget = artifact["budget"]
    try:
        budget_sha256 = (
            _canonical_content_sha256(budget) if type(budget) is dict else None
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "industrial budget observation has a foreign budget"
        ) from exc
    if budget_sha256 != experiment_budget_sha256:
        raise RuntimeError("industrial budget observation has a foreign budget")
    observed_rows = artifact["observed_component_ms"]
    if (
        type(observed_rows) is not list
        or tuple(row[0] for row in observed_rows if type(row) is list and len(row) == 2)
        != _BUDGET_OBSERVATION_COMPONENTS
        or any(
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            or row[1] < 0
            for row in observed_rows
        )
    ):
        raise RuntimeError("industrial budget observation has malformed components")
    integer_fields = (
        "measured_gpu_ms",
        "fixed_instance_billed_gpu_ms",
        "observed_wall_ms",
        "registered_wall_delta_ms",
        "registered_gpu_delta_ms",
        "registered_billed_delta_ms",
    )
    if any(type(artifact[name]) is not int for name in integer_fields):
        raise RuntimeError("industrial budget observation has malformed accounting")
    gpu_count = budget.get("gpu_count")
    if type(gpu_count) is not int or gpu_count < 1:
        raise RuntimeError("industrial budget observation has a malformed GPU count")
    if budget.get("measured_gpu_ms") is not None:
        raise RuntimeError("industrial observation does not bind a pre-run budget")
    observed_wall_ms = sum(row[1] for row in observed_rows)
    registered_wall_ms = sum(
        _registered_budget_milliseconds(budget, name)
        for name in _BUDGET_OBSERVATION_COMPONENTS
    )
    registered_billed_ms = _registered_budget_milliseconds(
        budget, "fixed_instance_billed_gpu_ms"
    )
    measured_gpu_ms = artifact["measured_gpu_ms"]
    fixed_instance_billed_gpu_ms = artifact["fixed_instance_billed_gpu_ms"]
    if registered_wall_ms <= 0 or registered_billed_ms % registered_wall_ms != 0:
        raise RuntimeError(
            "industrial budget does not identify its fixed-instance billing size"
        )
    fixed_instance_gpu_count = registered_billed_ms // registered_wall_ms
    if (
        measured_gpu_ms < 0
        or fixed_instance_billed_gpu_ms < measured_gpu_ms
        or fixed_instance_gpu_count < gpu_count
        or measured_gpu_ms != observed_wall_ms * gpu_count
        or fixed_instance_billed_gpu_ms != observed_wall_ms * fixed_instance_gpu_count
        or artifact["observed_wall_ms"] != observed_wall_ms
        or artifact["registered_wall_delta_ms"] != observed_wall_ms - registered_wall_ms
        or artifact["registered_gpu_delta_ms"]
        != measured_gpu_ms - (registered_wall_ms * gpu_count)
        or artifact["registered_billed_delta_ms"]
        != fixed_instance_billed_gpu_ms - registered_billed_ms
    ):
        raise RuntimeError("industrial budget observation accounting is inconsistent")
    semantic_receipt = {
        "schema_version": artifact["schema_version"],
        "budget": budget,
        "observed_component_ms": observed_rows,
        "measured_gpu_ms": measured_gpu_ms,
        "fixed_instance_billed_gpu_ms": fixed_instance_billed_gpu_ms,
        "terminal_evidence_sha256": artifact["terminal_evidence_sha256"],
    }
    try:
        semantic_sha256 = _canonical_content_sha256(semantic_receipt)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "industrial budget observation content is non-canonical"
        ) from exc
    if artifact[
        "budget_observation_sha256"
    ] != semantic_sha256 or sidecar_body != f"{semantic_sha256}\n".encode("ascii"):
        raise RuntimeError("budget observation content binding is invalid")
    return {
        "directory": directory.name,
        "receipt_name": receipt.name,
        "receipt_sha256": hashlib.sha256(receipt_body).hexdigest(),
        "receipt_size": len(receipt_body),
        "sidecar_name": sidecar.name,
        "sidecar_sha256": hashlib.sha256(sidecar_body).hexdigest(),
        "sidecar_size": len(sidecar_body),
        "budget_observation_sha256": semantic_sha256,
    }


def _validate_industrial_terminal_binding(
    root: Path,
    *,
    run_id: str,
    rank: int,
    terminal_receipt: dict[str, object],
    resolved: dict[str, Path],
    experiment_budget_sha256: str,
) -> None:
    prepared_name = terminal_receipt.get("prepared_receipt_name")
    prepared_sha256 = terminal_receipt.get("prepared_receipt_sha256")
    prepared_size = terminal_receipt.get("prepared_receipt_size")
    observation_binding = terminal_receipt.get("budget_observation")
    expected_prepared_name = f"{run_id}.rank{rank}.prepared.json"
    if (
        prepared_name != expected_prepared_name
        or type(prepared_sha256) is not str
        or _LOWER_SHA256.fullmatch(prepared_sha256) is None
        or type(prepared_size) is not int
        or prepared_size < 1
        or type(observation_binding) is not dict
        or set(observation_binding) != _BUDGET_OBSERVATION_BINDING_FIELDS
    ):
        raise RuntimeError("industrial terminal receipt lacks its exact post-binding")
    prepared = root / prepared_name
    if (
        prepared.is_symlink()
        or not prepared.is_file()
        or prepared.stat().st_size != prepared_size
        or _sha256(prepared) != prepared_sha256
    ):
        raise RuntimeError("industrial terminal receipt does not bind its preparation")
    try:
        prepared_body = prepared.read_bytes()
        prepared_receipt = json.loads(prepared_body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("industrial prepared receipt is invalid") from exc
    if (
        type(prepared_receipt) is not dict
        or (
            json.dumps(prepared_receipt, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        != prepared_body
    ):
        raise RuntimeError("industrial prepared receipt is not canonical")
    prepared_resolved = _load_receipt(prepared, run_id=run_id, rank=rank)
    if prepared_resolved != resolved or _sha256(prepared) != prepared_sha256:
        raise RuntimeError("industrial preparation differs from terminal evidence")
    terminal_base = dict(terminal_receipt)
    for name in (
        "prepared_receipt_name",
        "prepared_receipt_sha256",
        "prepared_receipt_size",
        "budget_observation",
    ):
        terminal_base.pop(name, None)
    if terminal_base != prepared_receipt:
        raise RuntimeError("industrial terminal envelope changed prepared evidence")
    expected_observation = _validate_budget_observation_binding(
        root,
        run_id=run_id,
        rank=rank,
        experiment_budget_sha256=experiment_budget_sha256,
        terminal_evidence_sha256=prepared_sha256,
    )
    if observation_binding != expected_observation:
        raise RuntimeError("industrial terminal receipt does not bind its observation")
    if _sha256(prepared) != prepared_sha256:
        raise RuntimeError("industrial preparation changed during terminal validation")


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
    completed: list[dict[str, Path]] = []
    for receipt in receipts:
        receipt_body = receipt.read_bytes()
        receipt_sha256 = hashlib.sha256(receipt_body).hexdigest()
        resolved = _load_receipt(receipt, run_id=run_id, rank=rank)
        receipt_value = json.loads(receipt_body.decode("utf-8"))
        if _sha256(receipt) != receipt_sha256:
            raise RuntimeError("completion receipt changed during validation")
        if receipt_value["schema_version"] == 3:
            if (
                json.dumps(receipt_value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8") != receipt_body:
                raise RuntimeError("schema-v3 completion receipt is not canonical")
            workload_contract, experiment_budget_sha256 = _run_completion_binding(
                resolved
            )
        else:
            workload_contract = experiment_budget_sha256 = None
        if workload_contract in _BUDGET_OBSERVED_WORKLOAD_CONTRACTS:
            if not isinstance(experiment_budget_sha256, str):
                raise RuntimeError(
                    "industrial evidence lacks its ExperimentBudget binding"
                )
            _validate_industrial_terminal_binding(
                directory,
                run_id=run_id,
                rank=rank,
                terminal_receipt=receipt_value,
                resolved=resolved,
                experiment_budget_sha256=experiment_budget_sha256,
            )
        if _sha256(receipt) != receipt_sha256:
            raise RuntimeError(
                "completion receipt changed during post-binding validation"
            )
        completed.append(resolved)
    if len(completed) > 1:
        raise RuntimeError(f"multiple completed attempts exist for run {run_id}")
    return completed[0] if completed else None


def publish_prepared_evidence_completion(
    root: str | Path,
    *,
    run_id: str,
    rank: int,
    expected_receipt_sha256: str | None = None,
    validate: Callable[[dict[str, Path]], None] | None = None,
    validate_post_binding: Callable[[], None] | None = None,
) -> dict[str, Path]:
    """Publish a final envelope over prepared evidence and its post-binding."""

    if not _SAFE_COMPONENT.fullmatch(run_id):
        raise ValueError("run_id must be a safe non-empty path component")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if expected_receipt_sha256 is not None and (
        not isinstance(expected_receipt_sha256, str)
        or _LOWER_SHA256.fullmatch(expected_receipt_sha256) is None
    ):
        raise ValueError("expected receipt must be a lowercase SHA-256")
    directory = Path(root)
    prepared = directory / f"{run_id}.rank{rank}.prepared.json"
    canonical = directory / f"{run_id}.rank{rank}.complete.json"
    if os.path.lexists(canonical):
        raise RuntimeError(f"completed evidence already exists for run {run_id}")
    if prepared.is_symlink() or not prepared.is_file():
        raise RuntimeError(f"invalid prepared receipt {prepared}")
    try:
        prepared_body = prepared.read_bytes()
        receipt = json.loads(prepared_body.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid prepared receipt {prepared}") from exc
    canonical_body = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    prepared_sha256 = hashlib.sha256(prepared_body).hexdigest()
    if canonical_body != prepared_body:
        raise RuntimeError("prepared evidence receipt is not canonical")
    if (
        expected_receipt_sha256 is not None
        and prepared_sha256 != expected_receipt_sha256
    ):
        raise RuntimeError("prepared evidence receipt has the wrong content binding")
    resolved = _load_receipt(prepared, run_id=run_id, rank=rank)
    if _sha256(prepared) != prepared_sha256:
        raise RuntimeError("prepared evidence receipt changed during validation")
    if validate is not None:
        validate(dict(resolved))
    workload_contract, experiment_budget_sha256 = _run_completion_binding(resolved)
    requires_post_binding = workload_contract in _BUDGET_OBSERVED_WORKLOAD_CONTRACTS
    observation_binding: dict[str, object] | None = None
    if requires_post_binding:
        if not isinstance(experiment_budget_sha256, str):
            raise RuntimeError("industrial evidence lacks its ExperimentBudget binding")
        observation_binding = _validate_budget_observation_binding(
            directory,
            run_id=run_id,
            rank=rank,
            experiment_budget_sha256=experiment_budget_sha256,
            terminal_evidence_sha256=prepared_sha256,
        )
    if validate_post_binding is not None:
        validate_post_binding()
    if requires_post_binding and observation_binding != (
        _validate_budget_observation_binding(
            directory,
            run_id=run_id,
            rank=rank,
            experiment_budget_sha256=experiment_budget_sha256,
            terminal_evidence_sha256=prepared_sha256,
        )
    ):
        raise RuntimeError("industrial budget observation changed before publication")
    if _sha256(prepared) != prepared_sha256:
        raise RuntimeError("prepared evidence receipt changed before publication")
    terminal_receipt = dict(receipt)
    if requires_post_binding:
        terminal_receipt.update(
            {
                "prepared_receipt_name": prepared.name,
                "prepared_receipt_sha256": prepared_sha256,
                "prepared_receipt_size": len(prepared_body),
                "budget_observation": observation_binding,
            }
        )
    terminal_body = (
        json.dumps(terminal_receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    terminal_sha256 = hashlib.sha256(terminal_body).hexdigest()
    _publish_receipt_exclusive(canonical, terminal_receipt)
    if _sha256(canonical) != terminal_sha256:
        raise RuntimeError("published completion differs from its terminal envelope")
    loaded = _load_receipt(canonical, run_id=run_id, rank=rank)
    if loaded != resolved:
        raise RuntimeError("published completion changed prepared evidence bindings")
    if requires_post_binding:
        _validate_industrial_terminal_binding(
            directory,
            run_id=run_id,
            rank=rank,
            terminal_receipt=terminal_receipt,
            resolved=loaded,
            experiment_budget_sha256=experiment_budget_sha256,
        )
    return loaded


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
        registered_policy: EvidenceWriterPolicy | None = None,
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
        if registered_policy is not None:
            if type(registered_policy) is not EvidenceWriterPolicy:
                raise TypeError(
                    "registered evidence policy must be an EvidenceWriterPolicy"
                )
            registered_policy.validate()
            expected_checkpoint_s = registered_policy.checkpoint_interval_ms / 1000
            if (
                max_queued_rows != registered_policy.writer_queue_rows
                or row_group_rows != registered_policy.parquet_row_group_rows
                or checkpoint_interval_s != expected_checkpoint_s
                or overflow_policy != registered_policy.overflow_policy
            ):
                raise ValueError(
                    "evidence writer settings differ from the registered policy"
                )

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
        self.registered_policy = registered_policy
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
        self._request_hash_formats: set[str] = set()
        self._run_record: dict[str, object] | None = None
        self._flushes = 0
        self._fsync_time_ns = 0
        self._backpressure_events = 0
        self._backpressured_rows = 0
        self._dropped_rows = 0
        self._max_observed_queued_rows = 0
        self._last_checkpoint = time.monotonic()
        self._closed = False
        self._prepared_receipt: dict[str, object] | None = None
        self._prepared_files: dict[str, Path] | None = None
        self._native_terminal_binding: dict[str, object] | None = None
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

    @property
    def fsync_time_ns(self) -> int:
        return self._fsync_time_ns

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
            "fsync_time_ns": self._fsync_time_ns,
            "max_observed_queued_rows": self._max_observed_queued_rows,
        }

    def persist_native_terminal_artifact(
        self,
        artifact: Mapping[str, object],
    ) -> dict[str, object]:
        """Exclusively publish one canonical begin/reset/final bundle."""

        if self._closed or self._prepared_receipt is not None:
            raise RuntimeError("evidence writer is closed or prepared")
        if self._native_terminal_binding is not None:
            raise RuntimeError("native terminal artifact is already persisted")
        value = dict(artifact)
        if (
            set(value) != _NATIVE_TERMINAL_ARTIFACT_FIELDS
            or value.get("schema_version") != 2
            or value.get("artifact_kind") != "native_terminal_evidence_bundle_v2"
            or value.get("run_id") != self.run_id
            or value.get("rank") != self.rank
        ):
            raise ValueError("native terminal artifact differs from this writer")
        try:
            body = (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("native terminal artifact is not strict JSON") from exc
        terminal_sha256 = value.get("terminal_sha256")
        policy_sha256 = value.get("trusted_attester_policy_sha256")
        if (
            not isinstance(terminal_sha256, str)
            or _LOWER_SHA256.fullmatch(terminal_sha256) is None
            or not isinstance(policy_sha256, str)
            or _LOWER_SHA256.fullmatch(policy_sha256) is None
        ):
            raise ValueError("native terminal artifact lacks release digest bindings")
        path = self.root / f"{self.prefix}.native-terminal.json"
        _publish_receipt_exclusive(path, value)
        binding: dict[str, object] = {
            "path": path.name,
            "size": len(body),
            "raw_sha256": hashlib.sha256(body).hexdigest(),
            "terminal_sha256": terminal_sha256,
            "trusted_attester_policy_sha256": policy_sha256,
        }
        _validate_native_terminal_artifact_binding(
            self.root,
            binding=binding,
            run_id=self.run_id,
            rank=self.rank,
            expected_prefix=self.prefix,
        )
        self._native_terminal_binding = binding
        self._write_checkpoint()
        return dict(binding)

    def _identity(self, table: str, row: dict[str, object]) -> tuple[object, ...]:
        if table == "run":
            return ("run",)
        if table == "request":
            return (row["request_id"],)
        if table == "round":
            return (
                row.get("request_epoch"),
                row["request_id"],
                row["round_index"],
            )
        if table == "update":
            return (
                row.get("request_epoch"),
                row["cohort_sha256"],
                row["update_index"],
            )
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
            "native_terminal_artifact": self._native_terminal_binding,
            "writer_policy": (
                None
                if self.registered_policy is None
                else self.registered_policy.to_dict()
            ),
            "writer_policy_sha256": (
                None
                if self.registered_policy is None
                else self.registered_policy.sha256
            ),
        }

    def _write_checkpoint(self) -> None:
        started = time.perf_counter_ns()
        _atomic_json(self._checkpoint_path, self._checkpoint_value())
        self._fsync_time_ns += time.perf_counter_ns() - started
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
        started = time.perf_counter_ns()
        _fsync_file(temporary)
        os.replace(temporary, output)
        _fsync_directory(self.root)
        self._fsync_time_ns += time.perf_counter_ns() - started

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
        if table == "request":
            self._request_hash_formats.add(str(row["output_hash_format"]))
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
            or self._request_hash_formats != {OUTPUT_HASH_FORMAT}
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
            if workload_contract in _INDUSTRIAL_WORKLOAD_CONTRACTS:
                segments = sorted(
                    self.root.glob(f"{self.prefix}.request.wal.*.parquet")
                )
                if len(segments) != self._row_groups["request"]:
                    raise RuntimeError("industrial request WAL coverage changed")
                for segment in segments:
                    for row in pq.read_table(
                        segment,
                        columns=[
                            "output_tokens",
                            "output_hash_format",
                            "output_sha256",
                            "output_token_ids",
                            "output_token_ids_sha256",
                        ],
                    ).to_pylist():
                        _validate_output_token_identity(row)
            if workload_contract in _BUDGET_OBSERVED_WORKLOAD_CONTRACTS:
                if self._native_terminal_binding is None:
                    raise RuntimeError(
                        "serving completion lacks its native terminal artifact"
                    )
                run_binding = {
                    "path": self._run_record.get("native_terminal_artifact_path"),
                    "size": self._run_record.get("native_terminal_artifact_size"),
                    "raw_sha256": self._run_record.get("native_terminal_raw_sha256"),
                    "terminal_sha256": self._run_record.get("native_terminal_sha256"),
                    "trusted_attester_policy_sha256": self._run_record.get(
                        "trusted_attester_policy_sha256"
                    ),
                }
                if run_binding != self._native_terminal_binding:
                    raise RuntimeError(
                        "run evidence changed its native terminal binding"
                    )
                _validate_native_terminal_artifact_binding(
                    self.root,
                    binding=run_binding,
                    run_id=self.run_id,
                    rank=self.rank,
                    expected_prefix=self.prefix,
                )
            elif self._native_terminal_binding is not None:
                raise RuntimeError(
                    "non-serving completion carries a native terminal artifact"
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

    def prepare_close(self) -> tuple[dict[str, Path], Path]:
        """Durably prepare all completion bytes without publishing completion.

        The prepared receipt lets a caller finish dependent terminal work and
        publish a post-bound observation before the canonical completion
        receipt becomes visible.  This removes the completed-without-
        observation crash window while keeping the receipt as the final claim
        mutation.
        """

        if self._closed or self._prepared_receipt is not None:
            raise RuntimeError("evidence writer is already closed or prepared")
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
            "writer_policy": (
                None
                if self.registered_policy is None
                else self.registered_policy.to_dict()
            ),
            "writer_policy_sha256": (
                None
                if self.registered_policy is None
                else self.registered_policy.sha256
            ),
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
        experiment_budget_sha256 = self._run_record.get("experiment_budget_sha256")
        if experiment_budget_sha256 is not None:
            receipt["experiment_budget_sha256"] = experiment_budget_sha256
        if self._native_terminal_binding is not None:
            receipt["native_terminal_artifact"] = dict(self._native_terminal_binding)
        prepared_path = self.root / f"{self.run_id}.rank{self.rank}.prepared.json"
        if os.path.lexists(prepared_path):
            raise RuntimeError(
                f"prepared evidence already exists for run {self.run_id}"
            )
        _publish_receipt_exclusive(prepared_path, receipt)
        self._index.commit()
        self._index.close()
        self._prepared_receipt = receipt
        self._prepared_files = written
        self._state = "prepared"
        return dict(written), prepared_path

    def publish_close(
        self,
        *,
        validate_post_binding: Callable[[], None] | None = None,
    ) -> dict[str, Path]:
        """Publish a prepared receipt as the final completion mutation."""

        if self._closed:
            raise RuntimeError("evidence writer already closed")
        if self._prepared_receipt is None or self._prepared_files is None:
            raise RuntimeError("evidence writer has not prepared completion")
        body = (
            json.dumps(
                self._prepared_receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

        def validate_prepared(resolved: dict[str, Path]) -> None:
            if resolved != self._prepared_files:
                raise RuntimeError("prepared evidence changed before completion")

        published = publish_prepared_evidence_completion(
            self.root,
            run_id=self.run_id,
            rank=self.rank,
            expected_receipt_sha256=hashlib.sha256(body).hexdigest(),
            validate=validate_prepared,
            validate_post_binding=validate_post_binding,
        )
        self._state = "complete"
        self._closed = True
        return published

    def close(self) -> dict[str, Path]:
        self.prepare_close()
        return self.publish_close()

    def abort(self, reason: str | None = None) -> None:
        """Durably retain an inspectable attempt without completing it."""
        if self._closed or self._prepared_receipt is not None:
            raise RuntimeError("evidence writer is closed or prepared")
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
