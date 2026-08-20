"""Bounded, lossless storage adapters for large raw terminal evidence.

The patched runtime still emits the registered schema-1 terminal and native
ITL dictionaries.  These helpers only change their durable *container* when a
canonical file would exceed the proof-artifact size bound.  Reopening a
sharded container reconstructs the byte-semantic schema-1 value and verifies
its complete SHA-256 before any existing scientific validator sees it.

Small artifacts keep their historical schema and bytes.  Large artifacts use
deterministic contiguous sequence shards.  Native ITL events are nested below
each request pointer so one 40,928-token request cannot make a pointer row
exceed the shard limit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lightcone_spec.runtime.formal_sharded_artifact import (
    FormalCanonicalSequenceShardIndex,
    publish_formal_canonical_sequence_shards,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_INLINE_MAX_BYTES = 1_500_000
_SHARD_MAX_BYTES = 1_000_000
_SHARD_MAX_ROWS = 256
_INLINE_POINTER_EVENTS_MAX_BYTES = 200_000
_SHA256_CHARS = frozenset("0123456789abcdef")

LEGACY_SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND = (
    "native_terminal_evidence_bundle_v2_sharded"
)
SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND = "native_terminal_evidence_bundle_v3_sharded"
SHARDED_UNSIGNED_NATIVE_ITL_BUNDLE_KIND = (
    "unsigned_native_itl_result_pointer_bundle_v2_sharded"
)
LEGACY_SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND = (
    "unsigned_formal_gang_request_terminal_v2_sharded"
)
SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND = (
    "unsigned_formal_gang_request_terminal_v3_sharded"
)
SHARDED_FORMAL_GANG_ITL_BUNDLE_KIND = (
    "unsigned_formal_gang_native_itl_pointer_bundle_v2_sharded"
)
LEGACY_SHARDED_FORMAL_GANG_TERMINAL_KIND = (
    "sglang_formal_gang_all_rank_terminal_v2_sharded"
)
SHARDED_FORMAL_GANG_TERMINAL_KIND = "sglang_formal_gang_all_rank_terminal_v3_sharded"
SHARDED_CLIENT_REQUEST_LIFECYCLE_KIND = "registered_client_request_lifecycle_v1_sharded"

_TERMINAL_SEQUENCE_FIELDS = (
    "warmup_requests",
    "scored_requests",
    "terminal_expected_request_ids",
    "terminal_request_rows",
    "terminal_round_rows",
    "terminal_update_rows",
    "terminal_historical_kv_items",
)

_CURRENT_TERMINAL_SEQUENCE_FIELDS = (
    "warmup_requests",
    "scored_requests",
    "reset_warmup_request_rows",
    "reset_warmup_round_rows",
    "reset_warmup_update_rows",
    "reset_warmup_historical_kv_items",
    "reset_warmup_request_reset_receipts",
    "terminal_expected_request_ids",
    "terminal_request_rows",
    "terminal_round_rows",
    "terminal_update_rows",
    "terminal_historical_kv_items",
    "terminal_request_reset_receipts",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _copy_json(value: object) -> object:
    return json.loads(_canonical_bytes(value))


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _strict(label: str, value: object, expected: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} fields differ")
    return dict(value)


def _shard_root(output_path: str | Path) -> Path:
    output = Path(output_path)
    if not output.is_absolute() or output != output.resolve(strict=False):
        raise ValueError("terminal output path must be absolute and normalized")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("terminal output parent must be an existing directory")
    root = output.with_name(f"{output.name}.shards")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    return root


def _publish_sequence(
    *,
    rows: list[object],
    artifact_kind: str,
    artifact_id: str,
    root: Path,
) -> dict[str, object] | None:
    if not rows:
        return None
    directory = root / artifact_kind
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    binding, _index = publish_formal_canonical_sequence_shards(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        rows=rows,
        output_directory=directory,
        maximum_shard_bytes=_SHARD_MAX_BYTES,
        maximum_shard_rows=_SHARD_MAX_ROWS,
    )
    return binding.to_dict()


def _reopen_sequence(
    value: object,
    *,
    artifact_kind: str,
    artifact_id: str,
    expected_count: int,
) -> list[object]:
    if expected_count == 0:
        if value is not None:
            raise ValueError("empty terminal sequence unexpectedly binds shards")
        return []
    binding = CanonicalJsonProofBinding.from_dict(value)
    index = FormalCanonicalSequenceShardIndex.from_dict(binding.reopen())
    if (
        binding.semantic_sha256 != _sha256(index.to_dict())
        or index.artifact_kind != artifact_kind
        or index.artifact_id != artifact_id
        or index.total_rows != expected_count
    ):
        raise ValueError("terminal sequence shard lineage differs")
    rows = list(index.revalidate().iter_rows())
    if len(rows) != expected_count:
        raise ValueError("terminal sequence shard coverage differs")
    return rows


def publish_scalable_client_request_lifecycle(
    *,
    output_path: str | Path,
    run_binding_sha256: str,
    execution_policy_sha256: str,
    rows: list[object],
) -> CanonicalJsonProofBinding:
    """Publish lossless client admission/terminal rows with bounded shards."""

    _require_sha256("client lifecycle run binding", run_binding_sha256)
    _require_sha256("client lifecycle execution policy", execution_policy_sha256)
    if not rows:
        raise ValueError("client lifecycle requires request rows")
    for row in rows:
        if type(row) is not dict or row.get("phase") not in {"warmup", "scored"}:
            raise ValueError("client lifecycle row shape differs")
    request_ids = tuple(row.get("request_id") for row in rows)
    if any(type(value) is not str or not value for value in request_ids) or len(
        request_ids
    ) != len(set(request_ids)):
        raise ValueError("client lifecycle request coverage differs")
    artifact_id = _sha256(
        {
            "kind": SHARDED_CLIENT_REQUEST_LIFECYCLE_KIND,
            "run_binding_sha256": run_binding_sha256,
            "execution_policy_sha256": execution_policy_sha256,
            "request_ids": request_ids,
        }
    )
    root = _shard_root(output_path)
    index_value = _publish_sequence(
        rows=rows,
        artifact_kind="registered_client_request_lifecycle_rows",
        artifact_id=artifact_id,
        root=root,
    )
    assert index_value is not None
    container = {
        "schema_version": 1,
        "kind": SHARDED_CLIENT_REQUEST_LIFECYCLE_KIND,
        "run_binding_sha256": run_binding_sha256,
        "execution_policy_sha256": execution_policy_sha256,
        "artifact_id": artifact_id,
        "request_count": len(rows),
        "phase_counts": {
            phase: sum(row.get("phase") == phase for row in rows)
            for phase in ("warmup", "scored")
        },
        "rows_index": index_value,
    }
    publish_canonical_json_no_replace(output_path, container)
    binding = CanonicalJsonProofBinding.bind(output_path)
    if reopen_scalable_client_request_lifecycle(binding) != rows:
        raise RuntimeError("client lifecycle sharded reconstruction changed")
    return binding


def reopen_scalable_client_request_lifecycle(
    value: object,
    *,
    expected_run_binding_sha256: str | None = None,
    expected_execution_policy_sha256: str | None = None,
) -> list[object]:
    """Deep-open a client lifecycle artifact and verify every shard."""

    raw_value = value.reopen() if type(value) is CanonicalJsonProofBinding else value
    container = _strict(
        "client lifecycle container",
        raw_value,
        {
            "schema_version",
            "kind",
            "run_binding_sha256",
            "execution_policy_sha256",
            "artifact_id",
            "request_count",
            "phase_counts",
            "rows_index",
        },
    )
    run_binding_sha256 = _require_sha256(
        "client lifecycle run binding", container["run_binding_sha256"]
    )
    execution_policy_sha256 = _require_sha256(
        "client lifecycle execution policy", container["execution_policy_sha256"]
    )
    if (
        container["schema_version"] != 1
        or container["kind"] != SHARDED_CLIENT_REQUEST_LIFECYCLE_KIND
        or (
            expected_run_binding_sha256 is not None
            and run_binding_sha256 != expected_run_binding_sha256
        )
        or (
            expected_execution_policy_sha256 is not None
            and execution_policy_sha256 != expected_execution_policy_sha256
        )
        or type(container["request_count"]) is not int
        or container["request_count"] < 1
    ):
        raise ValueError("client lifecycle identity differs")
    phase_counts = _strict(
        "client lifecycle phase counts",
        container["phase_counts"],
        {"warmup", "scored"},
    )
    if any(type(value) is not int or value < 0 for value in phase_counts.values()):
        raise ValueError("client lifecycle phase counts are invalid")
    rows = _reopen_sequence(
        container["rows_index"],
        artifact_kind="registered_client_request_lifecycle_rows",
        artifact_id=str(container["artifact_id"]),
        expected_count=int(container["request_count"]),
    )
    if (
        sum(phase_counts.values()) != len(rows)
        or any(type(row) is not dict for row in rows)
        or {
            phase: sum(row.get("phase") == phase for row in rows)
            for phase in ("warmup", "scored")
        }
        != phase_counts
    ):
        raise ValueError("client lifecycle phase coverage differs")
    request_ids = tuple(row.get("request_id") for row in rows)
    expected_artifact_id = _sha256(
        {
            "kind": SHARDED_CLIENT_REQUEST_LIFECYCLE_KIND,
            "run_binding_sha256": run_binding_sha256,
            "execution_policy_sha256": execution_policy_sha256,
            "request_ids": request_ids,
        }
    )
    if container["artifact_id"] != expected_artifact_id:
        raise ValueError("client lifecycle artifact identity differs")
    return rows


def _publish_legacy_scalable_native_terminal_artifact(
    *,
    output_path: str | Path,
    legacy_artifact: object,
) -> CanonicalJsonProofBinding:
    """Publish a schema-1 terminal inline or a lossless sharded container."""

    artifact = _strict(
        "native terminal artifact",
        legacy_artifact,
        {
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
        },
    )
    if (
        artifact["schema_version"] != 1
        or artifact["artifact_kind"] != "native_terminal_evidence_bundle_v1"
    ):
        raise ValueError("native terminal artifact is not registered schema 1")
    if len(_canonical_bytes(artifact)) + 1 <= _INLINE_MAX_BYTES:
        publish_canonical_json_no_replace(output_path, artifact)
        return CanonicalJsonProofBinding.bind(output_path)

    terminal = _strict(
        "native terminal envelope",
        artifact["terminal"],
        {
            "schema_version",
            "hook",
            "run_id",
            "run_nonce_sha256",
            "execution_plan_sha256",
            "rank_config_sha256",
            "server_process_id",
            "server_process_started_ns",
            "attempt_id",
            "session_id",
            "session_epoch",
            "previous_run_id",
            "challenge_nonce_sha256",
            "method",
            "expected_request_ids",
            "reset_receipt_sha256",
            "request_round_rows",
            "update_rows",
            "performance_counters",
            "historical_kv_source_versions",
            "final_state",
            "completion_marker",
            "terminal_sha256",
            "attestation",
        },
    )
    request_round = _strict(
        "native terminal request/round rows",
        terminal.pop("request_round_rows"),
        {"requests", "rounds"},
    )
    sequences: dict[str, list[object]] = {
        "warmup_requests": list(artifact.pop("warmup_requests")),
        "scored_requests": list(artifact.pop("scored_requests")),
        "terminal_expected_request_ids": list(terminal.pop("expected_request_ids")),
        "terminal_request_rows": list(request_round["requests"]),
        "terminal_round_rows": list(request_round["rounds"]),
        "terminal_update_rows": list(terminal.pop("update_rows")),
        "terminal_historical_kv_items": [
            {"request_id": key, "versions": value}
            for key, value in sorted(
                dict(terminal.pop("historical_kv_source_versions")).items()
            )
        ],
    }
    artifact.pop("terminal")
    legacy_sha256 = _sha256(legacy_artifact)
    root = _shard_root(output_path)
    indexes = {
        field: _publish_sequence(
            rows=rows,
            artifact_kind=f"formal_{field}",
            artifact_id=legacy_sha256,
            root=root,
        )
        for field, rows in sequences.items()
    }
    container = {
        "schema_version": 2,
        "artifact_kind": LEGACY_SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND,
        "legacy_artifact_sha256": legacy_sha256,
        "legacy_artifact_static": _copy_json(artifact),
        "terminal_static": _copy_json(terminal),
        "sequence_counts": {
            field: len(sequences[field]) for field in _TERMINAL_SEQUENCE_FIELDS
        },
        "sequence_indexes": indexes,
    }
    # Exercise the same deep reconstruction that all later consumers use.
    if _reopen_legacy_scalable_native_terminal_artifact(container) != legacy_artifact:
        raise RuntimeError("sharded native terminal reconstruction changed")
    publish_canonical_json_no_replace(output_path, container)
    return CanonicalJsonProofBinding.bind(output_path)


def _reopen_legacy_scalable_native_terminal_artifact(
    value: object,
) -> dict[str, object]:
    """Return the exact registered schema-1 terminal from either container."""

    if (
        type(value) is dict
        and value.get("schema_version") == 1
        and value.get("artifact_kind") == "native_terminal_evidence_bundle_v1"
    ):
        return dict(value)
    row = _strict(
        "sharded native terminal artifact",
        value,
        {
            "schema_version",
            "artifact_kind",
            "legacy_artifact_sha256",
            "legacy_artifact_static",
            "terminal_static",
            "sequence_counts",
            "sequence_indexes",
        },
    )
    if (
        row["schema_version"] != 2
        or row["artifact_kind"] != LEGACY_SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND
    ):
        raise ValueError("sharded native terminal container schema differs")
    legacy_sha256 = _require_sha256(
        "sharded native terminal legacy artifact",
        row["legacy_artifact_sha256"],
    )
    counts = _strict(
        "sharded native terminal sequence counts",
        row["sequence_counts"],
        set(_TERMINAL_SEQUENCE_FIELDS),
    )
    indexes = _strict(
        "sharded native terminal sequence indexes",
        row["sequence_indexes"],
        set(_TERMINAL_SEQUENCE_FIELDS),
    )
    sequences: dict[str, list[object]] = {}
    for field in _TERMINAL_SEQUENCE_FIELDS:
        count = counts[field]
        if type(count) is not int or count < 0:
            raise ValueError("sharded native terminal sequence count differs")
        sequences[field] = _reopen_sequence(
            indexes[field],
            artifact_kind=f"formal_{field}",
            artifact_id=legacy_sha256,
            expected_count=count,
        )
    artifact = dict(
        _strict(
            "sharded native terminal static artifact",
            row["legacy_artifact_static"],
            {
                "schema_version",
                "artifact_kind",
                "run_id",
                "rank",
                "trusted_attester_policy_sha256",
                "begin_sha256",
                "reset_sha256",
                "terminal_sha256",
                "binding",
                "begin",
                "reset",
            },
        )
    )
    terminal = dict(row["terminal_static"])
    history: dict[str, object] = {}
    for raw_item in sequences["terminal_historical_kv_items"]:
        item = _strict(
            "sharded native terminal historical KV item",
            raw_item,
            {"request_id", "versions"},
        )
        request_id = item["request_id"]
        if type(request_id) is not str or request_id in history:
            raise ValueError("sharded native terminal historical KV keys differ")
        history[request_id] = item["versions"]
    terminal.update(
        {
            "expected_request_ids": sequences["terminal_expected_request_ids"],
            "request_round_rows": {
                "requests": sequences["terminal_request_rows"],
                "rounds": sequences["terminal_round_rows"],
            },
            "update_rows": sequences["terminal_update_rows"],
            "historical_kv_source_versions": history,
        }
    )
    artifact.update(
        {
            "warmup_requests": sequences["warmup_requests"],
            "scored_requests": sequences["scored_requests"],
            "terminal": terminal,
        }
    )
    if _sha256(artifact) != legacy_sha256:
        raise ValueError("sharded native terminal complete digest differs")
    return artifact


def _history_items(value: object, *, label: str) -> list[object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be a string-keyed object")
    return [
        {"request_id": request_id, "versions": versions}
        for request_id, versions in sorted(value.items())
    ]


def _history_from_items(rows: list[object], *, label: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw_item in rows:
        item = _strict(label, raw_item, {"request_id", "versions"})
        request_id = item["request_id"]
        if type(request_id) is not str or not request_id or request_id in result:
            raise ValueError(f"{label} keys differ")
        result[request_id] = item["versions"]
    return result


def publish_scalable_native_terminal_artifact(
    *,
    output_path: str | Path,
    legacy_artifact: object,
) -> CanonicalJsonProofBinding:
    """Publish only the current schema-2 terminal, inline or as schema-3 shards.

    The parameter name is retained for call compatibility.  A schema-1 bundle
    remains reopenable below but is intentionally no longer publishable by a
    production path.
    """

    artifact = _strict(
        "current native terminal artifact",
        _copy_json(legacy_artifact),
        {
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
        },
    )
    if (
        artifact["schema_version"] != 2
        or artifact["artifact_kind"] != "native_terminal_evidence_bundle_v2"
    ):
        raise ValueError("production native terminal artifact must be current schema 2")
    if len(_canonical_bytes(artifact)) + 1 <= _INLINE_MAX_BYTES:
        publish_canonical_json_no_replace(output_path, artifact)
        return CanonicalJsonProofBinding.bind(output_path)

    reset = artifact.pop("reset")
    terminal = artifact.pop("terminal")
    if type(reset) is not dict or type(terminal) is not dict:
        raise TypeError("current native terminal reset/terminal must be objects")
    reset = dict(reset)
    terminal = dict(terminal)
    warmup_resets = reset.pop("warmup_request_source_point_resets", None)
    terminal_resets = terminal.pop("request_source_point_resets", None)
    request_round = terminal.pop("request_round_rows", None)
    if (
        type(warmup_resets) is not dict
        or type(terminal_resets) is not dict
        or type(request_round) is not dict
        or set(request_round) != {"requests", "rounds"}
    ):
        raise ValueError("current native terminal reset/round envelopes differ")
    warmup_resets = dict(warmup_resets)
    terminal_resets = dict(terminal_resets)
    sequences: dict[str, list[object]] = {
        "warmup_requests": list(artifact.pop("warmup_requests")),
        "scored_requests": list(artifact.pop("scored_requests")),
        "reset_warmup_request_rows": list(reset.pop("warmup_request_rows")),
        "reset_warmup_round_rows": list(reset.pop("warmup_round_rows")),
        "reset_warmup_update_rows": list(reset.pop("warmup_update_rows")),
        "reset_warmup_historical_kv_items": _history_items(
            reset.pop("warmup_historical_kv_source_versions"),
            label="warmup historical KV",
        ),
        "reset_warmup_request_reset_receipts": list(warmup_resets.pop("receipts")),
        "terminal_expected_request_ids": list(terminal.pop("expected_request_ids")),
        "terminal_request_rows": list(request_round["requests"]),
        "terminal_round_rows": list(request_round["rounds"]),
        "terminal_update_rows": list(terminal.pop("update_rows")),
        "terminal_historical_kv_items": _history_items(
            terminal.pop("historical_kv_source_versions"),
            label="terminal historical KV",
        ),
        "terminal_request_reset_receipts": list(terminal_resets.pop("receipts")),
    }
    artifact_sha256 = _sha256(legacy_artifact)
    root = _shard_root(output_path)
    indexes = {
        field: _publish_sequence(
            rows=rows,
            artifact_kind=f"formal_{field}",
            artifact_id=artifact_sha256,
            root=root,
        )
        for field, rows in sequences.items()
    }
    container = {
        "schema_version": 3,
        "artifact_kind": SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND,
        "semantic_artifact_sha256": artifact_sha256,
        "artifact_static": artifact,
        "reset_static": reset,
        "warmup_request_resets_static": warmup_resets,
        "terminal_static": terminal,
        "terminal_request_resets_static": terminal_resets,
        "sequence_counts": {
            field: len(sequences[field]) for field in _CURRENT_TERMINAL_SEQUENCE_FIELDS
        },
        "sequence_indexes": indexes,
    }
    if reopen_scalable_native_terminal_artifact(container) != legacy_artifact:
        raise RuntimeError("sharded current native terminal reconstruction changed")
    publish_canonical_json_no_replace(output_path, container)
    return CanonicalJsonProofBinding.bind(output_path)


def reopen_scalable_native_terminal_artifact(value: object) -> dict[str, object]:
    """Deep-reopen current containers and read legacy containers compatibly."""

    if type(value) is not dict:
        raise TypeError("native terminal artifact must be an object")
    if value.get("schema_version") == 1 or (
        value.get("schema_version") == 2
        and value.get("artifact_kind") == LEGACY_SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND
    ):
        return _reopen_legacy_scalable_native_terminal_artifact(value)
    if (
        value.get("schema_version") == 2
        and value.get("artifact_kind") == "native_terminal_evidence_bundle_v2"
    ):
        return dict(value)
    row = _strict(
        "sharded current native terminal artifact",
        value,
        {
            "schema_version",
            "artifact_kind",
            "semantic_artifact_sha256",
            "artifact_static",
            "reset_static",
            "warmup_request_resets_static",
            "terminal_static",
            "terminal_request_resets_static",
            "sequence_counts",
            "sequence_indexes",
        },
    )
    if (
        row["schema_version"] != 3
        or row["artifact_kind"] != SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND
    ):
        raise ValueError("sharded current native terminal container schema differs")
    artifact_sha256 = _require_sha256(
        "sharded current native terminal semantic artifact",
        row["semantic_artifact_sha256"],
    )
    counts = _strict(
        "sharded current native terminal sequence counts",
        row["sequence_counts"],
        set(_CURRENT_TERMINAL_SEQUENCE_FIELDS),
    )
    indexes = _strict(
        "sharded current native terminal sequence indexes",
        row["sequence_indexes"],
        set(_CURRENT_TERMINAL_SEQUENCE_FIELDS),
    )
    sequences: dict[str, list[object]] = {}
    for field in _CURRENT_TERMINAL_SEQUENCE_FIELDS:
        count = counts[field]
        if type(count) is not int or count < 0:
            raise ValueError("sharded current native terminal sequence count differs")
        sequences[field] = _reopen_sequence(
            indexes[field],
            artifact_kind=f"formal_{field}",
            artifact_id=artifact_sha256,
            expected_count=count,
        )
    artifact = dict(row["artifact_static"])
    reset = dict(row["reset_static"])
    warmup_resets = dict(row["warmup_request_resets_static"])
    terminal = dict(row["terminal_static"])
    terminal_resets = dict(row["terminal_request_resets_static"])
    warmup_resets["receipts"] = sequences["reset_warmup_request_reset_receipts"]
    terminal_resets["receipts"] = sequences["terminal_request_reset_receipts"]
    reset.update(
        {
            "warmup_request_rows": sequences["reset_warmup_request_rows"],
            "warmup_round_rows": sequences["reset_warmup_round_rows"],
            "warmup_update_rows": sequences["reset_warmup_update_rows"],
            "warmup_historical_kv_source_versions": _history_from_items(
                sequences["reset_warmup_historical_kv_items"],
                label="sharded warmup historical KV item",
            ),
            "warmup_request_source_point_resets": warmup_resets,
        }
    )
    terminal.update(
        {
            "expected_request_ids": sequences["terminal_expected_request_ids"],
            "request_round_rows": {
                "requests": sequences["terminal_request_rows"],
                "rounds": sequences["terminal_round_rows"],
            },
            "update_rows": sequences["terminal_update_rows"],
            "historical_kv_source_versions": _history_from_items(
                sequences["terminal_historical_kv_items"],
                label="sharded terminal historical KV item",
            ),
            "request_source_point_resets": terminal_resets,
        }
    )
    artifact.update(
        {
            "warmup_requests": sequences["warmup_requests"],
            "scored_requests": sequences["scored_requests"],
            "reset": reset,
            "terminal": terminal,
        }
    )
    if _sha256(artifact) != artifact_sha256:
        raise ValueError("sharded current native terminal complete digest differs")
    return artifact


def publish_scalable_unsigned_native_itl_bundle(
    *,
    output_path: str | Path,
    legacy_bundle: object,
) -> CanonicalJsonProofBinding:
    """Publish native ITL pointers with nested per-request event shards."""

    bundle = _strict(
        "unsigned native ITL pointer bundle",
        legacy_bundle,
        {
            "schema_version",
            "kind",
            "run_binding_sha256",
            "terminal_artifact_raw_sha256",
            "terminal_artifact_semantic_sha256",
            "scored_request_inputs_sha256",
            "native_result_pointers",
        },
    )
    if (
        bundle["schema_version"] != 1
        or bundle["kind"] != "unsigned_native_itl_result_pointer_bundle"
    ):
        raise ValueError("unsigned native ITL bundle is not registered schema 1")
    if len(_canonical_bytes(bundle)) + 1 <= _INLINE_MAX_BYTES:
        publish_canonical_json_no_replace(output_path, bundle)
        return CanonicalJsonProofBinding.bind(output_path)
    raw_pointers = bundle.pop("native_result_pointers")
    if type(raw_pointers) is not list or not raw_pointers:
        raise ValueError("unsigned native ITL bundle has no pointers")
    legacy_sha256 = _sha256(legacy_bundle)
    root = _shard_root(output_path)
    pointer_rows: list[object] = []
    for ordinal, raw_pointer in enumerate(raw_pointers):
        if type(raw_pointer) is not dict:
            raise TypeError("unsigned native ITL pointer must be an object")
        pointer = dict(raw_pointer)
        events = pointer.pop("events", None)
        if type(events) is not list or not events:
            raise ValueError("unsigned native ITL pointer has no events")
        event_id: str | None = None
        event_binding: CanonicalJsonProofBinding | None = None
        inline_events: list[object] | None = events
        if len(_canonical_bytes(events)) + 1 > _INLINE_POINTER_EVENTS_MAX_BYTES:
            event_id = _sha256(
                {
                    "legacy_bundle_sha256": legacy_sha256,
                    "pointer_ordinal": ordinal,
                    "request_id": pointer.get("request_id"),
                }
            )
            event_directory = root / f"events-{ordinal:06d}"
            event_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
            event_binding, _event_index = publish_formal_canonical_sequence_shards(
                artifact_kind="formal_unsigned_native_itl_events",
                artifact_id=event_id,
                rows=events,
                output_directory=event_directory,
                maximum_shard_bytes=_SHARD_MAX_BYTES,
                maximum_shard_rows=_SHARD_MAX_ROWS,
            )
            inline_events = None
        pointer_rows.append(
            {
                "pointer_static": pointer,
                "event_artifact_id": event_id,
                "event_count": len(events),
                "events_inline": inline_events,
                "events_shard_index": (
                    None if event_binding is None else event_binding.to_dict()
                ),
            }
        )
    pointer_binding = _publish_sequence(
        rows=pointer_rows,
        artifact_kind="formal_unsigned_native_itl_pointers",
        artifact_id=legacy_sha256,
        root=root,
    )
    assert pointer_binding is not None
    container = {
        "schema_version": 2,
        "kind": SHARDED_UNSIGNED_NATIVE_ITL_BUNDLE_KIND,
        "legacy_bundle_sha256": legacy_sha256,
        "legacy_bundle_static": _copy_json(bundle),
        "pointer_count": len(pointer_rows),
        "pointers_shard_index": pointer_binding,
    }
    if reopen_scalable_unsigned_native_itl_bundle(container) != legacy_bundle:
        raise RuntimeError("sharded unsigned native ITL reconstruction changed")
    publish_canonical_json_no_replace(output_path, container)
    return CanonicalJsonProofBinding.bind(output_path)


def reopen_scalable_unsigned_native_itl_bundle(
    value: object,
) -> dict[str, object]:
    """Return the exact registered schema-1 ITL bundle from either container."""

    if (
        type(value) is dict
        and value.get("schema_version") == 1
        and value.get("kind") == "unsigned_native_itl_result_pointer_bundle"
    ):
        return dict(value)
    row = _strict(
        "sharded unsigned native ITL bundle",
        value,
        {
            "schema_version",
            "kind",
            "legacy_bundle_sha256",
            "legacy_bundle_static",
            "pointer_count",
            "pointers_shard_index",
        },
    )
    if (
        row["schema_version"] != 2
        or row["kind"] != SHARDED_UNSIGNED_NATIVE_ITL_BUNDLE_KIND
    ):
        raise ValueError("sharded unsigned native ITL bundle schema differs")
    legacy_sha256 = _require_sha256(
        "sharded unsigned native ITL legacy bundle", row["legacy_bundle_sha256"]
    )
    pointer_count = row["pointer_count"]
    if type(pointer_count) is not int or pointer_count < 1:
        raise ValueError("sharded unsigned native ITL pointer count differs")
    pointer_rows = _reopen_sequence(
        row["pointers_shard_index"],
        artifact_kind="formal_unsigned_native_itl_pointers",
        artifact_id=legacy_sha256,
        expected_count=pointer_count,
    )
    pointers: list[object] = []
    for raw_pointer_row in pointer_rows:
        pointer_row = _strict(
            "sharded unsigned native ITL pointer row",
            raw_pointer_row,
            {
                "pointer_static",
                "event_artifact_id",
                "event_count",
                "events_inline",
                "events_shard_index",
            },
        )
        event_count = pointer_row["event_count"]
        if type(event_count) is not int or event_count < 1:
            raise ValueError("sharded unsigned native ITL event count differs")
        pointer = dict(pointer_row["pointer_static"])
        inline_events = pointer_row["events_inline"]
        event_binding = pointer_row["events_shard_index"]
        event_id = pointer_row["event_artifact_id"]
        if inline_events is not None:
            if (
                type(inline_events) is not list
                or len(inline_events) != event_count
                or event_binding is not None
                or event_id is not None
                or len(_canonical_bytes(inline_events)) + 1
                > _INLINE_POINTER_EVENTS_MAX_BYTES
            ):
                raise ValueError("inline unsigned native ITL events differ")
            events = inline_events
        else:
            event_id = _require_sha256(
                "sharded unsigned native ITL event artifact", event_id
            )
            expected_event_id = _sha256(
                {
                    "legacy_bundle_sha256": legacy_sha256,
                    "pointer_ordinal": len(pointers),
                    "request_id": pointer.get("request_id"),
                }
            )
            if event_id != expected_event_id or event_binding is None:
                raise ValueError("sharded unsigned native ITL event lineage differs")
            events = _reopen_sequence(
                event_binding,
                artifact_kind="formal_unsigned_native_itl_events",
                artifact_id=event_id,
                expected_count=event_count,
            )
        pointer["events"] = events
        pointers.append(pointer)
    bundle = dict(row["legacy_bundle_static"])
    bundle["native_result_pointers"] = pointers
    if _sha256(bundle) != legacy_sha256:
        raise ValueError("sharded unsigned native ITL complete digest differs")
    return bundle


def publish_scalable_formal_gang_request_terminal(
    *,
    output_path: str | Path,
    legacy_terminal: object,
) -> CanonicalJsonProofBinding:
    """Publish legacy request rows or the current deep warmup-reset binding."""

    common_fields = {
        "schema_version",
        "kind",
        "protocol_sha256",
        "formal_execution_authorized",
        "plan_sha256",
        "formal_launch_admission",
        "formal_launch_consumption",
        "budget_consumption",
        "capability_sha256",
        "begin_sha256",
        "reset_sha256",
        "finalize_sha256",
        "warmup_requests",
        "scored_requests",
    }
    if type(legacy_terminal) is not dict:
        raise TypeError("formal gang request terminal must be an object")
    current = (
        legacy_terminal.get("schema_version") == 2
        and legacy_terminal.get("kind") == "unsigned_formal_gang_request_terminal_v2"
    )
    terminal = _strict(
        "formal gang request terminal",
        legacy_terminal,
        common_fields | ({"formal_gang_reset"} if current else set()),
    )
    legacy = (
        terminal["schema_version"] == 1
        and terminal["kind"] == "unsigned_formal_gang_request_terminal"
    )
    if not current and not legacy:
        raise ValueError("formal gang request terminal schema differs")
    if current:
        reset_binding = CanonicalJsonProofBinding.from_dict(
            terminal["formal_gang_reset"]
        )
        reset = reopen_scalable_formal_gang_terminal(reset_binding.reopen())
        if (
            reset.get("schema_version") != 2
            or reset.get("kind") != "sglang_formal_gang_all_rank_terminal"
            or reset.get("action") != "formal_gang_reset"
            or reset.get("aggregate_sha256") != terminal["reset_sha256"]
        ):
            raise ValueError("current formal gang warmup reset binding differs")
    if len(_canonical_bytes(terminal)) + 1 <= _INLINE_MAX_BYTES:
        publish_canonical_json_no_replace(output_path, terminal)
        return CanonicalJsonProofBinding.bind(output_path)
    terminal_sha256 = _sha256(legacy_terminal)
    root = _shard_root(output_path)
    sequences = {
        phase: list(terminal.pop(f"{phase}_requests")) for phase in ("warmup", "scored")
    }
    indexes = {
        phase: _publish_sequence(
            rows=rows,
            artifact_kind=f"formal_gang_{phase}_request_terminals",
            artifact_id=terminal_sha256,
            root=root,
        )
        for phase, rows in sequences.items()
    }
    if current:
        container = {
            "schema_version": 3,
            "kind": SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND,
            "semantic_terminal_sha256": terminal_sha256,
            "terminal_static": _copy_json(terminal),
            "request_counts": {phase: len(rows) for phase, rows in sequences.items()},
            "request_shard_indexes": indexes,
        }
    else:
        container = {
            "schema_version": 2,
            "kind": LEGACY_SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND,
            "legacy_terminal_sha256": terminal_sha256,
            "legacy_terminal_static": _copy_json(terminal),
            "request_counts": {phase: len(rows) for phase, rows in sequences.items()},
            "request_shard_indexes": indexes,
        }
    if reopen_scalable_formal_gang_request_terminal(container) != legacy_terminal:
        raise RuntimeError("sharded formal gang request terminal changed")
    publish_canonical_json_no_replace(output_path, container)
    return CanonicalJsonProofBinding.bind(output_path)


def reopen_scalable_formal_gang_request_terminal(
    value: object,
) -> dict[str, object]:
    if type(value) is dict and (
        (
            value.get("schema_version") == 1
            and value.get("kind") == "unsigned_formal_gang_request_terminal"
        )
        or (
            value.get("schema_version") == 2
            and value.get("kind") == "unsigned_formal_gang_request_terminal_v2"
        )
    ):
        return dict(value)
    if type(value) is not dict:
        raise TypeError("sharded formal gang request terminal must be an object")
    current = (
        value.get("schema_version") == 3
        and value.get("kind") == SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND
    )
    row = _strict(
        "sharded formal gang request terminal",
        value,
        {
            "schema_version",
            "kind",
            "semantic_terminal_sha256" if current else "legacy_terminal_sha256",
            "terminal_static" if current else "legacy_terminal_static",
            "request_counts",
            "request_shard_indexes",
        },
    )
    legacy = (
        row["schema_version"] == 2
        and row["kind"] == LEGACY_SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND
    )
    if not current and not legacy:
        raise ValueError("sharded formal gang request terminal schema differs")
    digest_field = "semantic_terminal_sha256" if current else "legacy_terminal_sha256"
    static_field = "terminal_static" if current else "legacy_terminal_static"
    terminal_sha256 = _require_sha256(
        "sharded formal gang request terminal", row[digest_field]
    )
    counts = _strict(
        "sharded formal gang request counts",
        row["request_counts"],
        {"warmup", "scored"},
    )
    indexes = _strict(
        "sharded formal gang request indexes",
        row["request_shard_indexes"],
        {"warmup", "scored"},
    )
    terminal = dict(row[static_field])
    for phase in ("warmup", "scored"):
        count = counts[phase]
        if type(count) is not int or count < 0:
            raise ValueError("sharded formal gang request count differs")
        terminal[f"{phase}_requests"] = _reopen_sequence(
            indexes[phase],
            artifact_kind=f"formal_gang_{phase}_request_terminals",
            artifact_id=terminal_sha256,
            expected_count=count,
        )
    if _sha256(terminal) != terminal_sha256:
        raise ValueError("sharded formal gang request terminal digest differs")
    return terminal


def _publish_gang_pointer_rows(
    *,
    raw_pointers: object,
    phase: str,
    legacy_sha256: str,
    root: Path,
) -> tuple[dict[str, object] | None, int]:
    if type(raw_pointers) is not list:
        raise TypeError("formal gang native ITL pointers must be arrays")
    pointer_rows: list[object] = []
    for ordinal, raw_pointer in enumerate(raw_pointers):
        if type(raw_pointer) is not dict:
            raise TypeError("formal gang native ITL pointer must be an object")
        pointer = dict(raw_pointer)
        events = pointer.pop("events", None)
        if type(events) is not list or not events:
            raise ValueError("formal gang native ITL pointer has no events")
        event_id: str | None = None
        event_binding: CanonicalJsonProofBinding | None = None
        inline_events: list[object] | None = events
        if len(_canonical_bytes(events)) + 1 > _INLINE_POINTER_EVENTS_MAX_BYTES:
            event_id = _sha256(
                {
                    "legacy_bundle_sha256": legacy_sha256,
                    "phase": phase,
                    "pointer_ordinal": ordinal,
                    "request_id": pointer.get("request_id"),
                }
            )
            directory = root / f"{phase}-events-{ordinal:06d}"
            directory.mkdir(mode=0o700, parents=False, exist_ok=False)
            event_binding, _index = publish_formal_canonical_sequence_shards(
                artifact_kind=f"formal_gang_{phase}_native_itl_events",
                artifact_id=event_id,
                rows=events,
                output_directory=directory,
                maximum_shard_bytes=_SHARD_MAX_BYTES,
                maximum_shard_rows=_SHARD_MAX_ROWS,
            )
            inline_events = None
        pointer_rows.append(
            {
                "pointer_static": pointer,
                "event_artifact_id": event_id,
                "event_count": len(events),
                "events_inline": inline_events,
                "events_shard_index": (
                    None if event_binding is None else event_binding.to_dict()
                ),
            }
        )
    return (
        _publish_sequence(
            rows=pointer_rows,
            artifact_kind=f"formal_gang_{phase}_native_itl_pointers",
            artifact_id=legacy_sha256,
            root=root,
        ),
        len(pointer_rows),
    )


def _reopen_gang_pointer_rows(
    *,
    binding: object,
    phase: str,
    legacy_sha256: str,
    pointer_count: int,
) -> list[object]:
    pointer_rows = _reopen_sequence(
        binding,
        artifact_kind=f"formal_gang_{phase}_native_itl_pointers",
        artifact_id=legacy_sha256,
        expected_count=pointer_count,
    )
    pointers: list[object] = []
    for ordinal, raw_pointer_row in enumerate(pointer_rows):
        pointer_row = _strict(
            "sharded formal gang native ITL pointer row",
            raw_pointer_row,
            {
                "pointer_static",
                "event_artifact_id",
                "event_count",
                "events_inline",
                "events_shard_index",
            },
        )
        event_count = pointer_row["event_count"]
        if type(event_count) is not int or event_count < 1:
            raise ValueError("sharded formal gang native ITL event count differs")
        pointer = dict(pointer_row["pointer_static"])
        inline_events = pointer_row["events_inline"]
        event_id = pointer_row["event_artifact_id"]
        event_binding = pointer_row["events_shard_index"]
        if inline_events is not None:
            if (
                type(inline_events) is not list
                or len(inline_events) != event_count
                or event_id is not None
                or event_binding is not None
                or len(_canonical_bytes(inline_events)) + 1
                > _INLINE_POINTER_EVENTS_MAX_BYTES
            ):
                raise ValueError("inline formal gang native ITL events differ")
            events = inline_events
        else:
            event_id = _require_sha256(
                "sharded formal gang native ITL event artifact", event_id
            )
            expected_id = _sha256(
                {
                    "legacy_bundle_sha256": legacy_sha256,
                    "phase": phase,
                    "pointer_ordinal": ordinal,
                    "request_id": pointer.get("request_id"),
                }
            )
            if event_id != expected_id or event_binding is None:
                raise ValueError("sharded formal gang native ITL event lineage differs")
            events = _reopen_sequence(
                event_binding,
                artifact_kind=f"formal_gang_{phase}_native_itl_events",
                artifact_id=event_id,
                expected_count=event_count,
            )
        pointer["events"] = events
        pointers.append(pointer)
    return pointers


def publish_scalable_formal_gang_itl_bundle(
    *,
    output_path: str | Path,
    legacy_bundle: object,
) -> CanonicalJsonProofBinding:
    """Publish distributed native pointers with bounded per-request events."""

    bundle = _strict(
        "formal gang native ITL bundle",
        legacy_bundle,
        {
            "schema_version",
            "kind",
            "protocol_sha256",
            "formal_execution_authorized",
            "plan_sha256",
            "formal_launch_admission",
            "formal_launch_consumption",
            "budget_consumption",
            "warmup_pointers",
            "scored_pointers",
        },
    )
    if (
        bundle["schema_version"] != 1
        or bundle["kind"] != "unsigned_formal_gang_native_itl_pointer_bundle"
    ):
        raise ValueError("formal gang native ITL bundle is not registered schema 1")
    if len(_canonical_bytes(bundle)) + 1 <= _INLINE_MAX_BYTES:
        publish_canonical_json_no_replace(output_path, bundle)
        return CanonicalJsonProofBinding.bind(output_path)
    legacy_sha256 = _sha256(legacy_bundle)
    raw_phases = {
        phase: bundle.pop(f"{phase}_pointers") for phase in ("warmup", "scored")
    }
    root = _shard_root(output_path)
    published = {
        phase: _publish_gang_pointer_rows(
            raw_pointers=raw_phases[phase],
            phase=phase,
            legacy_sha256=legacy_sha256,
            root=root,
        )
        for phase in ("warmup", "scored")
    }
    container = {
        "schema_version": 2,
        "kind": SHARDED_FORMAL_GANG_ITL_BUNDLE_KIND,
        "legacy_bundle_sha256": legacy_sha256,
        "legacy_bundle_static": _copy_json(bundle),
        "pointer_counts": {
            phase: published[phase][1] for phase in ("warmup", "scored")
        },
        "pointer_shard_indexes": {
            phase: published[phase][0] for phase in ("warmup", "scored")
        },
    }
    if reopen_scalable_formal_gang_itl_bundle(container) != legacy_bundle:
        raise RuntimeError("sharded formal gang native ITL bundle changed")
    publish_canonical_json_no_replace(output_path, container)
    return CanonicalJsonProofBinding.bind(output_path)


def reopen_scalable_formal_gang_itl_bundle(value: object) -> dict[str, object]:
    if (
        type(value) is dict
        and value.get("schema_version") == 1
        and value.get("kind") == "unsigned_formal_gang_native_itl_pointer_bundle"
    ):
        return dict(value)
    row = _strict(
        "sharded formal gang native ITL bundle",
        value,
        {
            "schema_version",
            "kind",
            "legacy_bundle_sha256",
            "legacy_bundle_static",
            "pointer_counts",
            "pointer_shard_indexes",
        },
    )
    if row["schema_version"] != 2 or row["kind"] != SHARDED_FORMAL_GANG_ITL_BUNDLE_KIND:
        raise ValueError("sharded formal gang native ITL bundle schema differs")
    legacy_sha256 = _require_sha256(
        "sharded formal gang native ITL bundle", row["legacy_bundle_sha256"]
    )
    counts = _strict(
        "sharded formal gang native ITL counts",
        row["pointer_counts"],
        {"warmup", "scored"},
    )
    indexes = _strict(
        "sharded formal gang native ITL indexes",
        row["pointer_shard_indexes"],
        {"warmup", "scored"},
    )
    bundle = dict(row["legacy_bundle_static"])
    for phase in ("warmup", "scored"):
        count = counts[phase]
        if type(count) is not int or count < 0:
            raise ValueError("sharded formal gang native ITL count differs")
        bundle[f"{phase}_pointers"] = _reopen_gang_pointer_rows(
            binding=indexes[phase],
            phase=phase,
            legacy_sha256=legacy_sha256,
            pointer_count=count,
        )
    if _sha256(bundle) != legacy_sha256:
        raise ValueError("sharded formal gang native ITL digest differs")
    return bundle


def _publish_rank_terminal_sequences(
    *,
    rank: dict[str, object],
    rank_ordinal: int,
    legacy_sha256: str,
    root: Path,
    current: bool,
) -> dict[str, object]:
    native_state = dict(rank.pop("native_state"))
    sequences = {
        "request_terminals": list(rank.pop("request_terminals")),
        "request_terminal_sha256s": list(rank.pop("request_terminal_sha256s")),
        "round_rows": list(native_state.pop("round_rows")),
        "update_rows": list(native_state.pop("update_rows")),
        "historical_kv_items": [
            {"request_id": key, "versions": value}
            for key, value in sorted(
                dict(native_state.pop("historical_kv_source_versions")).items()
            )
        ],
    }
    if current:
        resets = native_state.pop("request_source_point_resets")
        if type(resets) is not dict:
            raise TypeError("current formal gang request resets must be an object")
        resets = dict(resets)
        sequences["request_reset_receipts"] = list(resets.pop("receipts"))
        native_state["request_source_point_resets"] = resets
        adaptation = native_state.get("adaptation")
        if adaptation is None:
            sequences.update(
                {
                    "adaptation_rounds": [],
                    "adaptation_updates": [],
                    "adaptation_request_reset_receipts": [],
                }
            )
        elif type(adaptation) is dict:
            adaptation = dict(adaptation)
            sequences.update(
                {
                    "adaptation_rounds": list(adaptation.pop("rounds")),
                    "adaptation_updates": list(adaptation.pop("updates")),
                    "adaptation_request_reset_receipts": list(
                        adaptation.pop("request_source_point_reset_receipts")
                    ),
                }
            )
            native_state["adaptation"] = adaptation
        else:
            raise TypeError("current formal gang adaptation must be an object or null")
    published: dict[str, object] = {}
    for name, rows in sequences.items():
        published[name] = _publish_sequence(
            rows=rows,
            artifact_kind=f"formal_gang_rank_{rank_ordinal}_{name}",
            artifact_id=legacy_sha256,
            root=root,
        )
    return {
        "rank_static": rank,
        "native_state_static": native_state,
        "sequence_counts": {name: len(rows) for name, rows in sequences.items()},
        "sequence_indexes": published,
    }


def _reopen_rank_terminal_sequences(
    *,
    value: object,
    rank_ordinal: int,
    legacy_sha256: str,
    current: bool,
) -> dict[str, object]:
    row = _strict(
        "sharded formal gang rank terminal",
        value,
        {
            "rank_static",
            "native_state_static",
            "sequence_counts",
            "sequence_indexes",
        },
    )
    names = {
        "request_terminals",
        "request_terminal_sha256s",
        "round_rows",
        "update_rows",
        "historical_kv_items",
    }
    if current:
        names.update(
            {
                "request_reset_receipts",
                "adaptation_rounds",
                "adaptation_updates",
                "adaptation_request_reset_receipts",
            }
        )
    counts = _strict("sharded formal gang rank counts", row["sequence_counts"], names)
    indexes = _strict(
        "sharded formal gang rank indexes", row["sequence_indexes"], names
    )
    sequences: dict[str, list[object]] = {}
    for name in names:
        count = counts[name]
        if type(count) is not int or count < 0:
            raise ValueError("sharded formal gang rank sequence count differs")
        sequences[name] = _reopen_sequence(
            indexes[name],
            artifact_kind=f"formal_gang_rank_{rank_ordinal}_{name}",
            artifact_id=legacy_sha256,
            expected_count=count,
        )
    history: dict[str, object] = {}
    for raw_item in sequences["historical_kv_items"]:
        item = _strict(
            "sharded formal gang historical KV item",
            raw_item,
            {"request_id", "versions"},
        )
        request_id = item["request_id"]
        if type(request_id) is not str or request_id in history:
            raise ValueError("sharded formal gang historical KV keys differ")
        history[request_id] = item["versions"]
    native_state = dict(row["native_state_static"])
    native_state.update(
        {
            "round_rows": sequences["round_rows"],
            "update_rows": sequences["update_rows"],
            "historical_kv_source_versions": history,
        }
    )
    if current:
        resets = dict(native_state["request_source_point_resets"])
        resets["receipts"] = sequences["request_reset_receipts"]
        native_state["request_source_point_resets"] = resets
        adaptation = native_state.get("adaptation")
        if adaptation is not None:
            adaptation = dict(adaptation)
            adaptation.update(
                {
                    "rounds": sequences["adaptation_rounds"],
                    "updates": sequences["adaptation_updates"],
                    "request_source_point_reset_receipts": sequences[
                        "adaptation_request_reset_receipts"
                    ],
                }
            )
            native_state["adaptation"] = adaptation
        elif any(
            sequences[name]
            for name in (
                "adaptation_rounds",
                "adaptation_updates",
                "adaptation_request_reset_receipts",
            )
        ):
            raise ValueError("allocation-free formal gang carries adaptation shards")
    rank = dict(row["rank_static"])
    rank.update(
        {
            "request_terminals": sequences["request_terminals"],
            "request_terminal_sha256s": sequences["request_terminal_sha256s"],
            "native_state": native_state,
        }
    )
    return rank


def publish_scalable_formal_gang_terminal(
    *,
    output_path: str | Path,
    legacy_terminal: object,
) -> CanonicalJsonProofBinding:
    """Publish legacy or current all-rank gang evidence with rank-local shards."""

    if type(legacy_terminal) is not dict:
        raise TypeError("formal gang terminal must be an object")
    terminal = dict(legacy_terminal)
    schema_version = terminal.get("schema_version")
    if terminal.get("kind") != "sglang_formal_gang_all_rank_terminal" or (
        schema_version not in {1, 2}
    ):
        raise ValueError("formal gang terminal schema differs")
    raw_rank_values = terminal.get("rank_terminals")
    current_flags = (
        tuple(
            type(rank) is dict
            and type(rank.get("native_state")) is dict
            and "request_source_point_resets" in rank["native_state"]
            for rank in raw_rank_values
        )
        if type(raw_rank_values) is list
        else ()
    )
    if (
        not current_flags
        or len(set(current_flags)) != 1
        or (schema_version == 2) is not current_flags[0]
    ):
        raise ValueError("formal gang terminal schema/native-state generation differs")
    if len(_canonical_bytes(terminal)) + 1 <= _INLINE_MAX_BYTES:
        publish_canonical_json_no_replace(output_path, terminal)
        return CanonicalJsonProofBinding.bind(output_path)
    raw_ranks = terminal.pop("rank_terminals", None)
    if type(raw_ranks) is not list or not raw_ranks:
        raise ValueError("formal gang terminal has no rank terminals")
    current = current_flags[0]
    legacy_sha256 = _sha256(legacy_terminal)
    root = _shard_root(output_path)
    rank_rows: list[object] = []
    for ordinal, raw_rank in enumerate(raw_ranks):
        if type(raw_rank) is not dict:
            raise TypeError("formal gang rank terminal must be an object")
        rank_rows.append(
            _publish_rank_terminal_sequences(
                rank=dict(raw_rank),
                rank_ordinal=ordinal,
                legacy_sha256=legacy_sha256,
                root=root,
                current=current,
            )
        )
    rank_binding = _publish_sequence(
        rows=rank_rows,
        artifact_kind="formal_gang_rank_terminals",
        artifact_id=legacy_sha256,
        root=root,
    )
    assert rank_binding is not None
    container = {
        "schema_version": 3 if current else 2,
        "kind": (
            SHARDED_FORMAL_GANG_TERMINAL_KIND
            if current
            else LEGACY_SHARDED_FORMAL_GANG_TERMINAL_KIND
        ),
        "legacy_terminal_sha256": legacy_sha256,
        "legacy_terminal_static": _copy_json(terminal),
        "rank_count": len(rank_rows),
        "rank_terminals_shard_index": rank_binding,
    }
    if reopen_scalable_formal_gang_terminal(container) != legacy_terminal:
        raise RuntimeError("sharded formal gang terminal reconstruction changed")
    publish_canonical_json_no_replace(output_path, container)
    return CanonicalJsonProofBinding.bind(output_path)


def reopen_scalable_formal_gang_terminal(value: object) -> dict[str, object]:
    if (
        type(value) is dict
        and value.get("schema_version") in {1, 2}
        and value.get("kind") == "sglang_formal_gang_all_rank_terminal"
    ):
        return dict(value)
    row = _strict(
        "sharded formal gang terminal",
        value,
        {
            "schema_version",
            "kind",
            "legacy_terminal_sha256",
            "legacy_terminal_static",
            "rank_count",
            "rank_terminals_shard_index",
        },
    )
    current = (
        row["schema_version"] == 3 and row["kind"] == SHARDED_FORMAL_GANG_TERMINAL_KIND
    )
    legacy = (
        row["schema_version"] == 2
        and row["kind"] == LEGACY_SHARDED_FORMAL_GANG_TERMINAL_KIND
    )
    if not current and not legacy:
        raise ValueError("sharded formal gang terminal schema differs")
    legacy_sha256 = _require_sha256(
        "sharded formal gang terminal", row["legacy_terminal_sha256"]
    )
    rank_count = row["rank_count"]
    if type(rank_count) is not int or rank_count < 1:
        raise ValueError("sharded formal gang rank count differs")
    raw_ranks = _reopen_sequence(
        row["rank_terminals_shard_index"],
        artifact_kind="formal_gang_rank_terminals",
        artifact_id=legacy_sha256,
        expected_count=rank_count,
    )
    terminal = dict(row["legacy_terminal_static"])
    terminal["rank_terminals"] = [
        _reopen_rank_terminal_sequences(
            value=raw_rank,
            rank_ordinal=ordinal,
            legacy_sha256=legacy_sha256,
            current=current,
        )
        for ordinal, raw_rank in enumerate(raw_ranks)
    ]
    if _sha256(terminal) != legacy_sha256:
        raise ValueError("sharded formal gang terminal digest differs")
    return terminal


__all__ = [
    "LEGACY_SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND",
    "LEGACY_SHARDED_FORMAL_GANG_TERMINAL_KIND",
    "LEGACY_SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND",
    "SHARDED_FORMAL_GANG_ITL_BUNDLE_KIND",
    "SHARDED_FORMAL_GANG_REQUEST_TERMINAL_KIND",
    "SHARDED_FORMAL_GANG_TERMINAL_KIND",
    "SHARDED_NATIVE_TERMINAL_ARTIFACT_KIND",
    "SHARDED_UNSIGNED_NATIVE_ITL_BUNDLE_KIND",
    "publish_scalable_formal_gang_itl_bundle",
    "publish_scalable_formal_gang_request_terminal",
    "publish_scalable_formal_gang_terminal",
    "publish_scalable_native_terminal_artifact",
    "publish_scalable_unsigned_native_itl_bundle",
    "reopen_scalable_formal_gang_itl_bundle",
    "reopen_scalable_formal_gang_request_terminal",
    "reopen_scalable_formal_gang_terminal",
    "reopen_scalable_native_terminal_artifact",
    "reopen_scalable_unsigned_native_itl_bundle",
]
