"""Evidence-derived reduction for schema-v3 industrial GPU runs.

This module deliberately accepts paths to terminal receipts, not caller-provided
metric summaries.  Every statistic is rebuilt from the receipt-bound Parquet
rows.  The small JSON locks accepted alongside those receipts contain only
pre-run request qualification and post-run hardware identity that cannot be
represented in the normalized request/performance tables.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    ConfirmationBlockPlan,
    ExperimentCell,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import (
    PRIMARY_CONTRASTS,
    BootstrapInterval,
    HardwareBlockObservation,
    HardwareEnvelope,
    MultiplicityDecision,
    P99ClaimGuard,
    PairedBcaContrast,
    PilotBlock,
    PowerSizingPlan,
    SloAccounting,
    SloRequest,
    account_slo,
    guard_p99_claim,
    hierarchical_block_request_bootstrap,
    holm_primary_contrasts,
    paired_bca_contrast,
    preregister_power_sizing,
    time_block_bootstrap,
    validate_hardware_block,
)
from lightcone_spec.telemetry.records import OUTPUT_HASH_FORMAT
from lightcone_spec.telemetry.writer import load_completed_evidence

type BootstrapStatistic = Callable[[np.ndarray], float | np.ndarray]
type ReducerStatus = Literal["UNRESOLVED"]

_LOWER_SHA256_LENGTH = 64
_METHODS = tuple(CORE_METHODS)
_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
    "evidence_dropped_rows",
)
_INDUSTRIAL_DOCTOR_CHECKS = frozenset(
    {
        "compatibility_manifest",
        "project_patch_binding",
        "project_sglang_roots_distinct",
        "project_source_tree",
        "python",
        "linux_host",
        "torch",
        "triton",
        "flashinfer",
        "flashinfer_cuda_flavor",
        "cuda_build",
        "cuda_toolkit",
        "torch_cuda_visibility",
        "driver",
        "gpu_count",
        "gpu_identity",
        "gpu_memory",
        "gpu_topology",
        "compiler",
        "disk",
        "network",
        "sglang_commit_lineage",
        "sglang_tree",
        "sglang_import",
    }
)
_REQUEST_METRICS = frozenset(
    {
        "ttft_ms",
        "within_request_p99_itl_ms",
        "latency_ms",
        "output_tokens",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _LOWER_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_file(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} digest must be lower-case SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"{label} digest mismatch")
    return path.read_bytes()


def _bound_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    body = _bound_file(path, expected_sha256, label=label)
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain one JSON object")
    return value


@dataclass(frozen=True)
class BoundArtifact:
    """One immutable file reference supplied to the reducer."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("bound artifact paths must be pathlib.Path values")
        if not _is_sha256(self.sha256):
            raise ValueError("bound artifact digest must be lower-case SHA-256")


@dataclass(frozen=True)
class IndustrialCellEvidence:
    """All rank-terminal and hardware receipts for one registry cell."""

    cell_id: str
    terminal_receipts: tuple[BoundArtifact, ...]
    hardware_receipt: BoundArtifact

    def __post_init__(self) -> None:
        if not _is_sha256(self.cell_id):
            raise ValueError("cell_id must be lower-case SHA-256")
        if not self.terminal_receipts:
            raise ValueError("cell evidence requires terminal rank receipts")


@dataclass(frozen=True)
class IndustrialBlockEvidence:
    """One paired block with all four methods and its pre-run qualification lock."""

    block: int
    cells: tuple[IndustrialCellEvidence, ...]
    qualification_lock: BoundArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.block, int) or isinstance(self.block, bool):
            raise TypeError("industrial block must be an integer")
        if not self.cells:
            raise ValueError("industrial block evidence requires cells")


@dataclass(frozen=True)
class MethodReduction:
    """Evidence-derived aggregate for one method over final blocks."""

    method: str
    block_ids: tuple[str, ...]
    mean_output_goodput_tps: float
    mean_slo_qualified_goodput_tps: float
    slo: SloAccounting
    aggregate_latency_p99: P99ClaimGuard


@dataclass(frozen=True)
class IndustrialRunBinding:
    """Explicit immutable identity recovered from one paired cell's run rows."""

    block: int
    method: str
    cell_id: str
    config_sha256: str
    rank_config_sha256s: tuple[str, ...]
    run_id: str
    rank_count: int
    model_pair: str
    corpus_sha256: str
    arrival_trace_sha256: str
    request_ids_sha256: str
    sampling_profile_sha256: str
    model_lock_sha256: str
    patched_sglang_tree: str
    run_nonce_sha256: str
    topology_sha256: str
    gpu_uuids: tuple[str, ...]
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256: str


@dataclass(frozen=True)
class IndustrialReducerArtifact:
    """Canonical, claim-safe reducer output."""

    status: ReducerStatus
    gpu_evidence: str
    reasons: tuple[str, ...]
    registry_sha256: str
    experiment: str
    runtime_sha256: str
    split_sha256: str
    confirmation_plan_sha256: str
    patched_sglang_tree: str | None
    model_lock_sha256: str | None
    hardware_envelope_sha256: str
    gpu_attestation_sha256: str | None
    doctor_report_sha256: str | None
    pilot_evidence_sha256: str
    completed_pilot_cells_sha256: str
    terminal_receipt_sha256s: tuple[str, ...]
    qualification_lock_sha256s: tuple[str, ...]
    hardware_receipt_sha256s: tuple[str, ...]
    run_bindings: tuple[IndustrialRunBinding, ...]
    power_plan: PowerSizingPlan | None
    hardware_validity: tuple[tuple[str, str, tuple[str, ...]], ...]
    methods: tuple[MethodReduction, ...]
    primary_contrasts: tuple[PairedBcaContrast, ...]
    holm_family: tuple[MultiplicityDecision, ...]
    bootstrap_hooks: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "industrial_schema_v3_reducer",
            "status": self.status,
            "gpu_evidence": self.gpu_evidence,
            "reasons": list(self.reasons),
            "identity": {
                "registry_sha256": self.registry_sha256,
                "experiment": self.experiment,
                "runtime_sha256": self.runtime_sha256,
                "split_sha256": self.split_sha256,
                "confirmation_plan_sha256": self.confirmation_plan_sha256,
                "patched_sglang_tree": self.patched_sglang_tree,
                "model_lock_sha256": self.model_lock_sha256,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
                "gpu_attestation_sha256": self.gpu_attestation_sha256,
                "doctor_report_sha256": self.doctor_report_sha256,
            },
            "evidence": {
                "pilot_evidence_sha256": self.pilot_evidence_sha256,
                "completed_pilot_cells_sha256": self.completed_pilot_cells_sha256,
                "terminal_receipt_sha256s": list(self.terminal_receipt_sha256s),
                "qualification_lock_sha256s": list(self.qualification_lock_sha256s),
                "hardware_receipt_sha256s": list(self.hardware_receipt_sha256s),
                "run_bindings": [asdict(binding) for binding in self.run_bindings],
            },
            "power_plan": None if self.power_plan is None else asdict(self.power_plan),
            "hardware_validity": [
                {"identity": identity, "status": status, "reasons": list(reasons)}
                for identity, status, reasons in self.hardware_validity
            ],
            "methods": [asdict(method) for method in self.methods],
            "primary_contrasts": [
                asdict(contrast) for contrast in self.primary_contrasts
            ],
            "holm_family": [asdict(decision) for decision in self.holm_family],
            "bootstrap_hooks": [
                {"name": name, "independent_units": list(units)}
                for name, units in self.bootstrap_hooks
            ],
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class _RequestMetric:
    request_id: str
    output_tokens: int
    completed: bool
    error: bool
    ttft_ms: float | None
    within_request_p99_itl_ms: float | None
    latency_ms: float


@dataclass(frozen=True)
class _LoadedCell:
    cell: ExperimentCell
    run_rows: tuple[dict[str, Any], ...]
    request_rows: tuple[dict[str, Any], ...]
    performance_rows_by_rank: tuple[tuple[dict[str, Any], ...], ...]
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256: str
    hardware_validity: tuple[tuple[str, str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class _BlockReduction:
    block: int
    qualification_sha256: str
    cells: Mapping[str, _LoadedCell]
    request_metrics: Mapping[str, tuple[_RequestMetric, ...]]
    goodput_tps: Mapping[str, float]
    slo_goodput_tps: Mapping[str, float]
    slo_requests: Mapping[str, tuple[SloRequest, ...]]


@dataclass(frozen=True)
class IndustrialReduction:
    """Reducer artifact plus strictly observed rows for registered bootstraps."""

    artifact: IndustrialReducerArtifact
    _request_metrics: Mapping[str, Mapping[str, tuple[_RequestMetric, ...]]]

    def _bootstrap_rows(
        self,
        method: str,
        metric: str,
    ) -> dict[str, np.ndarray]:
        if method not in _METHODS:
            raise ValueError("bootstrap method must be Target-only, Static, TTS, or L0")
        if metric not in _REQUEST_METRICS:
            raise ValueError("unknown request bootstrap metric")
        blocks = self._request_metrics.get(method)
        if not blocks:
            raise ValueError("reducer has no measured final rows for this method")
        result: dict[str, np.ndarray] = {}
        for block_id, rows in blocks.items():
            values = tuple(getattr(row, metric) for row in rows)
            if any(value is None for value in values):
                raise ValueError(
                    f"{metric} is incomplete; bootstrap refuses to impute missing rows"
                )
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("bootstrap evidence must be finite")
            result[block_id] = array
        return result

    def hierarchical_block_request_bootstrap(
        self,
        method: str,
        metric: str,
        statistic: BootstrapStatistic,
        *,
        repetitions: int = 10_000,
        seed: int = 0,
    ) -> BootstrapInterval:
        """Resample final blocks and requests from observed reducer rows."""

        return hierarchical_block_request_bootstrap(
            self._bootstrap_rows(method, metric),
            statistic,
            repetitions=repetitions,
            seed=seed,
        )

    def whole_time_block_bootstrap(
        self,
        method: str,
        metric: str,
        statistic: BootstrapStatistic,
        *,
        repetitions: int = 10_000,
        seed: int = 0,
    ) -> BootstrapInterval:
        """Resample complete final arrival blocks without splitting their rows."""

        return time_block_bootstrap(
            self._bootstrap_rows(method, metric),
            statistic,
            repetitions=repetitions,
            seed=seed,
        )


def _phase_variant(cell: ExperimentCell, block: int) -> str:
    variant = cell.identity.variant
    expected = "excluded_pilot" if block in PILOT_BLOCKS else "final_candidate"
    if variant is None or not variant.startswith(f"{expected}:"):
        raise ValueError("registry cell does not carry its immutable block phase")
    return variant.split(":", 1)[1]


def _pairing_identity(cell: ExperimentCell, block: int) -> tuple[object, ...]:
    identity = cell.identity
    return (
        identity.experiment,
        identity.model,
        identity.task,
        identity.context,
        identity.regime,
        identity.arrival,
        identity.slo,
        identity.seed,
        identity.concurrency,
        identity.load_factor,
        identity.cohort_count,
        identity.topology,
        _phase_variant(cell, block),
    )


def _expected_topology(cell: ExperimentCell) -> tuple[int, int, int, str]:
    topology = cell.identity.topology
    tensor_parallel_size = 2 if topology == "tp2_dp1" else 1
    data_parallel_size = 2 if topology == "two_replica_tp1_dp2" else 1
    world_size = len(cell.resources.gpu_uuids)
    topology_sha256 = content_sha256(
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "topology": topology,
            "gpu_uuids": list(cell.resources.gpu_uuids),
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "world_size": world_size,
        }
    )
    return tensor_parallel_size, data_parallel_size, world_size, topology_sha256


def _read_terminal_receipt(
    reference: BoundArtifact,
) -> tuple[dict[str, Any], dict[str, Path]]:
    body = _bound_file(reference.path, reference.sha256, label="terminal receipt")
    try:
        receipt = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("terminal receipt is not valid JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 3:
        raise ValueError("industrial reduction requires schema-v3 terminal receipts")
    run_id = receipt.get("run_id")
    rank = receipt.get("rank")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank < 0
        or reference.path.name != f"{run_id}.rank{rank}.complete.json"
    ):
        raise ValueError("terminal receipt path/run/rank identity mismatch")
    try:
        evidence = load_completed_evidence(
            reference.path.parent,
            run_id=run_id,
            rank=rank,
        )
    except RuntimeError as exc:
        raise ValueError("terminal receipt failed durable evidence validation") from exc
    if evidence is None:
        raise ValueError("terminal receipt has no completed evidence")
    return receipt, evidence


def _read_table(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        rows = pq.read_table(path).to_pylist()
    except (OSError, pa.ArrowException) as exc:
        raise ValueError("receipt-bound Parquet evidence is unreadable") from exc
    if not rows:
        raise ValueError("receipt-bound Parquet evidence is empty")
    return tuple(rows)


def _parse_string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError(f"{label} is missing")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise ValueError(f"{label} must be a unique JSON string list")
    return tuple(parsed)


def _parse_output_token_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    if row.get("output_hash_format") != OUTPUT_HASH_FORMAT:
        raise ValueError("request uses an unknown output token hash format")
    value = row.get("output_token_ids")
    digest = row.get("output_token_ids_sha256")
    if not isinstance(value, str) or not _is_sha256(digest):
        raise ValueError("claim reduction requires ordered output token IDs")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("output token IDs are not valid JSON") from exc
    output_tokens = row.get("output_tokens")
    if (
        not isinstance(parsed, list)
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or len(parsed) != output_tokens
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in parsed
        )
    ):
        raise ValueError("output token IDs do not cover the request output")
    expected = hashlib.sha256(
        json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if digest != expected or row.get("output_sha256") != expected:
        raise ValueError("output token-ID digest disagrees with raw IDs")
    return tuple(parsed)


def _parse_itl(value: object, row: Mapping[str, Any]) -> float | None:
    output_tokens = row.get("output_tokens")
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 0
    ):
        raise ValueError("request output_tokens must be a non-negative integer")
    coverage = row.get("token_timing_coverage")
    coalesced = row.get("coalesced_intervals")
    raw_timestamps = row.get("token_timestamps_ns")
    if raw_timestamps is None:
        if value is not None:
            raise ValueError("request ITL summary lacks raw token timestamps")
        return None
    if not isinstance(raw_timestamps, str):
        raise TypeError("request token_timestamps_ns must be a JSON array")
    try:
        parsed_timestamps = json.loads(raw_timestamps)
    except json.JSONDecodeError as exc:
        raise ValueError("request token_timestamps_ns must be a JSON array") from exc
    if (
        not isinstance(parsed_timestamps, list)
        or len(parsed_timestamps) != output_tokens
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in parsed_timestamps
        )
    ):
        raise ValueError("request token timestamps have invalid coverage")
    first_token = row.get("first_token_ns")
    if parsed_timestamps and parsed_timestamps[0] != first_token:
        raise ValueError("raw token timestamps disagree with first_token_ns")
    arrival = row.get("arrival_ns")
    terminal = row.get("completed_ns")
    if parsed_timestamps and (
        not isinstance(arrival, int)
        or isinstance(arrival, bool)
        or not isinstance(terminal, int)
        or isinstance(terminal, bool)
        or parsed_timestamps[0] < arrival
        or parsed_timestamps[-1] > terminal
    ):
        raise ValueError("raw token timestamps are outside the request lifetime")
    raw_intervals = np.diff(np.asarray(parsed_timestamps, dtype=np.int64)) / 1_000_000
    if raw_intervals.size and np.any(raw_intervals <= 0.0):
        raise ValueError("raw token timestamps are not strictly increasing")
    if coverage != 1.0 or coalesced != 0:
        if value is not None:
            raise ValueError(
                "partial token timing cannot carry a claimable ITL summary"
            )
        return None
    if value is None:
        if raw_intervals.size:
            raise ValueError("full raw token timing lacks its persisted ITL summary")
        return None
    if not isinstance(value, str):
        raise TypeError("request inter_token_ms must be a JSON array")
    try:
        parsed_intervals = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("request inter_token_ms must be a JSON array") from exc
    intervals = np.asarray(parsed_intervals, dtype=np.float64)
    if (
        not isinstance(parsed_intervals, list)
        or intervals.ndim != 1
        or intervals.size != raw_intervals.size
        or not np.isfinite(intervals).all()
        or np.any(intervals < 0.0)
        or not np.allclose(intervals, raw_intervals, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("request ITL summary disagrees with raw token timestamps")
    return float(np.quantile(raw_intervals, 0.99)) if raw_intervals.size else None


def _request_metric(row: Mapping[str, Any]) -> _RequestMetric:
    request_id = row.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request evidence lacks a request_id")
    arrival = row.get("arrival_ns")
    terminal = row.get("completed_ns")
    first_token = row.get("first_token_ns")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (arrival, terminal)
    ):
        raise ValueError(
            "request terminal accounting requires arrival/completion times"
        )
    assert isinstance(arrival, int) and isinstance(terminal, int)
    if terminal < arrival:
        raise ValueError("request completion precedes arrival")
    observed_ttft: float | None = None
    if first_token is not None:
        if (
            not isinstance(first_token, int)
            or isinstance(first_token, bool)
            or not arrival <= first_token <= terminal
        ):
            raise ValueError("request first-token time is outside its lifetime")
        observed_ttft = (first_token - arrival) / 1_000_000.0
        recorded_ttft = row.get("ttft_ms")
        if recorded_ttft is None or not math.isclose(
            float(recorded_ttft), observed_ttft, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("request TTFT summary disagrees with raw timestamps")
    elif row.get("ttft_ms") is not None:
        raise ValueError("request TTFT summary lacks a raw first-token timestamp")
    output_tokens = row.get("output_tokens")
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 0
    ):
        raise ValueError("request output_tokens must be a non-negative integer")
    finished = row.get("finished")
    outcome_status = row.get("outcome_status")
    if not isinstance(finished, bool):
        raise TypeError("request finished flag must be boolean")
    if outcome_status not in {"completed", "rejected", "timed_out", "cancelled"}:
        raise ValueError("claim reduction requires an exact terminal request outcome")
    if finished != (outcome_status == "completed"):
        raise ValueError("request finished flag disagrees with terminal outcome")
    error = outcome_status != "completed"
    return _RequestMetric(
        request_id=request_id,
        output_tokens=output_tokens,
        completed=finished,
        error=error,
        ttft_ms=observed_ttft,
        within_request_p99_itl_ms=_parse_itl(row.get("inter_token_ms"), row),
        latency_ms=(terminal - arrival) / 1_000_000.0,
    )


def _validate_run_row(
    row: Mapping[str, Any],
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    cell: ExperimentCell,
    rank: int,
) -> None:
    tensor_parallel_size, data_parallel_size, world_size, topology_sha256 = (
        _expected_topology(cell)
    )
    expected = {
        "manifest_sha256": registry.sha256,
        "config_sha256": cell.cell_id,
        "industrial_cell_id": cell.cell_id,
        "runtime_sha256": plan.runtime_sha256,
        "split_sha256": plan.split_sha256,
        "method": cell.identity.method,
        "model_pair": cell.identity.model,
        "repetition_block": cell.identity.block,
        "patched_sglang_tree": row.get("patched_sglang_tree"),
        "topology_sha256": topology_sha256,
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "world_size": world_size,
        "rank": rank,
        "status": "complete",
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError("run evidence differs from its registry/runtime identity")
    for field in (
        "rank_config_sha256",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "run_nonce_sha256",
        "topology_sha256",
    ):
        if not _is_sha256(row.get(field)):
            raise ValueError(f"run evidence lacks immutable {field}")
    patched_tree = row.get("patched_sglang_tree")
    if (
        not isinstance(patched_tree, str)
        or len(patched_tree) != 40
        or any(character not in "0123456789abcdef" for character in patched_tree)
    ):
        raise ValueError("run evidence lacks a lower-case patched SGLang tree")
    for field in (
        "expected_request_rows",
        "expected_round_rows",
        "expected_update_rows",
        "expected_performance_rows",
    ):
        value = row.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"run evidence lacks exact {field}")
    adapted = cell.identity.method in {"tts", "l0"}
    if (row["expected_round_rows"] > 0) is not adapted or (
        row["expected_update_rows"] > 0
    ) is not adapted:
        raise ValueError(
            "run detail-table coverage disagrees with the method allocation contract"
        )


def _load_hardware_receipt(
    reference: BoundArtifact,
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    cell: ExperimentCell,
    terminal_receipts: tuple[BoundArtifact, ...],
    performance_rows_by_rank: tuple[tuple[dict[str, Any], ...], ...],
    envelope: HardwareEnvelope,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    value = _bound_json(reference.path, reference.sha256, label="hardware receipt")
    required = {
        "schema_version",
        "kind",
        "registry_sha256",
        "runtime_sha256",
        "split_sha256",
        "cell_id",
        "block",
        "topology_sha256",
        "hardware_envelope_sha256",
        "terminal_receipt_sha256s",
        "rank_contexts",
    }
    if set(value) != required:
        raise ValueError("hardware receipt has an ambiguous schema")
    _, _, _, topology_sha256 = _expected_topology(cell)
    if any(
        value.get(name) != expected
        for name, expected in (
            ("schema_version", 1),
            ("kind", "industrial_hardware_receipt"),
            ("registry_sha256", registry.sha256),
            ("runtime_sha256", plan.runtime_sha256),
            ("split_sha256", plan.split_sha256),
            ("cell_id", cell.cell_id),
            ("block", cell.identity.block),
            ("topology_sha256", topology_sha256),
            ("hardware_envelope_sha256", content_sha256(envelope)),
            (
                "terminal_receipt_sha256s",
                [receipt.sha256 for receipt in terminal_receipts],
            ),
        )
    ):
        raise ValueError("hardware receipt differs from terminal/run identity")
    contexts = value.get("rank_contexts")
    if not isinstance(contexts, list) or len(contexts) != len(terminal_receipts):
        raise ValueError("hardware receipt lacks exact rank coverage")
    validities: list[tuple[str, str, tuple[str, ...]]] = []
    for rank, (context, rows) in enumerate(
        zip(contexts, performance_rows_by_rank, strict=True)
    ):
        if not isinstance(context, dict) or set(context) != {
            "rank",
            "gpu_uuid",
            "power_state",
            "background_processes",
        }:
            raise ValueError("hardware rank context has an ambiguous schema")
        processes = context.get("background_processes")
        if (
            context.get("rank") != rank
            or context.get("gpu_uuid") != cell.resources.gpu_uuids[rank]
            or not isinstance(context.get("power_state"), str)
            or not context["power_state"]
            or not isinstance(processes, list)
            or any(not isinstance(item, str) or not item for item in processes)
            or len(set(processes)) != len(processes)
        ):
            raise ValueError("hardware rank context is incomplete")
        for row_index, row in enumerate(rows):
            throttling = _parse_string_list(
                row.get("throttling_reasons"),
                label="performance throttling_reasons",
            )
            observation_id = (
                f"block-{cell.identity.block}:{cell.identity.method}:"
                f"rank-{rank}:sample-{row_index}"
            )
            validity = validate_hardware_block(
                envelope,
                HardwareBlockObservation(
                    block_id=observation_id,
                    gpu_clock_mhz=row.get("gpu_clock_mhz"),
                    memory_clock_mhz=row.get("memory_clock_mhz"),
                    temperature_c=row.get("temperature_c"),
                    power_watts=row.get("power_watts"),
                    power_state=str(context["power_state"]),
                    throttling_reasons=throttling,
                    background_processes=tuple(processes),
                ),
            )
            validities.append((validity.block_id, validity.status, validity.reasons))
    return tuple(validities)


def _load_cell(
    reference: IndustrialCellEvidence,
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    cells_by_id: Mapping[str, ExperimentCell],
    envelope: HardwareEnvelope,
) -> _LoadedCell:
    try:
        cell = cells_by_id[reference.cell_id]
    except KeyError as exc:
        raise ValueError("cell evidence is absent from the registry") from exc
    if cell.identity.experiment != plan.experiment or not cell.runnable:
        raise ValueError(
            "reducer accepts only runnable cells from the planned experiment"
        )
    evidence_root = Path(cell.resources.evidence_root).resolve()
    if (
        any(
            receipt.path.resolve().parent != evidence_root
            for receipt in reference.terminal_receipts
        )
        or reference.hardware_receipt.path.resolve().parent != evidence_root
    ):
        raise ValueError("cell receipts must live in the registry evidence root")
    _, _, world_size, _ = _expected_topology(cell)
    if len(reference.terminal_receipts) != world_size:
        raise ValueError("cell evidence lacks complete rank coverage")

    run_rows: list[dict[str, Any]] = []
    request_rows_by_rank: list[tuple[dict[str, Any], ...]] = []
    performance_rows_by_rank: list[tuple[dict[str, Any], ...]] = []
    run_ids: set[str] = set()
    for expected_rank, receipt_reference in enumerate(reference.terminal_receipts):
        receipt, evidence = _read_terminal_receipt(receipt_reference)
        if receipt.get("rank") != expected_rank:
            raise ValueError("terminal rank receipts must be complete and ordered")
        run = _read_table(evidence["run"])
        if len(run) != 1:
            raise ValueError("rank evidence must contain exactly one run row")
        run_row = run[0]
        _validate_run_row(
            run_row,
            registry=registry,
            plan=plan,
            cell=cell,
            rank=expected_rank,
        )
        if run_row.get("run_id") != receipt.get("run_id"):
            raise ValueError("terminal receipt and run row disagree")
        run_ids.add(str(run_row["run_id"]))
        request_rows = _read_table(evidence["request"])
        performance_rows = _read_table(evidence["performance"])
        adapted = cell.identity.method in {"tts", "l0"}
        if adapted:
            for table_name, expected_rows in (
                ("round", run_row["expected_round_rows"]),
                ("update", run_row["expected_update_rows"]),
            ):
                if table_name not in evidence:
                    raise ValueError("adapted run lacks terminal detail-table evidence")
                detail_rows = _read_table(evidence[table_name])
                if len(detail_rows) != expected_rows or any(
                    row.get("run_id") != run_row["run_id"] for row in detail_rows
                ):
                    raise ValueError(
                        "adapted detail-table evidence disagrees with run coverage"
                    )
        elif (
            run_row["expected_round_rows"] != 0
            or run_row["expected_update_rows"] != 0
            or "round" in evidence
            or "update" in evidence
        ):
            raise ValueError(
                "Target-only/Static must allocate no round or update trace tables"
            )
        if (
            len(request_rows) != run_row["expected_request_rows"]
            or len(performance_rows) != run_row["expected_performance_rows"]
        ):
            raise ValueError("terminal tables disagree with declared row coverage")
        if any(
            row.get("run_id") != run_row["run_id"]
            or row.get("method") != cell.identity.method
            or row.get("repetition_block") != cell.identity.block
            for row in (*request_rows, *performance_rows)
        ):
            raise ValueError("terminal rows cross a run/method/block boundary")
        for performance in performance_rows:
            for counter in _SAFETY_COUNTERS:
                value = performance.get(counter)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"performance evidence lacks {counter}")
        run_rows.append(run_row)
        request_rows_by_rank.append(request_rows)
        performance_rows_by_rank.append(performance_rows)
    if len(run_ids) != 1:
        raise ValueError("all ranks of one topology must share one run_id")
    rank_zero_requests = request_rows_by_rank[0]
    rank_zero_identity = tuple(
        (row.get("request_id"), row.get("output_sha256")) for row in rank_zero_requests
    )
    if any(
        tuple((row.get("request_id"), row.get("output_sha256")) for row in rows)
        != rank_zero_identity
        for rows in request_rows_by_rank[1:]
    ):
        raise ValueError("rank request/trajectory evidence is inconsistent")
    outcome_statuses = tuple(row.get("outcome_status") for row in rank_zero_requests)
    if any(
        status not in {"completed", "rejected", "timed_out", "cancelled"}
        for status in outcome_statuses
    ):
        raise ValueError("claim reduction requires terminal request outcomes")
    aggregate_rows = [
        row
        for row in performance_rows_by_rank[0]
        if row.get("offered_requests") is not None
    ]
    if len(aggregate_rows) != 1:
        raise ValueError("claim reduction requires one exact load-accounting row")
    aggregate = aggregate_rows[0]
    expected_accounting = {
        "offered_requests": len(rank_zero_requests),
        "admitted_requests": sum(
            row.get("admitted_ns") is not None for row in rank_zero_requests
        ),
        "completed_requests": outcome_statuses.count("completed"),
        "admission_rejections": outcome_statuses.count("rejected"),
        "timeouts": outcome_statuses.count("timed_out"),
        "cancellations": outcome_statuses.count("cancelled"),
        "unfinished_requests": 0,
    }
    if any(
        not isinstance(aggregate.get(name), int)
        or isinstance(aggregate.get(name), bool)
        or aggregate.get(name) != expected
        for name, expected in expected_accounting.items()
    ):
        raise ValueError("performance accounting differs from terminal outcomes")
    bound_fields = (
        "industrial_cell_id",
        "runtime_sha256",
        "split_sha256",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "patched_sglang_tree",
        "run_nonce_sha256",
        "topology_sha256",
        "world_size",
    )
    if any(
        tuple(row.get(field) for field in bound_fields)
        != tuple(run_rows[0].get(field) for field in bound_fields)
        for row in run_rows[1:]
    ):
        raise ValueError("rank run identities are inconsistent")
    hardware_validity = _load_hardware_receipt(
        reference.hardware_receipt,
        registry=registry,
        plan=plan,
        cell=cell,
        terminal_receipts=reference.terminal_receipts,
        performance_rows_by_rank=tuple(performance_rows_by_rank),
        envelope=envelope,
    )
    return _LoadedCell(
        cell=cell,
        run_rows=tuple(run_rows),
        request_rows=rank_zero_requests,
        performance_rows_by_rank=tuple(performance_rows_by_rank),
        terminal_receipt_sha256s=tuple(
            receipt.sha256 for receipt in reference.terminal_receipts
        ),
        hardware_receipt_sha256=reference.hardware_receipt.sha256,
        hardware_validity=hardware_validity,
    )


def _qualification_rows(
    reference: BoundArtifact,
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    block: int,
    loaded: Mapping[str, _LoadedCell],
) -> tuple[tuple[str, str, bool], ...]:
    value = _bound_json(reference.path, reference.sha256, label="qualification lock")
    required = {
        "schema_version",
        "kind",
        "registry_sha256",
        "runtime_sha256",
        "split_sha256",
        "block",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "rows",
    }
    if set(value) != required:
        raise ValueError("qualification lock has an ambiguous schema")
    run = loaded[_METHODS[0]].run_rows[0]
    if any(
        value.get(name) != expected
        for name, expected in (
            ("schema_version", 1),
            ("kind", "industrial_request_qualification_lock"),
            ("registry_sha256", registry.sha256),
            ("runtime_sha256", plan.runtime_sha256),
            ("split_sha256", plan.split_sha256),
            ("block", block),
            ("corpus_sha256", run["corpus_sha256"]),
            ("arrival_trace_sha256", run["arrival_trace_sha256"]),
            ("request_ids_sha256", run["request_ids_sha256"]),
            ("sampling_profile_sha256", run["sampling_profile_sha256"]),
            ("model_lock_sha256", run["model_lock_sha256"]),
        )
    ):
        raise ValueError("qualification lock differs from terminal run identity")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("qualification lock requires request rows")
    parsed: list[tuple[str, str, bool]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "request_id",
            "prompt_bucket",
            "eligible",
        }:
            raise ValueError("qualification row has an ambiguous schema")
        request_id = row.get("request_id")
        bucket = row.get("prompt_bucket")
        eligible = row.get("eligible")
        if (
            not isinstance(request_id, str)
            or not request_id
            or bucket not in {"short", "medium", "long"}
            or not isinstance(eligible, bool)
        ):
            raise ValueError("qualification row is incomplete")
        parsed.append((request_id, str(bucket), eligible))
    request_ids = tuple(row[0] for row in parsed)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("qualification request IDs must be unique")
    if content_sha256(list(request_ids)) != run["request_ids_sha256"]:
        raise ValueError("qualification order does not match request_ids_sha256")
    for method, cell in loaded.items():
        cell_run = cell.run_rows[0]
        for field in (
            "corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "sampling_profile_sha256",
            "model_lock_sha256",
            "patched_sglang_tree",
        ):
            if cell_run[field] != run[field]:
                raise ValueError(f"paired {method} run differs in immutable {field}")
        actual_ids = {str(row.get("request_id")) for row in cell.request_rows}
        if str(cell.cell.identity.arrival).startswith("closed_loop"):
            concurrency = cell.cell.identity.concurrency
            if concurrency is None and cell.cell.identity.arrival == "closed_loop_c1":
                concurrency = 1
            if (
                not isinstance(concurrency, int)
                or isinstance(concurrency, bool)
                or concurrency < 1
                or not actual_ids
                or actual_ids - set(request_ids)
            ):
                raise ValueError("closed-loop request coverage is invalid")
            request_by_id = {
                str(row.get("request_id")): row for row in cell.request_rows
            }
            for lane in range(concurrency):
                seen_gap = False
                previous_completed_ns: int | None = None
                lane_request_ids = request_ids[lane::concurrency]
                if not lane_request_ids or lane_request_ids[0] not in actual_ids:
                    raise ValueError("closed-loop request coverage omits a client lane")
                for request_id in lane_request_ids:
                    present = request_id in actual_ids
                    if seen_gap and present:
                        raise ValueError(
                            "closed-loop request coverage is not a client prefix"
                        )
                    if present:
                        request = request_by_id[request_id]
                        arrival_ns = request.get("arrival_ns")
                        completed_ns = request.get("completed_ns")
                        if (
                            not isinstance(arrival_ns, int)
                            or isinstance(arrival_ns, bool)
                            or not isinstance(completed_ns, int)
                            or isinstance(completed_ns, bool)
                            or (
                                previous_completed_ns is None
                                and arrival_ns != cell_run["started_ns"]
                            )
                            or (
                                previous_completed_ns is not None
                                and arrival_ns != previous_completed_ns
                            )
                        ):
                            raise ValueError(
                                "closed-loop request offers are not zero-think"
                            )
                        previous_completed_ns = completed_ns
                    seen_gap = seen_gap or not present
        elif tuple(row.get("request_id") for row in cell.request_rows) != request_ids:
            raise ValueError("paired request coverage/order is incomplete")
    return tuple(parsed)


def _reduce_block(
    block_reference: IndustrialBlockEvidence,
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    cells_by_id: Mapping[str, ExperimentCell],
    envelope: HardwareEnvelope,
) -> _BlockReduction:
    block = block_reference.block
    loaded_sequence = tuple(
        _load_cell(
            reference,
            registry=registry,
            plan=plan,
            cells_by_id=cells_by_id,
            envelope=envelope,
        )
        for reference in block_reference.cells
    )
    if any(cell.cell.identity.block != block for cell in loaded_sequence):
        raise ValueError("block evidence contains a cell from another block")
    loaded = {cell.cell.identity.method: cell for cell in loaded_sequence}
    if len(loaded) != len(loaded_sequence) or set(loaded) != set(_METHODS):
        raise ValueError("paired blocks require exactly Target-only/Static/TTS/L0")
    loaded = {method: loaded[method] for method in _METHODS}
    pairing = {_pairing_identity(cell.cell, block) for cell in loaded.values()}
    if len(pairing) != 1:
        raise ValueError("paired methods do not share one registry workload slice")
    qualification = _qualification_rows(
        block_reference.qualification_lock,
        registry=registry,
        plan=plan,
        block=block,
        loaded=loaded,
    )
    qualification_by_id = {
        request_id: (bucket, eligible) for request_id, bucket, eligible in qualification
    }
    reference_rows = {
        str(row["request_id"]): row for row in loaded["target_only"].request_rows
    }
    metrics: dict[str, tuple[_RequestMetric, ...]] = {}
    goodput: dict[str, float] = {}
    slo_goodput: dict[str, float] = {}
    slo_rows: dict[str, tuple[SloRequest, ...]] = {}
    for method in _METHODS:
        cell = loaded[method]
        method_rows = {str(row["request_id"]): row for row in cell.request_rows}
        for request_id, row in method_rows.items():
            method_ids = _parse_output_token_ids(row)
            reference = reference_rows.get(request_id)
            if reference is None:
                continue
            reference_ids = _parse_output_token_ids(reference)
            both_complete = (
                reference.get("outcome_status") == "completed"
                and row.get("outcome_status") == "completed"
            )
            common = min(len(reference_ids), len(method_ids))
            if (
                both_complete
                and reference_ids != method_ids
                or reference_ids[:common] != method_ids[:common]
            ):
                raise ValueError(
                    "paired methods do not preserve target token trajectories"
                )
        rows = tuple(_request_metric(row) for row in cell.request_rows)
        score_started = min(int(row["arrival_ns"]) for row in cell.request_rows)
        score_ended = max(int(row["completed_ns"]) for row in cell.request_rows)
        elapsed_s = (score_ended - score_started) / 1_000_000_000.0
        if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
            raise ValueError("request score interval must be finite and positive")
        method_slo_rows = tuple(
            SloRequest(
                request_id=metric.request_id,
                prompt_bucket=qualification_by_id[metric.request_id][0],
                eligible=qualification_by_id[metric.request_id][1],
                completed=metric.completed,
                error=metric.error,
                ttft_ms=metric.ttft_ms,
                within_request_p99_itl_ms=metric.within_request_p99_itl_ms,
            )
            for metric in rows
        )
        slo_accounting = account_slo(method_slo_rows)
        ttft_limits = dict(slo_accounting.ttft_limits_ms)
        qualified_ids = {
            row.request_id
            for row in method_slo_rows
            if row.eligible
            and row.completed
            and not row.error
            and row.ttft_ms is not None
            and row.within_request_p99_itl_ms is not None
            and row.ttft_ms <= ttft_limits[row.prompt_bucket]
            and row.within_request_p99_itl_ms
            <= slo_accounting.within_request_p99_itl_limit_ms
        }
        metrics[method] = rows
        goodput[method] = (
            sum(metric.output_tokens for metric in rows if metric.completed) / elapsed_s
        )
        slo_goodput[method] = (
            sum(
                metric.output_tokens
                for metric in rows
                if metric.request_id in qualified_ids
            )
            / elapsed_s
        )
        slo_rows[method] = method_slo_rows
    return _BlockReduction(
        block=block,
        qualification_sha256=block_reference.qualification_lock.sha256,
        cells=MappingProxyType(loaded),
        request_metrics=MappingProxyType(metrics),
        goodput_tps=MappingProxyType(goodput),
        slo_goodput_tps=MappingProxyType(slo_goodput),
        slo_requests=MappingProxyType(slo_rows),
    )


def _pilot_bindings(
    blocks: Sequence[IndustrialBlockEvidence],
) -> tuple[str, str]:
    pilots = tuple(
        sorted(
            (row for row in blocks if row.block in PILOT_BLOCKS),
            key=lambda row: row.block,
        )
    )
    if tuple(row.block for row in pilots) != PILOT_BLOCKS:
        raise ValueError("industrial analysis requires exactly four excluded pilots")
    cell_rows = [
        {
            "block": block.block,
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "terminal_receipt_sha256s": [
                        receipt.sha256 for receipt in cell.terminal_receipts
                    ],
                    "hardware_receipt_sha256": cell.hardware_receipt.sha256,
                }
                for cell in sorted(block.cells, key=lambda item: item.cell_id)
            ],
            "qualification_lock_sha256": block.qualification_lock.sha256,
        }
        for block in pilots
    ]
    pilot_evidence_sha256 = content_sha256(
        {"schema_version": 1, "kind": "industrial_pilot_evidence", "blocks": cell_rows}
    )
    completed_pilot_cells_sha256 = content_sha256(
        sorted(cell.cell_id for block in pilots for cell in block.cells)
    )
    return pilot_evidence_sha256, completed_pilot_cells_sha256


def industrial_pilot_evidence_sha256(
    blocks: Sequence[IndustrialBlockEvidence],
) -> str:
    """Precompute the exact pilot receipt binding used by a block plan."""

    return _pilot_bindings(blocks)[0]


def industrial_completed_pilot_cells_sha256(
    blocks: Sequence[IndustrialBlockEvidence],
) -> str:
    """Precompute the exact completed pilot-cell identity used by a block plan."""

    return _pilot_bindings(blocks)[1]


def _unresolved_artifact(
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    pilot_evidence_sha256: str,
    completed_pilot_cells_sha256: str,
    blocks: Sequence[_BlockReduction],
    hardware_envelope: HardwareEnvelope,
    patched_sglang_tree: str,
    model_lock_sha256: str,
    gpu_attestation_sha256: str | None,
    doctor_report_sha256: str | None,
    power_plan: PowerSizingPlan | None,
    reasons: tuple[str, ...],
) -> IndustrialReduction:
    terminal = tuple(
        digest
        for block in blocks
        for cell in block.cells.values()
        for digest in cell.terminal_receipt_sha256s
    )
    hardware = tuple(
        cell.hardware_receipt_sha256
        for block in blocks
        for cell in block.cells.values()
    )
    validity = tuple(
        row
        for block in blocks
        for cell in block.cells.values()
        for row in cell.hardware_validity
    )
    run_bindings = _run_bindings(blocks)
    artifact = IndustrialReducerArtifact(
        status="UNRESOLVED",
        gpu_evidence="INVALIDATED" if validity else "UNMEASURED",
        reasons=reasons,
        registry_sha256=registry.sha256,
        experiment=plan.experiment,
        runtime_sha256=plan.runtime_sha256,
        split_sha256=plan.split_sha256,
        confirmation_plan_sha256=plan.sha256,
        patched_sglang_tree=patched_sglang_tree,
        model_lock_sha256=model_lock_sha256,
        hardware_envelope_sha256=content_sha256(hardware_envelope),
        gpu_attestation_sha256=gpu_attestation_sha256,
        doctor_report_sha256=doctor_report_sha256,
        pilot_evidence_sha256=pilot_evidence_sha256,
        completed_pilot_cells_sha256=completed_pilot_cells_sha256,
        terminal_receipt_sha256s=tuple(sorted(terminal)),
        qualification_lock_sha256s=tuple(
            sorted(block.qualification_sha256 for block in blocks)
        ),
        hardware_receipt_sha256s=tuple(sorted(hardware)),
        run_bindings=run_bindings,
        power_plan=power_plan,
        hardware_validity=validity,
        methods=(),
        primary_contrasts=(),
        holm_family=(),
        bootstrap_hooks=(
            ("hierarchical_block_request", ("block", "request")),
            ("whole_time_block", ("time_block",)),
        ),
    )
    return IndustrialReduction(artifact=artifact, _request_metrics=MappingProxyType({}))


def _run_bindings(
    blocks: Sequence[_BlockReduction],
) -> tuple[IndustrialRunBinding, ...]:
    bindings: list[IndustrialRunBinding] = []
    for block in sorted(blocks, key=lambda row: row.block):
        for method in _METHODS:
            cell = block.cells[method]
            run = cell.run_rows[0]
            bindings.append(
                IndustrialRunBinding(
                    block=block.block,
                    method=method,
                    cell_id=cell.cell.cell_id,
                    config_sha256=str(run["config_sha256"]),
                    rank_config_sha256s=tuple(
                        str(rank_run["rank_config_sha256"])
                        for rank_run in cell.run_rows
                    ),
                    run_id=str(run["run_id"]),
                    rank_count=len(cell.run_rows),
                    model_pair=str(run["model_pair"]),
                    corpus_sha256=str(run["corpus_sha256"]),
                    arrival_trace_sha256=str(run["arrival_trace_sha256"]),
                    request_ids_sha256=str(run["request_ids_sha256"]),
                    sampling_profile_sha256=str(run["sampling_profile_sha256"]),
                    model_lock_sha256=str(run["model_lock_sha256"]),
                    patched_sglang_tree=str(run["patched_sglang_tree"]),
                    run_nonce_sha256=str(run["run_nonce_sha256"]),
                    topology_sha256=str(run["topology_sha256"]),
                    gpu_uuids=cell.cell.resources.gpu_uuids,
                    terminal_receipt_sha256s=cell.terminal_receipt_sha256s,
                    hardware_receipt_sha256=cell.hardware_receipt_sha256,
                )
            )
    return tuple(bindings)


def _validate_industrial_doctor(
    reference: BoundArtifact,
    *,
    registry: ExperimentRegistry,
) -> None:
    """Validate one exact, content-bound schema-v1 GPU readiness report."""

    report = _bound_json(reference.path, reference.sha256, label="doctor report")
    readiness = report.get("readiness")
    compatibility = report.get("compatibility")
    if (
        report.get("schema_version") != 1
        or report.get("status") != "PASS"
        or not isinstance(readiness, dict)
        or readiness.get("status") != "PASS"
        or not isinstance(compatibility, dict)
        or compatibility.get("status") != "PASS"
    ):
        raise ValueError("industrial GPU attestation requires a schema-v1 PASS doctor")

    checks = report.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != _INDUSTRIAL_DOCTOR_CHECKS
        or any(
            not isinstance(check, dict) or check.get("status") != "PASS"
            for check in checks.values()
        )
        or readiness.get("pass_count") != len(checks)
        or readiness.get("fail_count") != 0
        or readiness.get("unknown_count") != 0
    ):
        raise ValueError("industrial doctor must contain only complete PASS checks")

    runtime_manifest = report.get("runtime_manifest")
    manifest_sha256 = (
        runtime_manifest.get("sha256") if isinstance(runtime_manifest, dict) else None
    )
    if (
        not isinstance(runtime_manifest, dict)
        or runtime_manifest.get("valid") is not True
        or not _is_sha256(manifest_sha256)
        or runtime_manifest.get("sidecar_sha256") != manifest_sha256
        or runtime_manifest.get("error") is not None
        or compatibility.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("industrial doctor runtime-manifest digests are inconsistent")

    source_tree = report.get("source_tree")
    roots = report.get("roots")
    source_head = source_tree.get("head") if isinstance(source_tree, dict) else None
    if (
        not isinstance(roots, dict)
        or roots.get("distinct") is not True
        or not isinstance(roots.get("project"), str)
        or not roots["project"]
        or not isinstance(roots.get("patched_sglang"), str)
        or not roots["patched_sglang"]
        or roots["project"] == roots["patched_sglang"]
        or not isinstance(source_tree, dict)
        or source_tree.get("path") != roots.get("patched_sglang")
        or source_tree.get("is_git_checkout") is not True
        or source_tree.get("root_matches_toplevel") is not True
        or not isinstance(source_head, str)
        or len(source_head) != 40
        or any(character not in "0123456789abcdef" for character in source_head)
        or source_tree.get("tree") != PINNED_SGLANG_TREE
        or source_tree.get("dirty") is not False
        or source_tree.get("pinned_ancestor") is not True
        or source_tree.get("patch_commits") != PINNED_SGLANG_PATCH_COUNT
        or compatibility.get("sglang_commit") != PINNED_SGLANG_COMMIT
        or compatibility.get("sglang_tree") != PINNED_SGLANG_TREE
        or compatibility.get("patch_count") != PINNED_SGLANG_PATCH_COUNT
        or compatibility.get("single_node_only") is not True
        or compatibility.get("multi_node_supported") is not False
    ):
        raise ValueError("industrial doctor patched-tree identity is not exact")

    gpu = report.get("gpu")
    inventory = gpu.get("parsed_inventory") if isinstance(gpu, dict) else None
    devices = inventory.get("devices") if isinstance(inventory, dict) else None
    expected_device_fields = {
        "uuid",
        "name",
        "memory_total_mib",
        "driver_version",
        "compute_capability",
        "pci_bus_id",
    }
    if (
        not isinstance(gpu, dict)
        or gpu.get("two_gpu_visible") is not True
        or not isinstance(inventory, dict)
        or set(inventory) != {"devices", "parse_error"}
        or inventory.get("parse_error") is not None
        or not isinstance(devices, list)
        or len(devices) != 2
        or any(
            not isinstance(device, dict) or set(device) != expected_device_fields
            for device in devices
        )
        or [device["uuid"] for device in devices] != list(registry.gpu_uuids)
        or len({device["pci_bus_id"] for device in devices}) != 2
    ):
        raise ValueError("industrial doctor does not bind the two registry GPU UUIDs")

    topology = gpu.get("parsed_topology")
    gpu_topology_check = checks.get("gpu_topology")
    gpu_identity_check = checks.get("gpu_identity")
    commands = report.get("commands")
    if (
        not isinstance(topology, dict)
        or set(topology) != {"gpu_rows", "pair_link", "reciprocal_link", "parse_error"}
        or topology.get("parse_error") is not None
        or topology.get("gpu_rows") != ["GPU0", "GPU1"]
        or not isinstance(topology.get("pair_link"), str)
        or not topology["pair_link"]
        or topology.get("reciprocal_link") != topology["pair_link"]
        or not isinstance(gpu_topology_check, dict)
        or gpu_topology_check.get("observed") != topology
        or not isinstance(gpu_identity_check, dict)
        or gpu_identity_check.get("observed") != devices
        or not isinstance(gpu.get("inventory"), str)
        or not gpu["inventory"].strip()
        or not isinstance(commands, dict)
        or commands.get("nvidia_smi") != gpu["inventory"]
    ):
        raise ValueError("industrial doctor two-GPU topology is not exact")


def _attestation_chain(
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
    hardware_envelope: HardwareEnvelope,
    pilot_evidence_sha256: str,
    completed_pilot_cells_sha256: str,
    patched_sglang_tree: str,
    model_lock_sha256: str,
    blocks: Sequence[_BlockReduction],
    run_bindings: tuple[IndustrialRunBinding, ...],
) -> dict[str, Any]:
    terminal_receipts = tuple(
        digest
        for block in blocks
        for cell in block.cells.values()
        for digest in cell.terminal_receipt_sha256s
    )
    hardware_receipts = tuple(
        cell.hardware_receipt_sha256
        for block in blocks
        for cell in block.cells.values()
    )
    return {
        "registry_sha256": registry.sha256,
        "experiment": plan.experiment,
        "runtime_sha256": plan.runtime_sha256,
        "split_sha256": plan.split_sha256,
        "confirmation_plan_sha256": plan.sha256,
        "patched_sglang_tree": patched_sglang_tree,
        "model_lock_sha256": model_lock_sha256,
        "hardware_envelope_sha256": content_sha256(hardware_envelope),
        "pilot_evidence_sha256": pilot_evidence_sha256,
        "completed_pilot_cells_sha256": completed_pilot_cells_sha256,
        "gpu_uuids": list(registry.gpu_uuids),
        "terminal_receipt_sha256s": sorted(terminal_receipts),
        "qualification_lock_sha256s": sorted(
            block.qualification_sha256 for block in blocks
        ),
        "hardware_receipt_sha256s": sorted(hardware_receipts),
        "run_bindings": [asdict(binding) for binding in run_bindings],
    }


def _validate_industrial_gpu_attestation(
    reference: BoundArtifact,
    *,
    doctor_report: BoundArtifact,
    expected_chain: Mapping[str, Any],
) -> None:
    """Require the attester to bind every reducer-derived run/evidence identity."""

    attestation = _bound_json(
        reference.path,
        reference.sha256,
        label="industrial GPU attestation",
    )
    expected = {
        "schema_version": 1,
        "kind": "industrial_gpu_attestation",
        "status": "PASS",
        "doctor_report_sha256": doctor_report.sha256,
        **expected_chain,
    }
    if set(attestation) != set(expected) or content_sha256(
        attestation
    ) != content_sha256(expected):
        raise ValueError(
            "industrial GPU attestation does not bind the exact doctor/run evidence chain"
        )


def reduce_industrial_schema_v3(
    *,
    registry: ExperimentRegistry,
    confirmation_plan: ConfirmationBlockPlan,
    blocks: Sequence[IndustrialBlockEvidence],
    hardware_envelope: HardwareEnvelope,
    gpu_attestation: BoundArtifact | None = None,
    doctor_report: BoundArtifact | None = None,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 0,
) -> IndustrialReduction:
    """Reduce a powered paired confirmation directly from terminal evidence.

    Missing/partial inputs raise before inference.  A valid but out-of-envelope
    hardware block returns ``UNRESOLVED`` with no contrasts.  This distinction
    prevents a failed environment gate from becoming a performance claim.
    """

    if confirmation_plan.registry_sha256 != registry.sha256:
        raise ValueError("confirmation plan belongs to another registry")
    if (gpu_attestation is None) != (doctor_report is None):
        raise ValueError(
            "industrial GPU attestation and doctor report must be supplied together"
        )
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise TypeError("bootstrap seed must be an integer")
    if (
        not isinstance(bootstrap_repetitions, int)
        or isinstance(bootstrap_repetitions, bool)
        or bootstrap_repetitions < 100
    ):
        raise ValueError("bootstrap repetitions must be at least 100")
    block_references = tuple(blocks)
    if not block_references:
        raise ValueError("industrial reducer requires terminal block evidence")
    block_ids = tuple(reference.block for reference in block_references)
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("industrial block evidence must be unique")
    pilot_evidence_sha256, completed_pilot_cells_sha256 = _pilot_bindings(
        block_references
    )
    if (
        confirmation_plan.pilot_evidence_sha256 != pilot_evidence_sha256
        or confirmation_plan.completed_pilot_cells_sha256
        != completed_pilot_cells_sha256
    ):
        raise ValueError("confirmation plan is not bound to these exact pilot receipts")

    if confirmation_plan.status != "POWERED":
        raise ValueError("industrial final analysis requires a POWERED block plan")
    assert confirmation_plan.selected_final_blocks is not None
    expected_blocks = (
        PILOT_BLOCKS + FINAL_BLOCKS[: confirmation_plan.selected_final_blocks]
    )
    if tuple(sorted(block_ids)) != expected_blocks:
        raise ValueError("evidence must cover four pilots and the locked final prefix")

    cells_by_id = {cell.cell_id: cell for cell in registry.cells}
    reduced = tuple(
        _reduce_block(
            reference,
            registry=registry,
            plan=confirmation_plan,
            cells_by_id=cells_by_id,
            envelope=hardware_envelope,
        )
        for reference in sorted(block_references, key=lambda item: item.block)
    )
    nonces: dict[str, str] = {}
    run_ids: dict[str, str] = {}
    for block in reduced:
        for cell in block.cells.values():
            nonce = str(cell.run_rows[0]["run_nonce_sha256"])
            owner = nonces.setdefault(nonce, cell.cell.cell_id)
            if owner != cell.cell.cell_id:
                raise ValueError("run nonce is reused across registry cells")
            run_id = str(cell.run_rows[0]["run_id"])
            run_owner = run_ids.setdefault(run_id, cell.cell.cell_id)
            if run_owner != cell.cell.cell_id:
                raise ValueError("run identity is reused across registry cells")

    pilots = tuple(block for block in reduced if block.block in PILOT_BLOCKS)
    pilot_metric = (
        "slo_goodput_tps" if confirmation_plan.experiment == "E5" else "goodput_tps"
    )
    power_plan = preregister_power_sizing(
        tuple(
            PilotBlock(
                block_id=f"block-{block.block}",
                static_goodput=getattr(block, pilot_metric)["static"],
                tts_goodput=getattr(block, pilot_metric)["tts"],
                l0_goodput=getattr(block, pilot_metric)["l0"],
            )
            for block in pilots
        )
    )
    if (
        power_plan.underpowered
        or power_plan.selected_final_blocks != confirmation_plan.selected_final_blocks
    ):
        raise ValueError("powered block plan disagrees with excluded pilot evidence")

    patched_trees = {
        str(cell.run_rows[0]["patched_sglang_tree"])
        for block in reduced
        for cell in block.cells.values()
    }
    model_locks = {
        str(cell.run_rows[0]["model_lock_sha256"])
        for block in reduced
        for cell in block.cells.values()
    }
    if len(patched_trees) != 1 or len(model_locks) != 1:
        raise ValueError("confirmation crosses patch or model-lock identities")
    patched_sglang_tree = next(iter(patched_trees))
    model_lock_sha256 = next(iter(model_locks))
    if patched_sglang_tree != PINNED_SGLANG_TREE:
        raise ValueError("confirmation does not use the exact patched SGLang tree")
    run_bindings = _run_bindings(reduced)
    bound_attestation = gpu_attestation is not None and doctor_report is not None
    if bound_attestation:
        assert gpu_attestation is not None and doctor_report is not None
        _validate_industrial_doctor(doctor_report, registry=registry)
        _validate_industrial_gpu_attestation(
            gpu_attestation,
            doctor_report=doctor_report,
            expected_chain=_attestation_chain(
                registry=registry,
                plan=confirmation_plan,
                hardware_envelope=hardware_envelope,
                pilot_evidence_sha256=pilot_evidence_sha256,
                completed_pilot_cells_sha256=completed_pilot_cells_sha256,
                patched_sglang_tree=patched_sglang_tree,
                model_lock_sha256=model_lock_sha256,
                blocks=reduced,
                run_bindings=run_bindings,
            ),
        )

    invalid_hardware = tuple(
        row
        for block in reduced
        for cell in block.cells.values()
        for row in cell.hardware_validity
        if row[1] != "VALID"
    )
    safety_violations = tuple(
        f"block-{block.block}:{method}:{counter}"
        for block in reduced
        for method, cell in block.cells.items()
        for rows in cell.performance_rows_by_rank
        for row in rows
        for counter in _SAFETY_COUNTERS
        if row[counter] != 0
    )
    if invalid_hardware or safety_violations:
        reasons = tuple(
            sorted(
                {
                    *(f"hardware:{identity}" for identity, _, _ in invalid_hardware),
                    *(f"safety:{reason}" for reason in safety_violations),
                }
            )
        )
        return _unresolved_artifact(
            registry=registry,
            plan=confirmation_plan,
            pilot_evidence_sha256=pilot_evidence_sha256,
            completed_pilot_cells_sha256=completed_pilot_cells_sha256,
            blocks=reduced,
            hardware_envelope=hardware_envelope,
            patched_sglang_tree=patched_sglang_tree,
            model_lock_sha256=model_lock_sha256,
            gpu_attestation_sha256=(
                None if gpu_attestation is None else gpu_attestation.sha256
            ),
            doctor_report_sha256=(
                None if doctor_report is None else doctor_report.sha256
            ),
            power_plan=power_plan,
            reasons=reasons,
        )

    final = tuple(block for block in reduced if block.block not in PILOT_BLOCKS)
    final_ids = tuple(f"block-{block.block}" for block in final)
    if len(final) != confirmation_plan.selected_final_blocks:
        raise AssertionError("validated final prefix changed during reduction")
    metric_name = (
        "slo_goodput_tps" if confirmation_plan.experiment == "E5" else "goodput_tps"
    )
    primary: dict[str, PairedBcaContrast] = {}
    for contrast, denominator in (
        ("l0_vs_static", "static"),
        ("l0_vs_tts", "tts"),
    ):
        paired = {
            f"block-{block.block}": (
                getattr(block, metric_name)["l0"],
                getattr(block, metric_name)[denominator],
            )
            for block in final
        }
        primary[contrast] = paired_bca_contrast(
            contrast,
            paired,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + PRIMARY_CONTRASTS.index(contrast),
        )
    holm = holm_primary_contrasts(primary)

    methods: list[MethodReduction] = []
    request_metrics: dict[str, dict[str, tuple[_RequestMetric, ...]]] = defaultdict(
        dict
    )
    for method in _METHODS:
        combined_slo = tuple(
            SloRequest(
                request_id=f"block-{block.block}:{row.request_id}",
                prompt_bucket=row.prompt_bucket,
                eligible=row.eligible,
                completed=row.completed,
                error=row.error,
                ttft_ms=row.ttft_ms,
                within_request_p99_itl_ms=row.within_request_p99_itl_ms,
            )
            for block in final
            for row in block.slo_requests[method]
        )
        slo = account_slo(combined_slo)
        completed_latencies = tuple(
            metric.latency_ms
            for block in final
            for metric in block.request_metrics[method]
            if metric.completed
        )
        observed_p99 = (
            float(np.quantile(np.asarray(completed_latencies), 0.99))
            if completed_latencies
            else None
        )
        p99_guard = guard_p99_claim(
            f"{confirmation_plan.experiment}:{method}",
            completed_requests=len(completed_latencies),
            observed_p99_ms=observed_p99,
        )
        methods.append(
            MethodReduction(
                method=method,
                block_ids=final_ids,
                mean_output_goodput_tps=float(
                    np.mean([block.goodput_tps[method] for block in final])
                ),
                mean_slo_qualified_goodput_tps=float(
                    np.mean([block.slo_goodput_tps[method] for block in final])
                ),
                slo=slo,
                aggregate_latency_p99=p99_guard,
            )
        )
        for block in final:
            request_metrics[method][f"block-{block.block}"] = block.request_metrics[
                method
            ]

    terminal = tuple(
        digest
        for block in reduced
        for cell in block.cells.values()
        for digest in cell.terminal_receipt_sha256s
    )
    hardware = tuple(
        cell.hardware_receipt_sha256
        for block in reduced
        for cell in block.cells.values()
    )
    validity = tuple(
        row
        for block in reduced
        for cell in block.cells.values()
        for row in cell.hardware_validity
    )
    # Hash/field agreement proves integrity, not that a GPU produced the files.
    # This release has no trusted hardware-rooted attester, so even an exactly
    # bound caller artifact remains diagnostic-only.
    artifact = IndustrialReducerArtifact(
        status="UNRESOLVED",
        gpu_evidence="UNMEASURED",
        reasons=(
            ("gpu_attestation:untrusted_attester",)
            if bound_attestation
            else ("gpu_attestation:missing",)
        ),
        registry_sha256=registry.sha256,
        experiment=confirmation_plan.experiment,
        runtime_sha256=confirmation_plan.runtime_sha256,
        split_sha256=confirmation_plan.split_sha256,
        confirmation_plan_sha256=confirmation_plan.sha256,
        patched_sglang_tree=patched_sglang_tree,
        model_lock_sha256=model_lock_sha256,
        hardware_envelope_sha256=content_sha256(hardware_envelope),
        gpu_attestation_sha256=(
            None if gpu_attestation is None else gpu_attestation.sha256
        ),
        doctor_report_sha256=(None if doctor_report is None else doctor_report.sha256),
        pilot_evidence_sha256=pilot_evidence_sha256,
        completed_pilot_cells_sha256=completed_pilot_cells_sha256,
        terminal_receipt_sha256s=tuple(sorted(terminal)),
        qualification_lock_sha256s=tuple(
            sorted(block.qualification_sha256 for block in reduced)
        ),
        hardware_receipt_sha256s=tuple(sorted(hardware)),
        run_bindings=run_bindings,
        power_plan=power_plan,
        hardware_validity=validity,
        methods=tuple(methods),
        primary_contrasts=tuple(primary[name] for name in PRIMARY_CONTRASTS),
        holm_family=holm,
        bootstrap_hooks=(
            ("hierarchical_block_request", ("block", "request")),
            ("whole_time_block", ("time_block",)),
        ),
    )
    frozen_metrics = MappingProxyType(
        {
            method: MappingProxyType(dict(block_rows))
            for method, block_rows in request_metrics.items()
        }
    )
    return IndustrialReduction(artifact=artifact, _request_metrics=frozen_metrics)
