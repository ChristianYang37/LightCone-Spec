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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning import (
    CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256,
    E2_HALVING_PROTOCOL_SHA256,
    ConfirmationFamilyIdentity,
    ConfirmationFamilyPowerReductionArtifact,
    E1ParetoArtifact,
    E2CandidateEvaluation,
    E2CandidateIdentity,
    E2StageEvidenceArtifact,
    E2StageReductionArtifact,
    EvidenceDependenceMap,
    FamilyActivationArtifact,
    RawEvidenceRunBinding,
    ReducerActivationArtifact,
    _reduce_e2_successive_halving,
    _seal_confirmation_family_power,
    family_pilot_block_id,
    materialize_confirmation_prefix,
    reduce_e2_activation,
    verify_confirmation_pilot_activation,
)
from lightcone_spec.experiments.planning_artifacts import experiment_budget_from_dict
from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    PILOT_BLOCKS,
    ExperimentCell,
    ExperimentReceipt,
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
_BUDGET_OBSERVATION_KIND = "industrial_budget_observation_receipt_v1"
_RESERVED_GANG_MEASUREMENT = "exclusive_reserved_gang_wall_ms_x_gpu_count"
_WHOLE_INSTANCE_BILLING = "whole_inventory_wall_clock_v1"
_DISABLED_SESSION_RUN_FIELDS = (
    "session_plan_sha256",
    "session_open_receipt_sha256",
    "reset_receipt_sha256",
    "session_epoch",
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
    budget_observation: BoundArtifact

    def __post_init__(self) -> None:
        if not _is_sha256(self.cell_id):
            raise ValueError("cell_id must be lower-case SHA-256")
        if not self.terminal_receipts:
            raise ValueError("cell evidence requires terminal rank receipts")
        if not isinstance(self.budget_observation, BoundArtifact):
            raise TypeError("cell evidence requires a bound budget observation")


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
    experiment_budget_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    fixed_instance_gpu_count: int
    physical_host_id: str
    gpu_uuids: tuple[str, ...]
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256: str
    budget_observation_sha256: str


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
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    fixed_instance_gpu_count: int
    inventory_host_id: str
    confirmation_family_sha256: str
    pilot_activation_sha256: str
    final_activation_sha256: str
    confirmation_plan_sha256: str
    evidence_dependence_map_sha256: str | None
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
    budget_observation_sha256s: tuple[str, ...]
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
                "inventory_sha256": self.inventory_sha256,
                "inventory_source_receipt_sha256": (
                    self.inventory_source_receipt_sha256
                ),
                "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
                "inventory_host_id": self.inventory_host_id,
                "confirmation_family_sha256": self.confirmation_family_sha256,
                "pilot_activation_sha256": self.pilot_activation_sha256,
                "final_activation_sha256": self.final_activation_sha256,
                "confirmation_plan_sha256": self.confirmation_plan_sha256,
                "evidence_dependence_map_sha256": (self.evidence_dependence_map_sha256),
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
                "budget_observation_sha256s": list(self.budget_observation_sha256s),
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


class _RunEvidenceIdentity(Protocol):
    experiment: str
    runtime_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class _E2RunIdentity:
    experiment: Literal["E2"]
    runtime_sha256: str
    split_sha256: str


@dataclass(frozen=True)
class _LoadedCell:
    cell: ExperimentCell
    run_rows: tuple[dict[str, Any], ...]
    request_rows: tuple[dict[str, Any], ...]
    performance_rows_by_rank: tuple[tuple[dict[str, Any], ...], ...]
    update_rows_by_rank: tuple[tuple[dict[str, Any], ...], ...]
    terminal_receipt_sha256s: tuple[str, ...]
    hardware_receipt_sha256: str
    physical_gpu_uuids: tuple[str, ...]
    experiment_budget_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    fixed_instance_gpu_count: int
    physical_host_id: str
    budget_observation_sha256: str
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
    _uses_evidence_dependence_units: bool = False

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

        interval = hierarchical_block_request_bootstrap(
            self._bootstrap_rows(method, metric),
            statistic,
            repetitions=repetitions,
            seed=seed,
        )
        if self._uses_evidence_dependence_units:
            return replace(
                interval,
                independent_units=("evidence_dependence_unit", "request"),
            )
        return interval

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

        interval = time_block_bootstrap(
            self._bootstrap_rows(method, metric),
            statistic,
            repetitions=repetitions,
            seed=seed,
        )
        if self._uses_evidence_dependence_units:
            return replace(
                interval,
                independent_units=("evidence_dependence_unit",),
            )
        return interval


def _dependence_unit(
    dependence_map: EvidenceDependenceMap | None,
    *,
    cell_id: str,
    fallback: str,
) -> str:
    if dependence_map is None:
        return fallback
    return dependence_map.unit_for(cell_id)


def _dependence_source_by_unit(
    dependence_map: EvidenceDependenceMap | None,
) -> Mapping[str, str]:
    if dependence_map is None:
        return MappingProxyType({})
    return MappingProxyType(
        {unit.unit_sha256: unit.source_cell_id for unit in dependence_map.units}
    )


def _validate_analysis_dependence_map(
    dependence_map: EvidenceDependenceMap,
    *,
    active_cell_ids: Sequence[str],
) -> None:
    """Reject caller-asserted aliases until evidence semantics are recomputed."""

    active_units = {dependence_map.unit_for(cell_id) for cell_id in active_cell_ids}
    by_id = {unit.unit_sha256: unit for unit in dependence_map.units}
    if any(len(by_id[unit_id].member_cell_ids) != 1 for unit_id in active_units):
        raise ValueError(
            "non-singleton dependence units require evidence-recomputed alias receipts"
        )


def _independent_method_blocks(
    blocks: Sequence[_BlockReduction],
    *,
    method: str,
    dependence_map: EvidenceDependenceMap | None,
) -> tuple[tuple[str, _BlockReduction], ...]:
    """Select one representative for every independent evidence unit."""

    source_by_unit = _dependence_source_by_unit(dependence_map)
    selected: dict[str, _BlockReduction] = {}
    for block in blocks:
        cell_id = block.cells[method].cell.cell_id
        unit = _dependence_unit(
            dependence_map,
            cell_id=cell_id,
            fallback=f"block-{block.block}",
        )
        current = selected.get(unit)
        if current is None:
            selected[unit] = block
            continue
        source_cell_id = source_by_unit[unit]
        current_is_source = current.cells[method].cell.cell_id == source_cell_id
        candidate_is_source = cell_id == source_cell_id
        if candidate_is_source and not current_is_source:
            selected[unit] = block
    return tuple(sorted(selected.items(), key=lambda row: row[0]))


def _paired_dependence_components(
    blocks: Sequence[_BlockReduction],
    *,
    numerator: str,
    denominator: str,
    dependence_map: EvidenceDependenceMap | None,
) -> tuple[tuple[str, tuple[_BlockReduction, ...]], ...]:
    """Collapse blocks connected by any aliased numerator/denominator evidence."""

    ordered = tuple(blocks)
    if dependence_map is None:
        return tuple((f"block-{block.block}", (block,)) for block in ordered)
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    owner_by_unit: dict[str, int] = {}
    for index, block in enumerate(ordered):
        for method in (numerator, denominator):
            unit = dependence_map.unit_for(block.cells[method].cell.cell_id)
            owner = owner_by_unit.setdefault(unit, index)
            union(index, owner)
    grouped: dict[int, list[_BlockReduction]] = defaultdict(list)
    for index, block in enumerate(ordered):
        grouped[find(index)].append(block)
    result: list[tuple[str, tuple[_BlockReduction, ...]]] = []
    for component in grouped.values():
        members = tuple(sorted(component, key=lambda row: row.block))
        units = sorted(
            {
                dependence_map.unit_for(block.cells[method].cell.cell_id)
                for block in members
                for method in (numerator, denominator)
            }
        )
        component_id = content_sha256(
            {
                "schema_version": 1,
                "kind": "paired_evidence_dependence_component",
                "member_units": units,
            }
        )
        result.append((component_id, members))
    return tuple(sorted(result, key=lambda row: row[0]))


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


def _expected_topology(cell: ExperimentCell) -> tuple[int, int, int]:
    topology = cell.identity.topology
    tensor_parallel_size = 2 if topology == "tp2_dp1" else 1
    data_parallel_size = 2 if topology == "two_replica_tp1_dp2" else 1
    world_size = tensor_parallel_size * data_parallel_size
    return tensor_parallel_size, data_parallel_size, world_size


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
    family: _RunEvidenceIdentity,
    cell: ExperimentCell,
    rank: int,
) -> None:
    _validate_disabled_session_run_fields(row)
    tensor_parallel_size, data_parallel_size, world_size = _expected_topology(cell)
    expected_workload_contract = (
        f"industrial_{cell.identity.method}"
        if cell.identity.method in {"target_only", "static"}
        else "industrial_adapted"
    )
    expected = {
        "manifest_sha256": registry.sha256,
        "config_sha256": cell.cell_id,
        "industrial_cell_id": cell.cell_id,
        "runtime_sha256": family.runtime_sha256,
        "split_sha256": family.split_sha256,
        "method": cell.identity.method,
        "model_pair": cell.identity.model,
        "repetition_block": cell.identity.block,
        "patched_sglang_tree": row.get("patched_sglang_tree"),
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "world_size": world_size,
        "rank": rank,
        "status": "complete",
        "workload_contract": expected_workload_contract,
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
        "experiment_budget_sha256",
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


def _validate_disabled_session_run_fields(row: Mapping[str, Any]) -> None:
    """Reject shared-session evidence while this release blocks that execution path."""

    unknown_session_fields = {
        field
        for field in row
        if field.startswith("session_") and field not in _DISABLED_SESSION_RUN_FIELDS
    }
    if unknown_session_fields or any(
        row.get(field) is not None for field in _DISABLED_SESSION_RUN_FIELDS
    ):
        raise ValueError(
            "shared-session evidence is unavailable in this pre-mutation release"
        )


def _validate_allocation_free_performance(
    row: Mapping[str, Any],
    *,
    method: str,
) -> None:
    """Require structural zero/null adaptation state for Target-only and Static."""

    if method not in {"target_only", "static"}:
        return
    zero_fields = (
        "optimizer_bytes",
        "trainable_parameters",
        "updates_launched",
        "updates_published",
    )
    null_fields = (
        "adaptation_memory_ledger",
        "training_cuda_ms",
        "optimizer_cuda_ms",
        "merge_cuda_ms",
        "publish_cuda_ms",
        "barrier_cuda_ms",
        "exposed_update_ms",
        "main_side_overlap_ratio",
    )
    if any(
        type(row.get(field)) is not int or row.get(field) != 0 for field in zero_fields
    ):
        raise ValueError(
            "Target-only/Static performance allocated adaptation state or work"
        )
    if any(row.get(field) is not None for field in null_fields):
        raise ValueError(
            "Target-only/Static performance reports adaptation-only timing or state"
        )


def _load_hardware_receipt(
    reference: BoundArtifact,
    *,
    registry: ExperimentRegistry,
    family: _RunEvidenceIdentity,
    cell: ExperimentCell,
    terminal_receipts: tuple[BoundArtifact, ...],
    topology_sha256: str,
    performance_rows_by_rank: tuple[tuple[dict[str, Any], ...], ...],
    envelope: HardwareEnvelope,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, tuple[str, ...]], ...]]:
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
    if any(
        value.get(name) != expected
        for name, expected in (
            ("schema_version", 1),
            ("kind", "industrial_hardware_receipt"),
            ("registry_sha256", registry.sha256),
            ("runtime_sha256", family.runtime_sha256),
            ("split_sha256", family.split_sha256),
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
    physical_gpu_uuids: list[str] = []
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
            or not isinstance(context.get("gpu_uuid"), str)
            or not context["gpu_uuid"].strip()
            or not isinstance(context.get("power_state"), str)
            or not context["power_state"]
            or not isinstance(processes, list)
            or any(not isinstance(item, str) or not item for item in processes)
            or len(set(processes)) != len(processes)
        ):
            raise ValueError("hardware rank context is incomplete")
        physical_gpu_uuids.append(context["gpu_uuid"])
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
    if len(set(physical_gpu_uuids)) != len(physical_gpu_uuids):
        raise ValueError("hardware receipt reuses one physical GPU across ranks")
    return tuple(physical_gpu_uuids), tuple(validities)


def _inventory_host_id(inventory: GpuInventory) -> str:
    if not isinstance(inventory, GpuInventory):
        raise TypeError("raw industrial evidence requires an exact GPU inventory")
    if len(inventory.host_ids) != 1:
        raise ValueError("industrial raw evidence requires one whole-instance host")
    return inventory.host_ids[0]


def _validate_cell_inventory_authority(
    *,
    cell: ExperimentCell,
    physical_gpu_uuids: tuple[str, ...],
    inventory: GpuInventory,
) -> str:
    """Bind observed ranks to the complete inventory and its TP fabric."""

    host_id = _inventory_host_id(inventory)
    devices = {device.uuid: device for device in inventory.devices}
    if any(uuid not in devices for uuid in physical_gpu_uuids):
        raise ValueError("hardware evidence names a GPU outside the bound inventory")
    if any(devices[uuid].host_id != host_id for uuid in physical_gpu_uuids):
        raise ValueError("hardware evidence crosses the bound inventory host")
    tensor_parallel_size, data_parallel_size, world_size = _expected_topology(cell)
    if len(physical_gpu_uuids) != world_size:
        raise ValueError("hardware evidence differs from the registry gang shape")
    groups = {group.group_id: group for group in inventory.topology_groups}
    for replica in range(data_parallel_size):
        rank_group = frozenset(
            physical_gpu_uuids[
                replica * tensor_parallel_size : (replica + 1) * tensor_parallel_size
            ]
        )
        if tensor_parallel_size == 1:
            continue
        common = set.intersection(
            *(set(devices[uuid].allowed_topology_groups) for uuid in rank_group)
        )
        if not any(
            group_id in groups
            and groups[group_id].host_id == host_id
            and rank_group <= frozenset(groups[group_id].gpu_uuids)
            for group_id in common
        ):
            raise ValueError(
                "hardware evidence lacks an inventory-authorized TP topology group"
            )
    return host_id


def _load_budget_observation(
    reference: BoundArtifact,
    *,
    experiment_budget_sha256: str,
    terminal_receipt_sha256: str,
    fixed_instance_gpu_count: int,
) -> str:
    """Validate required planned-versus-observed timing without making a claim."""

    value = _bound_json(
        reference.path,
        reference.sha256,
        label="budget observation",
    )
    required = {
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
    if set(value) != required:
        raise ValueError("budget observation has an ambiguous schema")
    budget = value.get("budget")
    rows = value.get("observed_component_ms")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_kind") != _BUDGET_OBSERVATION_KIND
        or value.get("gpu_measurement_semantics") != _RESERVED_GANG_MEASUREMENT
        or value.get("fixed_instance_billing_semantics") != _WHOLE_INSTANCE_BILLING
        or not isinstance(budget, dict)
        or value.get("experiment_budget_sha256") != experiment_budget_sha256
        or value.get("terminal_evidence_sha256") != terminal_receipt_sha256
        or not _is_sha256(value.get("budget_observation_sha256"))
        or not isinstance(rows, list)
        or tuple(row[0] for row in rows if isinstance(row, list) and len(row) == 2)
        != _BUDGET_OBSERVATION_COMPONENTS
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], int)
            or isinstance(row[1], bool)
            or row[1] < 0
            for row in rows
        )
    ):
        raise ValueError("budget observation identity or component coverage is invalid")
    try:
        parsed_budget = experiment_budget_from_dict(
            {
                "artifact_kind": "experiment_budget",
                "artifact_sha256": experiment_budget_sha256,
                **budget,
            }
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "budget observation does not contain an exact ExperimentBudget"
        ) from exc
    integer_fields = (
        "measured_gpu_ms",
        "fixed_instance_billed_gpu_ms",
        "observed_wall_ms",
        "registered_wall_delta_ms",
        "registered_gpu_delta_ms",
        "registered_billed_delta_ms",
    )
    if any(
        not isinstance(value.get(name), int) or isinstance(value.get(name), bool)
        for name in integer_fields
    ):
        raise ValueError("budget observation accounting must use integral milliseconds")
    observed_wall_ms = sum(row[1] for row in rows)
    gpu_count = parsed_budget.gpu_count
    registered_wall_ms = parsed_budget.wall_time.registered
    registered_gpu_ms = parsed_budget.compute_gpu_ms.registered
    registered_billed_ms = parsed_budget.fixed_instance_billed_gpu_ms.registered
    scenario_pairs = tuple(
        (
            getattr(parsed_budget.wall_time, scenario),
            getattr(parsed_budget.fixed_instance_billed_gpu_ms, scenario),
        )
        for scenario in ("optimistic", "registered", "quota_envelope")
    )
    if registered_wall_ms <= 0 or any(
        wall_ms <= 0 or billed_ms % wall_ms != 0
        for wall_ms, billed_ms in scenario_pairs
    ):
        raise ValueError(
            "budget whole-instance factor requires integral positive scenario anchors"
        )
    scenario_factors = tuple(
        billed_ms // wall_ms for wall_ms, billed_ms in scenario_pairs
    )
    whole_instance_factor = registered_billed_ms // registered_wall_ms
    if (
        set(scenario_factors) != {whole_instance_factor}
        or whole_instance_factor != fixed_instance_gpu_count
        or fixed_instance_gpu_count < gpu_count
    ):
        raise ValueError(
            "budget whole-instance factor must equal the bound inventory count"
        )
    expected_observed_billed_ms = observed_wall_ms * whole_instance_factor
    if (
        value["observed_wall_ms"] != observed_wall_ms
        or value["registered_wall_delta_ms"] != observed_wall_ms - registered_wall_ms
        or value["registered_gpu_delta_ms"]
        != value["measured_gpu_ms"] - registered_gpu_ms
        or value["registered_billed_delta_ms"]
        != value["fixed_instance_billed_gpu_ms"] - registered_billed_ms
        or value["measured_gpu_ms"] != observed_wall_ms * gpu_count
        or value["fixed_instance_billed_gpu_ms"] != expected_observed_billed_ms
    ):
        raise ValueError("budget observation derived accounting is inconsistent")
    observation_content = {
        "schema_version": value["schema_version"],
        "budget": budget,
        "observed_component_ms": rows,
        "measured_gpu_ms": value["measured_gpu_ms"],
        "fixed_instance_billed_gpu_ms": value["fixed_instance_billed_gpu_ms"],
        "terminal_evidence_sha256": value["terminal_evidence_sha256"],
    }
    observation_sha256 = content_sha256(observation_content)
    if value["budget_observation_sha256"] != observation_sha256:
        raise ValueError("budget observation content digest is inconsistent")
    return observation_sha256


def _load_cell(
    reference: IndustrialCellEvidence,
    *,
    registry: ExperimentRegistry,
    family: _RunEvidenceIdentity,
    cells_by_id: Mapping[str, ExperimentCell],
    envelope: HardwareEnvelope,
    inventory: GpuInventory,
) -> _LoadedCell:
    try:
        cell = cells_by_id[reference.cell_id]
    except KeyError as exc:
        raise ValueError("cell evidence is absent from the registry") from exc
    if cell.identity.experiment != family.experiment or not cell.runnable:
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
    _, _, world_size = _expected_topology(cell)
    if len(reference.terminal_receipts) != world_size:
        raise ValueError("cell evidence lacks complete rank coverage")

    run_rows: list[dict[str, Any]] = []
    request_rows_by_rank: list[tuple[dict[str, Any], ...]] = []
    performance_rows_by_rank: list[tuple[dict[str, Any], ...]] = []
    update_rows_by_rank: list[tuple[dict[str, Any], ...]] = []
    terminal_receipt_values: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for expected_rank, receipt_reference in enumerate(reference.terminal_receipts):
        receipt, evidence = _read_terminal_receipt(receipt_reference)
        terminal_receipt_values.append(receipt)
        if receipt.get("rank") != expected_rank:
            raise ValueError("terminal rank receipts must be complete and ordered")
        run = _read_table(evidence["run"])
        if len(run) != 1:
            raise ValueError("rank evidence must contain exactly one run row")
        run_row = run[0]
        _validate_run_row(
            run_row,
            registry=registry,
            family=family,
            cell=cell,
            rank=expected_rank,
        )
        if run_row.get("run_id") != receipt.get("run_id"):
            raise ValueError("terminal receipt and run row disagree")
        if receipt.get("experiment_budget_sha256") != run_row.get(
            "experiment_budget_sha256"
        ):
            raise ValueError(
                "terminal receipt and run row disagree on ExperimentBudget"
            )
        run_ids.add(str(run_row["run_id"]))
        request_rows = _read_table(evidence["request"])
        performance_rows = _read_table(evidence["performance"])
        adapted = cell.identity.method in {"tts", "l0"}
        if adapted:
            update_rows: tuple[dict[str, Any], ...] = ()
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
                if table_name == "update":
                    update_rows = detail_rows
        elif (
            run_row["expected_round_rows"] != 0
            or run_row["expected_update_rows"] != 0
            or "round" in evidence
            or "update" in evidence
        ):
            raise ValueError(
                "Target-only/Static must allocate no round or update trace tables"
            )
        else:
            update_rows = ()
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
            _validate_allocation_free_performance(
                performance,
                method=cell.identity.method,
            )
        run_rows.append(run_row)
        request_rows_by_rank.append(request_rows)
        performance_rows_by_rank.append(performance_rows)
        update_rows_by_rank.append(update_rows)
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
        "experiment_budget_sha256",
        "world_size",
    )
    if any(
        tuple(row.get(field) for field in bound_fields)
        != tuple(run_rows[0].get(field) for field in bound_fields)
        for row in run_rows[1:]
    ):
        raise ValueError("rank run identities are inconsistent")
    physical_gpu_uuids, hardware_validity = _load_hardware_receipt(
        reference.hardware_receipt,
        registry=registry,
        family=family,
        cell=cell,
        terminal_receipts=reference.terminal_receipts,
        topology_sha256=str(run_rows[0]["topology_sha256"]),
        performance_rows_by_rank=tuple(performance_rows_by_rank),
        envelope=envelope,
    )
    physical_host_id = _validate_cell_inventory_authority(
        cell=cell,
        physical_gpu_uuids=physical_gpu_uuids,
        inventory=inventory,
    )
    run_id = str(run_rows[0]["run_id"])
    observation_path = reference.budget_observation.path.resolve()
    if (
        observation_path.name != "observation.json"
        or observation_path.parent.name != f"{run_id}.rank0.budget-observation"
        or observation_path.parent.parent != evidence_root
    ):
        raise ValueError("budget observation must live in its run evidence directory")
    prepared_receipt_sha256 = terminal_receipt_values[0].get("prepared_receipt_sha256")
    if not _is_sha256(prepared_receipt_sha256):
        raise ValueError("terminal receipt lacks its prepared-evidence binding")
    budget_observation_sha256 = _load_budget_observation(
        reference.budget_observation,
        experiment_budget_sha256=str(run_rows[0]["experiment_budget_sha256"]),
        terminal_receipt_sha256=str(prepared_receipt_sha256),
        fixed_instance_gpu_count=len(inventory.devices),
    )
    return _LoadedCell(
        cell=cell,
        run_rows=tuple(run_rows),
        request_rows=rank_zero_requests,
        performance_rows_by_rank=tuple(performance_rows_by_rank),
        update_rows_by_rank=tuple(update_rows_by_rank),
        terminal_receipt_sha256s=tuple(
            receipt.sha256 for receipt in reference.terminal_receipts
        ),
        hardware_receipt_sha256=reference.hardware_receipt.sha256,
        physical_gpu_uuids=physical_gpu_uuids,
        experiment_budget_sha256=str(run_rows[0]["experiment_budget_sha256"]),
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        fixed_instance_gpu_count=len(inventory.devices),
        physical_host_id=physical_host_id,
        budget_observation_sha256=budget_observation_sha256,
        hardware_validity=hardware_validity,
    )


def _qualification_rows(
    reference: BoundArtifact,
    *,
    registry: ExperimentRegistry,
    family: ConfirmationFamilyIdentity,
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
            ("runtime_sha256", family.runtime_sha256),
            ("split_sha256", family.split_sha256),
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
    family: ConfirmationFamilyIdentity,
    cells_by_id: Mapping[str, ExperimentCell],
    envelope: HardwareEnvelope,
    inventory: GpuInventory,
) -> _BlockReduction:
    block = block_reference.block
    loaded_sequence = tuple(
        _load_cell(
            reference,
            registry=registry,
            family=family,
            cells_by_id=cells_by_id,
            envelope=envelope,
            inventory=inventory,
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
        family=family,
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
    *,
    inventory: GpuInventory,
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
                    "budget_observation_sha256": cell.budget_observation.sha256,
                }
                for cell in sorted(block.cells, key=lambda item: item.cell_id)
            ],
            "qualification_lock_sha256": block.qualification_lock.sha256,
        }
        for block in pilots
    ]
    pilot_evidence_sha256 = content_sha256(
        {
            "schema_version": 2,
            "kind": "industrial_pilot_evidence",
            "inventory_sha256": inventory.sha256,
            "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
            "fixed_instance_gpu_count": len(inventory.devices),
            "inventory_host_id": _inventory_host_id(inventory),
            "blocks": cell_rows,
        }
    )
    completed_pilot_cells_sha256 = content_sha256(
        sorted(cell.cell_id for block in pilots for cell in block.cells)
    )
    return pilot_evidence_sha256, completed_pilot_cells_sha256


def industrial_pilot_evidence_sha256(
    blocks: Sequence[IndustrialBlockEvidence],
    *,
    inventory: GpuInventory,
) -> str:
    """Precompute the exact pilot receipt binding used by a block plan."""

    return _pilot_bindings(blocks, inventory=inventory)[0]


def industrial_completed_pilot_cells_sha256(
    blocks: Sequence[IndustrialBlockEvidence],
    *,
    inventory: GpuInventory,
) -> str:
    """Precompute the exact completed pilot-cell identity used by a block plan."""

    return _pilot_bindings(blocks, inventory=inventory)[1]


def _unresolved_artifact(
    *,
    registry: ExperimentRegistry,
    pilot_activation: FamilyActivationArtifact,
    final_activation: FamilyActivationArtifact,
    reduction: ConfirmationFamilyPowerReductionArtifact,
    evidence_dependence_map: EvidenceDependenceMap | None,
    pilot_evidence_sha256: str,
    completed_pilot_cells_sha256: str,
    blocks: Sequence[_BlockReduction],
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    patched_sglang_tree: str,
    model_lock_sha256: str,
    gpu_attestation_sha256: str | None,
    doctor_report_sha256: str | None,
    power_plan: PowerSizingPlan | None,
    reasons: tuple[str, ...],
) -> IndustrialReduction:
    plan = reduction.plan
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
    budget_observations = tuple(
        cell.budget_observation_sha256
        for block in blocks
        for cell in block.cells.values()
    )
    run_bindings = _run_bindings(blocks)
    independent_unit = (
        "evidence_dependence_unit" if evidence_dependence_map is not None else "block"
    )
    artifact = IndustrialReducerArtifact(
        status="UNRESOLVED",
        gpu_evidence=(
            "INVALIDATED"
            if any(status != "VALID" for _, status, _ in validity)
            else "UNMEASURED"
        ),
        reasons=reasons,
        registry_sha256=registry.sha256,
        experiment=plan.family.experiment,
        runtime_sha256=plan.family.runtime_sha256,
        split_sha256=plan.family.split_sha256,
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        fixed_instance_gpu_count=len(inventory.devices),
        inventory_host_id=_inventory_host_id(inventory),
        confirmation_family_sha256=plan.family.sha256,
        pilot_activation_sha256=pilot_activation.sha256,
        final_activation_sha256=final_activation.sha256,
        confirmation_plan_sha256=reduction.sha256,
        evidence_dependence_map_sha256=(
            None if evidence_dependence_map is None else evidence_dependence_map.sha256
        ),
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
        budget_observation_sha256s=tuple(sorted(budget_observations)),
        run_bindings=run_bindings,
        power_plan=power_plan,
        hardware_validity=validity,
        methods=(),
        primary_contrasts=(),
        holm_family=(),
        bootstrap_hooks=(
            ("hierarchical_block_request", (independent_unit, "request")),
            ("whole_time_block", (independent_unit,)),
        ),
    )
    return IndustrialReduction(
        artifact=artifact,
        _request_metrics=MappingProxyType({}),
        _uses_evidence_dependence_units=evidence_dependence_map is not None,
    )


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
                    experiment_budget_sha256=str(run["experiment_budget_sha256"]),
                    inventory_sha256=cell.inventory_sha256,
                    inventory_source_receipt_sha256=(
                        cell.inventory_source_receipt_sha256
                    ),
                    fixed_instance_gpu_count=cell.fixed_instance_gpu_count,
                    physical_host_id=cell.physical_host_id,
                    gpu_uuids=cell.physical_gpu_uuids,
                    terminal_receipt_sha256s=cell.terminal_receipt_sha256s,
                    hardware_receipt_sha256=cell.hardware_receipt_sha256,
                    budget_observation_sha256=(cell.budget_observation_sha256),
                )
            )
    return tuple(bindings)


def _loaded_cell_raw_run_binding(
    cell: _LoadedCell,
    *,
    scientific_unit: str,
) -> RawEvidenceRunBinding:
    run = cell.run_rows[0]
    return RawEvidenceRunBinding(
        schema_version=1,
        cell_id=cell.cell.cell_id,
        experiment=cell.cell.identity.experiment,
        method=cell.cell.identity.method,
        scientific_unit=scientific_unit,
        config_sha256=str(run["config_sha256"]),
        rank_config_sha256s=tuple(
            str(rank_run["rank_config_sha256"]) for rank_run in cell.run_rows
        ),
        run_id=str(run["run_id"]),
        rank_count=len(cell.run_rows),
        model_pair=str(run["model_pair"]),
        runtime_sha256=str(run["runtime_sha256"]),
        split_sha256=str(run["split_sha256"]),
        corpus_sha256=str(run["corpus_sha256"]),
        arrival_trace_sha256=str(run["arrival_trace_sha256"]),
        request_ids_sha256=str(run["request_ids_sha256"]),
        sampling_profile_sha256=str(run["sampling_profile_sha256"]),
        model_lock_sha256=str(run["model_lock_sha256"]),
        patched_sglang_tree=str(run["patched_sglang_tree"]),
        run_nonce_sha256=str(run["run_nonce_sha256"]),
        topology_sha256=str(run["topology_sha256"]),
        experiment_budget_sha256=str(run["experiment_budget_sha256"]),
        physical_gpu_uuids=cell.physical_gpu_uuids,
        terminal_receipt_sha256s=cell.terminal_receipt_sha256s,
        hardware_receipt_sha256=cell.hardware_receipt_sha256,
        budget_observation_sha256=cell.budget_observation_sha256,
    )


def _validate_industrial_doctor(
    reference: BoundArtifact,
    *,
    physical_gpu_uuids: Sequence[str],
) -> None:
    """Validate one exact, content-bound schema-v1 GPU readiness report."""

    physical = tuple(physical_gpu_uuids)
    if (
        not physical
        or len(physical) != len(set(physical))
        or any(not isinstance(value, str) or not value.strip() for value in physical)
    ):
        raise ValueError("industrial doctor requires exact physical GPU UUIDs")
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
        or not set(physical) <= {device["uuid"] for device in devices}
        or len({device["pci_bus_id"] for device in devices}) != 2
    ):
        raise ValueError("industrial doctor does not bind the physical GPU UUIDs")

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
    pilot_activation: FamilyActivationArtifact,
    final_activation: FamilyActivationArtifact,
    reduction: ConfirmationFamilyPowerReductionArtifact,
    evidence_dependence_map: EvidenceDependenceMap | None,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    pilot_evidence_sha256: str,
    completed_pilot_cells_sha256: str,
    patched_sglang_tree: str,
    model_lock_sha256: str,
    blocks: Sequence[_BlockReduction],
    run_bindings: tuple[IndustrialRunBinding, ...],
) -> dict[str, Any]:
    plan = reduction.plan
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
    budget_observations = tuple(
        cell.budget_observation_sha256
        for block in blocks
        for cell in block.cells.values()
    )
    return {
        "registry_sha256": registry.sha256,
        "experiment": plan.family.experiment,
        "runtime_sha256": plan.family.runtime_sha256,
        "split_sha256": plan.family.split_sha256,
        "confirmation_family_sha256": plan.family.sha256,
        "pilot_activation_sha256": pilot_activation.sha256,
        "final_activation_sha256": final_activation.sha256,
        "confirmation_plan_sha256": reduction.sha256,
        "evidence_dependence_map_sha256": (
            None if evidence_dependence_map is None else evidence_dependence_map.sha256
        ),
        "patched_sglang_tree": patched_sglang_tree,
        "model_lock_sha256": model_lock_sha256,
        "hardware_envelope_sha256": content_sha256(hardware_envelope),
        "inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "fixed_instance_gpu_count": len(inventory.devices),
        "inventory_host_id": _inventory_host_id(inventory),
        "pilot_evidence_sha256": pilot_evidence_sha256,
        "completed_pilot_cells_sha256": completed_pilot_cells_sha256,
        "gpu_uuids": sorted(
            {gpu_uuid for binding in run_bindings for gpu_uuid in binding.gpu_uuids}
        ),
        "terminal_receipt_sha256s": sorted(terminal_receipts),
        "qualification_lock_sha256s": sorted(
            block.qualification_sha256 for block in blocks
        ),
        "hardware_receipt_sha256s": sorted(hardware_receipts),
        "budget_observation_sha256s": sorted(budget_observations),
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


def _e2_evidence_manifest_sha256(
    *,
    registry: ExperimentRegistry,
    e1_receipt: ExperimentReceipt,
    pareto: E1ParetoArtifact,
    activation: ReducerActivationArtifact,
    prior_stage_reduction: E2StageReductionArtifact | None,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    cells: Sequence[IndustrialCellEvidence],
) -> str:
    return content_sha256(
        {
            "schema_version": 2,
            "kind": "industrial_e2_raw_stage_evidence",
            "registry_sha256": registry.sha256,
            "e1_receipt_sha256": e1_receipt.sha256,
            "pareto_sha256": pareto.sha256,
            "activation_sha256": activation.sha256,
            "prior_stage_reduction_sha256": (
                None if prior_stage_reduction is None else prior_stage_reduction.sha256
            ),
            "hardware_envelope_sha256": content_sha256(hardware_envelope),
            "inventory_sha256": inventory.sha256,
            "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
            "fixed_instance_gpu_count": len(inventory.devices),
            "inventory_host_id": _inventory_host_id(inventory),
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "terminal_receipt_sha256s": [
                        receipt.sha256 for receipt in cell.terminal_receipts
                    ],
                    "hardware_receipt_sha256": cell.hardware_receipt.sha256,
                    "budget_observation_sha256": cell.budget_observation.sha256,
                }
                for cell in sorted(cells, key=lambda row: row.cell_id)
            ],
        }
    )


def _raw_request_goodput(
    request_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[_RequestMetric],
) -> float:
    completed = tuple(row for row in metrics if row.completed and not row.error)
    if not completed:
        raise ValueError("E2 raw evidence has no completed request output")
    completed_ids = {row.request_id for row in completed}
    arrivals = tuple(
        row.get("arrival_ns")
        for row in request_rows
        if row.get("request_id") in completed_ids
    )
    completions = tuple(
        row.get("completed_ns")
        for row in request_rows
        if row.get("request_id") in completed_ids
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (*arrivals, *completions)
    ):
        raise ValueError("E2 goodput lacks raw arrival/completion timestamps")
    elapsed_ns = max(completions) - min(arrivals)
    if elapsed_ns <= 0:
        raise ValueError("E2 raw request duration is not finite and positive")
    goodput = sum(row.output_tokens for row in completed) / (elapsed_ns / 1e9)
    if not math.isfinite(goodput) or goodput <= 0:
        raise ValueError("E2 raw goodput is not finite and positive")
    return goodput


def _paired_request_confidence_lower(
    numerator: Sequence[_RequestMetric],
    denominator: Sequence[_RequestMetric],
) -> float:
    numerator_by_id = {
        row.request_id: row
        for row in numerator
        if row.completed and not row.error and row.output_tokens > 0
    }
    denominator_by_id = {
        row.request_id: row
        for row in denominator
        if row.completed and not row.error and row.output_tokens > 0
    }
    if set(numerator_by_id) != set(denominator_by_id) or not numerator_by_id:
        raise ValueError(
            "E2 confidence reduction requires exact paired completed requests"
        )
    log_ratios: list[float] = []
    for request_id in sorted(numerator_by_id):
        numerator_row = numerator_by_id[request_id]
        denominator_row = denominator_by_id[request_id]
        numerator_rate = numerator_row.output_tokens / numerator_row.latency_ms
        denominator_rate = denominator_row.output_tokens / denominator_row.latency_ms
        if (
            not math.isfinite(numerator_rate)
            or numerator_rate <= 0
            or not math.isfinite(denominator_rate)
            or denominator_rate <= 0
        ):
            raise ValueError("E2 paired request rate is not finite and positive")
        log_ratios.append(math.log(numerator_rate / denominator_rate))
    mean = float(np.mean(np.asarray(log_ratios, dtype=np.float64)))
    if len(log_ratios) == 1:
        lower_log_ratio = mean
    else:
        standard_error = float(
            np.std(np.asarray(log_ratios, dtype=np.float64), ddof=1)
            / math.sqrt(len(log_ratios))
        )
        lower_log_ratio = mean - 1.959963984540054 * standard_error
    result = math.exp(lower_log_ratio)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("E2 confidence lower bound is not finite and positive")
    return result


def _required_nonnegative_number(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"E2 raw evidence lacks non-negative {label}")
    return float(value)


def _e2_update_summary(cell: _LoadedCell) -> tuple[int, int]:
    update_rows = cell.update_rows_by_rank[0]
    statuses = tuple(row.get("candidate_status") for row in update_rows)
    if any(not isinstance(status, str) or not status for status in statuses):
        raise ValueError("E2 update evidence lacks candidate status")
    published_rows = tuple(
        row for row in update_rows if row.get("candidate_status") == "published"
    )
    exposed_values = tuple(
        _required_nonnegative_number(
            row.get("exposed_update_ms"),
            label="published exposed-update time",
        )
        for row in published_rows
    )
    aggregate_rows = tuple(
        row
        for row in cell.performance_rows_by_rank[0]
        if row.get("offered_requests") is not None
    )
    if len(aggregate_rows) != 1:
        raise ValueError("E2 update reduction requires one aggregate performance row")
    aggregate = aggregate_rows[0]
    if aggregate.get("updates_launched") != len(update_rows) or aggregate.get(
        "updates_published"
    ) != len(published_rows):
        raise ValueError("E2 raw update rows disagree with aggregate counters")
    if published_rows:
        observed_exposed_ms = max(exposed_values)
        aggregate_exposed_ms = _required_nonnegative_number(
            aggregate.get("exposed_update_ms"),
            label="aggregate exposed-update time",
        )
        if not math.isclose(
            aggregate_exposed_ms,
            observed_exposed_ms,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "E2 exposed-update summary disagrees with raw published updates"
            )
    elif aggregate.get("exposed_update_ms") is not None:
        raise ValueError("E2 zero-publish evidence reports exposed-update time")
    exposed_update_us = (
        0 if not exposed_values else math.ceil(max(exposed_values) * 1_000.0)
    )
    return len(published_rows), exposed_update_us


def _mark_e2_confidence_pareto(
    rows: Sequence[E2CandidateEvaluation],
) -> tuple[E2CandidateEvaluation, ...]:
    safe = tuple(row for row in rows if row.safety_passed)

    def dominates(
        left: E2CandidateEvaluation,
        right: E2CandidateEvaluation,
    ) -> bool:
        weak = (
            left.confidence_lower_goodput_ratio >= right.confidence_lower_goodput_ratio
            and left.hbm_bytes <= right.hbm_bytes
            and left.p99_itl_us <= right.p99_itl_us
            and left.exposed_update_us <= right.exposed_update_us
        )
        strict = (
            left.confidence_lower_goodput_ratio > right.confidence_lower_goodput_ratio
            or left.hbm_bytes < right.hbm_bytes
            or left.p99_itl_us < right.p99_itl_us
            or left.exposed_update_us < right.exposed_update_us
        )
        return weak and strict

    return tuple(
        replace(
            row,
            confidence_pareto=(
                row.safety_passed
                and not any(
                    other.candidate_id != row.candidate_id and dominates(other, row)
                    for other in safe
                )
            ),
        )
        for row in sorted(rows, key=lambda item: item.candidate_id)
    )


def reduce_e2_stage_from_raw(
    *,
    registry: ExperimentRegistry,
    e1_receipt: ExperimentReceipt,
    pareto: E1ParetoArtifact,
    stage_index: int,
    cells: Sequence[IndustrialCellEvidence],
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    prior_stage_reduction: E2StageReductionArtifact | None = None,
    confirmation_data_visible: bool,
) -> E2StageReductionArtifact:
    """Recompute an E2 decision from schema-v3 terminal evidence only."""

    if not isinstance(confirmation_data_visible, bool):
        raise TypeError("confirmation_data_visible must be boolean")
    if confirmation_data_visible:
        raise ValueError("E2 tuning cannot reduce visible confirmation evidence")
    inventory_host_id = _inventory_host_id(inventory)
    if prior_stage_reduction is not None and (
        prior_stage_reduction.stage_evidence.inventory_sha256 != inventory.sha256
        or prior_stage_reduction.stage_evidence.inventory_source_receipt_sha256
        != inventory.source_receipt_sha256
        or prior_stage_reduction.stage_evidence.fixed_instance_gpu_count
        != len(inventory.devices)
        or prior_stage_reduction.stage_evidence.inventory_host_id != inventory_host_id
    ):
        raise ValueError("E2 stages must retain one exact GPU inventory authority")
    activation = reduce_e2_activation(
        registry,
        e1_receipt=e1_receipt,
        pareto=pareto,
        stage_index=stage_index,
        prior_reduction=prior_stage_reduction,
    )
    references = tuple(cells)
    reference_ids = tuple(row.cell_id for row in references)
    if len(reference_ids) != len(set(reference_ids)) or set(reference_ids) != set(
        activation.plan.activated_cell_ids
    ):
        raise ValueError("E2 raw evidence must exactly cover the activated stage")
    cells_by_id = {cell.cell_id: cell for cell in registry.cells_for("E2")}
    run_identity = _E2RunIdentity(
        experiment="E2",
        runtime_sha256=activation.plan.runtime_sha256,
        split_sha256=activation.plan.split_sha256,
    )
    loaded = {
        reference.cell_id: _load_cell(
            reference,
            registry=registry,
            family=run_identity,
            cells_by_id=cells_by_id,
            envelope=hardware_envelope,
            inventory=inventory,
        )
        for reference in references
    }
    run_ids = [str(row.run_rows[0]["run_id"]) for row in loaded.values()]
    nonces = [str(row.run_rows[0]["run_nonce_sha256"]) for row in loaded.values()]
    if len(run_ids) != len(set(run_ids)) or len(nonces) != len(set(nonces)):
        raise ValueError("E2 raw stage reuses a run identity or nonce")
    common_fields = (
        "runtime_sha256",
        "split_sha256",
        "model_pair",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "patched_sglang_tree",
    )
    first_run = next(iter(loaded.values())).run_rows[0]
    if any(
        tuple(row.run_rows[0][field] for field in common_fields)
        != tuple(first_run[field] for field in common_fields)
        for row in loaded.values()
    ):
        raise ValueError("E2 stage crosses an immutable tuning-run identity")
    if first_run["patched_sglang_tree"] != PINNED_SGLANG_TREE:
        raise ValueError("E2 stage does not use the pinned patched SGLang tree")

    by_method: dict[str, list[_LoadedCell]] = defaultdict(list)
    for row in loaded.values():
        by_method[row.cell.identity.method].append(row)
    if len(by_method["target_only"]) != 1 or len(by_method["static"]) != 1:
        raise ValueError(
            "E2 raw stage requires one Target-only and one Static reference"
        )
    target = by_method["target_only"][0]
    static = by_method["static"][0]
    target_metrics = tuple(_request_metric(row) for row in target.request_rows)
    static_metrics = tuple(_request_metric(row) for row in static.request_rows)
    if any(
        not row.completed or row.error for row in (*target_metrics, *static_metrics)
    ):
        raise ValueError("E2 reference baselines require complete terminal requests")
    target_tokens = {
        str(row["request_id"]): _parse_output_token_ids(row)
        for row in target.request_rows
    }
    static_tokens = {
        str(row["request_id"]): _parse_output_token_ids(row)
        for row in static.request_rows
    }
    if target_tokens != static_tokens:
        raise ValueError(
            "E2 Static reference differs from Target-only token trajectories"
        )
    static_goodput = _raw_request_goodput(static.request_rows, static_metrics)
    reference_failures = tuple(
        (row.cell.identity.method, counter)
        for row in (target, static)
        for rows in row.performance_rows_by_rank
        for performance in rows
        for counter in _SAFETY_COUNTERS
        if performance[counter] != 0
    )
    reference_invalid_hardware = tuple(
        identity
        for row in (target, static)
        for identity, status, _ in row.hardware_validity
        if status != "VALID"
    )
    if reference_failures or reference_invalid_hardware:
        raise ValueError("E2 reference safety or hardware evidence is invalid")

    candidate_cells: dict[str, dict[str, _LoadedCell]] = defaultdict(dict)
    for method in ("tts", "l0"):
        for row in by_method[method]:
            candidate_id = E2CandidateIdentity.from_cell(row.cell).sha256
            if method in candidate_cells[candidate_id]:
                raise ValueError("E2 candidate repeats one adaptive method")
            candidate_cells[candidate_id][method] = row
    if not candidate_cells or any(
        set(pair) != {"tts", "l0"} for pair in candidate_cells.values()
    ):
        raise ValueError("E2 raw stage lacks exact matched TTS/L0 candidate pairs")

    evaluations: list[E2CandidateEvaluation] = []
    for candidate_id, pair in sorted(candidate_cells.items()):
        method_metrics: dict[str, tuple[_RequestMetric, ...]] = {}
        reasons: set[str] = set()
        hbm_values: list[int] = []
        published_counts: list[int] = []
        exposed_update_us: list[int] = []
        p99_itl_ms: list[float] = []
        for method in ("tts", "l0"):
            cell = pair[method]
            metrics = tuple(_request_metric(row) for row in cell.request_rows)
            method_metrics[method] = metrics
            if any(not row.completed or row.error for row in metrics):
                reasons.add(f"{method}:incomplete_request")
            completed_rows = tuple(
                row for row in metrics if row.completed and not row.error
            )
            if not completed_rows or any(
                row.within_request_p99_itl_ms is None for row in completed_rows
            ):
                raise ValueError("E2 candidate lacks complete raw ITL timing")
            p99_itl_ms.extend(
                float(row.within_request_p99_itl_ms) for row in completed_rows
            )
            candidate_tokens = {
                str(row["request_id"]): _parse_output_token_ids(row)
                for row in cell.request_rows
                if row.get("outcome_status") == "completed"
            }
            if any(
                request_id not in target_tokens
                or target_tokens[request_id] != token_ids
                for request_id, token_ids in candidate_tokens.items()
            ):
                reasons.add(f"{method}:target_token_mismatch")
            for rows in cell.performance_rows_by_rank:
                for performance in rows:
                    peak_hbm = performance.get("peak_hbm_bytes")
                    if (
                        not isinstance(peak_hbm, int)
                        or isinstance(peak_hbm, bool)
                        or peak_hbm < 0
                    ):
                        raise ValueError("E2 performance lacks exact peak HBM")
                    hbm_values.append(peak_hbm)
                    for counter in _SAFETY_COUNTERS:
                        if performance[counter] != 0:
                            reasons.add(f"{method}:{counter}")
            for identity, status, _ in cell.hardware_validity:
                if status != "VALID":
                    reasons.add(f"{method}:hardware:{identity}")
            published, exposed = _e2_update_summary(cell)
            published_counts.append(published)
            exposed_update_us.append(exposed)
            if published < 1:
                reasons.add(f"{method}:no_published_update")
        goodput_ratios = tuple(
            _raw_request_goodput(pair[method].request_rows, method_metrics[method])
            / static_goodput
            for method in ("tts", "l0")
        )
        point_score = min(goodput_ratios)
        confidence_lower = min(
            point_score,
            *(
                _paired_request_confidence_lower(
                    method_metrics[method],
                    static_metrics,
                )
                for method in ("tts", "l0")
            ),
        )
        evidence_sha256 = content_sha256(
            {
                "schema_version": 1,
                "kind": "industrial_e2_candidate_raw_evidence",
                "candidate_id": candidate_id,
                "reference_cell_ids": sorted(
                    (target.cell.cell_id, static.cell.cell_id)
                ),
                "candidate_cell_ids": sorted(row.cell.cell_id for row in pair.values()),
                "terminal_receipt_sha256s": sorted(
                    digest
                    for row in (target, static, *pair.values())
                    for digest in row.terminal_receipt_sha256s
                ),
                "hardware_receipt_sha256s": sorted(
                    row.hardware_receipt_sha256
                    for row in (target, static, *pair.values())
                ),
                "budget_observation_sha256s": sorted(
                    row.budget_observation_sha256
                    for row in (target, static, *pair.values())
                ),
                "run_binding_sha256s": sorted(
                    _loaded_cell_raw_run_binding(
                        row,
                        scientific_unit=f"halving_{stage_index}",
                    ).sha256
                    for row in (target, static, *pair.values())
                ),
            }
        )
        evaluations.append(
            E2CandidateEvaluation(
                candidate_id=candidate_id,
                evidence_sha256=evidence_sha256,
                safety_passed=not reasons,
                confidence_pareto=False,
                min_tts_l0_static_goodput_ratio=point_score,
                confidence_lower_goodput_ratio=confidence_lower,
                hbm_bytes=max(hbm_values),
                p99_itl_us=math.ceil(max(p99_itl_ms) * 1_000.0),
                exposed_update_us=max(exposed_update_us),
                minimum_published_updates=min(published_counts),
                safety_reason_codes=tuple(sorted(reasons)),
            )
        )
    evaluations_with_pareto = _mark_e2_confidence_pareto(evaluations)
    ordered_loaded = tuple(loaded[cell_id] for cell_id in sorted(loaded))
    raw_run_bindings = tuple(
        _loaded_cell_raw_run_binding(
            row,
            scientific_unit=f"halving_{stage_index}",
        )
        for row in ordered_loaded
    )
    stage_evidence = E2StageEvidenceArtifact(
        schema_version=3,
        registry_sha256=registry.sha256,
        runtime_sha256=activation.plan.runtime_sha256,
        split_sha256=activation.plan.split_sha256,
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        fixed_instance_gpu_count=len(inventory.devices),
        inventory_host_id=inventory_host_id,
        activation_sha256=activation.sha256,
        stage_index=stage_index,
        prior_stage_reduction_sha256=(
            None if prior_stage_reduction is None else prior_stage_reduction.sha256
        ),
        raw_evidence_manifest_sha256=_e2_evidence_manifest_sha256(
            registry=registry,
            e1_receipt=e1_receipt,
            pareto=pareto,
            activation=activation,
            prior_stage_reduction=prior_stage_reduction,
            hardware_envelope=hardware_envelope,
            inventory=inventory,
            cells=references,
        ),
        completed_cell_ids=tuple(sorted(loaded)),
        terminal_receipt_sha256s=tuple(
            sorted(
                digest
                for row in ordered_loaded
                for digest in row.terminal_receipt_sha256s
            )
        ),
        hardware_receipt_sha256s=tuple(
            sorted(row.hardware_receipt_sha256 for row in ordered_loaded)
        ),
        budget_observation_sha256s=tuple(
            sorted(row.budget_observation_sha256 for row in ordered_loaded)
        ),
        run_bindings=raw_run_bindings,
        evaluations=evaluations_with_pareto,
        reducer_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        data_source="tuning_only",
        confirmation_data_visible=False,
    )
    return _reduce_e2_successive_halving(
        activation,
        registry=registry,
        stage_evidence=stage_evidence,
    )


def reduce_confirmation_family_power(
    *,
    registry: ExperimentRegistry,
    pilot_activation: FamilyActivationArtifact,
    blocks: Sequence[IndustrialBlockEvidence],
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    confirmation_data_visible: bool,
) -> ConfirmationFamilyPowerReductionArtifact:
    """Seal family power only from exact excluded-pilot terminal evidence.

    Callers cannot provide metric summaries.  The reducer rebuilds goodput from
    schema-v3 request rows, checks safety/hardware and immutable run identity,
    and rejects any attempt to run after confirmation data became visible.
    """

    if not isinstance(confirmation_data_visible, bool):
        raise TypeError("confirmation_data_visible must be boolean")
    if confirmation_data_visible:
        raise ValueError("family power cannot use visible confirmation data")
    inventory_host_id = _inventory_host_id(inventory)
    if not isinstance(pilot_activation, FamilyActivationArtifact):
        raise TypeError("family power requires an exact pilot activation")
    family = pilot_activation.family
    if family.registry_sha256 != registry.sha256:
        raise ValueError("pilot activation belongs to another registry")
    if family.hardware_envelope_sha256 != content_sha256(hardware_envelope):
        raise ValueError("pilot activation belongs to another hardware envelope")
    verify_confirmation_pilot_activation(
        registry,
        family=family,
        artifact=pilot_activation,
    )
    references = tuple(blocks)
    if tuple(sorted(block.block for block in references)) != PILOT_BLOCKS or len(
        {block.block for block in references}
    ) != len(PILOT_BLOCKS):
        raise ValueError("family power requires exactly four excluded pilot blocks")
    evidence_cell_ids = tuple(
        cell.cell_id for block in references for cell in block.cells
    )
    if len(evidence_cell_ids) != len(set(evidence_cell_ids)) or set(
        evidence_cell_ids
    ) != set(pilot_activation.activated_cell_ids):
        raise ValueError("family power requires exact activated pilot-cell evidence")
    pilot_evidence_sha256, completed_pilot_cells_sha256 = _pilot_bindings(
        references,
        inventory=inventory,
    )
    expected_completed = content_sha256(
        tuple(sorted(pilot_activation.activated_cell_ids))
    )
    if completed_pilot_cells_sha256 != expected_completed:
        raise ValueError("pilot receipt completion differs from pilot activation")

    cells_by_id = {cell.cell_id: cell for cell in registry.cells}
    reduced = tuple(
        _reduce_block(
            reference,
            registry=registry,
            family=family,
            cells_by_id=cells_by_id,
            envelope=hardware_envelope,
            inventory=inventory,
        )
        for reference in sorted(references, key=lambda item: item.block)
    )
    nonces: set[str] = set()
    run_ids: set[str] = set()
    for block in reduced:
        for cell in block.cells.values():
            nonce = str(cell.run_rows[0]["run_nonce_sha256"])
            run_id = str(cell.run_rows[0]["run_id"])
            if nonce in nonces or run_id in run_ids:
                raise ValueError("pilot run identity or nonce is reused")
            nonces.add(nonce)
            run_ids.add(run_id)
    invalid_hardware = tuple(
        identity
        for block in reduced
        for cell in block.cells.values()
        for identity, status, _ in cell.hardware_validity
        if status != "VALID"
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
        raise ValueError("family power refuses invalid hardware or safety evidence")
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
    if patched_trees != {PINNED_SGLANG_TREE} or len(model_locks) != 1:
        raise ValueError("pilot family crosses patch or model-lock identities")
    pilot_metric = "slo_goodput_tps" if family.experiment == "E5" else "goodput_tps"
    power_sizing = preregister_power_sizing(
        tuple(
            PilotBlock(
                block_id=family_pilot_block_id(family, block.block),
                static_goodput=getattr(block, pilot_metric)["static"],
                tts_goodput=getattr(block, pilot_metric)["tts"],
                l0_goodput=getattr(block, pilot_metric)["l0"],
            )
            for block in reduced
        )
    )
    plan = _seal_confirmation_family_power(
        registry=registry,
        family=family,
        pilot_activation=pilot_activation,
        completed_pilot_cell_ids=pilot_activation.activated_cell_ids,
        pilot_evidence_sha256=pilot_evidence_sha256,
        power_sizing=power_sizing,
        confirmation_data_visible=False,
    )
    raw_run_bindings = tuple(
        sorted(
            (
                _loaded_cell_raw_run_binding(
                    cell,
                    scientific_unit=f"excluded_pilot_{block.block}",
                )
                for block in reduced
                for cell in block.cells.values()
            ),
            key=lambda binding: binding.cell_id,
        )
    )
    return ConfirmationFamilyPowerReductionArtifact(
        schema_version=2,
        plan=plan,
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        fixed_instance_gpu_count=len(inventory.devices),
        inventory_host_id=inventory_host_id,
        raw_evidence_manifest_sha256=pilot_evidence_sha256,
        terminal_receipt_sha256s=tuple(
            sorted(
                digest
                for block in reduced
                for cell in block.cells.values()
                for digest in cell.terminal_receipt_sha256s
            )
        ),
        hardware_receipt_sha256s=tuple(
            sorted(
                cell.hardware_receipt_sha256
                for block in reduced
                for cell in block.cells.values()
            )
        ),
        budget_observation_sha256s=tuple(
            sorted(
                cell.budget_observation_sha256
                for block in reduced
                for cell in block.cells.values()
            )
        ),
        run_bindings=raw_run_bindings,
        reducer_protocol_sha256=(CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256),
        data_source="excluded_pilots_only",
        confirmation_data_visible=False,
    )


def reduce_industrial_schema_v3(
    *,
    registry: ExperimentRegistry,
    pilot_activation: FamilyActivationArtifact,
    final_activation: FamilyActivationArtifact,
    confirmation_reduction: ConfirmationFamilyPowerReductionArtifact,
    blocks: Sequence[IndustrialBlockEvidence],
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    evidence_dependence_map: EvidenceDependenceMap | None = None,
    gpu_attestation: BoundArtifact | None = None,
    doctor_report: BoundArtifact | None = None,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 0,
) -> IndustrialReduction:
    """Reduce one strictly activated confirmation family from terminal evidence.

    Missing/partial inputs raise before inference.  A valid but out-of-envelope
    hardware block and a valid UNDERPOWERED family both return ``UNRESOLVED``
    with no contrasts.  Pilots are never reused as final confirmation rows.
    """

    if not isinstance(pilot_activation, FamilyActivationArtifact) or not isinstance(
        final_activation, FamilyActivationArtifact
    ):
        raise TypeError("industrial analysis requires exact family activations")
    if not isinstance(
        confirmation_reduction,
        ConfirmationFamilyPowerReductionArtifact,
    ):
        raise TypeError("industrial analysis requires a raw family-power reduction")
    if evidence_dependence_map is not None and not isinstance(
        evidence_dependence_map, EvidenceDependenceMap
    ):
        raise TypeError("evidence_dependence_map must be an EvidenceDependenceMap")
    confirmation_plan = confirmation_reduction.plan
    family = confirmation_reduction.family
    inventory_host_id = _inventory_host_id(inventory)
    if family.registry_sha256 != registry.sha256:
        raise ValueError("confirmation plan belongs to another registry")
    if family.hardware_envelope_sha256 != content_sha256(hardware_envelope):
        raise ValueError("confirmation family belongs to another hardware envelope")
    if (
        confirmation_reduction.inventory_sha256 != inventory.sha256
        or confirmation_reduction.inventory_source_receipt_sha256
        != inventory.source_receipt_sha256
        or confirmation_reduction.fixed_instance_gpu_count != len(inventory.devices)
        or confirmation_reduction.inventory_host_id != inventory_host_id
    ):
        raise ValueError("confirmation reduction belongs to another GPU inventory")
    verify_confirmation_pilot_activation(
        registry,
        family=family,
        artifact=pilot_activation,
    )
    expected_final_activation = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=confirmation_reduction,
        pilot_activation=pilot_activation,
    )
    if final_activation != expected_final_activation:
        raise ValueError(
            "family final activation is not reducer-generated from its power plan"
        )
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
    evidence_cell_ids = tuple(
        cell.cell_id for block in block_references for cell in block.cells
    )
    if len(evidence_cell_ids) != len(set(evidence_cell_ids)):
        raise ValueError("industrial cell evidence must be unique")
    activated_cell_ids = (
        pilot_activation.activated_cell_ids + final_activation.activated_cell_ids
    )
    if set(evidence_cell_ids) != set(activated_cell_ids):
        raise ValueError(
            "industrial evidence contains missing or unactivated family candidates"
        )
    if evidence_dependence_map is not None:
        _validate_analysis_dependence_map(
            evidence_dependence_map,
            active_cell_ids=activated_cell_ids,
        )
    pilot_evidence_sha256, completed_pilot_cells_sha256 = _pilot_bindings(
        block_references,
        inventory=inventory,
    )

    expected_blocks = PILOT_BLOCKS + confirmation_plan.selected_final_prefix
    if tuple(sorted(block_ids)) != expected_blocks:
        raise ValueError("evidence must cover four pilots and the locked final prefix")
    expected_reduction = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot_activation,
        blocks=tuple(
            block for block in block_references if block.block in PILOT_BLOCKS
        ),
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    if confirmation_reduction != expected_reduction:
        raise ValueError(
            "confirmation family power reduction differs from pilot terminal evidence"
        )

    cells_by_id = {cell.cell_id: cell for cell in registry.cells}
    reduced = tuple(
        _reduce_block(
            reference,
            registry=registry,
            family=family,
            cells_by_id=cells_by_id,
            envelope=hardware_envelope,
            inventory=inventory,
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
    if evidence_dependence_map is not None and any(
        len(
            _paired_dependence_components(
                pilots,
                numerator="l0",
                denominator=denominator,
                dependence_map=evidence_dependence_map,
            )
        )
        != len(PILOT_BLOCKS)
        for denominator in ("static", "tts")
    ):
        raise ValueError("family power requires four independent excluded pilot units")
    power_plan = expected_reduction.plan.power_sizing

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
        if gpu_attestation is None or doctor_report is None:
            raise RuntimeError(
                "industrial attestation binding changed during reduction"
            )
        _validate_industrial_doctor(
            doctor_report,
            physical_gpu_uuids=tuple(
                sorted(
                    {
                        gpu_uuid
                        for binding in run_bindings
                        for gpu_uuid in binding.gpu_uuids
                    }
                )
            ),
        )
        _validate_industrial_gpu_attestation(
            gpu_attestation,
            doctor_report=doctor_report,
            expected_chain=_attestation_chain(
                registry=registry,
                pilot_activation=pilot_activation,
                final_activation=final_activation,
                reduction=confirmation_reduction,
                evidence_dependence_map=evidence_dependence_map,
                hardware_envelope=hardware_envelope,
                inventory=inventory,
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
            pilot_activation=pilot_activation,
            final_activation=final_activation,
            reduction=confirmation_reduction,
            evidence_dependence_map=evidence_dependence_map,
            pilot_evidence_sha256=pilot_evidence_sha256,
            completed_pilot_cells_sha256=completed_pilot_cells_sha256,
            blocks=reduced,
            hardware_envelope=hardware_envelope,
            inventory=inventory,
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

    attestation_reason = (
        "gpu_attestation:untrusted_attester"
        if bound_attestation
        else "gpu_attestation:missing"
    )
    if confirmation_plan.status == "UNDERPOWERED":
        return _unresolved_artifact(
            registry=registry,
            pilot_activation=pilot_activation,
            final_activation=final_activation,
            reduction=confirmation_reduction,
            evidence_dependence_map=evidence_dependence_map,
            pilot_evidence_sha256=pilot_evidence_sha256,
            completed_pilot_cells_sha256=completed_pilot_cells_sha256,
            blocks=reduced,
            hardware_envelope=hardware_envelope,
            inventory=inventory,
            patched_sglang_tree=patched_sglang_tree,
            model_lock_sha256=model_lock_sha256,
            gpu_attestation_sha256=(
                None if gpu_attestation is None else gpu_attestation.sha256
            ),
            doctor_report_sha256=(
                None if doctor_report is None else doctor_report.sha256
            ),
            power_plan=power_plan,
            reasons=("confirmation_family:underpowered", attestation_reason),
        )

    final = tuple(block for block in reduced if block.block not in PILOT_BLOCKS)
    if confirmation_plan.selected_final_blocks is None:
        raise RuntimeError("POWERED confirmation lost its selected final prefix")
    if len(final) != confirmation_plan.selected_final_blocks:
        raise RuntimeError("validated final prefix changed during reduction")
    metric_name = "slo_goodput_tps" if family.experiment == "E5" else "goodput_tps"
    primary: dict[str, PairedBcaContrast] = {}
    for contrast, denominator in (
        ("l0_vs_static", "static"),
        ("l0_vs_tts", "tts"),
    ):
        components = _paired_dependence_components(
            final,
            numerator="l0",
            denominator=denominator,
            dependence_map=evidence_dependence_map,
        )
        if len(components) < 2:
            raise ValueError(
                "evidence dependence leaves fewer than two covariance units"
            )
        paired: dict[str, tuple[float, float]] = {}
        for component_id, component in components:
            numerator_values = np.asarray(
                [getattr(block, metric_name)["l0"] for block in component],
                dtype=np.float64,
            )
            denominator_values = np.asarray(
                [getattr(block, metric_name)[denominator] for block in component],
                dtype=np.float64,
            )
            paired[component_id] = (
                float(np.exp(np.mean(np.log(numerator_values)))),
                float(np.exp(np.mean(np.log(denominator_values)))),
            )
        contrast_result = paired_bca_contrast(
            contrast,
            paired,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + PRIMARY_CONTRASTS.index(contrast),
        )
        primary[contrast] = (
            replace(
                contrast_result,
                independent_unit="evidence_dependence_component",
            )
            if evidence_dependence_map is not None
            else contrast_result
        )
    holm = holm_primary_contrasts(primary)

    methods: list[MethodReduction] = []
    request_metrics: dict[str, dict[str, tuple[_RequestMetric, ...]]] = defaultdict(
        dict
    )
    for method in _METHODS:
        independent_blocks = _independent_method_blocks(
            final,
            method=method,
            dependence_map=evidence_dependence_map,
        )
        independent_ids = tuple(unit for unit, _ in independent_blocks)
        combined_slo = tuple(
            SloRequest(
                request_id=f"{unit}:{row.request_id}",
                prompt_bucket=row.prompt_bucket,
                eligible=row.eligible,
                completed=row.completed,
                error=row.error,
                ttft_ms=row.ttft_ms,
                within_request_p99_itl_ms=row.within_request_p99_itl_ms,
            )
            for unit, block in independent_blocks
            for row in block.slo_requests[method]
        )
        slo = account_slo(combined_slo)
        completed_latencies = tuple(
            metric.latency_ms
            for _, block in independent_blocks
            for metric in block.request_metrics[method]
            if metric.completed
        )
        observed_p99 = (
            float(np.quantile(np.asarray(completed_latencies), 0.99))
            if completed_latencies
            else None
        )
        p99_guard = guard_p99_claim(
            f"{family.experiment}:{method}",
            completed_requests=len(completed_latencies),
            observed_p99_ms=observed_p99,
        )
        methods.append(
            MethodReduction(
                method=method,
                block_ids=independent_ids,
                mean_output_goodput_tps=float(
                    np.mean(
                        [block.goodput_tps[method] for _, block in independent_blocks]
                    )
                ),
                mean_slo_qualified_goodput_tps=float(
                    np.mean(
                        [
                            block.slo_goodput_tps[method]
                            for _, block in independent_blocks
                        ]
                    )
                ),
                slo=slo,
                aggregate_latency_p99=p99_guard,
            )
        )
        for unit, block in independent_blocks:
            request_metrics[method][unit] = block.request_metrics[method]

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
    budget_observations = tuple(
        cell.budget_observation_sha256
        for block in reduced
        for cell in block.cells.values()
    )
    independent_unit = (
        "evidence_dependence_unit" if evidence_dependence_map is not None else "block"
    )
    # Hash/field agreement proves integrity, not that a GPU produced the files.
    # This release has no trusted hardware-rooted attester, so even an exactly
    # bound caller artifact remains diagnostic-only.
    artifact = IndustrialReducerArtifact(
        status="UNRESOLVED",
        gpu_evidence="UNMEASURED",
        reasons=(attestation_reason,),
        registry_sha256=registry.sha256,
        experiment=family.experiment,
        runtime_sha256=family.runtime_sha256,
        split_sha256=family.split_sha256,
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        fixed_instance_gpu_count=len(inventory.devices),
        inventory_host_id=inventory_host_id,
        confirmation_family_sha256=family.sha256,
        pilot_activation_sha256=pilot_activation.sha256,
        final_activation_sha256=final_activation.sha256,
        confirmation_plan_sha256=confirmation_reduction.sha256,
        evidence_dependence_map_sha256=(
            None if evidence_dependence_map is None else evidence_dependence_map.sha256
        ),
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
        budget_observation_sha256s=tuple(sorted(budget_observations)),
        run_bindings=run_bindings,
        power_plan=power_plan,
        hardware_validity=validity,
        methods=tuple(methods),
        primary_contrasts=tuple(primary[name] for name in PRIMARY_CONTRASTS),
        holm_family=holm,
        bootstrap_hooks=(
            ("hierarchical_block_request", (independent_unit, "request")),
            ("whole_time_block", (independent_unit,)),
        ),
    )
    frozen_metrics = MappingProxyType(
        {
            method: MappingProxyType(dict(block_rows))
            for method, block_rows in request_metrics.items()
        }
    )
    return IndustrialReduction(
        artifact=artifact,
        _request_metrics=frozen_metrics,
        _uses_evidence_dependence_units=evidence_dependence_map is not None,
    )
