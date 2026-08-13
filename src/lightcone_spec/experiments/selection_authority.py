"""Raw-evidence authority for the E3a and E1 locked selections.

The planning dataclasses are intentionally not scientific authority on their
own.  This module derives them from path-bearing terminal evidence manifests.
The manifest references are reopened by :mod:`industrial_analysis`, which
validates the terminal -> Parquet -> native-terminal chain, the hardware
receipt, the immutable :class:`ExperimentBudget`, and its terminal-bound
``BudgetObservationReceipt``.

Formal callers must additionally obtain ``registry``, ``inventory``, the E3a
receipt, and the E1 activation inputs from their release-owned raw authorities.
Passing content-consistent typed values directly is useful for deterministic
CPU tests, but is not by itself a launch or stage-sealing authority.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.planning import (
    E1_RAW_PARETO_PROTOCOL_SHA256,
    E1GeometryIdentity,
    E1ParetoArtifact,
    ReducerActivationArtifact,
    SealedE3aSelection,
    reduce_e1_activation,
)
from lightcone_spec.experiments.registry import (
    DRAFT_WIDTHS,
    E1_OPTIMIZER_ANCHORS,
    ExperimentReceipt,
    ExperimentRegistry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.orchestration.native_terminal import (
    validate_native_terminal_artifact,
)
from lightcone_spec.runtime.attestation import RELEASE_TRUSTED_ATTESTER_POLICY

if TYPE_CHECKING:
    from lightcone_spec.experiments.industrial_analysis import (
        RawE1ParetoEvidenceManifest,
        RawE3aSelectionEvidenceManifest,
        _LoadedCell,
        _RequestMetric,
    )

_SHA256_LENGTH = 64
_NORMAL_95_LOWER_Z = 1.959963984540054
_E1_ACTIVATED_CELL_COUNT = 130
E3A_SELECTION_POLICY_UNREGISTERED_REASON = "e3a_selection_policy_unregistered"
E3A_LOCKED_OUTPUT_REDUCTION_UNREGISTERED_REASON = (
    "e3a_locked_output_typed_reduction_unregistered"
)
E3A_CAPACITY_SURFACE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e3a_raw_capacity_surface_reduction_protocol",
        "inputs": (
            "exact_runnable_e3a_terminal_evidence",
            "source_owned_trusted_native_terminal_authority",
            "exact_static_target_request_and_token_pairing",
        ),
        "outputs": (
            "per_cell_raw_goodput_tps",
            "per_cell_peak_hbm_bytes",
            "per_static_cell_target_goodput_ratio",
        ),
        "scientific_selection": "forbidden",
    }
)
E3A_SCIENTIFIC_SELECTION_AUTHORITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "e3a_source_owned_scientific_selection_authority_protocol",
        "capacity_input": E3A_CAPACITY_SURFACE_PROTOCOL_SHA256,
        "policy": "source_owned_typed_policy_without_caller_override",
        "unregistered_policy": E3A_SELECTION_POLICY_UNREGISTERED_REASON,
        "locked_outputs": "separate_typed_first_party_reduction_required",
    }
)


class SelectionReductionAuthorityUnavailableError(RuntimeError):
    """Raw evidence is bound but cannot be trusted by this source release."""

    def __init__(self, reason_code: str) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("selection authority reason code must be text")
        self.reason_code = reason_code
        super().__init__(f"selection reduction is BLOCKED: {reason_code}")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(label: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


@dataclass(frozen=True)
class E3aScientificSelectionPolicy:
    """Source-owned, reviewable scientific choices for one E3a selection.

    The reducer supports this one explicit rule vocabulary, but this release
    deliberately registers no instance.  In particular, neither a context
    floor, a goodput fraction, nor a tie-break is a default.
    """

    schema_version: int
    source_authority: str
    source_authority_sha256: str
    primary_contexts: tuple[int, ...]
    reference_load_goodput_fraction: float
    reference_load_statistic: Literal["maximum_width_median_static_goodput"]
    reference_load_choice: Literal["smallest_concurrency_reaching_registered_fraction"]
    width_primary_objective: Literal["maximum_worst_static_target_goodput_ratio"]
    width_secondary_objective: Literal["maximum_median_static_goodput"]
    width_final_tiebreak: Literal["smallest_width"]
    locked_output_reducer_protocol_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E3a scientific-selection policy schema 1 works")
        if (
            type(self.source_authority) is not str
            or not self.source_authority
            or "\n" in self.source_authority
            or "\r" in self.source_authority
        ):
            raise ValueError("E3a policy source authority must be single-line text")
        _require_sha256("E3a policy source authority", self.source_authority_sha256)
        _require_sha256(
            "E3a locked-output reducer protocol",
            self.locked_output_reducer_protocol_sha256,
        )
        if (
            not self.primary_contexts
            or self.primary_contexts != tuple(sorted(set(self.primary_contexts)))
            or any(
                type(value) is not int or value < 1 for value in self.primary_contexts
            )
        ):
            raise ValueError("E3a policy primary contexts must be sorted unique ints")
        fraction = self.reference_load_goodput_fraction
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError("E3a reference-load fraction must lie in (0, 1]")
        expected = {
            "reference_load_statistic": "maximum_width_median_static_goodput",
            "reference_load_choice": (
                "smallest_concurrency_reaching_registered_fraction"
            ),
            "width_primary_objective": ("maximum_worst_static_target_goodput_ratio"),
            "width_secondary_objective": "maximum_median_static_goodput",
            "width_final_tiebreak": "smallest_width",
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("E3a scientific-selection rule vocabulary is unsupported")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


# Scientific defaults are forbidden.  A future release must replace this
# source-owned constant with one reviewed typed policy and implement the
# independently registered six-output reducer before completion can proceed.
RELEASE_E3A_SCIENTIFIC_SELECTION_POLICY: E3aScientificSelectionPolicy | None = None


@dataclass(frozen=True)
class E3aCapacityObservation:
    """One policy-free E3a capacity observation derived from raw evidence."""

    cell_id: str
    method: Literal["target_only", "static"]
    context: int
    regime: str
    concurrency: int
    width: int | None
    raw_goodput_tps: float
    peak_hbm_bytes: int
    static_target_goodput_ratio: float | None
    terminal_rank_count: int
    raw_run_binding_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("E3a capacity cell", self.cell_id)
        if self.method not in {"target_only", "static"}:
            raise ValueError("E3a capacity method must be Target-only or Static")
        if type(self.context) is not int or self.context < 1:
            raise ValueError("E3a capacity context must be a positive integer")
        if type(self.regime) is not str or not self.regime:
            raise ValueError("E3a capacity regime must be text")
        if type(self.concurrency) is not int or self.concurrency < 1:
            raise ValueError("E3a capacity concurrency must be a positive integer")
        if (
            not isinstance(self.raw_goodput_tps, (int, float))
            or isinstance(self.raw_goodput_tps, bool)
            or not math.isfinite(float(self.raw_goodput_tps))
            or float(self.raw_goodput_tps) <= 0.0
        ):
            raise ValueError("E3a raw goodput must be finite and positive")
        if type(self.peak_hbm_bytes) is not int or self.peak_hbm_bytes < 0:
            raise ValueError("E3a peak HBM must be a nonnegative integer")
        if type(self.terminal_rank_count) is not int or self.terminal_rank_count < 1:
            raise ValueError("E3a capacity must bind every terminal rank")
        _require_sha256("E3a capacity raw-run binding", self.raw_run_binding_sha256)
        if self.method == "target_only":
            if self.width is not None or self.static_target_goodput_ratio is not None:
                raise ValueError("E3a Target-only capacity cannot claim Static fields")
        else:
            ratio = self.static_target_goodput_ratio
            if self.width not in DRAFT_WIDTHS:
                raise ValueError("E3a Static capacity width is outside the grid")
            if (
                not isinstance(ratio, (int, float))
                or isinstance(ratio, bool)
                or not math.isfinite(float(ratio))
                or float(ratio) <= 0.0
            ):
                raise ValueError("E3a Static/Target ratio must be finite and positive")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3aNativeTerminalLineage:
    """One exact cell/rank native-terminal binding in a capacity surface."""

    cell_id: str
    rank: int
    binding_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("E3a native-terminal cell", self.cell_id)
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("E3a native-terminal rank must be nonnegative")
        _require_sha256("E3a native-terminal binding", self.binding_sha256)


@dataclass(frozen=True)
class E3aCapacitySurface:
    """Complete policy-free E3a capacity facts and their raw lineage."""

    schema_version: int
    registry_sha256: str
    runtime_sha256: str
    split_sha256: str
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    hardware_envelope_sha256: str
    raw_manifest_sha256: str
    observations: tuple[E3aCapacityObservation, ...]
    native_terminal_lineage: tuple[E3aNativeTerminalLineage, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E3a capacity-surface schema 1 is supported")
        for name in (
            "registry_sha256",
            "runtime_sha256",
            "split_sha256",
            "inventory_sha256",
            "inventory_source_receipt_sha256",
            "hardware_envelope_sha256",
            "raw_manifest_sha256",
        ):
            _require_sha256(f"E3a capacity {name}", getattr(self, name))
        if (
            not self.observations
            or any(type(row) is not E3aCapacityObservation for row in self.observations)
            or tuple(row.cell_id for row in self.observations)
            != tuple(sorted({row.cell_id for row in self.observations}))
        ):
            raise ValueError("E3a capacity observations must be cell-sorted and unique")
        if any(
            type(row) is not E3aNativeTerminalLineage
            for row in self.native_terminal_lineage
        ):
            raise TypeError("E3a native-terminal lineage must be typed")
        actual_ranks = tuple(
            (row.cell_id, row.rank) for row in self.native_terminal_lineage
        )
        expected_ranks = tuple(
            (observation.cell_id, rank)
            for observation in self.observations
            for rank in range(observation.terminal_rank_count)
        )
        if actual_ranks != expected_ranks:
            raise ValueError(
                "E3a native-terminal lineage must exactly cover every cell/rank"
            )

    @property
    def sha256(self) -> str:
        return content_sha256(self)


def _require_e3a_scientific_selection_policy() -> E3aScientificSelectionPolicy:
    policy = RELEASE_E3A_SCIENTIFIC_SELECTION_POLICY
    if type(policy) is not E3aScientificSelectionPolicy:
        raise SelectionReductionAuthorityUnavailableError(
            E3A_SELECTION_POLICY_UNREGISTERED_REASON
        )
    policy.__post_init__()
    return policy


def require_e3a_locked_output_reduction_authority() -> None:
    """Fail closed until all six E3a outputs have one typed raw replay.

    Registering a width/load selection policy alone must never make opaque
    caller-authored JSON eligible for an E3a completion receipt.
    """

    _require_e3a_scientific_selection_policy()
    raise SelectionReductionAuthorityUnavailableError(
        E3A_LOCKED_OUTPUT_REDUCTION_UNREGISTERED_REASON
    )


@dataclass(frozen=True)
class _ExactRunIdentity:
    experiment: Literal["E3a", "E1"]
    runtime_sha256: str
    split_sha256: str


def _analysis_api():
    """Import the shared raw loader lazily to keep module imports acyclic."""

    from lightcone_spec.experiments import industrial_analysis as analysis

    return analysis


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _require_finite_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is forbidden")
    if type(value) is dict:
        for item in value.values():
            _require_finite_json(item)
    elif type(value) is list:
        for item in value:
            _require_finite_json(item)


def _strict_json_object(body: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    _require_finite_json(value)
    if type(value) is not dict:
        raise TypeError(f"{label} must contain one JSON object")
    return value


def _validate_native_terminal_authority(
    loaded: Mapping[str, _LoadedCell],
    *,
    references: Sequence[object],
) -> tuple[dict[str, object], ...]:
    """Reopen every signed native artifact and bind it to F/P/run evidence."""

    policy = RELEASE_TRUSTED_ATTESTER_POLICY
    if not policy.release_ready:
        raise SelectionReductionAuthorityUnavailableError(
            "trusted_hardware_attester_unavailable"
        )
    analysis = _analysis_api()
    references_by_id = {reference.cell_id: reference for reference in references}
    if set(references_by_id) != set(loaded):
        raise ValueError("native terminal authority coverage differs from raw evidence")
    bindings: list[dict[str, object]] = []
    for cell_id in sorted(loaded):
        cell = loaded[cell_id]
        reference = references_by_id[cell_id]
        if len(reference.terminal_receipts) != len(cell.run_rows):
            raise ValueError("native terminal authority lacks exact rank coverage")
        for rank, (run, terminal_reference) in enumerate(
            zip(cell.run_rows, reference.terminal_receipts, strict=True)
        ):
            terminal_body = analysis._bound_file(
                terminal_reference.path,
                terminal_reference.sha256,
                label="selection terminal receipt",
                require_sidecar=True,
            )
            terminal = _strict_json_object(
                terminal_body, label="selection terminal receipt"
            )
            native_binding = terminal.get("native_terminal_artifact")
            if type(native_binding) is not dict or set(native_binding) != {
                "path",
                "size",
                "raw_sha256",
                "terminal_sha256",
                "trusted_attester_policy_sha256",
            }:
                raise ValueError(
                    "selection terminal receipt lacks native artifact authority"
                )
            name = run.get("native_terminal_artifact_path")
            size = run.get("native_terminal_artifact_size")
            raw_sha256 = run.get("native_terminal_raw_sha256")
            terminal_sha256 = run.get("native_terminal_sha256")
            policy_sha256 = run.get("trusted_attester_policy_sha256")
            if (
                type(name) is not str
                or Path(name).name != name
                or type(size) is not int
                or size < 1
                or not _is_sha256(raw_sha256)
                or not _is_sha256(terminal_sha256)
                or policy_sha256 != policy.sha256
                or native_binding
                != {
                    "path": name,
                    "size": size,
                    "raw_sha256": raw_sha256,
                    "terminal_sha256": terminal_sha256,
                    "trusted_attester_policy_sha256": policy_sha256,
                }
            ):
                raise ValueError("selection run has a foreign native terminal binding")
            native_path = terminal_reference.path.parent / name
            if not native_path.is_absolute() or native_path.parent != (
                terminal_reference.path.parent
            ):
                raise ValueError("selection native terminal escaped the evidence root")
            native_body = analysis._bound_file(
                native_path,
                raw_sha256,
                label="selection native terminal artifact",
                require_sidecar=True,
            )
            if len(native_body) != size:
                raise ValueError("selection native terminal size binding changed")
            native = _strict_json_object(
                native_body, label="selection native terminal artifact"
            )
            canonical = (
                json.dumps(
                    native,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            if native_body != canonical:
                raise ValueError("selection native terminal artifact is not canonical")
            try:
                validated = validate_native_terminal_artifact(
                    native,
                    trusted_attester_policy=policy,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "selection native terminal fails trusted validation"
                ) from error
            binding = validated.binding
            request_rows = cell.request_rows
            request_ids = tuple(row.get("request_id") for row in request_rows)
            if (
                any(
                    type(request_id) is not str or not request_id
                    for request_id in request_ids
                )
                or len(request_ids) != len(set(request_ids))
                or binding.scored_request_ids != request_ids
                or len(validated.requests) != len(request_rows)
            ):
                raise ValueError(
                    "selection native terminal scored coverage differs from Parquet"
                )
            for expectation, request_row in zip(
                validated.requests, request_rows, strict=True
            ):
                input_tokens = request_row.get("input_tokens")
                submitted = request_row.get("admitted_ns") is not None
                outcome = request_row.get("outcome_status")
                if (
                    type(input_tokens) is not int
                    or input_tokens < 1
                    or len(expectation.input_token_ids) != input_tokens
                    or (outcome == "completed" and not submitted)
                    or (outcome == "rejected" and submitted)
                ):
                    raise ValueError(
                        "selection native terminal request identity differs from Parquet"
                    )
                expected_status = (
                    "completed"
                    if outcome == "completed"
                    else "aborted"
                    if submitted
                    else outcome
                )
                expected_outputs = (
                    analysis._parse_output_token_ids(request_row) if submitted else None
                )
                if (
                    expectation.request_id != request_row.get("request_id")
                    or expectation.submitted_to_server is not submitted
                    or expectation.terminal_status != expected_status
                    or expectation.output_token_ids != expected_outputs
                ):
                    raise ValueError(
                        "selection native terminal outcome differs from Parquet"
                    )
            if (
                not validated.trusted_attestation
                or validated.terminal_sha256 != terminal_sha256
                or binding.run_id != run.get("run_id")
                or binding.run_nonce_sha256 != run.get("run_nonce_sha256")
                or binding.execution_plan_sha256 != run.get("runtime_sha256")
                or binding.rank_config_sha256 != run.get("rank_config_sha256")
                or binding.method != cell.cell.identity.method
                or terminal.get("run_id") != run.get("run_id")
                or terminal.get("rank") != rank
            ):
                raise ValueError(
                    "selection native terminal signature has a foreign run identity"
                )
            bindings.append(
                {
                    "cell_id": cell_id,
                    "rank": rank,
                    "path": str(native_path),
                    "raw_sha256": raw_sha256,
                    "terminal_sha256": terminal_sha256,
                    "trusted_attester_policy_sha256": policy_sha256,
                    "run_id": binding.run_id,
                    "run_nonce_sha256": binding.run_nonce_sha256,
                    "scored_request_ids": list(binding.scored_request_ids),
                }
            )
    return tuple(bindings)


def _require_exact_public_inputs(
    *,
    registry: ExperimentRegistry,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
) -> None:
    if type(registry) is not ExperimentRegistry:
        raise TypeError("selection reduction requires an exact ExperimentRegistry")
    if type(hardware_envelope) is not HardwareEnvelope:
        raise TypeError("selection reduction requires an exact HardwareEnvelope")
    if type(inventory) is not GpuInventory:
        raise TypeError("selection reduction requires an exact GpuInventory")
    _require_sha256("selection runtime", runtime_sha256)
    _require_sha256("selection split", split_sha256)
    if len(inventory.host_ids) != 1:
        raise ValueError("selection reduction requires one same-host GPU inventory")


def _load_exact_cells(
    *,
    registry: ExperimentRegistry,
    manifest: object,
    expected_manifest_type: type,
    experiment: Literal["E3a", "E1"],
    expected_cell_ids: tuple[str, ...],
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
) -> dict[str, _LoadedCell]:
    if type(manifest) is not expected_manifest_type:
        raise TypeError(f"{experiment} reduction requires its exact raw manifest")
    if manifest.schema_version != 2:
        raise ValueError(
            f"{experiment} formal reduction requires raw evidence schema 2"
        )
    references = tuple(manifest.cells)
    reference_ids = tuple(reference.cell_id for reference in references)
    if reference_ids != expected_cell_ids:
        raise ValueError(
            f"{experiment} raw evidence must exactly cover the reducer-owned cells"
        )
    cells_by_id = {cell.cell_id: cell for cell in registry.cells_for(experiment)}
    family = _ExactRunIdentity(
        experiment=experiment,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    analysis = _analysis_api()
    loaded = {
        reference.cell_id: analysis._load_cell(
            reference,
            registry=registry,
            family=family,
            cells_by_id=cells_by_id,
            envelope=hardware_envelope,
            inventory=inventory,
        )
        for reference in references
    }
    run_ids = tuple(str(row.run_rows[0]["run_id"]) for row in loaded.values())
    nonces = tuple(str(row.run_rows[0]["run_nonce_sha256"]) for row in loaded.values())
    if len(set(run_ids)) != len(run_ids) or len(set(nonces)) != len(nonces):
        raise ValueError(f"{experiment} raw evidence reuses a run identity or nonce")
    return loaded


def _metrics(
    cell: _LoadedCell, *, require_complete: bool = True
) -> tuple[_RequestMetric, ...]:
    analysis = _analysis_api()
    rows = tuple(analysis._request_metric(row) for row in cell.request_rows)
    if not rows:
        raise ValueError("selection evidence requires request rows")
    if require_complete and any(not row.completed or row.error for row in rows):
        raise ValueError("selection evidence requires complete successful requests")
    return rows


def _tokens(cell: _LoadedCell) -> dict[str, tuple[int, ...]]:
    analysis = _analysis_api()
    result: dict[str, tuple[int, ...]] = {}
    for row in cell.request_rows:
        request_id = row.get("request_id")
        if type(request_id) is not str or not request_id or request_id in result:
            raise ValueError("selection evidence has duplicate or invalid request IDs")
        result[request_id] = analysis._parse_output_token_ids(row)
    return result


def _safety_reasons(cell: _LoadedCell) -> tuple[str, ...]:
    analysis = _analysis_api()
    failures = {
        f"safety:{counter}"
        for rank_rows in cell.performance_rows_by_rank
        for performance in rank_rows
        for counter in analysis._SAFETY_COUNTERS
        if performance[counter] != 0
    }
    invalid_hardware = {
        f"hardware:{identity}"
        for identity, status, _ in cell.hardware_validity
        if status != "VALID"
    }
    return tuple(sorted(failures | invalid_hardware))


def _require_zero_safety_and_valid_hardware(cell: _LoadedCell) -> None:
    reasons = _safety_reasons(cell)
    if reasons:
        raise ValueError(
            "selection reference evidence is unsafe or has invalid hardware"
        )


def _raw_goodput(
    request_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[_RequestMetric],
) -> float:
    completed = tuple(row for row in metrics if row.completed and not row.error)
    if not completed:
        raise ValueError("selection evidence has no completed request output")
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
    if (
        len(arrivals) != len(completed_ids)
        or len(completions) != len(completed_ids)
        or any(type(value) is not int for value in (*arrivals, *completions))
    ):
        raise ValueError("selection goodput lacks exact arrival/completion timestamps")
    elapsed_ns = max(completions) - min(arrivals)
    if elapsed_ns <= 0:
        raise ValueError("selection request duration must be positive")
    goodput = sum(row.output_tokens for row in completed) / (elapsed_ns / 1e9)
    if not math.isfinite(goodput) or goodput <= 0:
        raise ValueError("selection raw goodput must be finite and positive")
    return goodput


def _paired_confidence_lower(
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
        raise ValueError("E1 confidence reduction requires exact paired requests")
    log_ratios: list[float] = []
    for request_id in sorted(numerator_by_id):
        left = numerator_by_id[request_id]
        right = denominator_by_id[request_id]
        left_rate = left.output_tokens / left.latency_ms
        right_rate = right.output_tokens / right.latency_ms
        if (
            not math.isfinite(left_rate)
            or left_rate <= 0
            or not math.isfinite(right_rate)
            or right_rate <= 0
        ):
            raise ValueError("E1 paired request rate must be finite and positive")
        log_ratios.append(math.log(left_rate / right_rate))
    mean = statistics.fmean(log_ratios)
    lower = (
        mean
        if len(log_ratios) == 1
        else mean
        - _NORMAL_95_LOWER_Z * statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
    )
    result = math.exp(lower)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("E1 confidence lower bound must be finite and positive")
    return result


def _peak_hbm(cell: _LoadedCell) -> int:
    values: list[int] = []
    for rank_rows in cell.performance_rows_by_rank:
        for row in rank_rows:
            value = row.get("peak_hbm_bytes")
            if type(value) is not int or value < 0:
                raise ValueError("E1 performance lacks exact peak HBM")
            values.append(value)
    if not values:
        raise ValueError("E1 performance lacks peak-HBM evidence")
    return max(values)


def _published_update_summary(cell: _LoadedCell) -> tuple[int, int]:
    update_rows = cell.update_rows_by_rank[0]
    statuses = tuple(row.get("candidate_status") for row in update_rows)
    if any(type(status) is not str or not status for status in statuses):
        raise ValueError("E1 update evidence lacks candidate status")
    published = tuple(
        row for row in update_rows if row.get("candidate_status") == "published"
    )
    exposed_ms: list[float] = []
    for row in published:
        value = row.get("exposed_update_ms")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("E1 published update lacks exposed-update time")
        exposed_ms.append(float(value))
    aggregate = tuple(
        row
        for row in cell.performance_rows_by_rank[0]
        if row.get("offered_requests") is not None
    )
    if len(aggregate) != 1:
        raise ValueError("E1 update reduction requires one aggregate row")
    row = aggregate[0]
    if row.get("updates_launched") != len(update_rows) or row.get(
        "updates_published"
    ) != len(published):
        raise ValueError("E1 raw update rows disagree with aggregate counters")
    aggregate_exposed = row.get("exposed_update_ms")
    if not published:
        if aggregate_exposed is not None:
            raise ValueError("E1 zero-publish evidence reports exposed-update time")
        return 0, 0
    if (
        not isinstance(aggregate_exposed, (int, float))
        or isinstance(aggregate_exposed, bool)
        or not math.isclose(
            float(aggregate_exposed),
            max(exposed_ms),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("E1 exposed-update summary differs from raw updates")
    return len(published), math.ceil(max(exposed_ms) * 1_000.0)


def _raw_run_bindings(
    loaded: Mapping[str, _LoadedCell],
    *,
    scientific_unit: str,
    lineage_runtime_sha256: str,
    lineage_split_sha256: str,
) -> tuple[object, ...]:
    analysis = _analysis_api()
    return tuple(
        analysis._loaded_cell_raw_run_binding(
            loaded[cell_id],
            scientific_unit=scientific_unit,
            lineage_runtime_sha256=lineage_runtime_sha256,
            lineage_split_sha256=lineage_split_sha256,
        )
        for cell_id in sorted(loaded)
    )


def _common_run_fields(
    loaded: Mapping[str, _LoadedCell], fields: tuple[str, ...]
) -> dict[str, object]:
    first = loaded[min(loaded)].run_rows[0]
    expected = {field: first[field] for field in fields}
    if any(
        any(row.run_rows[0][field] != value for field, value in expected.items())
        for row in loaded.values()
    ):
        raise ValueError("selection evidence crosses an immutable run identity")
    return expected


def reduce_e3a_capacity_surface_from_raw(
    *,
    registry: ExperimentRegistry,
    manifest: RawE3aSelectionEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    confirmation_data_visible: bool,
) -> E3aCapacitySurface:
    """Reduce exact E3a terminal evidence to policy-free capacity facts.

    A formal caller must source the typed registry/inventory/runtime/split from
    path-bound release authorities before invoking this reducer.  This function
    never chooses a context subset, reference load, width, crossover, or drift
    rule.
    """

    analysis = _analysis_api()
    if type(manifest) is not analysis.RawE3aSelectionEvidenceManifest:
        raise TypeError("E3a reduction requires its exact raw evidence manifest")
    if type(confirmation_data_visible) is not bool:
        raise TypeError("confirmation_data_visible must be boolean")
    if confirmation_data_visible:
        raise ValueError("E3a selection cannot inspect confirmation data")
    _require_exact_public_inputs(
        registry=registry,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    runnable = tuple(
        sorted(cell.cell_id for cell in registry.cells_for("E3a") if cell.runnable)
    )
    loaded = _load_exact_cells(
        registry=registry,
        manifest=manifest,
        expected_manifest_type=analysis.RawE3aSelectionEvidenceManifest,
        experiment="E3a",
        expected_cell_ids=runnable,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    native_bindings = _validate_native_terminal_authority(
        loaded, references=manifest.cells
    )
    _common_run_fields(
        loaded,
        (
            "model_pair",
            "sampling_profile_sha256",
            "model_lock_sha256",
            "patched_sglang_tree",
        ),
    )

    metrics: dict[str, tuple[_RequestMetric, ...]] = {}
    tokens: dict[str, dict[str, tuple[int, ...]]] = {}
    goodput: dict[str, float] = {}
    peak_hbm: dict[str, int] = {}
    target_by_slice: dict[tuple[int, str, int], _LoadedCell] = {}
    static_by_slice: dict[tuple[int, str, int, int], _LoadedCell] = {}
    for cell_id in sorted(loaded):
        row = loaded[cell_id]
        _require_zero_safety_and_valid_hardware(row)
        metrics[cell_id] = _metrics(row)
        tokens[cell_id] = _tokens(row)
        goodput[cell_id] = _raw_goodput(row.request_rows, metrics[cell_id])
        peak_hbm[cell_id] = _peak_hbm(row)
        identity = row.cell.identity
        if identity.context is None or identity.concurrency is None:
            raise ValueError("E3a registry cell lacks context/load identity")
        if identity.method == "target_only":
            if identity.width is not None:
                raise ValueError("E3a Target-only cell unexpectedly carries width")
            key = (identity.context, identity.regime, identity.concurrency)
            if key in target_by_slice:
                raise ValueError("E3a repeats a Target-only comparison slice")
            target_by_slice[key] = row
        elif identity.method == "static":
            if identity.width not in DRAFT_WIDTHS:
                raise ValueError("E3a Static cell lacks a registered width")
            key = (
                identity.context,
                identity.regime,
                identity.concurrency,
                identity.width,
            )
            if key in static_by_slice:
                raise ValueError("E3a repeats a Static comparison slice")
            static_by_slice[key] = row
        else:
            raise ValueError("E3a selection accepts only Target-only and Static")

    for key, static in static_by_slice.items():
        target = target_by_slice.get(key[:3])
        if target is None:
            raise ValueError("E3a Static evidence lacks its exact Target-only pair")
        paired_fields = (
            "corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "sampling_profile_sha256",
            "model_lock_sha256",
        )
        if any(
            static.run_rows[0][field] != target.run_rows[0][field]
            for field in paired_fields
        ):
            raise ValueError("E3a Static/Target evidence is not exactly paired")
        if tokens[static.cell.cell_id] != tokens[target.cell.cell_id]:
            raise ValueError("E3a Static differs from Target-only token trajectories")
    if any(
        not any(static_key[:3] == target_key for static_key in static_by_slice)
        for target_key in target_by_slice
    ):
        raise ValueError("E3a Target-only evidence lacks a Static comparison")

    run_bindings = _raw_run_bindings(
        loaded,
        scientific_unit="e3a_capacity",
        lineage_runtime_sha256=runtime_sha256,
        lineage_split_sha256=split_sha256,
    )
    run_binding_by_cell = {
        cell_id: binding
        for cell_id, binding in zip(sorted(loaded), run_bindings, strict=True)
    }
    if any(
        binding.cell_id != cell_id
        or binding.rank_count != len(loaded[cell_id].run_rows)
        for cell_id, binding in run_binding_by_cell.items()
    ):
        raise ValueError("E3a raw-run lineage differs from exact cell/rank coverage")
    observations: list[E3aCapacityObservation] = []
    for cell_id in sorted(loaded):
        row = loaded[cell_id]
        identity = row.cell.identity
        ratio = None
        if identity.method == "static":
            target = target_by_slice[
                (identity.context, identity.regime, identity.concurrency)
            ]
            ratio = goodput[cell_id] / goodput[target.cell.cell_id]
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("E3a Static/Target goodput ratio is invalid")
        observations.append(
            E3aCapacityObservation(
                cell_id=cell_id,
                method=identity.method,
                context=identity.context,
                regime=identity.regime,
                concurrency=identity.concurrency,
                width=identity.width,
                raw_goodput_tps=goodput[cell_id],
                peak_hbm_bytes=peak_hbm[cell_id],
                static_target_goodput_ratio=ratio,
                terminal_rank_count=len(row.run_rows),
                raw_run_binding_sha256=run_binding_by_cell[cell_id].sha256,
            )
        )
    native_lineage: list[E3aNativeTerminalLineage] = []
    for binding in native_bindings:
        cell_id = binding.get("cell_id")
        rank = binding.get("rank")
        if type(cell_id) is not str or type(rank) is not int:
            raise ValueError("E3a native-terminal binding lost cell/rank identity")
        native_lineage.append(
            E3aNativeTerminalLineage(
                cell_id=cell_id,
                rank=rank,
                binding_sha256=content_sha256(binding),
            )
        )
    return E3aCapacitySurface(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        hardware_envelope_sha256=content_sha256(hardware_envelope),
        raw_manifest_sha256=manifest.sha256,
        observations=tuple(observations),
        native_terminal_lineage=tuple(native_lineage),
    )


def _select_e3a_capacity_surface(
    surface: E3aCapacitySurface,
    policy: E3aScientificSelectionPolicy,
) -> SealedE3aSelection:
    """Apply only the exact source-owned typed policy to one capacity surface."""

    if type(surface) is not E3aCapacitySurface:
        raise TypeError("E3a selection requires an exact capacity surface")
    if type(policy) is not E3aScientificSelectionPolicy:
        raise TypeError("E3a selection requires its source-owned typed policy")
    observed_contexts = {
        row.context for row in surface.observations if row.method == "static"
    }
    if not set(policy.primary_contexts) <= observed_contexts:
        raise ValueError("E3a policy names a context absent from the capacity surface")
    primary_static = tuple(
        row
        for row in surface.observations
        if row.method == "static" and row.context in policy.primary_contexts
    )
    if not primary_static:
        raise ValueError("E3a has no registered primary-context Static evidence")
    by_width_load: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in primary_static:
        if row.width is None:  # pragma: no cover - guarded by the typed row
            raise ValueError("E3a Static capacity row lost its width")
        by_width_load[(row.width, row.concurrency)].append(row.raw_goodput_tps)
    median_by_width_load = {
        key: statistics.median(values) for key, values in by_width_load.items()
    }
    by_load: dict[int, list[float]] = defaultdict(list)
    for (_, concurrency), median in median_by_width_load.items():
        by_load[concurrency].append(median)
    best_median_by_load = {
        concurrency: max(values) for concurrency, values in by_load.items()
    }
    maximum = max(best_median_by_load.values())
    threshold = float(policy.reference_load_goodput_fraction) * maximum
    eligible_loads = tuple(
        concurrency
        for concurrency, value in sorted(best_median_by_load.items())
        if value >= threshold
    )
    if not eligible_loads:
        raise ValueError("E3a reference-load reducer has no eligible concurrency")
    selected_concurrency = min(eligible_loads)

    width_scores: list[tuple[float, float, int]] = []
    for width in DRAFT_WIDTHS:
        rows = tuple(
            row
            for row in primary_static
            if row.concurrency == selected_concurrency and row.width == width
        )
        if not rows:
            continue
        ratios = tuple(row.static_target_goodput_ratio for row in rows)
        static_goodputs = tuple(row.raw_goodput_tps for row in rows)
        if any(
            value is None or not math.isfinite(value) or value <= 0 for value in ratios
        ):
            raise ValueError("E3a Static/Target goodput ratio is invalid")
        width_scores.append(
            (
                min(value for value in ratios if value is not None),
                statistics.median(static_goodputs),
                width,
            )
        )
    if not width_scores:
        raise ValueError("E3a selected load has no registered width evidence")
    _, _, selected_width = max(
        width_scores,
        key=lambda row: (row[0], row[1], -row[2]),
    )

    evidence_sha256 = content_sha256(
        {
            "schema_version": 2,
            "kind": "e3a_scientific_selection_reduction_evidence",
            "authority_protocol_sha256": (
                E3A_SCIENTIFIC_SELECTION_AUTHORITY_PROTOCOL_SHA256
            ),
            "capacity_surface_sha256": surface.sha256,
            "selection_policy_sha256": policy.sha256,
            "median_static_goodput_by_width_load": [
                {
                    "width": width,
                    "concurrency": concurrency,
                    "median_goodput_tps": median_by_width_load[(width, concurrency)],
                }
                for width, concurrency in sorted(median_by_width_load)
            ],
            "width_scores_at_selected_load": [
                {
                    "width": width,
                    "worst_static_target_goodput_ratio": worst,
                    "median_static_goodput_tps": median,
                }
                for worst, median, width in sorted(width_scores, key=lambda row: row[2])
            ],
            "selected_width": selected_width,
            "selected_concurrency": selected_concurrency,
        }
    )
    return SealedE3aSelection(
        schema_version=1,
        registry_sha256=surface.registry_sha256,
        runtime_sha256=surface.runtime_sha256,
        split_sha256=surface.split_sha256,
        width=selected_width,
        concurrency=selected_concurrency,
        reducer_evidence_sha256=evidence_sha256,
    )


def reduce_e3a_selection_from_raw(
    *,
    registry: ExperimentRegistry,
    manifest: RawE3aSelectionEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
    confirmation_data_visible: bool,
) -> SealedE3aSelection:
    """Apply the source-owned E3a policy to a separate raw capacity surface."""

    # Policy registration is checked before reopening the large evidence set.
    # Callers that only need descriptive capacity facts use the surface reducer.
    policy = _require_e3a_scientific_selection_policy()
    surface = reduce_e3a_capacity_surface_from_raw(
        registry=registry,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        confirmation_data_visible=confirmation_data_visible,
    )
    return _select_e3a_capacity_surface(surface, policy)


@dataclass(frozen=True)
class _E1GeometryEvaluation:
    geometry: E1GeometryIdentity
    confidence_lower_goodput_ratio: float
    peak_hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int
    cell_ids: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "geometry_sha256": self.geometry.sha256,
                "confidence_lower_goodput_ratio": (self.confidence_lower_goodput_ratio),
                "peak_hbm_bytes": self.peak_hbm_bytes,
                "p99_itl_us": self.p99_itl_us,
                "exposed_update_us": self.exposed_update_us,
                "cell_ids": self.cell_ids,
            }
        )


def _non_dominated_e1(
    rows: Sequence[_E1GeometryEvaluation],
) -> tuple[_E1GeometryEvaluation, ...]:
    def dominates(left: _E1GeometryEvaluation, right: _E1GeometryEvaluation) -> bool:
        weak = (
            left.confidence_lower_goodput_ratio >= right.confidence_lower_goodput_ratio
            and left.peak_hbm_bytes <= right.peak_hbm_bytes
            and left.p99_itl_us <= right.p99_itl_us
            and left.exposed_update_us <= right.exposed_update_us
        )
        strict = (
            left.confidence_lower_goodput_ratio > right.confidence_lower_goodput_ratio
            or left.peak_hbm_bytes < right.peak_hbm_bytes
            or left.p99_itl_us < right.p99_itl_us
            or left.exposed_update_us < right.exposed_update_us
        )
        return weak and strict

    return tuple(
        row
        for row in sorted(rows, key=lambda item: item.geometry.sha256)
        if not any(
            other.geometry.sha256 != row.geometry.sha256 and dominates(other, row)
            for other in rows
        )
    )


def reduce_e1_pareto_from_raw(
    *,
    registry: ExperimentRegistry,
    activation: ReducerActivationArtifact,
    e3a_receipt: ExperimentReceipt,
    e3a_selection: SealedE3aSelection,
    source_activation_authority_sha256: str,
    manifest: RawE1ParetoEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    confirmation_data_visible: bool,
) -> E1ParetoArtifact:
    """Recompute E1's safe four-objective Pareto set from the exact 130 cells.

    Formal callers must first replay the path-bound E3a selection and E3a stage
    completion authorities.  ``source_activation_authority_sha256`` binds that
    replay into this reduction; a typed E3a selection or receipt alone is not
    accepted as formal authority by downstream integration.
    """

    analysis = _analysis_api()
    if type(manifest) is not analysis.RawE1ParetoEvidenceManifest:
        raise TypeError("E1 reduction requires its exact raw evidence manifest")
    if type(confirmation_data_visible) is not bool:
        raise TypeError("confirmation_data_visible must be boolean")
    if confirmation_data_visible:
        raise ValueError("E1 Pareto selection cannot inspect downstream data")
    if type(activation) is not ReducerActivationArtifact:
        raise TypeError("E1 reduction requires an exact reducer activation")
    if type(e3a_receipt) is not ExperimentReceipt:
        raise TypeError("E1 reduction requires an exact E3a receipt")
    if type(e3a_selection) is not SealedE3aSelection:
        raise TypeError("E1 reduction requires an exact E3a selection")
    _require_sha256(
        "E1 source activation authority", source_activation_authority_sha256
    )
    expected_activation = reduce_e1_activation(
        registry,
        e3a_receipt=e3a_receipt,
        selection=e3a_selection,
    )
    if activation != expected_activation:
        raise ValueError("E1 activation differs from raw-authority replay")
    runtime_sha256 = activation.plan.runtime_sha256
    split_sha256 = activation.plan.split_sha256
    _require_exact_public_inputs(
        registry=registry,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    expected_ids = activation.plan.activated_cell_ids
    if len(expected_ids) != _E1_ACTIVATED_CELL_COUNT:
        raise ValueError("E1 activation must contain exactly 130 cells")
    loaded = _load_exact_cells(
        registry=registry,
        manifest=manifest,
        expected_manifest_type=analysis.RawE1ParetoEvidenceManifest,
        experiment="E1",
        expected_cell_ids=expected_ids,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    native_bindings = _validate_native_terminal_authority(
        loaded, references=manifest.cells
    )
    common = _common_run_fields(
        loaded,
        (
            "model_pair",
            "corpus_sha256",
            "arrival_trace_sha256",
            "request_ids_sha256",
            "sampling_profile_sha256",
            "model_lock_sha256",
            "patched_sglang_tree",
        ),
    )
    concurrencies = {row.cell.identity.concurrency for row in loaded.values()}
    widths_valid = all(
        row.cell.identity.width
        == (None if row.cell.identity.method == "target_only" else e3a_selection.width)
        for row in loaded.values()
    )
    if not widths_valid or concurrencies != {e3a_selection.concurrency}:
        raise ValueError("E1 raw evidence differs from the E3a width/load lock")

    by_method: dict[str, list[_LoadedCell]] = defaultdict(list)
    metrics: dict[str, tuple[_RequestMetric, ...]] = {}
    tokens: dict[str, dict[str, tuple[int, ...]]] = {}
    for cell_id in sorted(loaded):
        row = loaded[cell_id]
        metrics[cell_id] = _metrics(row, require_complete=False)
        tokens[cell_id] = _tokens(row)
        by_method[row.cell.identity.method].append(row)
    if len(by_method["target_only"]) != 1 or len(by_method["static"]) != 1:
        raise ValueError("E1 requires one Target-only and one Static reference")
    if set(by_method) != {"target_only", "static", "tts", "l0"}:
        raise ValueError("E1 evidence has an unsupported method")
    target = by_method["target_only"][0]
    static = by_method["static"][0]
    _require_zero_safety_and_valid_hardware(target)
    _require_zero_safety_and_valid_hardware(static)
    if any(
        not row.completed or row.error
        for row in (
            *metrics[target.cell.cell_id],
            *metrics[static.cell.cell_id],
        )
    ):
        raise ValueError("E1 reference evidence requires complete requests")
    target_tokens = tokens[target.cell.cell_id]
    if tokens[static.cell.cell_id] != target_tokens:
        raise ValueError("E1 Static differs from Target-only token trajectories")
    static_metrics = metrics[static.cell.cell_id]

    grouped: dict[
        str, tuple[E1GeometryIdentity, dict[tuple[str, str], _LoadedCell]]
    ] = {}
    for method in ("tts", "l0"):
        for row in by_method[method]:
            geometry = E1GeometryIdentity.from_cell(row.cell)
            current = grouped.setdefault(geometry.sha256, (geometry, {}))
            key = (str(row.cell.identity.optimizer), method)
            if key in current[1]:
                raise ValueError("E1 repeats one geometry/optimizer/method cell")
            current[1][key] = row
    required_pairs = {
        (optimizer, method)
        for optimizer in E1_OPTIMIZER_ANCHORS
        for method in ("tts", "l0")
    }
    if len(grouped) != 32 or any(
        set(cells) != required_pairs for _, cells in grouped.values()
    ):
        raise ValueError(
            "E1 requires 32 geometries with TTS/L0 at both optimizer anchors"
        )

    evaluations: list[_E1GeometryEvaluation] = []
    geometry_dispositions: list[dict[str, object]] = []
    for geometry_sha256 in sorted(grouped):
        geometry, cells = grouped[geometry_sha256]
        confidence: list[float] = []
        hbm: list[int] = []
        p99_itl_us: list[int] = []
        exposed_update_us: list[int] = []
        reasons: set[str] = set()
        for key in sorted(cells):
            row = cells[key]
            cell_id = row.cell.cell_id
            label = f"{key[0]}:{key[1]}"
            reasons.update(f"{label}:{reason}" for reason in _safety_reasons(row))
            cell_metrics = metrics[cell_id]
            if any(not metric.completed or metric.error for metric in cell_metrics):
                reasons.add(f"{label}:incomplete_request")
            if tokens[cell_id] != target_tokens:
                reasons.add(f"{label}:target_token_mismatch")
            hbm.append(_peak_hbm(row))
            completed_p99 = tuple(
                metric.within_request_p99_itl_ms
                for metric in cell_metrics
                if metric.completed and not metric.error
            )
            if not completed_p99 or any(value is None for value in completed_p99):
                reasons.add(f"{label}:incomplete_itl_timing")
            else:
                p99_itl_us.append(
                    math.ceil(max(float(value) for value in completed_p99) * 1_000.0)
                )
            published, exposed = _published_update_summary(row)
            if published < 1:
                reasons.add(f"{label}:no_published_update")
            exposed_update_us.append(exposed)
            if not reasons:
                confidence.append(
                    _paired_confidence_lower(cell_metrics, static_metrics)
                )
        cell_ids = tuple(sorted(row.cell.cell_id for row in cells.values()))
        geometry_dispositions.append(
            {
                "geometry_sha256": geometry.sha256,
                "cell_ids": list(cell_ids),
                "status": "SAFE" if not reasons else "EXCLUDED_UNSAFE",
                "reason_codes": sorted(reasons),
            }
        )
        if reasons:
            continue
        evaluations.append(
            _E1GeometryEvaluation(
                geometry=geometry,
                confidence_lower_goodput_ratio=min(confidence),
                peak_hbm_bytes=max(hbm),
                p99_itl_us=max(p99_itl_us),
                exposed_update_us=max(exposed_update_us),
                cell_ids=cell_ids,
            )
        )
    survivors = _non_dominated_e1(evaluations)
    if not survivors:
        raise ValueError("E1 has no safe non-dominated geometry")

    common_load_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e1_common_downstream_load",
            "width": e3a_selection.width,
            "concurrency": e3a_selection.concurrency,
            "runtime_sha256": runtime_sha256,
            "split_sha256": split_sha256,
            "corpus_sha256": common["corpus_sha256"],
            "arrival_trace_sha256": common["arrival_trace_sha256"],
            "request_ids_sha256": common["request_ids_sha256"],
            "sampling_profile_sha256": common["sampling_profile_sha256"],
            "model_lock_sha256": common["model_lock_sha256"],
        }
    )
    run_bindings = _raw_run_bindings(
        loaded,
        scientific_unit="e1_geometry_screen",
        lineage_runtime_sha256=runtime_sha256,
        lineage_split_sha256=split_sha256,
    )
    reducer_evidence_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e1_raw_pareto_reduction_evidence",
            "reducer_protocol_sha256": E1_RAW_PARETO_PROTOCOL_SHA256,
            "registry_sha256": registry.sha256,
            "runtime_sha256": runtime_sha256,
            "split_sha256": split_sha256,
            "e3a_receipt_sha256": e3a_receipt.sha256,
            "e3a_selection_sha256": e3a_selection.sha256,
            "source_activation_authority_sha256": (source_activation_authority_sha256),
            "activation_sha256": activation.sha256,
            "inventory_sha256": inventory.sha256,
            "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
            "fixed_instance_gpu_count": len(inventory.devices),
            "hardware_envelope_sha256": content_sha256(hardware_envelope),
            "raw_manifest_sha256": manifest.sha256,
            "common_load_sha256": common_load_sha256,
            "geometry_dispositions": geometry_dispositions,
            "geometry_evaluation_sha256s": [row.sha256 for row in evaluations],
            "surviving_geometry_sha256s": [row.geometry.sha256 for row in survivors],
            "raw_run_binding_sha256s": [row.sha256 for row in run_bindings],
            "native_terminal_bindings": list(native_bindings),
        }
    )
    return E1ParetoArtifact(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        e1_activation_sha256=activation.sha256,
        reducer_evidence_sha256=reducer_evidence_sha256,
        common_load_sha256=common_load_sha256,
        surviving_geometries=tuple(row.geometry for row in survivors),
        selection_state="sealed_before_e2_unblinding",
    )


@dataclass(frozen=True)
class E3aSelectionReductionAuthority:
    """Path-bearing replay handle for one exact E3a selection."""

    schema_version: int
    registry: ExperimentRegistry
    manifest: RawE3aSelectionEvidenceManifest
    hardware_envelope: HardwareEnvelope
    inventory: GpuInventory
    runtime_sha256: str
    split_sha256: str
    selection_policy_sha256: str
    selection_sha256: str

    def __post_init__(self) -> None:
        analysis = _analysis_api()
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E3a selection authority schema 1 is supported")
        if type(self.manifest) is not analysis.RawE3aSelectionEvidenceManifest:
            raise TypeError("E3a authority requires its exact raw manifest")
        _require_exact_public_inputs(
            registry=self.registry,
            hardware_envelope=self.hardware_envelope,
            inventory=self.inventory,
            runtime_sha256=self.runtime_sha256,
            split_sha256=self.split_sha256,
        )
        _require_sha256("E3a authority policy", self.selection_policy_sha256)
        _require_sha256("E3a authority selection", self.selection_sha256)

    def revalidate(self) -> SealedE3aSelection:
        policy = _require_e3a_scientific_selection_policy()
        if policy.sha256 != self.selection_policy_sha256:
            raise RuntimeError("E3a source-owned selection policy changed")
        _analysis_api().validate_raw_evidence_manifest_sidecars(self.manifest)
        selection = reduce_e3a_selection_from_raw(
            registry=self.registry,
            manifest=self.manifest,
            hardware_envelope=self.hardware_envelope,
            inventory=self.inventory,
            runtime_sha256=self.runtime_sha256,
            split_sha256=self.split_sha256,
            confirmation_data_visible=False,
        )
        if selection.sha256 != self.selection_sha256:
            raise RuntimeError("E3a selection authority changed after binding")
        return selection

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": "e3a_selection_reduction_authority",
                "registry_sha256": self.registry.sha256,
                "raw_manifest_sha256": self.manifest.sha256,
                "hardware_envelope_sha256": content_sha256(self.hardware_envelope),
                "inventory_sha256": self.inventory.sha256,
                "inventory_source_receipt_sha256": (
                    self.inventory.source_receipt_sha256
                ),
                "runtime_sha256": self.runtime_sha256,
                "split_sha256": self.split_sha256,
                "selection_policy_sha256": self.selection_policy_sha256,
                "selection_sha256": self.selection_sha256,
                "reducer_protocol_sha256": (
                    E3A_SCIENTIFIC_SELECTION_AUTHORITY_PROTOCOL_SHA256
                ),
            }
        )


@dataclass(frozen=True)
class E1ParetoReductionAuthority:
    """Path-bearing replay handle for one exact E1 Pareto reduction."""

    schema_version: int
    registry: ExperimentRegistry
    activation: ReducerActivationArtifact
    e3a_receipt: ExperimentReceipt
    e3a_selection: SealedE3aSelection
    source_activation_authority_sha256: str
    manifest: RawE1ParetoEvidenceManifest
    hardware_envelope: HardwareEnvelope
    inventory: GpuInventory
    pareto_sha256: str

    def __post_init__(self) -> None:
        analysis = _analysis_api()
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E1 Pareto authority schema 1 is supported")
        if type(self.activation) is not ReducerActivationArtifact:
            raise TypeError("E1 authority requires an exact reducer activation")
        if type(self.e3a_receipt) is not ExperimentReceipt:
            raise TypeError("E1 authority requires an exact E3a receipt")
        if type(self.e3a_selection) is not SealedE3aSelection:
            raise TypeError("E1 authority requires an exact E3a selection")
        if type(self.manifest) is not analysis.RawE1ParetoEvidenceManifest:
            raise TypeError("E1 authority requires its exact raw manifest")
        _require_exact_public_inputs(
            registry=self.registry,
            hardware_envelope=self.hardware_envelope,
            inventory=self.inventory,
            runtime_sha256=self.activation.plan.runtime_sha256,
            split_sha256=self.activation.plan.split_sha256,
        )
        _require_sha256(
            "E1 authority source activation", self.source_activation_authority_sha256
        )
        _require_sha256("E1 authority Pareto", self.pareto_sha256)

    def revalidate(self) -> E1ParetoArtifact:
        _analysis_api().validate_raw_evidence_manifest_sidecars(self.manifest)
        artifact = reduce_e1_pareto_from_raw(
            registry=self.registry,
            activation=self.activation,
            e3a_receipt=self.e3a_receipt,
            e3a_selection=self.e3a_selection,
            source_activation_authority_sha256=(
                self.source_activation_authority_sha256
            ),
            manifest=self.manifest,
            hardware_envelope=self.hardware_envelope,
            inventory=self.inventory,
            confirmation_data_visible=False,
        )
        if artifact.sha256 != self.pareto_sha256:
            raise RuntimeError("E1 Pareto authority changed after binding")
        return artifact

    @property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": "e1_pareto_reduction_authority",
                "registry_sha256": self.registry.sha256,
                "activation_sha256": self.activation.sha256,
                "e3a_receipt_sha256": self.e3a_receipt.sha256,
                "e3a_selection_sha256": self.e3a_selection.sha256,
                "source_activation_authority_sha256": (
                    self.source_activation_authority_sha256
                ),
                "raw_manifest_sha256": self.manifest.sha256,
                "hardware_envelope_sha256": content_sha256(self.hardware_envelope),
                "inventory_sha256": self.inventory.sha256,
                "inventory_source_receipt_sha256": (
                    self.inventory.source_receipt_sha256
                ),
                "pareto_sha256": self.pareto_sha256,
                "reducer_protocol_sha256": E1_RAW_PARETO_PROTOCOL_SHA256,
            }
        )


def bind_e3a_selection_reduction_authority(
    *,
    registry: ExperimentRegistry,
    manifest: RawE3aSelectionEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    runtime_sha256: str,
    split_sha256: str,
) -> E3aSelectionReductionAuthority:
    policy = _require_e3a_scientific_selection_policy()
    _analysis_api().validate_raw_evidence_manifest_sidecars(manifest)
    selection = reduce_e3a_selection_from_raw(
        registry=registry,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        confirmation_data_visible=False,
    )
    authority = E3aSelectionReductionAuthority(
        schema_version=1,
        registry=registry,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        selection_policy_sha256=policy.sha256,
        selection_sha256=selection.sha256,
    )
    authority.revalidate()
    return authority


def bind_e1_pareto_reduction_authority(
    *,
    registry: ExperimentRegistry,
    activation: ReducerActivationArtifact,
    e3a_receipt: ExperimentReceipt,
    e3a_selection: SealedE3aSelection,
    source_activation_authority_sha256: str,
    manifest: RawE1ParetoEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
) -> E1ParetoReductionAuthority:
    _analysis_api().validate_raw_evidence_manifest_sidecars(manifest)
    artifact = reduce_e1_pareto_from_raw(
        registry=registry,
        activation=activation,
        e3a_receipt=e3a_receipt,
        e3a_selection=e3a_selection,
        source_activation_authority_sha256=source_activation_authority_sha256,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    authority = E1ParetoReductionAuthority(
        schema_version=1,
        registry=registry,
        activation=activation,
        e3a_receipt=e3a_receipt,
        e3a_selection=e3a_selection,
        source_activation_authority_sha256=source_activation_authority_sha256,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        pareto_sha256=artifact.sha256,
    )
    authority.revalidate()
    return authority


__all__ = [
    "E3A_CAPACITY_SURFACE_PROTOCOL_SHA256",
    "E3A_LOCKED_OUTPUT_REDUCTION_UNREGISTERED_REASON",
    "E3A_SCIENTIFIC_SELECTION_AUTHORITY_PROTOCOL_SHA256",
    "E3A_SELECTION_POLICY_UNREGISTERED_REASON",
    "RELEASE_E3A_SCIENTIFIC_SELECTION_POLICY",
    "E1ParetoReductionAuthority",
    "E3aCapacityObservation",
    "E3aCapacitySurface",
    "E3aNativeTerminalLineage",
    "E3aScientificSelectionPolicy",
    "E3aSelectionReductionAuthority",
    "SelectionReductionAuthorityUnavailableError",
    "bind_e1_pareto_reduction_authority",
    "bind_e3a_selection_reduction_authority",
    "reduce_e1_pareto_from_raw",
    "reduce_e3a_capacity_surface_from_raw",
    "reduce_e3a_selection_from_raw",
    "require_e3a_locked_output_reduction_authority",
]
