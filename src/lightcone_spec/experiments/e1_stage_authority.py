"""Staged-native E1 Pareto authority over exact controlled execution proofs.

The historical E1 reducer consumes an eager :class:`ExperimentRegistry` and
therefore cannot authorize the signed-staged protocol.  This module is the
formal replacement: its universe is the immutable 68-cell E1 materialization,
its terminal inputs are 68 durable release-controlled proof pairs, and its
67-row selection set excludes only the single L0-naive mechanism anchor.

No metric summary is accepted from a caller.  Request completion, token
trajectories, latency, ITL, safety, HBM, publication, and exposed-update timing
are rebuilt after deep-opening the first-party result and timestamp proofs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.itl_authority import (
    StageItlExecutionIdentity,
    StageItlTimestampAuthority,
)
from lightcone_spec.experiments.stage_materialization import (
    E1_OPTIMIZER_ANCHORS,
    E1Geometry,
    MaterializedCell,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.orchestration.formal_serving_lift import (
    validate_formal_serving_itl_proof,
)
from lightcone_spec.orchestration.formal_terminal_result import (
    FormalDistributedTerminalRequestResult,
    FormalDistributedTerminalResultProjection,
    validate_formal_terminal_result_proof_artifact,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalRequestResult,
    NativeTerminalResultProjection,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

E1_STAGED_PARETO_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_e1_staged_native_pareto_protocol",
        "universe": "exact_signed_68_cell_materialization",
        "execution_proof_rows": 68,
        "selection_rows": "67_cells_excluding_only_l0_naive_anchor",
        "references": ["Target-only", "Static", "TTS"],
        "candidate_grid": "32_geometries_x_2_optimizer_anchors",
        "proofs": [
            "durable_external_control_native_terminal_result",
            "admission_bound_formal_serving_itl_wrapper",
        ],
        "request_join": "exact_request_id_input_and_output_token_trajectories",
        "safety": "all_registered_native_safety_counters_zero",
        "publication": "tts_and_each_lightcone_candidate_publish_at_least_one_update",
        "paired_confidence": (
            "per_request_output_token_rate_log_ratio_lower_bound_against_"
            "static_and_frozen_tts"
        ),
        "pareto_dimensions": [
            "minimum_paired_95pct_lower_request_rate_ratio",
            "peak_hbm_bytes",
            "within_request_p99_itl_us",
            "exposed_update_us",
        ],
        "confirmation_data_visible": False,
    }
)

_NORMAL_95_LOWER_Z = 1.6448536269514722
_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
)


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_absolute_path(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"{label} must be absolute and resolved")
    return value


def _runtime_method(cell: MaterializedCell) -> str:
    try:
        return {
            "Target-only": "target_only",
            "Static": "static",
            "TTS": "tts",
            "L0-naive": "l0",
            "LightCone-candidate": "l0",
            "LightCone": "l0",
            "OnlineSPEC-OGD": "onlinespec_ogd",
            "OnlineSPEC-OPT": "onlinespec_opt",
            "OnlineSPEC-ENS": "onlinespec_ens",
            "OnlineSPEC-Optimistic-OGD": "onlinespec_opt",
            "OnlineSPEC-Hedge": "onlinespec_ens",
            "OnlineSPEC-OGD-candidate": "onlinespec_ogd",
            "OnlineSPEC-OPT-candidate": "onlinespec_opt",
            "OnlineSPEC-ENS-candidate": "onlinespec_ens",
            "OnlineSPEC-Optimistic-OGD-candidate": "onlinespec_opt",
            "OnlineSPEC-Hedge-candidate": "onlinespec_ens",
        }[cell.method_role]
    except KeyError as error:
        raise ValueError("E1 materialization contains an unsupported role") from error


@dataclass(frozen=True)
class E1CellExecutionEvidence:
    """Path-bound result and timing proofs for one materialized E1 cell."""

    schema_version: int
    materialized_cell_id: str
    execution_binding_sha256: str
    execution_identity: StageItlExecutionIdentity
    native_result_proof_path: str
    native_result_proof_raw_sha256: str
    native_result_proof_semantic_sha256: str
    stage_itl_proof_path: str
    stage_itl_proof_raw_sha256: str
    stage_itl_proof_semantic_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only E1 cell execution evidence schema 2 is supported")
        _require_sha256("E1 evidence materialized cell", self.materialized_cell_id)
        _require_sha256("E1 evidence execution binding", self.execution_binding_sha256)
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("E1 evidence requires an exact execution identity")
        self.execution_identity.__post_init__()
        if self.execution_identity.materialized_cell_id != self.materialized_cell_id:
            raise ValueError("E1 execution identity names another materialized cell")
        for label, path_value, raw_sha256, semantic_sha256 in (
            (
                "E1 native result proof",
                self.native_result_proof_path,
                self.native_result_proof_raw_sha256,
                self.native_result_proof_semantic_sha256,
            ),
            (
                "E1 stage ITL proof",
                self.stage_itl_proof_path,
                self.stage_itl_proof_raw_sha256,
                self.stage_itl_proof_semantic_sha256,
            ),
        ):
            _require_absolute_path(label, path_value)
            _require_sha256(f"{label} raw", raw_sha256)
            _require_sha256(f"{label} semantic", semantic_sha256)
            observed = CanonicalJsonProofBinding.bind(path_value)
            if (
                observed.raw_sha256 != raw_sha256
                or observed.semantic_sha256 != semantic_sha256
            ):
                raise ValueError(f"{label} content changed after binding")
        if self.native_result_proof_path == self.stage_itl_proof_path:
            raise ValueError("E1 result and timing proofs must be distinct artifacts")

    @classmethod
    def bind(
        cls,
        *,
        execution_binding: VerifiedFormalServingExecutionBinding,
        native_result_proof_path: str,
        stage_itl_proof_path: str,
    ) -> E1CellExecutionEvidence:
        verified = require_verified_formal_serving_execution_binding(execution_binding)
        if verified.subject.stage != "E1":
            raise ValueError("E1 evidence cannot consume another stage binding")
        result = CanonicalJsonProofBinding.bind(native_result_proof_path)
        timing = CanonicalJsonProofBinding.bind(stage_itl_proof_path)
        return cls(
            schema_version=2,
            materialized_cell_id=verified.subject.materialized_cell_id,
            execution_binding_sha256=verified.sha256,
            execution_identity=verified.subject.execution_identity,
            native_result_proof_path=result.absolute_path,
            native_result_proof_raw_sha256=result.raw_sha256,
            native_result_proof_semantic_sha256=result.semantic_sha256,
            stage_itl_proof_path=timing.absolute_path,
            stage_itl_proof_raw_sha256=timing.raw_sha256,
            stage_itl_proof_semantic_sha256=timing.semantic_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E1StagedParetoEvidenceManifest:
    """Exact 68-row proof manifest for the staged-native E1 reducer."""

    schema_version: int
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    e3a_selection_receipt_sha256: str
    inventory_sha256: str
    cells: tuple[E1CellExecutionEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E1 staged Pareto manifest schema 1 is supported")
        for label, digest in (
            ("E1 manifest protocol lock", self.protocol_lock_sha256),
            ("E1 manifest materialization", self.materialization_receipt_sha256),
            ("E1 manifest coverage", self.coverage_receipt_sha256),
            ("E1 manifest E3a selection", self.e3a_selection_receipt_sha256),
            ("E1 manifest inventory", self.inventory_sha256),
        ):
            _require_sha256(label, digest)
        if (
            type(self.cells) is not tuple
            or len(self.cells) != 68
            or any(type(row) is not E1CellExecutionEvidence for row in self.cells)
            or tuple(row.materialized_cell_id for row in self.cells)
            != tuple(sorted({row.materialized_cell_id for row in self.cells}))
        ):
            raise ValueError("E1 proof manifest must contain exactly 68 sorted cells")
        for row in self.cells:
            row.__post_init__()
        run_identities = tuple(
            (
                row.execution_identity.run_id,
                row.execution_identity.run_nonce_sha256,
                row.execution_identity.attempt_id,
            )
            for row in self.cells
        )
        if len(run_identities) != len(set(run_identities)):
            raise ValueError("E1 proof manifest reuses a run identity")
        result_proofs = tuple(row.native_result_proof_raw_sha256 for row in self.cells)
        timing_proofs = tuple(row.stage_itl_proof_raw_sha256 for row in self.cells)
        execution_bindings = tuple(row.execution_binding_sha256 for row in self.cells)
        if len(result_proofs) != len(set(result_proofs)):
            raise ValueError("E1 proof manifest reuses a terminal result proof")
        if len(timing_proofs) != len(set(timing_proofs)):
            raise ValueError("E1 proof manifest reuses an ITL proof")
        if len(execution_bindings) != len(set(execution_bindings)):
            raise ValueError("E1 proof manifest reuses an execution binding")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def e1_staged_pareto_evidence_manifest_to_dict(
    value: E1StagedParetoEvidenceManifest,
) -> dict[str, object]:
    """Encode the path-bearing 68-cell manifest without weakening its types."""

    if type(value) is not E1StagedParetoEvidenceManifest:
        raise TypeError("E1 Pareto evidence codec requires an exact manifest")
    value.__post_init__()
    row = asdict(value)
    row["cells"] = [asdict(cell) for cell in value.cells]
    return {**row, "manifest_sha256": value.sha256}


def e1_staged_pareto_evidence_manifest_from_dict(
    value: object,
) -> E1StagedParetoEvidenceManifest:
    """Strictly decode one durable staged-native E1 evidence manifest."""

    fields = {*E1StagedParetoEvidenceManifest.__dataclass_fields__, "manifest_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("E1 Pareto evidence manifest fields differ")
    row = dict(value)
    declared = row.pop("manifest_sha256")
    _require_sha256("E1 Pareto evidence manifest", declared)
    raw_cells = row["cells"]
    if type(raw_cells) is not list:
        raise TypeError("E1 Pareto evidence cells must be an array")
    cells = []
    for raw_cell in raw_cells:
        if type(raw_cell) is not dict or set(raw_cell) != set(
            E1CellExecutionEvidence.__dataclass_fields__
        ):
            raise ValueError("E1 Pareto evidence cell fields differ")
        cell = dict(raw_cell)
        identity = cell["execution_identity"]
        if type(identity) is not dict or set(identity) != set(
            StageItlExecutionIdentity.__dataclass_fields__
        ):
            raise ValueError("E1 Pareto execution identity fields differ")
        cell["execution_identity"] = StageItlExecutionIdentity(**identity)
        cells.append(E1CellExecutionEvidence(**cell))
    row["cells"] = tuple(cells)
    manifest = E1StagedParetoEvidenceManifest(**row)  # type: ignore[arg-type]
    if manifest.sha256 != declared:
        raise ValueError("E1 Pareto evidence manifest digest differs")
    return manifest


@dataclass(frozen=True)
class E1StagedGeometryEvaluation:
    geometry: E1Geometry
    confidence_lower_request_rate_ratio: float
    peak_hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int
    candidate_cell_ids: tuple[str, str]

    def __post_init__(self) -> None:
        if type(self.geometry) is not E1Geometry:
            raise TypeError("E1 evaluation requires an exact geometry")
        if (
            type(self.confidence_lower_request_rate_ratio) is not float
            or not math.isfinite(self.confidence_lower_request_rate_ratio)
            or self.confidence_lower_request_rate_ratio <= 0
        ):
            raise ValueError("E1 confidence lower request-rate ratio is invalid")
        for label, value in (
            ("peak HBM", self.peak_hbm_bytes),
            ("p99 ITL", self.p99_itl_us),
            ("exposed update", self.exposed_update_us),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"E1 {label} must be a non-negative integer")
        if len(self.candidate_cell_ids) != 2 or self.candidate_cell_ids != tuple(
            sorted(set(self.candidate_cell_ids))
        ):
            raise ValueError("E1 evaluation requires two exact candidate cells")
        for cell_id in self.candidate_cell_ids:
            _require_sha256("E1 evaluation candidate", cell_id)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E1StagedParetoArtifact:
    """Deterministic output of the staged-native 67-cell E1 reduction."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    e3a_selection_receipt_sha256: str
    inventory_sha256: str
    evidence_manifest_sha256: str
    protocol_sha256: str
    target_cell_id: str
    static_cell_id: str
    tts_cell_id: str
    excluded_l0_naive_cell_id: str
    evaluations: tuple[E1StagedGeometryEvaluation, ...]
    excluded_geometry_sha256s: tuple[str, ...]
    surviving_geometries: tuple[E1Geometry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E1 staged Pareto artifact schema 1 is supported")
        for label, digest in (
            ("E1 artifact protocol lock", self.protocol_lock_sha256),
            ("E1 artifact registry", self.registry_sha256),
            ("E1 artifact materialization", self.materialization_receipt_sha256),
            ("E1 artifact coverage", self.coverage_receipt_sha256),
            ("E1 artifact E3a selection", self.e3a_selection_receipt_sha256),
            ("E1 artifact inventory", self.inventory_sha256),
            ("E1 artifact evidence manifest", self.evidence_manifest_sha256),
            ("E1 artifact reducer protocol", self.protocol_sha256),
            ("E1 target cell", self.target_cell_id),
            ("E1 Static cell", self.static_cell_id),
            ("E1 TTS cell", self.tts_cell_id),
            ("E1 excluded L0-naive cell", self.excluded_l0_naive_cell_id),
        ):
            _require_sha256(label, digest)
        if self.protocol_sha256 != E1_STAGED_PARETO_PROTOCOL_SHA256:
            raise ValueError("E1 artifact uses another reducer protocol")
        if (
            type(self.evaluations) is not tuple
            or any(
                type(row) is not E1StagedGeometryEvaluation for row in self.evaluations
            )
            or tuple(row.geometry.sha256 for row in self.evaluations)
            != tuple(sorted({row.geometry.sha256 for row in self.evaluations}))
        ):
            raise ValueError("E1 evaluations must be canonical and unique")
        if self.excluded_geometry_sha256s != tuple(
            sorted(set(self.excluded_geometry_sha256s))
        ):
            raise ValueError("E1 excluded geometries must be canonical and unique")
        for value in self.excluded_geometry_sha256s:
            _require_sha256("E1 excluded geometry", value)
        if not self.surviving_geometries or tuple(
            row.sha256 for row in self.surviving_geometries
        ) != tuple(sorted({row.sha256 for row in self.surviving_geometries})):
            raise ValueError("E1 surviving geometries must be canonical and non-empty")
        if not set(self.surviving_geometries) <= {
            row.geometry for row in self.evaluations
        }:
            raise ValueError("E1 survivor is foreign to safe evaluations")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class _RequestMetric:
    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    latency_ns: int
    p99_itl_ns: Fraction

    @property
    def output_tokens(self) -> int:
        return len(self.output_token_ids)


@dataclass(frozen=True)
class _ValidatedCell:
    cell: MaterializedCell
    result: NativeTerminalResultProjection | FormalDistributedTerminalResultProjection
    timing: StageItlTimestampAuthority
    metrics: tuple[_RequestMetric, ...]
    peak_hbm_bytes: int
    exposed_update_us: int
    published_updates: int
    safety_reasons: tuple[str, ...]


def _linear_p99_ns(values: tuple[int, ...]) -> Fraction:
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("E1 ITL sample is empty or invalid")
    ordered = tuple(sorted(values))
    position = Fraction((len(ordered) - 1) * 99, 100)
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    remainder = position - lower
    return (
        Fraction(ordered[lower]) * (1 - remainder)
        + Fraction(ordered[upper]) * remainder
    )


def _metrics(
    result: NativeTerminalResultProjection | FormalDistributedTerminalResultProjection,
    timing: StageItlTimestampAuthority,
) -> tuple[_RequestMetric, ...]:
    result_by_id = {row.request_id: row for row in result.requests}
    timing_by_id = {row.request_id: row for row in timing.requests}
    if (
        not result_by_id
        or len(result_by_id) != len(result.requests)
        or len(timing_by_id) != len(timing.requests)
        or set(result_by_id) != set(timing_by_id)
        or set(result.scored_request_ids) != set(result_by_id)
    ):
        raise ValueError("E1 result/timing request coverage is not exact")
    rows: list[_RequestMetric] = []
    for request_id in sorted(result_by_id):
        terminal = result_by_id[request_id]
        clock = timing_by_id[request_id]
        if type(terminal) not in {
            NativeTerminalRequestResult,
            FormalDistributedTerminalRequestResult,
        }:
            raise TypeError("E1 formal request result is not verifier-sealed")
        if (
            not terminal.submitted_to_server
            or terminal.terminal_status != "completed"
            or terminal.output_token_ids is None
            or terminal.output_token_ids != clock.output_token_ids
        ):
            raise ValueError("E1 result/timing request is incomplete or differs")
        latency_ns = clock.request_terminal_ns - clock.request_started_ns
        if latency_ns <= 0:
            raise ValueError("E1 request latency must be positive")
        rows.append(
            _RequestMetric(
                request_id=request_id,
                input_token_ids=terminal.input_token_ids,
                output_token_ids=terminal.output_token_ids,
                latency_ns=latency_ns,
                p99_itl_ns=_linear_p99_ns(clock.inter_token_ns),
            )
        )
    return tuple(rows)


def _integer_counter(
    performance: dict[str, object],
    field: str,
    *,
    nullable_zero: bool = False,
) -> int:
    if field not in performance:
        raise ValueError(f"E1 performance counter {field} is missing")
    value = performance[field]
    if value is None and nullable_zero:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(f"E1 performance counter {field} is unavailable")
    return value


def _finite_nonnegative_counter(performance: dict[str, object], field: str) -> float:
    if field not in performance:
        raise ValueError(f"E1 performance counter {field} is missing")
    value = performance[field]
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"E1 performance counter {field} is unavailable")
    return float(value)


def _validated_cell(
    *,
    cell: MaterializedCell,
    evidence: E1CellExecutionEvidence,
    execution_binding: VerifiedFormalServingExecutionBinding,
    coverage_terminal_sha256: str,
    protocol_lock: ProtocolLock,
    inventory_sha256: str,
    now_ns: int,
    expected_stage: str = "E1",
) -> _ValidatedCell:
    verified_binding = require_verified_formal_serving_execution_binding(
        execution_binding
    )
    identity = evidence.execution_identity
    if (
        verified_binding.subject.stage != expected_stage
        or verified_binding.subject.protocol_lock_sha256 != protocol_lock.sha256
        or verified_binding.subject.materialized_cell_id != cell.cell_id
        or verified_binding.subject.inventory_sha256 != inventory_sha256
        or verified_binding.subject.execution_identity != identity
        or verified_binding.sha256 != evidence.execution_binding_sha256
        or identity.materialized_cell_id != cell.cell_id
        or identity.registry_sha256 != protocol_lock.registry_sha256
        or identity.inventory_sha256 != inventory_sha256
        or identity.method != _runtime_method(cell)
    ):
        raise ValueError("E1 execution identity differs from materialized cell")
    result = validate_formal_terminal_result_proof_artifact(
        evidence.native_result_proof_path,
        expected_inventory_sha256=inventory_sha256,
        expected_registry_sha256=protocol_lock.registry_sha256,
        expected_root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        expected_execution_plan_sha256=identity.execution_plan_sha256,
        expected_rank_config_sha256=identity.rank_config_sha256,
        expected_run_id=identity.run_id,
        expected_run_nonce_sha256=identity.run_nonce_sha256,
        expected_attempt_id=identity.attempt_id,
        expected_method=identity.method,
        expected_stage=expected_stage,
        expected_topology=verified_binding.subject.topology_mode,
        now_ns=now_ns,
    )
    timing = validate_formal_serving_itl_proof(
        evidence.stage_itl_proof_path,
        expected_registry_sha256=protocol_lock.registry_sha256,
        expected_root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        now_ns=now_ns,
    )
    if (
        result.terminal_sha256 != coverage_terminal_sha256
        or timing.execution_identity != identity
        or timing.native_result_proof_path != evidence.native_result_proof_path
        or timing.native_result_proof_raw_sha256
        != evidence.native_result_proof_raw_sha256
        or timing.native_result_proof_semantic_sha256
        != evidence.native_result_proof_semantic_sha256
    ):
        raise ValueError("E1 terminal/timing proof lineage differs from coverage")
    performance = result.performance_counters
    safety_reasons: tuple[str, ...]
    if identity.method == "target_only":
        for field in _SAFETY_COUNTERS:
            if field not in performance or performance[field] is not None:
                raise ValueError("E1 Target-only requires explicit N/A safety counters")
        safety_reasons = ()
    else:
        safety_reasons = tuple(
            field
            for field in _SAFETY_COUNTERS
            if _integer_counter(performance, field) != 0
        )
    published = _integer_counter(
        performance,
        "updates_published",
        nullable_zero=identity.method in {"target_only", "static"},
    )
    updates = getattr(result, "updates", None)
    adaptive_methods = {
        "tts",
        "l0",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    }
    if identity.method in adaptive_methods:
        if type(updates) is not tuple or not updates:
            raise ValueError("E1 adapted result lacks sealed update rows")
        published_rows = tuple(
            row
            for row in updates
            if row.status == "published"
            and row.published_version is not None
            and row.reconstruction_ok
        )
        if len(published_rows) != published:
            raise ValueError("E1 published-update counter differs from update rows")
    elif updates not in {None, ()}:
        raise ValueError("E1 allocation-free result contains update rows")
    peak_hbm = _integer_counter(performance, "peak_hbm_bytes")
    exposed_update_us = (
        math.ceil(_finite_nonnegative_counter(performance, "exposed_update_ms") * 1_000)
        if identity.method in adaptive_methods
        else 0
    )
    return _ValidatedCell(
        cell=cell,
        result=result,
        timing=timing,
        metrics=_metrics(result, timing),
        peak_hbm_bytes=peak_hbm,
        exposed_update_us=exposed_update_us,
        published_updates=published,
        safety_reasons=safety_reasons,
    )


def _request_identity(
    rows: tuple[_RequestMetric, ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (row.request_id, row.input_token_ids, row.output_token_ids) for row in rows
    )


def _paired_confidence_lower(
    numerator: tuple[_RequestMetric, ...],
    denominator: tuple[_RequestMetric, ...],
) -> float:
    left = {row.request_id: row for row in numerator}
    right = {row.request_id: row for row in denominator}
    if not left or set(left) != set(right):
        raise ValueError("E1 paired confidence requires exact request IDs")
    log_ratios = []
    for request_id in sorted(left):
        numerator_row = left[request_id]
        denominator_row = right[request_id]
        if (
            numerator_row.input_token_ids != denominator_row.input_token_ids
            or numerator_row.output_token_ids != denominator_row.output_token_ids
        ):
            raise ValueError("E1 paired requests differ in token trajectory")
        left_rate = numerator_row.output_tokens / numerator_row.latency_ns
        right_rate = denominator_row.output_tokens / denominator_row.latency_ns
        if left_rate <= 0 or right_rate <= 0:
            raise ValueError("E1 paired request rate is not positive")
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
        raise ValueError("E1 confidence lower bound is not finite and positive")
    return result


def _dominates(
    left: E1StagedGeometryEvaluation,
    right: E1StagedGeometryEvaluation,
) -> bool:
    weak = (
        left.confidence_lower_request_rate_ratio
        >= right.confidence_lower_request_rate_ratio
        and left.peak_hbm_bytes <= right.peak_hbm_bytes
        and left.p99_itl_us <= right.p99_itl_us
        and left.exposed_update_us <= right.exposed_update_us
    )
    strict = (
        left.confidence_lower_request_rate_ratio
        > right.confidence_lower_request_rate_ratio
        or left.peak_hbm_bytes < right.peak_hbm_bytes
        or left.p99_itl_us < right.p99_itl_us
        or left.exposed_update_us < right.exposed_update_us
    )
    return weak and strict


def reduce_e1_staged_pareto_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    e3a_selection_receipt_sha256: str,
    manifest: E1StagedParetoEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E1StagedParetoArtifact:
    """Deep-replay all 68 E1 proofs and seal the 67-row Pareto output."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E1 staged reduction requires an exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E1 staged reduction requires an exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E1 staged reduction requires an exact coverage receipt")
    if type(manifest) is not E1StagedParetoEvidenceManifest:
        raise TypeError("E1 staged reduction requires an exact proof manifest")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E1 staged reduction requires sealed execution bindings")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E1 staged reduction time must be non-negative")
    _require_sha256("E1 staged E3a selection", e3a_selection_receipt_sha256)
    if (
        materialization.stage != "E1"
        or materialization.expected_cell_count != 68
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("E1 staged reduction requires the exact 68-cell slice")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E1 staged reduction requires all-COMPLETE coverage")
    if (
        manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.e3a_selection_receipt_sha256 != e3a_selection_receipt_sha256
        or materialization.source_decision_sha256 != e3a_selection_receipt_sha256
    ):
        raise ValueError("E1 staged proof manifest differs from signed lineage")

    roles: dict[str, list[MaterializedCell]] = {}
    for cell in materialization.cells:
        roles.setdefault(cell.method_role, []).append(cell)
    if {role: len(rows) for role, rows in roles.items()} != {
        "Target-only": 1,
        "Static": 1,
        "TTS": 1,
        "L0-naive": 1,
        "LightCone-candidate": 64,
    }:
        raise ValueError("E1 materialization role cardinality differs from protocol")
    l0_naive = roles["L0-naive"][0]
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    if set(evidence_by_cell) != {cell.cell_id for cell in materialization.cells}:
        raise ValueError("E1 proof manifest must cover every materialized cell")
    execution_binding_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for row in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(row)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in execution_binding_by_cell:
            raise ValueError("E1 staged reduction reuses a cell execution binding")
        execution_binding_by_cell[cell_id] = verified
    if set(execution_binding_by_cell) != {
        cell.cell_id for cell in materialization.cells
    }:
        raise ValueError("E1 execution bindings must cover all 68 cells exactly")
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    validated: dict[str, _ValidatedCell] = {}
    for cell in materialization.cells:
        terminal_sha256 = terminal_by_cell[cell.cell_id]
        assert terminal_sha256 is not None
        validated[cell.cell_id] = _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],
            execution_binding=execution_binding_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_sha256,
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
        )

    target = validated[roles["Target-only"][0].cell_id]
    static = validated[roles["Static"][0].cell_id]
    tts = validated[roles["TTS"][0].cell_id]
    l0_naive_result = validated[l0_naive.cell_id]
    for label, row in (
        ("Target-only", target),
        ("Static", static),
        ("TTS", tts),
        ("L0-naive", l0_naive_result),
    ):
        if row.safety_reasons:
            raise ValueError(f"E1 {label} reference is unsafe")
    target_request_identity = _request_identity(target.metrics)
    if _request_identity(static.metrics) != target_request_identity:
        raise ValueError("E1 Static differs from Target-only token trajectories")
    if _request_identity(tts.metrics) != target_request_identity:
        raise ValueError("E1 TTS differs from Target-only token trajectories")
    if _request_identity(l0_naive_result.metrics) != target_request_identity:
        raise ValueError("E1 L0-naive differs from Target-only token trajectories")
    if tts.published_updates < 1 or l0_naive_result.published_updates < 1:
        raise ValueError("E1 frozen TTS/L0 reference has no published update")

    grouped: dict[str, tuple[E1Geometry, dict[str, _ValidatedCell]]] = {}
    for candidate_cell in roles["LightCone-candidate"]:
        dimensions = dict(candidate_cell.dimensions)
        geometry = E1Geometry(
            scope=str(dimensions["scope"]),
            parameterization=str(dimensions["parameterization"]),  # type: ignore[arg-type]
            rank=(None if dimensions["rank"] == "none" else int(dimensions["rank"])),
            alpha_over_rank=(
                None
                if dimensions["alpha_over_rank"] == "none"
                else float(dimensions["alpha_over_rank"])
            ),
        )
        optimizer = str(dimensions["optimizer_anchor"])
        entry = grouped.setdefault(geometry.sha256, (geometry, {}))
        if optimizer in entry[1]:
            raise ValueError("E1 repeats a geometry/optimizer candidate")
        entry[1][optimizer] = validated[candidate_cell.cell_id]
    if len(grouped) != 32 or any(
        set(candidates) != set(E1_OPTIMIZER_ANCHORS)
        for _, candidates in grouped.values()
    ):
        raise ValueError("E1 staged proof lacks the 32 x 2 candidate grid")

    evaluations: list[E1StagedGeometryEvaluation] = []
    excluded: list[str] = []
    for geometry_sha256 in sorted(grouped):
        geometry, candidates = grouped[geometry_sha256]
        reasons: set[str] = set()
        confidence: list[float] = []
        hbm: list[int] = []
        p99: list[int] = []
        exposed: list[int] = []
        for optimizer in E1_OPTIMIZER_ANCHORS:
            candidate = candidates[optimizer]
            reasons.update(candidate.safety_reasons)
            if _request_identity(candidate.metrics) != target_request_identity:
                reasons.add("target_token_trajectory_mismatch")
            if candidate.published_updates < 1:
                reasons.add("no_published_update")
            hbm.append(candidate.peak_hbm_bytes)
            p99.append(
                max(
                    math.ceil(metric.p99_itl_ns / 1_000) for metric in candidate.metrics
                )
            )
            exposed.append(candidate.exposed_update_us)
            if not reasons:
                confidence.extend(
                    (
                        _paired_confidence_lower(candidate.metrics, static.metrics),
                        _paired_confidence_lower(candidate.metrics, tts.metrics),
                    )
                )
        if reasons:
            excluded.append(geometry.sha256)
            continue
        evaluation = E1StagedGeometryEvaluation(
            geometry=geometry,
            confidence_lower_request_rate_ratio=min(confidence),
            peak_hbm_bytes=max(hbm),
            p99_itl_us=max(p99),
            exposed_update_us=max(exposed),
            candidate_cell_ids=tuple(
                sorted(candidate.cell.cell_id for candidate in candidates.values())
            ),
        )
        evaluations.append(evaluation)
    survivors = tuple(
        row
        for row in sorted(evaluations, key=lambda value: value.geometry.sha256)
        if not any(
            other.geometry.sha256 != row.geometry.sha256 and _dominates(other, row)
            for other in evaluations
        )
    )
    if not survivors:
        raise ValueError("E1 staged reduction has no safe non-dominated geometry")
    artifact = E1StagedParetoArtifact(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        e3a_selection_receipt_sha256=e3a_selection_receipt_sha256,
        inventory_sha256=manifest.inventory_sha256,
        evidence_manifest_sha256=manifest.sha256,
        protocol_sha256=E1_STAGED_PARETO_PROTOCOL_SHA256,
        target_cell_id=target.cell.cell_id,
        static_cell_id=static.cell.cell_id,
        tts_cell_id=tts.cell.cell_id,
        excluded_l0_naive_cell_id=l0_naive.cell_id,
        evaluations=tuple(evaluations),
        excluded_geometry_sha256s=tuple(sorted(excluded)),
        surviving_geometries=tuple(row.geometry for row in survivors),
    )
    artifact.__post_init__()
    return artifact


__all__ = [
    "E1_STAGED_PARETO_PROTOCOL_SHA256",
    "E1CellExecutionEvidence",
    "E1StagedGeometryEvaluation",
    "E1StagedParetoArtifact",
    "E1StagedParetoEvidenceManifest",
    "e1_staged_pareto_evidence_manifest_from_dict",
    "e1_staged_pareto_evidence_manifest_to_dict",
    "reduce_e1_staged_pareto_from_proofs",
]
