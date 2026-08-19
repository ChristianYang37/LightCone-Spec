"""Staged-native E3a reducer over exact controlled execution proofs.

The historical E3a selector reopens an eager diagnostic registry and depends
on an intentionally unregistered module global.  This module is the formal
replacement.  Its universe is the signed 360-cell materialization; every row
must carry a verifier-sealed execution binding plus durable native-result and
client-timestamp proofs.  No caller-supplied throughput, safety, width, load,
crossover, or drift scalar is accepted.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import Literal

from lightcone_spec.experiments.formal_protocol import (
    FormalRuntimeAuthorityManifest,
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.itl_authority import (
    StageItlExecutionIdentity,
    StageItlTimestampAuthority,
    validate_stage_itl_timestamp_proof_artifact,
)
from lightcone_spec.experiments.registry import (
    CONTEXT_REGIMES,
    DRAFT_WIDTHS,
    E3A_CONCURRENCY_GRID,
    LONG_CONTEXT_ANCHORS,
)
from lightcone_spec.experiments.stage_materialization import (
    MaterializedCell,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalRequestResult,
    NativeTerminalResultProjection,
    validate_native_terminal_result_proof_artifact,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

E3A_STAGED_REDUCTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e3a_staged_native_six_output_protocol",
        "universe": "exact_signed_360_cell_materialization",
        "proofs_per_cell": (
            "durable_external_control_native_terminal_result",
            "durable_external_control_stage_itl_timestamp",
            "verifier_sealed_cell_execution_binding",
        ),
        "pairing": (
            "exact_context_regime_concurrency_target_static_request_and_token_identity"
        ),
        "safety": "explicit_registered_counter_schema_missing_never_zero",
        "throughput": "committed_output_tokens_over_full_client_service_window",
        "primary_contexts": LONG_CONTEXT_ANCHORS,
        "reference_load": (
            "smallest_registered_concurrency_reaching_nine_tenths_of_"
            "maximum_width_median_static_throughput"
        ),
        "matched_width": (
            "maximum_worst_static_target_throughput_ratio_then_median_static_"
            "throughput_then_smallest_width"
        ),
        "crossover": "first_concurrency_with_static_target_ratio_lte_one",
        "locked_outputs": (
            "baseline_capacity_envelope",
            "e1_reference_load",
            "matched_width",
            "width_selection_rule",
            "static_target_crossover",
            "drift_witness",
        ),
        "confirmation_data_visible": False,
    }
)
E3A_STAGED_REDUCTION_RUNNER_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e3a_staged_native_reducer_runner",
        "module": "lightcone_spec.experiments.e3a_stage_authority",
        "entrypoint": "reduce_e3a_staged_selection_from_proofs",
    }
)
E3A_STAGED_REDUCTION_TEST_SET_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e3a_staged_native_reducer_acceptance_set",
        "requirements": (
            "exact_360_cell_and_proof_coverage",
            "foreign_or_reused_cell_run_terminal_and_timing_rejected",
            "explicit_safety_counter_schema",
            "exact_target_static_request_and_token_pairing",
            "deterministic_six_output_reduction",
            "legacy_reduction_authority_forbidden",
        ),
    }
)

_SAFETY_COUNTERS = (
    "communicator_failures",
    "exactness_violations",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "version_mismatches",
)
_LOCKED_OUTPUT_NAMES = (
    "baseline_capacity_envelope",
    "drift_witness",
    "e1_reference_load",
    "matched_width",
    "static_target_crossover",
    "width_selection_rule",
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


def _runtime_method(cell: MaterializedCell) -> Literal["target_only", "static"]:
    try:
        return {
            "Target-only": "target_only",
            "Static": "static",
        }[cell.method_role]  # type: ignore[return-value]
    except KeyError as error:
        raise ValueError("E3a materialization contains an unsupported role") from error


@dataclass(frozen=True)
class E3aCellExecutionEvidence:
    """Path-bound terminal and timing proofs for one E3a cell."""

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
        if self.schema_version != 1:
            raise ValueError("only E3a cell evidence schema 1 is supported")
        _require_sha256("E3a evidence cell", self.materialized_cell_id)
        _require_sha256("E3a evidence execution binding", self.execution_binding_sha256)
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("E3a evidence requires an exact execution identity")
        self.execution_identity.__post_init__()
        if self.execution_identity.materialized_cell_id != self.materialized_cell_id:
            raise ValueError("E3a execution identity names another cell")
        for label, path_value, raw_sha256, semantic_sha256 in (
            (
                "E3a native result proof",
                self.native_result_proof_path,
                self.native_result_proof_raw_sha256,
                self.native_result_proof_semantic_sha256,
            ),
            (
                "E3a stage ITL proof",
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
                raise ValueError(f"{label} changed after binding")
        if self.native_result_proof_path == self.stage_itl_proof_path:
            raise ValueError("E3a result and timing proofs must be distinct")

    @classmethod
    def bind(
        cls,
        *,
        execution_binding: VerifiedFormalServingExecutionBinding,
        native_result_proof_path: str,
        stage_itl_proof_path: str,
    ) -> E3aCellExecutionEvidence:
        verified = require_verified_formal_serving_execution_binding(execution_binding)
        if verified.subject.stage != "E3a":
            raise ValueError("E3a evidence cannot consume another stage binding")
        result = CanonicalJsonProofBinding.bind(native_result_proof_path)
        timing = CanonicalJsonProofBinding.bind(stage_itl_proof_path)
        return cls(
            schema_version=1,
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
class E3aStagedEvidenceManifest:
    """Exact 360-row input ledger for the staged-native reducer."""

    schema_version: int
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    inventory_sha256: str
    reducer_authority_member_sha256: str
    cells: tuple[E3aCellExecutionEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E3a evidence manifest schema 1 is supported")
        for label, digest in (
            ("E3a manifest protocol lock", self.protocol_lock_sha256),
            ("E3a manifest materialization", self.materialization_receipt_sha256),
            ("E3a manifest coverage", self.coverage_receipt_sha256),
            ("E3a manifest inventory", self.inventory_sha256),
            ("E3a manifest reducer member", self.reducer_authority_member_sha256),
        ):
            _require_sha256(label, digest)
        if (
            type(self.cells) is not tuple
            or len(self.cells) != 360
            or any(type(row) is not E3aCellExecutionEvidence for row in self.cells)
            or tuple(row.materialized_cell_id for row in self.cells)
            != tuple(sorted({row.materialized_cell_id for row in self.cells}))
        ):
            raise ValueError("E3a proof manifest must contain 360 sorted unique cells")
        runs = tuple(
            (
                row.execution_identity.run_id,
                row.execution_identity.run_nonce_sha256,
                row.execution_identity.attempt_id,
            )
            for row in self.cells
        )
        if len(runs) != len(set(runs)):
            raise ValueError("E3a proof manifest reuses a run identity")
        for label, values in (
            (
                "execution binding",
                tuple(row.execution_binding_sha256 for row in self.cells),
            ),
            (
                "terminal result proof",
                tuple(row.native_result_proof_raw_sha256 for row in self.cells),
            ),
            (
                "ITL proof",
                tuple(row.stage_itl_proof_raw_sha256 for row in self.cells),
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"E3a proof manifest reuses a {label}")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3aCapacityObservation:
    cell_id: str
    method_role: Literal["Target-only", "Static"]
    context: int
    regime: str
    concurrency: int
    width: int | None
    throughput_tokens: int
    throughput_window_ns: int
    peak_hbm_bytes: int
    target_cell_id: str | None
    static_target_ratio_numerator: int | None
    static_target_ratio_denominator: int | None
    execution_evidence_sha256: str
    terminal_sha256: str
    timing_authority_sha256: str

    def __post_init__(self) -> None:
        for label, digest in (
            ("E3a observation cell", self.cell_id),
            ("E3a observation execution evidence", self.execution_evidence_sha256),
            ("E3a observation terminal", self.terminal_sha256),
            ("E3a observation timing", self.timing_authority_sha256),
        ):
            _require_sha256(label, digest)
        if self.method_role not in {"Target-only", "Static"}:
            raise ValueError("E3a observation role is unsupported")
        if type(self.context) is not int or self.context < 1:
            raise ValueError("E3a observation context must be positive")
        if type(self.regime) is not str or not self.regime:
            raise ValueError("E3a observation regime must be text")
        if type(self.concurrency) is not int or self.concurrency < 1:
            raise ValueError("E3a observation concurrency must be positive")
        if type(self.throughput_tokens) is not int or self.throughput_tokens < 1:
            raise ValueError("E3a observation throughput numerator must be positive")
        if type(self.throughput_window_ns) is not int or self.throughput_window_ns < 1:
            raise ValueError("E3a observation throughput window must be positive")
        if type(self.peak_hbm_bytes) is not int or self.peak_hbm_bytes < 0:
            raise ValueError("E3a observation peak HBM must be non-negative")
        ratio_fields = (
            self.target_cell_id,
            self.static_target_ratio_numerator,
            self.static_target_ratio_denominator,
        )
        if self.method_role == "Target-only":
            if self.width is not None or any(
                value is not None for value in ratio_fields
            ):
                raise ValueError("E3a Target-only observation cannot claim Static data")
        else:
            if self.width not in DRAFT_WIDTHS:
                raise ValueError("E3a Static width is outside the registered grid")
            _require_sha256("E3a observation paired target", self.target_cell_id)
            if (
                type(self.static_target_ratio_numerator) is not int
                or self.static_target_ratio_numerator < 1
                or type(self.static_target_ratio_denominator) is not int
                or self.static_target_ratio_denominator < 1
            ):
                raise ValueError("E3a Static/Target ratio must be positive rational")

    @property
    def throughput(self) -> Fraction:
        return Fraction(self.throughput_tokens, self.throughput_window_ns)

    @property
    def static_target_ratio(self) -> Fraction | None:
        if self.static_target_ratio_numerator is None:
            return None
        assert self.static_target_ratio_denominator is not None
        return Fraction(
            self.static_target_ratio_numerator,
            self.static_target_ratio_denominator,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3aLockedOutput:
    name: str
    content_sha256: str

    def __post_init__(self) -> None:
        if self.name not in _LOCKED_OUTPUT_NAMES:
            raise ValueError("E3a locked-output name is unsupported")
        _require_sha256("E3a locked-output content", self.content_sha256)


@dataclass(frozen=True)
class E3aStagedSelectionArtifact:
    """The six exact locked outputs derived from all 360 proof rows."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    inventory_sha256: str
    evidence_manifest_sha256: str
    reducer_authority_member_sha256: str
    reducer_protocol_sha256: str
    model: str
    matched_width: int
    common_load: int
    observations: tuple[E3aCapacityObservation, ...]
    locked_outputs: tuple[E3aLockedOutput, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E3a staged selection schema 1 is supported")
        for label, digest in (
            ("E3a artifact protocol lock", self.protocol_lock_sha256),
            ("E3a artifact registry", self.registry_sha256),
            ("E3a artifact materialization", self.materialization_receipt_sha256),
            ("E3a artifact coverage", self.coverage_receipt_sha256),
            ("E3a artifact inventory", self.inventory_sha256),
            ("E3a artifact evidence manifest", self.evidence_manifest_sha256),
            ("E3a artifact reducer member", self.reducer_authority_member_sha256),
            ("E3a artifact reducer protocol", self.reducer_protocol_sha256),
        ):
            _require_sha256(label, digest)
        if self.reducer_protocol_sha256 != E3A_STAGED_REDUCTION_PROTOCOL_SHA256:
            raise ValueError("E3a artifact uses another reducer protocol")
        if self.matched_width not in DRAFT_WIDTHS:
            raise ValueError("E3a matched width is outside the registered grid")
        if self.common_load not in E3A_CONCURRENCY_GRID:
            raise ValueError("E3a common load is outside the registered grid")
        if len(self.observations) != 360 or tuple(
            row.cell_id for row in self.observations
        ) != tuple(sorted({row.cell_id for row in self.observations})):
            raise ValueError("E3a artifact requires 360 sorted observations")
        if tuple(row.name for row in self.locked_outputs) != _LOCKED_OUTPUT_NAMES:
            raise ValueError("E3a artifact must expose all six locked outputs exactly")
        reject_banned_model_identity(self)

    def locked_output(self, name: str) -> E3aLockedOutput:
        rows = tuple(row for row in self.locked_outputs if row.name == name)
        if len(rows) != 1:
            raise ValueError("E3a locked output is not exact")
        return rows[0]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3aStagedSelectionReceipt:
    """Signable source row for downstream TTS-Cal/E1 materialization."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    e3a_materialization_receipt_sha256: str
    e3a_coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    selection_artifact_sha256: str
    reducer_authority_member_sha256: str
    model: str
    matched_width: int
    common_load: int
    locked_outputs: tuple[E3aLockedOutput, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E3a staged receipt schema 1 is supported")
        for label, digest in (
            ("E3a receipt protocol lock", self.protocol_lock_sha256),
            ("E3a receipt registry", self.registry_sha256),
            ("E3a receipt materialization", self.e3a_materialization_receipt_sha256),
            ("E3a receipt coverage", self.e3a_coverage_receipt_sha256),
            ("E3a receipt evidence", self.evidence_manifest_sha256),
            ("E3a receipt selection", self.selection_artifact_sha256),
            ("E3a receipt reducer member", self.reducer_authority_member_sha256),
        ):
            _require_sha256(label, digest)
        if tuple(row.name for row in self.locked_outputs) != _LOCKED_OUTPUT_NAMES:
            raise ValueError("E3a staged receipt lacks one of six locked outputs")
        if self.matched_width not in DRAFT_WIDTHS:
            raise ValueError("E3a receipt width is outside the registered grid")
        if self.common_load not in E3A_CONCURRENCY_GRID:
            raise ValueError("E3a receipt load is outside the registered grid")
        reject_banned_model_identity(self)

    def validate_artifact(self, artifact: E3aStagedSelectionArtifact) -> None:
        if type(artifact) is not E3aStagedSelectionArtifact:
            raise TypeError("E3a receipt requires its exact selection artifact")
        if (
            self.protocol_lock_sha256 != artifact.protocol_lock_sha256
            or self.registry_sha256 != artifact.registry_sha256
            or self.e3a_materialization_receipt_sha256
            != artifact.materialization_receipt_sha256
            or self.e3a_coverage_receipt_sha256 != artifact.coverage_receipt_sha256
            or self.evidence_manifest_sha256 != artifact.evidence_manifest_sha256
            or self.selection_artifact_sha256 != artifact.sha256
            or self.reducer_authority_member_sha256
            != artifact.reducer_authority_member_sha256
            or self.model != artifact.model
            or self.matched_width != artifact.matched_width
            or self.common_load != artifact.common_load
            or self.locked_outputs != artifact.locked_outputs
        ):
            raise ValueError("E3a staged receipt differs from reducer artifact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE3aStagedSelectionReceipt:
    payload: E3aStagedSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        artifact: E3aStagedSelectionArtifact,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> E3aStagedSelectionReceipt:
        if type(self.payload) is not E3aStagedSelectionReceipt:
            raise TypeError("signed E3a staged payload has the wrong type")
        self.payload.validate_artifact(artifact)
        verify_signed_payload(
            self.payload,
            payload_sha256=self.payload_sha256,
            challenge=self.challenge,
            attestation=self.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        return self.payload

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "payload": asdict(self.payload),
                "payload_sha256": self.payload_sha256,
                "challenge": asdict(self.challenge),
                "attestation": asdict(self.attestation),
            }
        )


@dataclass(frozen=True)
class _ValidatedCell:
    cell: MaterializedCell
    evidence: E3aCellExecutionEvidence
    result: NativeTerminalResultProjection
    timing: StageItlTimestampAuthority
    request_identity: tuple[tuple[object, ...], ...]
    peak_hbm_bytes: int


def _integer_counter(performance: dict[str, object], field: str) -> int:
    if field not in performance:
        raise ValueError(f"E3a performance counter {field} is missing")
    value = performance[field]
    if type(value) is not int or value < 0:
        raise ValueError(f"E3a performance counter {field} is unavailable")
    return value


def _validate_requests(
    result: NativeTerminalResultProjection,
    timing: StageItlTimestampAuthority,
) -> tuple[tuple[object, ...], ...]:
    result_by_id = {row.request_id: row for row in result.requests}
    timing_by_id = {row.request_id: row for row in timing.requests}
    if (
        not result_by_id
        or len(result_by_id) != len(result.requests)
        or len(timing_by_id) != len(timing.requests)
        or set(result_by_id) != set(timing_by_id)
        or set(result.scored_request_ids) != set(result_by_id)
    ):
        raise ValueError("E3a result/timing request coverage is not exact")
    rows = []
    for request_id in sorted(result_by_id):
        terminal = result_by_id[request_id]
        clock = timing_by_id[request_id]
        if type(terminal) is not NativeTerminalRequestResult:
            raise TypeError("E3a native request result is not verifier-sealed")
        if (
            not terminal.submitted_to_server
            or terminal.terminal_status != "completed"
            or terminal.output_token_ids is None
            or terminal.output_token_ids != clock.output_token_ids
            or clock.request_terminal_ns <= clock.request_started_ns
        ):
            raise ValueError("E3a request is incomplete or differs across proofs")
        rows.append(
            (
                request_id,
                terminal.input_token_ids,
                terminal.output_token_ids,
            )
        )
    if result.output_token_count != timing.throughput_numerator_tokens:
        raise ValueError("E3a terminal/timing output-token totals differ")
    return tuple(rows)


def _validate_cell(
    *,
    cell: MaterializedCell,
    evidence: E3aCellExecutionEvidence,
    execution_binding: VerifiedFormalServingExecutionBinding,
    coverage_terminal_sha256: str | None,
    protocol_lock: ProtocolLock,
    inventory_sha256: str,
    now_ns: int,
) -> _ValidatedCell:
    binding = require_verified_formal_serving_execution_binding(execution_binding)
    identity = evidence.execution_identity
    if (
        binding.subject.stage != "E3a"
        or binding.subject.protocol_lock_sha256 != protocol_lock.sha256
        or binding.subject.materialized_cell_id != cell.cell_id
        or binding.subject.inventory_sha256 != inventory_sha256
        or binding.subject.execution_identity != identity
        or binding.sha256 != evidence.execution_binding_sha256
        or identity.registry_sha256 != protocol_lock.registry_sha256
        or identity.method != _runtime_method(cell)
    ):
        raise ValueError("E3a execution identity differs from materialized cell")
    for label, path_value, raw_sha256, semantic_sha256 in (
        (
            "E3a native result proof",
            evidence.native_result_proof_path,
            evidence.native_result_proof_raw_sha256,
            evidence.native_result_proof_semantic_sha256,
        ),
        (
            "E3a stage ITL proof",
            evidence.stage_itl_proof_path,
            evidence.stage_itl_proof_raw_sha256,
            evidence.stage_itl_proof_semantic_sha256,
        ),
    ):
        reopened = CanonicalJsonProofBinding.bind(path_value)
        if (
            reopened.raw_sha256 != raw_sha256
            or reopened.semantic_sha256 != semantic_sha256
        ):
            raise ValueError(f"{label} changed after evidence materialization")
    result = validate_native_terminal_result_proof_artifact(
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
        now_ns=now_ns,
    )
    timing = validate_stage_itl_timestamp_proof_artifact(
        evidence.stage_itl_proof_path,
        expected_execution_identity=identity,
        expected_root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        now_ns=now_ns,
    )
    if (
        (
            coverage_terminal_sha256 is not None
            and result.terminal_sha256 != coverage_terminal_sha256
        )
        or timing.native_result_proof_path != evidence.native_result_proof_path
        or timing.native_result_proof_raw_sha256
        != evidence.native_result_proof_raw_sha256
        or timing.native_result_proof_semantic_sha256
        != evidence.native_result_proof_semantic_sha256
    ):
        raise ValueError("E3a terminal/timing proof lineage differs from coverage")
    performance = result.performance_counters
    method = identity.method
    if method == "target_only":
        if any(
            field not in performance or performance[field] is not None
            for field in _SAFETY_COUNTERS
        ):
            raise ValueError("E3a Target-only requires explicit N/A safety counters")
        for field in ("updates_launched", "updates_published"):
            if field not in performance or performance[field] is not None:
                raise ValueError(
                    "E3a Target-only requires explicit N/A update counters"
                )
    else:
        if any(_integer_counter(performance, field) != 0 for field in _SAFETY_COUNTERS):
            raise ValueError("E3a Static evidence has a safety violation")
        if any(
            _integer_counter(performance, field) != 0
            for field in ("updates_launched", "updates_published")
        ):
            raise ValueError("E3a Static evidence unexpectedly launched an update")
    if result.updates:
        raise ValueError("E3a allocation-free baseline contains update rows")
    peak_hbm = _integer_counter(performance, "peak_hbm_bytes")
    return _ValidatedCell(
        cell=cell,
        evidence=evidence,
        result=result,
        timing=timing,
        request_identity=_validate_requests(result, timing),
        peak_hbm_bytes=peak_hbm,
    )


def _validate_e3a_execution_rows(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    evidence_cells: tuple[E3aCellExecutionEvidence, ...],
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    inventory_sha256: str,
    now_ns: int,
) -> dict[str, _ValidatedCell]:
    """Deep-reopen every E3a proof before a coverage summary can exist."""

    _require_sha256("E3a execution inventory", inventory_sha256)
    if (
        materialization.stage != "E3a"
        or materialization.expected_cell_count != 360
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.source_decision_sha256
        != protocol_lock.formal_workload_e3a_authorization_sha256
    ):
        raise ValueError("E3a proof coverage requires exact 360-cell materialization")
    cells = {row.cell_id: row for row in materialization.cells}
    if len(cells) != 360:
        raise ValueError("E3a materialization repeats a cell")
    if (
        type(evidence_cells) is not tuple
        or any(type(row) is not E3aCellExecutionEvidence for row in evidence_cells)
        or tuple(row.materialized_cell_id for row in evidence_cells)
        != tuple(sorted(cells))
    ):
        raise ValueError("E3a evidence must exactly cover 360 canonical cells")
    runs = tuple(
        (
            row.execution_identity.run_id,
            row.execution_identity.run_nonce_sha256,
            row.execution_identity.attempt_id,
        )
        for row in evidence_cells
    )
    if len(runs) != len(set(runs)):
        raise ValueError("E3a proof coverage reuses a run identity")
    for label, values in (
        (
            "execution binding",
            tuple(row.execution_binding_sha256 for row in evidence_cells),
        ),
        (
            "terminal result proof",
            tuple(row.native_result_proof_raw_sha256 for row in evidence_cells),
        ),
        (
            "ITL proof",
            tuple(row.stage_itl_proof_raw_sha256 for row in evidence_cells),
        ),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"E3a proof coverage reuses a {label}")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E3a proof coverage requires sealed execution bindings")
    bindings: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for row in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(row)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings:
            raise ValueError("E3a proof coverage reuses an execution binding")
        bindings[cell_id] = verified
    if set(bindings) != set(cells):
        raise ValueError("E3a proof coverage requires one binding per cell")
    evidence = {row.materialized_cell_id: row for row in evidence_cells}
    return {
        cell_id: _validate_cell(
            cell=cells[cell_id],
            evidence=evidence[cell_id],
            execution_binding=bindings[cell_id],
            coverage_terminal_sha256=None,
            protocol_lock=protocol_lock,
            inventory_sha256=inventory_sha256,
            now_ns=now_ns,
        )
        for cell_id in sorted(cells)
    }


def derive_e3a_stage_coverage_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    evidence_cells: tuple[E3aCellExecutionEvidence, ...],
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    inventory_sha256: str,
    now_ns: int,
) -> StageCoverageReceipt:
    """Produce the E3a coverage row only after all 360 proofs deep-reopen."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E3a proof coverage requires exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E3a proof coverage requires exact materialization")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E3a proof coverage time must be non-negative")
    _require_reducer_member(protocol_lock, formal_runtime_authority_manifest)
    validated = _validate_e3a_execution_rows(
        protocol_lock=protocol_lock,
        materialization=materialization,
        evidence_cells=evidence_cells,
        execution_bindings=execution_bindings,
        inventory_sha256=inventory_sha256,
        now_ns=now_ns,
    )
    return _coverage_from_validated(
        protocol_lock=protocol_lock,
        materialization=materialization,
        validated=validated,
    )


def _coverage_from_validated(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    validated: dict[str, _ValidatedCell],
) -> StageCoverageReceipt:
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E3a",
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="E3a",
                cell_id=cell_id,
                status="COMPLETE",
                reason_code="proof_derived_terminal_complete",
                terminal_receipt_sha256=validated[cell_id].result.terminal_sha256,
            )
            for cell_id in sorted(validated)
        ),
    )
    coverage.validate_against(materialization)
    return coverage


def _require_reducer_member(
    protocol_lock: ProtocolLock,
    runtime_manifest: FormalRuntimeAuthorityManifest,
) -> str:
    if type(runtime_manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("E3a reduction requires exact runtime authority manifest")
    if (
        runtime_manifest.sha256
        != protocol_lock.formal_runtime_authority_manifest_sha256
    ):
        raise ValueError("E3a runtime authority manifest differs from ProtocolLock")
    member = runtime_manifest.member("e3a_selection_reducer")
    if (
        member.protocol_sha256 != E3A_STAGED_REDUCTION_PROTOCOL_SHA256
        or member.runner_sha256 != E3A_STAGED_REDUCTION_RUNNER_SHA256
        or member.test_set_sha256 != E3A_STAGED_REDUCTION_TEST_SET_SHA256
    ):
        raise ValueError("E3a reducer source identity differs from ProtocolLock")
    return member.sha256


def _fraction_identity(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def reduce_e3a_staged_selection_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    manifest: E3aStagedEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E3aStagedSelectionArtifact:
    """Deep-replay 360 rows and derive all six E3a outputs."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E3a staged reduction requires exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E3a staged reduction requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E3a staged reduction requires exact coverage")
    if type(manifest) is not E3aStagedEvidenceManifest:
        raise TypeError("E3a staged reduction requires exact evidence manifest")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E3a reduction time must be non-negative")
    reducer_member_sha256 = _require_reducer_member(
        protocol_lock, formal_runtime_authority_manifest
    )
    if (
        materialization.stage != "E3a"
        or materialization.expected_cell_count != 360
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.source_decision_sha256
        != protocol_lock.formal_workload_e3a_authorization_sha256
    ):
        raise ValueError("E3a reduction requires exact 360-cell materialization")
    validated = _validate_e3a_execution_rows(
        protocol_lock=protocol_lock,
        materialization=materialization,
        evidence_cells=manifest.cells,
        execution_bindings=execution_bindings,
        inventory_sha256=manifest.inventory_sha256,
        now_ns=now_ns,
    )
    proof_derived_coverage = _coverage_from_validated(
        protocol_lock=protocol_lock,
        materialization=materialization,
        validated=validated,
    )
    if coverage != proof_derived_coverage:
        raise ValueError("E3a coverage is not derived from the exact execution proofs")
    if (
        manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.reducer_authority_member_sha256 != reducer_member_sha256
    ):
        raise ValueError("E3a evidence manifest differs from signed lineage")
    cells = {row.cell_id: row for row in materialization.cells}

    targets: dict[tuple[int, str, int], _ValidatedCell] = {}
    statics: dict[tuple[int, str, int, int], _ValidatedCell] = {}
    models = {cell.model for cell in cells.values()}
    for row in validated.values():
        dimensions = dict(row.cell.dimensions)
        context = dimensions.get("context")
        regime = dimensions.get("regime")
        concurrency = dimensions.get("concurrency")
        width = dimensions.get("width")
        if (
            type(context) is not int
            or type(regime) is not str
            or type(concurrency) is not int
        ):
            raise ValueError("E3a materialized cell lacks capacity axes")
        if row.cell.method_role == "Target-only":
            key = (context, regime, concurrency)
            if width is not None or key in targets:
                raise ValueError("E3a repeats a Target-only capacity slice")
            targets[key] = row
        elif row.cell.method_role == "Static":
            if type(width) is not int:
                raise ValueError("E3a Static cell lacks exact width")
            key = (context, regime, concurrency, width)
            if key in statics:
                raise ValueError("E3a repeats a Static capacity slice")
            statics[key] = row
        else:
            raise ValueError("E3a accepts only Target-only and Static")
    if len(models) != 1 or len(targets) != 96 or len(statics) != 264:
        raise ValueError("E3a role/axis cardinality differs from 360-row protocol")
    for context in LONG_CONTEXT_ANCHORS:
        for regime in CONTEXT_REGIMES:
            for concurrency in E3A_CONCURRENCY_GRID:
                if (context, regime, concurrency) not in targets or any(
                    (context, regime, concurrency, width) not in statics
                    for width in DRAFT_WIDTHS
                ):
                    raise ValueError("E3a full anchor grid is incomplete")

    observations: list[E3aCapacityObservation] = []
    for cell_id in sorted(validated):
        row = validated[cell_id]
        dimensions = dict(row.cell.dimensions)
        context = int(dimensions["context"])
        regime = str(dimensions["regime"])
        concurrency = int(dimensions["concurrency"])
        if row.cell.method_role == "Target-only":
            target_cell_id = None
            ratio_numerator = None
            ratio_denominator = None
            width = None
        else:
            width = int(dimensions["width"])
            target = targets[(context, regime, concurrency)]
            if row.request_identity != target.request_identity:
                raise ValueError("E3a Static differs from Target-only token trajectory")
            ratio = Fraction(
                row.timing.throughput_numerator_tokens
                * target.timing.throughput_window_ns,
                row.timing.throughput_window_ns
                * target.timing.throughput_numerator_tokens,
            )
            target_cell_id = target.cell.cell_id
            ratio_numerator, ratio_denominator = _fraction_identity(ratio)
        observations.append(
            E3aCapacityObservation(
                cell_id=row.cell.cell_id,
                method_role=row.cell.method_role,  # type: ignore[arg-type]
                context=context,
                regime=regime,
                concurrency=concurrency,
                width=width,
                throughput_tokens=row.timing.throughput_numerator_tokens,
                throughput_window_ns=row.timing.throughput_window_ns,
                peak_hbm_bytes=row.peak_hbm_bytes,
                target_cell_id=target_cell_id,
                static_target_ratio_numerator=ratio_numerator,
                static_target_ratio_denominator=ratio_denominator,
                execution_evidence_sha256=row.evidence.sha256,
                terminal_sha256=row.result.terminal_sha256,
                timing_authority_sha256=row.timing.sha256,
            )
        )
    observation_rows = tuple(observations)
    static_observations = tuple(
        row
        for row in observation_rows
        if row.method_role == "Static" and row.context in LONG_CONTEXT_ANCHORS
    )
    median_throughput: dict[tuple[int, int], Fraction] = {}
    for width in DRAFT_WIDTHS:
        for concurrency in E3A_CONCURRENCY_GRID:
            rows = tuple(
                row.throughput
                for row in static_observations
                if row.width == width and row.concurrency == concurrency
            )
            if len(rows) != len(LONG_CONTEXT_ANCHORS) * len(CONTEXT_REGIMES):
                raise ValueError("E3a width/load median lacks full primary contexts")
            median_throughput[(width, concurrency)] = statistics.median(rows)
    best_by_load = {
        concurrency: max(
            median_throughput[(width, concurrency)] for width in DRAFT_WIDTHS
        )
        for concurrency in E3A_CONCURRENCY_GRID
    }
    threshold = Fraction(9, 10) * max(best_by_load.values())
    eligible_loads = tuple(
        concurrency
        for concurrency in E3A_CONCURRENCY_GRID
        if best_by_load[concurrency] >= threshold
    )
    if not eligible_loads:
        raise ValueError("E3a reference-load reducer has no eligible load")
    common_load = min(eligible_loads)
    width_scores = []
    for width in DRAFT_WIDTHS:
        rows = tuple(
            row
            for row in static_observations
            if row.width == width and row.concurrency == common_load
        )
        ratios = tuple(row.static_target_ratio for row in rows)
        if len(rows) != len(LONG_CONTEXT_ANCHORS) * len(CONTEXT_REGIMES) or any(
            ratio is None for ratio in ratios
        ):
            raise ValueError("E3a matched-width reduction lacks paired ratios")
        width_scores.append(
            (
                min(ratio for ratio in ratios if ratio is not None),
                statistics.median(row.throughput for row in rows),
                width,
            )
        )
    _, _, matched_width = max(width_scores, key=lambda row: (row[0], row[1], -row[2]))

    crossovers = []
    for context in LONG_CONTEXT_ANCHORS:
        for regime in CONTEXT_REGIMES:
            for width in DRAFT_WIDTHS:
                rows = tuple(
                    row
                    for row in static_observations
                    if row.context == context
                    and row.regime == regime
                    and row.width == width
                )
                crossover = next(
                    (
                        row.concurrency
                        for row in sorted(rows, key=lambda item: item.concurrency)
                        if row.static_target_ratio is not None
                        and row.static_target_ratio <= 1
                    ),
                    None,
                )
                crossovers.append(
                    {
                        "context": context,
                        "regime": regime,
                        "width": width,
                        "first_static_target_ratio_lte_one": crossover,
                    }
                )
    drift_rows = tuple(
        row
        for row in static_observations
        if row.width == matched_width and row.concurrency == common_load
    )
    capacity_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e3a_baseline_capacity_envelope",
            "observation_sha256s": tuple(row.sha256 for row in observation_rows),
        }
    )
    reference_load_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e3a_reference_load",
            "common_load": common_load,
            "threshold_fraction": (9, 10),
            "best_static_throughput_by_load": tuple(
                (
                    load,
                    *_fraction_identity(best_by_load[load]),
                )
                for load in E3A_CONCURRENCY_GRID
            ),
            "capacity_sha256": capacity_sha256,
        }
    )
    matched_width_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e3a_matched_width",
            "matched_width": matched_width,
            "common_load": common_load,
            "scores": tuple(
                (
                    width,
                    *_fraction_identity(worst),
                    *_fraction_identity(median),
                )
                for worst, median, width in sorted(width_scores, key=lambda row: row[2])
            ),
            "capacity_sha256": capacity_sha256,
        }
    )
    crossover_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e3a_static_target_crossover",
            "rows": tuple(crossovers),
            "capacity_sha256": capacity_sha256,
        }
    )
    drift_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "e3a_drift_witness",
            "matched_width": matched_width,
            "common_load": common_load,
            "rows": tuple(
                (
                    row.context,
                    row.regime,
                    row.cell_id,
                    row.static_target_ratio_numerator,
                    row.static_target_ratio_denominator,
                )
                for row in drift_rows
            ),
            "capacity_sha256": capacity_sha256,
        }
    )
    locked_by_name = {
        "baseline_capacity_envelope": capacity_sha256,
        "drift_witness": drift_sha256,
        "e1_reference_load": reference_load_sha256,
        "matched_width": matched_width_sha256,
        "static_target_crossover": crossover_sha256,
        "width_selection_rule": E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
    }
    return E3aStagedSelectionArtifact(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        inventory_sha256=manifest.inventory_sha256,
        evidence_manifest_sha256=manifest.sha256,
        reducer_authority_member_sha256=reducer_member_sha256,
        reducer_protocol_sha256=E3A_STAGED_REDUCTION_PROTOCOL_SHA256,
        model=next(iter(models)),
        matched_width=matched_width,
        common_load=common_load,
        observations=observation_rows,
        locked_outputs=tuple(
            E3aLockedOutput(name=name, content_sha256=locked_by_name[name])
            for name in _LOCKED_OUTPUT_NAMES
        ),
    )


def build_e3a_staged_selection_receipt(
    artifact: E3aStagedSelectionArtifact,
) -> E3aStagedSelectionReceipt:
    if type(artifact) is not E3aStagedSelectionArtifact:
        raise TypeError("E3a receipt reducer requires exact staged artifact")
    receipt = E3aStagedSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=artifact.protocol_lock_sha256,
        registry_sha256=artifact.registry_sha256,
        e3a_materialization_receipt_sha256=(artifact.materialization_receipt_sha256),
        e3a_coverage_receipt_sha256=artifact.coverage_receipt_sha256,
        evidence_manifest_sha256=artifact.evidence_manifest_sha256,
        selection_artifact_sha256=artifact.sha256,
        reducer_authority_member_sha256=artifact.reducer_authority_member_sha256,
        model=artifact.model,
        matched_width=artifact.matched_width,
        common_load=artifact.common_load,
        locked_outputs=artifact.locked_outputs,
    )
    receipt.validate_artifact(artifact)
    return receipt


__all__ = [
    "E3A_STAGED_REDUCTION_PROTOCOL_SHA256",
    "E3A_STAGED_REDUCTION_RUNNER_SHA256",
    "E3A_STAGED_REDUCTION_TEST_SET_SHA256",
    "E3aCapacityObservation",
    "E3aCellExecutionEvidence",
    "E3aLockedOutput",
    "E3aStagedEvidenceManifest",
    "E3aStagedSelectionArtifact",
    "E3aStagedSelectionReceipt",
    "SignedE3aStagedSelectionReceipt",
    "build_e3a_staged_selection_receipt",
    "derive_e3a_stage_coverage_from_proofs",
    "reduce_e3a_staged_selection_from_proofs",
]
