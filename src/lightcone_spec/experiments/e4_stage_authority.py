"""Proof-derived two-phase E4 mechanism-selection authority.

The E4 screen and local factorial are separate immutable materializations.
Each reducer deep-opens the exact native-result and client-timestamp proofs
through verifier-sealed serving bindings; a signed summary alone is never an
input authority.  Profiler rows are kept outside the headline selection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import Literal

from lightcone_spec.experiments.e1_stage_authority import (
    _request_identity,
    _validated_cell,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    E4_LOADS,
    E4_SCREEN_FACTOR_LEVELS,
    E4_TRAFFIC,
    MaterializedCell,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

E4_SELECTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e4_two_phase_selection_protocol",
        "screen": "strength2_8_rows_x_3_loads_x_2_traffic",
        "local": "winner_neighborhood_2pow4_x_3_loads_x_2_traffic",
        "eligibility": "all_six_strata_safe_complete_and_published",
        "ranking": (
            "maximum_worst_stratum_request_token_rate",
            "minimum_peak_hbm_bytes",
            "minimum_p99_itl_us",
            "minimum_exposed_update_us",
            "configuration_sha256",
        ),
        "timing": "integer_native_client_timestamps_only",
        "profiler": "separate_non_headline_materialization",
    }
)

# The screen levels are deliberately wide.  The local factorial uses a fixed
# source-owned inward neighbour for numeric factors and the exact two priority
# levels.  These values are code/protocol identity, never caller-selected.
E4_LOCAL_NEIGHBORHOOD_BY_WINNER = (
    ("update_stride", ((1, (1, 5)), (50, (30, 50)))),
    ("microbatch", ((1, (1, 2)), (8, (4, 8)))),
    ("coalescing", ((1, (1, 2)), (8, (4, 8)))),
    (
        "stream_priority",
        (("default", ("default", "high")), ("high", ("default", "high"))),
    ),
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


E4Phase = Literal["screen", "local"]
E4FactorValue = str | int
E4FactorConfiguration = tuple[tuple[str, E4FactorValue], ...]
E4FactorNeighborhoods = tuple[tuple[str, E4FactorValue, E4FactorValue], ...]


def _configuration(cell: MaterializedCell, phase: E4Phase) -> E4FactorConfiguration:
    dimensions = dict(cell.dimensions)
    expected_names = tuple(name for name, _ in E4_SCREEN_FACTOR_LEVELS)
    configuration = tuple((name, dimensions.get(name)) for name in expected_names)
    if any(type(value) not in {str, int} for _, value in configuration):
        raise ValueError("E4 cell lacks an exact operational-factor configuration")
    row_field = "screen_row" if phase == "screen" else "local_row"
    if type(dimensions.get(row_field)) is not int:
        raise ValueError("E4 cell lacks its exact factorial row identity")
    return configuration  # type: ignore[return-value]


def _neighborhoods(configuration: E4FactorConfiguration) -> E4FactorNeighborhoods:
    source = dict(configuration)
    result = []
    for name, rows in E4_LOCAL_NEIGHBORHOOD_BY_WINNER:
        winner = source.get(name)
        matches = tuple(levels for level, levels in rows if level == winner)
        if len(matches) != 1:
            raise ValueError("E4 screen winner lies outside neighborhood policy")
        result.append((name, matches[0][0], matches[0][1]))
    return tuple(result)


@dataclass(frozen=True)
class E4CellExecutionEvidence:
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
            raise ValueError("only E4 cell evidence schema 1 is supported")
        _require_sha256("E4 evidence cell", self.materialized_cell_id)
        _require_sha256("E4 execution binding", self.execution_binding_sha256)
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("E4 evidence requires an exact execution identity")
        if self.execution_identity.materialized_cell_id != self.materialized_cell_id:
            raise ValueError("E4 execution identity names another cell")
        for label, path, raw_sha256, semantic_sha256 in (
            (
                "E4 native result proof",
                self.native_result_proof_path,
                self.native_result_proof_raw_sha256,
                self.native_result_proof_semantic_sha256,
            ),
            (
                "E4 stage ITL proof",
                self.stage_itl_proof_path,
                self.stage_itl_proof_raw_sha256,
                self.stage_itl_proof_semantic_sha256,
            ),
        ):
            _require_absolute_path(label, path)
            binding = CanonicalJsonProofBinding.bind(path)
            if (
                binding.raw_sha256 != raw_sha256
                or binding.semantic_sha256 != semantic_sha256
            ):
                raise ValueError(f"{label} content changed after binding")
        if self.native_result_proof_path == self.stage_itl_proof_path:
            raise ValueError("E4 result and timing proof paths must differ")

    @classmethod
    def bind(
        cls,
        *,
        execution_binding: VerifiedFormalServingExecutionBinding,
        native_result_proof_path: str,
        stage_itl_proof_path: str,
    ) -> E4CellExecutionEvidence:
        verified = require_verified_formal_serving_execution_binding(execution_binding)
        if verified.subject.stage != "E4":
            raise ValueError("E4 evidence cannot consume another stage binding")
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
class E4StagedEvidenceManifest:
    schema_version: int
    phase: E4Phase
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    upstream_signed_authority_sha256: str
    inventory_sha256: str
    cells: tuple[E4CellExecutionEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.phase not in {"screen", "local"}:
            raise ValueError("E4 evidence schema/phase is unsupported")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("upstream authority", self.upstream_signed_authority_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"E4 evidence {label}", value)
        ids = tuple(row.materialized_cell_id for row in self.cells)
        runs = tuple(
            (
                row.execution_identity.run_id,
                row.execution_identity.run_nonce_sha256,
                row.execution_identity.attempt_id,
            )
            for row in self.cells
        )
        if (
            not self.cells
            or any(type(row) is not E4CellExecutionEvidence for row in self.cells)
            or ids != tuple(sorted(set(ids)))
            or len(runs) != len(set(runs))
        ):
            raise ValueError("E4 evidence cells/runs are not exact unique rows")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E4ConfigurationEvaluation:
    configuration: E4FactorConfiguration
    cell_ids: tuple[str, ...]
    minimum_request_rate_numerator: int
    minimum_request_rate_denominator: int
    peak_hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int

    def __post_init__(self) -> None:
        expected_names = tuple(name for name, _ in E4_SCREEN_FACTOR_LEVELS)
        if tuple(name for name, _ in self.configuration) != expected_names:
            raise ValueError("E4 evaluation factors are not canonical")
        if len(self.cell_ids) != len(E4_LOADS) * len(
            E4_TRAFFIC
        ) or self.cell_ids != tuple(sorted(set(self.cell_ids))):
            raise ValueError("E4 evaluation lacks exact six-stratum cell coverage")
        for digest in self.cell_ids:
            _require_sha256("E4 evaluation cell", digest)
        for label, value in (
            ("request-rate numerator", self.minimum_request_rate_numerator),
            ("request-rate denominator", self.minimum_request_rate_denominator),
            ("peak HBM", self.peak_hbm_bytes),
            ("p99 ITL", self.p99_itl_us),
            ("exposed update", self.exposed_update_us),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"E4 {label} must be a positive integer")
        if (
            math.gcd(
                self.minimum_request_rate_numerator,
                self.minimum_request_rate_denominator,
            )
            != 1
        ):
            raise ValueError("E4 request-rate ratio must be reduced")

    @cached_property
    def configuration_sha256(self) -> str:
        return content_sha256(self.configuration)

    @property
    def minimum_request_rate(self) -> Fraction:
        return Fraction(
            self.minimum_request_rate_numerator,
            self.minimum_request_rate_denominator,
        )


@dataclass(frozen=True)
class E4StageSelectionReceipt:
    schema_version: int
    phase: E4Phase
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    upstream_signed_authority_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    model: str
    lightcone_recipe_sha256: str
    evaluations: tuple[E4ConfigurationEvaluation, ...]
    winner_configuration: E4FactorConfiguration
    factor_neighborhoods: E4FactorNeighborhoods | None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.phase not in {"screen", "local"}:
            raise ValueError("E4 selection schema/phase is unsupported")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("upstream authority", self.upstream_signed_authority_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("LightCone recipe", self.lightcone_recipe_sha256),
        ):
            _require_sha256(f"E4 selection {label}", value)
        evaluation_ids = tuple(row.configuration_sha256 for row in self.evaluations)
        if (
            not self.evaluations
            or evaluation_ids != tuple(sorted(set(evaluation_ids)))
            or self.winner_configuration
            not in {row.configuration for row in self.evaluations}
        ):
            raise ValueError("E4 selection evaluations/winner are not exact")
        expected_neighborhoods = (
            _neighborhoods(self.winner_configuration)
            if self.phase == "screen"
            else None
        )
        if self.factor_neighborhoods != expected_neighborhoods:
            raise ValueError("E4 selected neighborhood differs from source policy")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE4StageSelectionReceipt:
    payload: E4StageSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        manifest: E4StagedEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E4StageSelectionReceipt:
        if type(self.payload) is not E4StageSelectionReceipt:
            raise TypeError("signed E4 selection payload has the wrong type")
        expected = reduce_e4_stage_selection_from_proofs(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            upstream_signed_authority_sha256=(
                self.payload.upstream_signed_authority_sha256
            ),
            manifest=manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E4 selection differs from proof reducer")
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


E4_PROFILER_COMPLETION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e4_profiler_completion_protocol",
        "materialization": "exact_three_profiler_only_rows",
        "source": "signed_local_factorial_winner",
        "coverage": "registry_verified_all_complete_terminal_receipts",
        "claim": "profiler_completion_only_excluded_from_headline_timing",
    }
)


@dataclass(frozen=True)
class E4ProfilerTerminalCompletion:
    materialized_cell_id: str
    terminal_receipt_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("E4 profiler completion cell", self.materialized_cell_id)
        _require_sha256("E4 profiler completion terminal", self.terminal_receipt_sha256)


@dataclass(frozen=True)
class E4ProfilerCompletionReceipt:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    registry_verification_receipt_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    signed_local_selection_sha256: str
    terminals: tuple[E4ProfilerTerminalCompletion, ...]
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.protocol_sha256 != (
            E4_PROFILER_COMPLETION_PROTOCOL_SHA256
        ):
            raise ValueError("E4 profiler completion schema/protocol is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("registry verification", self.registry_verification_receipt_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("signed local selection", self.signed_local_selection_sha256),
        ):
            _require_sha256(f"E4 profiler completion {label}", digest)
        if (
            len(self.terminals) != 3
            or any(
                type(row) is not E4ProfilerTerminalCompletion for row in self.terminals
            )
            or tuple(row.materialized_cell_id for row in self.terminals)
            != tuple(sorted({row.materialized_cell_id for row in self.terminals}))
            or len({row.terminal_receipt_sha256 for row in self.terminals}) != 3
        ):
            raise ValueError("E4 profiler completion terminal coverage is not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE4ProfilerCompletionReceipt:
    payload: E4ProfilerCompletionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        registry_verification_receipt: object,
        materialization: StageMaterializationReceipt,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E4ProfilerCompletionReceipt:
        if type(self.payload) is not E4ProfilerCompletionReceipt:
            raise TypeError("signed E4 profiler completion payload is not exact")
        expected = reduce_e4_profiler_completion_from_registry(
            registry_verification_receipt=registry_verification_receipt,
            materialization=materialization,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E4 profiler completion differs from reducer")
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


def require_e4_profiler_completion_authority() -> None:
    """Fail closed until a release profiler and raw-proof adapter both exist."""

    from lightcone_spec.experiments.profiler_authority import (
        PROFILER_RAW_PROFILE_MISSING_REASON,
        PROFILER_TOOL_UNAVAILABLE_REASON,
        RELEASE_PROFILER_TOOL_ALLOWLIST,
        ProfilerAuthorityBlocked,
    )

    if not RELEASE_PROFILER_TOOL_ALLOWLIST:
        raise ProfilerAuthorityBlocked(PROFILER_TOOL_UNAVAILABLE_REASON)
    # A reviewed release tool is necessary but not sufficient.  This reducer's
    # historical registry-only interface cannot prove an nsys/ncu output, tool
    # version, exclusive assignment, or terminal pointer.  A future adapter
    # must add those typed inputs before removing this closed branch.
    raise ProfilerAuthorityBlocked(PROFILER_RAW_PROFILE_MISSING_REASON)


def reduce_e4_profiler_completion_from_registry(
    *,
    registry_verification_receipt: object,
    materialization: StageMaterializationReceipt,
    now_ns: int,
) -> E4ProfilerCompletionReceipt:
    """Reduce a completion-only receipt from the verified profiler append."""

    require_e4_profiler_completion_authority()

    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E4 profiler completion requires exact durable registry")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E4 profiler completion requires exact materialization")
    registry_verification_receipt.revalidate(current_ns=now_ns)
    lock = registry_verification_receipt.signed_protocol_lock.payload
    signed_materializations = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_materializations
        if row.payload == materialization
    )
    signed_coverages = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_coverage
        if row.payload.materialization_receipt_sha256 == materialization.sha256
    )
    local_selections = tuple(
        row
        for row in registry_verification_receipt.cumulative_signed_e4_stage_selections
        if row.payload.phase == "local"
        and row.sha256 == materialization.source_decision_sha256
    )
    expected_profilers = {"nvtx", "nsight_systems", "nsight_compute"}
    if (
        materialization.stage != "E4"
        or materialization.protocol_lock_sha256 != lock.sha256
        or materialization.materialization_rule
        != "three_profiler_only_rows_separate_from_headline"
        or len(materialization.cells) != 3
        or {dict(cell.dimensions).get("profiler") for cell in materialization.cells}
        != expected_profilers
        or any(
            cell.task != "mechanism_profile_only"
            or cell.publication_policy != "diagnostic_only"
            for cell in materialization.cells
        )
        or len(signed_materializations) != 1
        or len(signed_coverages) != 1
        or len(local_selections) != 1
    ):
        raise ValueError("E4 profiler completion lineage/materialization is not exact")
    coverage = signed_coverages[0].payload
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E4 profiler completion requires all-COMPLETE coverage")
    receipt = E4ProfilerCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        registry_verification_receipt_sha256=registry_verification_receipt.sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        signed_local_selection_sha256=local_selections[0].sha256,
        terminals=tuple(
            E4ProfilerTerminalCompletion(
                materialized_cell_id=row.cell_id,
                terminal_receipt_sha256=row.terminal_receipt_sha256,
            )
            for row in coverage.dispositions
        ),
        protocol_sha256=E4_PROFILER_COMPLETION_PROTOCOL_SHA256,
    )
    receipt.__post_init__()
    return receipt


def e4_profiler_completion_receipt_to_dict(
    value: E4ProfilerCompletionReceipt,
) -> dict[str, object]:
    if type(value) is not E4ProfilerCompletionReceipt:
        raise TypeError("E4 profiler completion codec requires an exact receipt")
    row = asdict(value)
    row["terminals"] = [asdict(item) for item in value.terminals]
    return {**row, "receipt_sha256": value.sha256}


def e4_profiler_completion_receipt_from_dict(
    value: object,
) -> E4ProfilerCompletionReceipt:
    if type(value) is not dict or set(value) != {
        *E4ProfilerCompletionReceipt.__dataclass_fields__,
        "receipt_sha256",
    }:
        raise ValueError("E4 profiler completion fields differ from schema")
    row = dict(value)
    declared = _require_sha256(
        "E4 profiler completion receipt", row.pop("receipt_sha256")
    )
    raw_terminals = row["terminals"]
    if type(raw_terminals) is not list:
        raise TypeError("E4 profiler completion terminals must be an array")
    row["terminals"] = tuple(
        E4ProfilerTerminalCompletion(**item)
        for item in raw_terminals
        if type(item) is dict
        and set(item) == set(E4ProfilerTerminalCompletion.__dataclass_fields__)
    )
    if len(row["terminals"]) != len(raw_terminals):
        raise ValueError("E4 profiler completion terminal fields differ")
    receipt = E4ProfilerCompletionReceipt(**row)  # type: ignore[arg-type]
    if receipt.sha256 != declared:
        raise ValueError("E4 profiler completion digest differs")
    return receipt


def signed_e4_profiler_completion_to_dict(
    value: SignedE4ProfilerCompletionReceipt,
) -> dict[str, object]:
    if type(value) is not SignedE4ProfilerCompletionReceipt:
        raise TypeError("signed E4 profiler completion codec requires exact wrapper")
    return {
        "payload": e4_profiler_completion_receipt_to_dict(value.payload),
        "payload_sha256": value.payload_sha256,
        "challenge": asdict(value.challenge),
        "attestation": asdict(value.attestation),
        "signed_receipt_sha256": value.sha256,
    }


def signed_e4_profiler_completion_from_dict(
    value: object,
) -> SignedE4ProfilerCompletionReceipt:
    if type(value) is not dict or set(value) != {
        "payload",
        "payload_sha256",
        "challenge",
        "attestation",
        "signed_receipt_sha256",
    }:
        raise ValueError("signed E4 profiler completion fields differ")
    row = dict(value)
    declared = _require_sha256(
        "signed E4 profiler completion", row.pop("signed_receipt_sha256")
    )
    for label in ("challenge", "attestation"):
        if type(row[label]) is not dict:
            raise TypeError(f"signed E4 profiler {label} must be an object")
    signed = SignedE4ProfilerCompletionReceipt(
        payload=e4_profiler_completion_receipt_from_dict(row["payload"]),
        payload_sha256=row["payload_sha256"],
        challenge=AttestationChallenge(**row["challenge"]),
        attestation=SignedAttestation(**row["attestation"]),
    )
    if signed.sha256 != declared:
        raise ValueError("signed E4 profiler completion digest differs")
    return signed


def _phase(materialization: StageMaterializationReceipt) -> E4Phase:
    if materialization.materialization_rule == (
        "strength2_8_rows_x_3_loads_x_2_traffic"
    ):
        return "screen"
    if materialization.materialization_rule == (
        "winner_neighborhood_2pow4_x_3_loads_x_2_traffic"
    ):
        return "local"
    raise ValueError("E4 headline selection cannot consume profiler/foreign rows")


def _ranking(row: E4ConfigurationEvaluation) -> tuple[object, ...]:
    return (
        -row.minimum_request_rate,
        row.peak_hbm_bytes,
        row.p99_itl_us,
        row.exposed_update_us,
        row.configuration_sha256,
    )


def _evaluate_configuration(
    configuration: E4FactorConfiguration,
    rows: tuple[object, ...],
) -> E4ConfigurationEvaluation:
    rates: list[Fraction] = []
    cell_ids: list[str] = []
    peak_hbm: list[int] = []
    p99_itl: list[int] = []
    exposed_update: list[int] = []
    for row in rows:
        metrics = row.metrics
        output_tokens = sum(metric.output_tokens for metric in metrics)
        latency_ns = sum(metric.latency_ns for metric in metrics)
        if output_tokens < 1 or latency_ns < 1:
            raise ValueError("E4 eligible cell has no completed request timing")
        rates.append(Fraction(output_tokens * 1_000_000_000, latency_ns))
        cell_ids.append(row.cell.cell_id)
        peak_hbm.append(row.peak_hbm_bytes)
        p99_itl.append(max(math.ceil(metric.p99_itl_ns / 1_000) for metric in metrics))
        exposed_update.append(row.exposed_update_us)
    worst = min(rates)
    return E4ConfigurationEvaluation(
        configuration=configuration,
        cell_ids=tuple(sorted(cell_ids)),
        minimum_request_rate_numerator=worst.numerator,
        minimum_request_rate_denominator=worst.denominator,
        peak_hbm_bytes=max(peak_hbm),
        p99_itl_us=max(p99_itl),
        exposed_update_us=max(exposed_update),
    )


def reduce_e4_stage_selection_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    upstream_signed_authority_sha256: str,
    manifest: E4StagedEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E4StageSelectionReceipt:
    """Deep-open one E4 headline phase and choose one configuration."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E4 reduction requires an exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E4 reduction requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E4 reduction requires exact coverage")
    if type(manifest) is not E4StagedEvidenceManifest:
        raise TypeError("E4 reduction requires exact evidence manifest")
    _require_sha256("E4 upstream signed authority", upstream_signed_authority_sha256)
    phase = _phase(materialization)
    expected_count = 48 if phase == "screen" else 96
    if (
        materialization.stage != "E4"
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.source_decision_sha256 != upstream_signed_authority_sha256
        or len(materialization.cells) != expected_count
        or manifest.phase != phase
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.upstream_signed_authority_sha256 != upstream_signed_authority_sha256
    ):
        raise ValueError("E4 staged evidence differs from exact lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E4 selection requires all-COMPLETE coverage")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell = {}
    for row in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(row)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E4 selection reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_cells = {cell.cell_id for cell in materialization.cells}
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    if (
        set(evidence_by_cell) != expected_cells
        or set(bindings_by_cell) != expected_cells
        or any(cell.method_role != "LightCone" for cell in materialization.cells)
    ):
        raise ValueError("E4 proof/binding/materialized coverage is not exact")
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E4",
        )
        for cell in materialization.cells
    }
    grouped: dict[E4FactorConfiguration, list[object]] = {}
    strata: dict[tuple[str, str], list[object]] = {}
    recipes = {cell.recipe_sha256 for cell in materialization.cells}
    models = {cell.model for cell in materialization.cells}
    if len(recipes) != 1 or None in recipes or len(models) != 1:
        raise ValueError("E4 phase must use one exact model/LightCone recipe")
    for cell in materialization.cells:
        row = validated[cell.cell_id]
        dimensions = dict(cell.dimensions)
        if row.safety_reasons or row.published_updates < 1:
            continue
        configuration = _configuration(cell, phase)
        grouped.setdefault(configuration, []).append(row)
        stratum = (str(dimensions.get("load")), str(dimensions.get("traffic")))
        strata.setdefault(stratum, []).append(row)
    expected_strata = {(load, traffic) for load in E4_LOADS for traffic in E4_TRAFFIC}
    if set(strata) != expected_strata:
        raise ValueError("E4 safe evidence lacks one or more traffic/load strata")
    for rows in strata.values():
        identities = {_request_identity(row.metrics) for row in rows}
        if len(identities) != 1:
            raise ValueError("E4 configurations use different requests/trajectories")
    expected_per_configuration = len(E4_LOADS) * len(E4_TRAFFIC)
    eligible = tuple(
        sorted(
            (
                _evaluate_configuration(configuration, tuple(rows))
                for configuration, rows in grouped.items()
                if len(rows) == expected_per_configuration
                and {
                    (
                        str(dict(row.cell.dimensions).get("load")),
                        str(dict(row.cell.dimensions).get("traffic")),
                    )
                    for row in rows
                }
                == expected_strata
            ),
            key=lambda row: row.configuration_sha256,
        )
    )
    if not eligible:
        raise ValueError("E4 has no safe complete operational configuration")
    winner = min(eligible, key=_ranking).configuration
    receipt = E4StageSelectionReceipt(
        schema_version=1,
        phase=phase,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        upstream_signed_authority_sha256=upstream_signed_authority_sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        model=next(iter(models)),
        lightcone_recipe_sha256=next(iter(recipes)),  # type: ignore[arg-type]
        evaluations=eligible,
        winner_configuration=winner,
        factor_neighborhoods=_neighborhoods(winner) if phase == "screen" else None,
    )
    receipt.__post_init__()
    return receipt


__all__ = [
    "E4_LOCAL_NEIGHBORHOOD_BY_WINNER",
    "E4_PROFILER_COMPLETION_PROTOCOL_SHA256",
    "E4_SELECTION_PROTOCOL_SHA256",
    "E4CellExecutionEvidence",
    "E4ConfigurationEvaluation",
    "E4ProfilerCompletionReceipt",
    "E4ProfilerTerminalCompletion",
    "E4StageSelectionReceipt",
    "E4StagedEvidenceManifest",
    "SignedE4ProfilerCompletionReceipt",
    "SignedE4StageSelectionReceipt",
    "e4_profiler_completion_receipt_from_dict",
    "e4_profiler_completion_receipt_to_dict",
    "reduce_e4_profiler_completion_from_registry",
    "reduce_e4_stage_selection_from_proofs",
    "require_e4_profiler_completion_authority",
    "signed_e4_profiler_completion_from_dict",
    "signed_e4_profiler_completion_to_dict",
]
