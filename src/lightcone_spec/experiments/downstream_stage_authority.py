"""Proof-derived authorities for the post-E4 formal DAG.

The confirmation stages cannot be materialized from a caller supplied block
count.  E3b begins with exactly four excluded pilot blocks, reopens every
terminal and client-timestamp proof through a verifier-sealed execution
binding, and deterministically applies the preregistered power rule.  The
resulting signed power-prefix receipt is the only value that may append the
12--20 final-block prefix.

Later post-E4 reducers build on the same proof-manifest shape.  Keeping the
shape here avoids reintroducing the historical eager registry or a signed
summary that never reopens its cell evidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

from lightcone_spec.experiments.e1_stage_authority import (
    _paired_confidence_lower,
    _request_identity,
    _validated_cell,
)
from lightcone_spec.experiments.formal_failure_execution import (
    VerifiedFormalFailureExecutionBinding,
    require_verified_formal_failure_execution_binding,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_slo_metrics import (
    FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
    FormalSloGoodputObservation,
    FormalSloRequestEvidence,
    reduce_formal_slo_goodput,
    require_paired_primary_goodputs,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    E1A_FIXED_VERIFICATION_BUDGET,
    E1A_NATIVE_VERIFICATION_BUDGET,
    E3B_CONTEXTS,
    E3B_LOADS,
    E3B_REGIMES,
    E3B_WIDTH_PANELS,
    E5_BACKENDS,
    E5_CLOSED_LOOP_CONCURRENCY,
    E5_COHORT_COUNTS,
    E5_COHORT_DISTRIBUTIONS,
    E5_OPEN_LOOP_LOAD_FACTORS,
    E5_TOPOLOGIES,
    E5_TRACE_AND_SOAK_ARRIVALS,
    FORMAL_METHOD_ROLES,
    E5SelectedP99Anchor,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.statistics import (
    PILOT_BLOCK_COUNT,
    PRIMARY_CONTRASTS,
    MultiplicityDecision,
    PairedBcaContrast,
    PilotBlock,
    PowerSizingPlan,
    holm_primary_contrasts,
    paired_bca_contrast,
    preregister_power_sizing,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

E3B_POWER_PREFIX_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_e3b_proof_derived_power_prefix_protocol",
        "pilot_blocks": 4,
        "pilot_disposition": "excluded_tuning_only",
        "pilot_cell_count": 4 * 480,
        "primary_roles": ("Static", "TTS", "LightCone"),
        "goodput_protocol_sha256": FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        "goodput": "slo_qualified_output_tokens_per_scored_wall_window_ns",
        "power_unit": "each_exact_scientific_stratum_has_four_pilots",
        "stage_final_N": "maximum_READY_N_across_all_96_strata",
        "pairing": "exact_block_stratum_request_and_token_trajectory",
        "power": "preregister_power_sizing_3pct_80pct_holm_first_threshold",
        "final_prefix": "first_N_of_12_to_20",
        "confirmation_data_visible": False,
    }
)
E3B_CONFIRMATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_e3b_proof_derived_confirmation_protocol",
        "final_rows": "96_exact_scientific_families_x_5_roles_x_powered_blocks",
        "primary_roles": ("Static", "TTS", "LightCone"),
        "goodput_protocol_sha256": FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        "goodput": "slo_qualified_output_tokens_per_scored_wall_window_ns",
        "pairing": "exact_family_block_request_and_token_trajectory",
        "independent_unit": "one_final_block_within_one_exact_family",
        "primary_family": PRIMARY_CONTRASTS,
        "multiplicity": "holm_within_each_exact_scientific_family",
        "stage_status": "CONFIRMED_only_when_all_96_families_are_CONFIRMED",
        "pooled_family_inference": "forbidden",
    }
)
E1A_VERIFICATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e1a_proof_derived_dspark_verification_protocol",
        "universe": "58_configurations_x_2_verification_modes",
        "adaptive_candidates": 56,
        "eligibility": "zero_safety_violation_and_at_least_one_published_update",
        "pairing": "exact_request_and_token_trajectory_with_static_per_mode",
        "ranking": (
            "maximum_worst_mode_paired_95pct_lower_request_rate_ratio",
            "minimum_peak_hbm_bytes",
            "minimum_p99_itl_us",
            "minimum_exposed_update_us",
            "configuration_sha256",
        ),
    }
)
E5_POWER_AND_ANCHOR_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_e5_proof_derived_power_and_anchor_protocol",
        "pilot_blocks": 4,
        "pilot_disposition": "excluded_tuning_only",
        "pilot_headline_cell_count": 4 * 450,
        "failure_diagnostics": "264_one_shot_rows_materialized_after_power",
        "goodput_protocol_sha256": FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        "goodput": "slo_qualified_output_tokens_per_scored_wall_window_ns",
        "power_unit": "each_exact_backend_scientific_family_has_four_pilots",
        "stage_final_N": "maximum_READY_N_across_all_90_families",
        "power": "preregister_power_sizing_3pct_80pct_holm_first_threshold",
        "final_prefix": "first_N_of_12_to_20",
        "p99_anchor_rule": (
            "one_worst_observed_lightcone_p99_family_per_backend_topology"
        ),
        "p99_anchor_count": len(E5_BACKENDS) * len(E5_TOPOLOGIES),
        "p99_minimum_completions": 10_000,
        "confirmation_data_visible": False,
    }
)
E5_CONFIRMATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_e5_proof_derived_confirmation_protocol",
        "headline": "450_rows_per_block_x_final_only_powered_prefix",
        "headline_family_count": 90,
        "goodput_protocol_sha256": FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        "primary_inference": (
            "per_exact_backend_scientific_family_paired_BCa_and_within_family_Holm"
        ),
        "pooled_family_inference": "forbidden",
        "stage_status": "CONFIRMED_only_when_all_90_families_are_CONFIRMED",
        "failure_diagnostics": "exact_264_one_shot_controlled_terminal_proofs",
        "safety": "all_explicit_counters_zero_and_adaptive_publication_present",
        "p99": "selected_anchor_families_each_reach_10000_completed_requests",
        "coverage": "sealed_serving_and_failure_bindings_plus_path_bound_proofs",
        "result": "confirmation_only_no_recipe_reselection",
    }
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _absolute_path(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"{label} must be absolute and resolved")
    return value


@dataclass(frozen=True)
class FormalDownstreamCellEvidence:
    """Path-bound result/timing evidence for one sealed downstream cell."""

    schema_version: int
    stage: str
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
        if self.schema_version != 1 or self.stage not in {
            "E3b",
            "E1a",
            "E5",
            "E6",
            "E0",
        }:
            raise ValueError("downstream cell evidence schema/stage is unsupported")
        _sha256("downstream evidence cell", self.materialized_cell_id)
        _sha256("downstream execution binding", self.execution_binding_sha256)
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("downstream evidence requires an exact execution identity")
        if self.execution_identity.materialized_cell_id != self.materialized_cell_id:
            raise ValueError("downstream execution identity names another cell")
        for label, path, raw_sha256, semantic_sha256 in (
            (
                "downstream native result proof",
                self.native_result_proof_path,
                self.native_result_proof_raw_sha256,
                self.native_result_proof_semantic_sha256,
            ),
            (
                "downstream stage ITL proof",
                self.stage_itl_proof_path,
                self.stage_itl_proof_raw_sha256,
                self.stage_itl_proof_semantic_sha256,
            ),
        ):
            binding = CanonicalJsonProofBinding.bind(_absolute_path(label, path))
            if (
                binding.raw_sha256 != raw_sha256
                or binding.semantic_sha256 != semantic_sha256
            ):
                raise ValueError(f"{label} content changed after binding")
        if self.native_result_proof_path == self.stage_itl_proof_path:
            raise ValueError("downstream result and timing proof paths must differ")

    @classmethod
    def bind(
        cls,
        *,
        stage: str,
        execution_binding: VerifiedFormalServingExecutionBinding,
        native_result_proof_path: str,
        stage_itl_proof_path: str,
    ) -> FormalDownstreamCellEvidence:
        verified = require_verified_formal_serving_execution_binding(execution_binding)
        if verified.subject.stage != stage:
            raise ValueError("downstream evidence cannot consume another stage binding")
        result = CanonicalJsonProofBinding.bind(native_result_proof_path)
        timing = CanonicalJsonProofBinding.bind(stage_itl_proof_path)
        return cls(
            schema_version=1,
            stage=stage,
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
class FormalDownstreamEvidenceManifest:
    """Exact cell universe reopened by a downstream reducer."""

    schema_version: int
    stage: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    source_authority_sha256: str
    inventory_sha256: str
    cells: tuple[FormalDownstreamCellEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.stage not in {
            "E3b",
            "E1a",
            "E5",
            "E6",
            "E0",
        }:
            raise ValueError("downstream evidence manifest schema/stage is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("source authority", self.source_authority_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha256(f"downstream evidence {label}", digest)
        ids = tuple(row.materialized_cell_id for row in self.cells)
        runs = tuple(
            (
                row.execution_identity.run_id,
                row.execution_identity.run_nonce_sha256,
                row.execution_identity.attempt_id,
            )
            for row in self.cells
        )
        result_proofs = tuple(row.native_result_proof_raw_sha256 for row in self.cells)
        timing_proofs = tuple(row.stage_itl_proof_raw_sha256 for row in self.cells)
        if (
            not self.cells
            or any(type(row) is not FormalDownstreamCellEvidence for row in self.cells)
            or any(row.stage != self.stage for row in self.cells)
            or ids != tuple(sorted(set(ids)))
            or len(runs) != len(set(runs))
            or len(result_proofs) != len(set(result_proofs))
            or len(timing_proofs) != len(set(timing_proofs))
        ):
            raise ValueError("downstream evidence cells/proofs/runs are not exact")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class FormalFamilyPowerCommitment:
    """One exact scientific family's four-pilot power commitment."""

    schema_version: int
    stage: str
    model: str
    task: str
    family_dimensions: tuple[tuple[str, str | int | float], ...]
    family_sha256: str
    slo_goodput_protocol_sha256: str
    pilot_goodput_observation_sha256s: tuple[tuple[int, str, str], ...]
    power_sizing: PowerSizingPlan

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.stage not in {"E3b", "E5", "E6", "E0"}:
            raise ValueError("formal family power commitment identity differs")
        if (
            type(self.model) is not str
            or not self.model
            or type(self.task) is not str
            or not self.task
            or self.family_dimensions != tuple(sorted(set(self.family_dimensions)))
            or any(
                type(key) is not str
                or not key
                or type(value) not in {str, int, float}
                or (type(value) is float and not math.isfinite(value))
                for key, value in self.family_dimensions
            )
        ):
            raise ValueError("formal family power identity is not canonical")
        expected_family = content_sha256(
            {
                "stage": self.stage,
                "model": self.model,
                "task": self.task,
                "dimensions": list(self.family_dimensions),
            }
        )
        if self.family_sha256 != expected_family:
            raise ValueError("formal family power digest differs from identity")
        if self.slo_goodput_protocol_sha256 != FORMAL_SLO_GOODPUT_PROTOCOL_SHA256:
            raise ValueError("formal family power uses another SLO-goodput protocol")
        expected_observation_keys = {
            (block, role)
            for block in range(PILOT_BLOCK_COUNT)
            for role in ("Static", "TTS", "LightCone")
        }
        if (
            type(self.pilot_goodput_observation_sha256s) is not tuple
            or self.pilot_goodput_observation_sha256s
            != tuple(sorted(set(self.pilot_goodput_observation_sha256s)))
            or {
                (block, role)
                for block, role, _digest in self.pilot_goodput_observation_sha256s
            }
            != expected_observation_keys
        ):
            raise ValueError("formal family power pilot observation coverage differs")
        for _block, _role, digest in self.pilot_goodput_observation_sha256s:
            _sha256("formal family pilot goodput observation", digest)
        if (
            type(self.power_sizing) is not PowerSizingPlan
            or self.power_sizing.status != "READY"
            or self.power_sizing.selected_final_blocks is None
        ):
            raise ValueError("formal family power commitment is not READY")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3bPowerPrefixReceipt:
    """Deterministic 12--20 final prefix selected from four excluded pilots."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    pilot_materialization_receipt_sha256: str
    pilot_coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    family_power_commitments: tuple[FormalFamilyPowerCommitment, ...]
    selected_final_blocks: int
    selected_final_prefix: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only E3b power-prefix schema 2 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("pilot materialization", self.pilot_materialization_receipt_sha256),
            ("pilot coverage", self.pilot_coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("power protocol", self.protocol_sha256),
        ):
            _sha256(f"E3b power {label}", digest)
        if self.protocol_sha256 != E3B_POWER_PREFIX_PROTOCOL_SHA256:
            raise ValueError("E3b power prefix uses another reducer protocol")
        if (
            len(self.family_power_commitments) != 96
            or any(
                type(row) is not FormalFamilyPowerCommitment or row.stage != "E3b"
                for row in self.family_power_commitments
            )
            or tuple(row.family_sha256 for row in self.family_power_commitments)
            != tuple(
                sorted({row.family_sha256 for row in self.family_power_commitments})
            )
            or max(
                row.power_sizing.selected_final_blocks or 0
                for row in self.family_power_commitments
            )
            != self.selected_final_blocks
            or not 12 <= self.selected_final_blocks <= 20
            or self.selected_final_prefix
            != tuple(
                range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + self.selected_final_blocks)
            )
        ):
            raise ValueError("E3b power prefix differs from preregistered sizing")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE3bPowerPrefixReceipt:
    payload: E3bPowerPrefixReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        pilot_materialization: StageMaterializationReceipt,
        pilot_coverage: StageCoverageReceipt,
        manifest: FormalDownstreamEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E3bPowerPrefixReceipt:
        if type(self.payload) is not E3bPowerPrefixReceipt:
            raise TypeError("signed E3b power-prefix payload has the wrong type")
        expected = reduce_e3b_power_prefix_from_proofs(
            protocol_lock=protocol_lock,
            pilot_materialization=pilot_materialization,
            pilot_coverage=pilot_coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E3b power prefix differs from proof reducer")
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
class FormalFamilyConfirmationResult:
    """Primary inference for one exact scientific family only."""

    schema_version: int
    stage: str
    model: str
    task: str
    family_dimensions: tuple[tuple[str, str | int | float], ...]
    family_sha256: str
    slo_goodput_protocol_sha256: str
    final_block_ids: tuple[str, ...]
    final_goodput_observation_sha256s: tuple[tuple[str, str, str], ...]
    primary_contrasts: tuple[PairedBcaContrast, ...]
    holm_decisions: tuple[MultiplicityDecision, ...]
    status: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.stage not in {"E3b", "E5", "E6", "E0"}:
            raise ValueError("formal family confirmation identity differs")
        if (
            type(self.model) is not str
            or not self.model
            or type(self.task) is not str
            or not self.task
            or self.family_dimensions != tuple(sorted(set(self.family_dimensions)))
            or any(
                type(key) is not str
                or not key
                or type(value) not in {str, int, float}
                or (type(value) is float and not math.isfinite(value))
                for key, value in self.family_dimensions
            )
        ):
            raise ValueError("formal family confirmation identity is not canonical")
        expected_family = content_sha256(
            {
                "stage": self.stage,
                "model": self.model,
                "task": self.task,
                "dimensions": list(self.family_dimensions),
            }
        )
        if self.family_sha256 != expected_family:
            raise ValueError("formal family confirmation digest differs from identity")
        if self.slo_goodput_protocol_sha256 != FORMAL_SLO_GOODPUT_PROTOCOL_SHA256:
            raise ValueError(
                "formal family confirmation uses another SLO-goodput protocol"
            )
        if (
            self.final_block_ids != tuple(sorted(set(self.final_block_ids)))
            or not 12 <= len(self.final_block_ids) <= 20
        ):
            raise ValueError("formal family confirmation block prefix is not exact")
        expected_observation_keys = {
            (block_id, role)
            for block_id in self.final_block_ids
            for role in ("Static", "TTS", "LightCone")
        }
        if (
            self.final_goodput_observation_sha256s
            != tuple(sorted(set(self.final_goodput_observation_sha256s)))
            or {
                (block_id, role)
                for block_id, role, _digest in (self.final_goodput_observation_sha256s)
            }
            != expected_observation_keys
        ):
            raise ValueError("formal family confirmation observation coverage differs")
        for _block_id, _role, digest in self.final_goodput_observation_sha256s:
            _sha256("formal family final goodput observation", digest)
        if (
            tuple(row.name for row in self.primary_contrasts) != PRIMARY_CONTRASTS
            or tuple(row.name for row in self.holm_decisions) != PRIMARY_CONTRASTS
            or any(
                row.block_ids != self.final_block_ids for row in self.primary_contrasts
            )
        ):
            raise ValueError("formal family primary inference is not exact")
        expected_status = (
            "CONFIRMED"
            if all(
                decision.rejected and contrast.ci_lower_relative_gain > 0
                for contrast, decision in zip(
                    self.primary_contrasts,
                    self.holm_decisions,
                    strict=True,
                )
            )
            else "NOT_CONFIRMED"
        )
        if self.status != expected_status:
            raise ValueError("formal family status differs from its Holm/CI decisions")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E3bConfirmationReceipt:
    """Proof-derived, non-pooled E3b family decisions consumed by E1a."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    model: str
    frozen_tts_recipe_sha256: str
    lightcone_recipe_sha256: str
    final_block_ids: tuple[str, ...]
    family_confirmations: tuple[FormalFamilyConfirmationResult, ...]
    status: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only E3b confirmation schema 2 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("protocol", self.protocol_sha256),
            ("frozen TTS recipe", self.frozen_tts_recipe_sha256),
            ("LightCone recipe", self.lightcone_recipe_sha256),
        ):
            _sha256(f"E3b confirmation {label}", digest)
        if self.protocol_sha256 != E3B_CONFIRMATION_PROTOCOL_SHA256:
            raise ValueError("E3b confirmation uses another reducer protocol")
        if (
            type(self.model) is not str
            or not self.model
            or self.final_block_ids != tuple(sorted(set(self.final_block_ids)))
            or not 12 <= len(self.final_block_ids) <= 20
            or len(self.family_confirmations) != 96
            or any(
                type(row) is not FormalFamilyConfirmationResult
                or row.stage != "E3b"
                or row.model != self.model
                or row.final_block_ids != self.final_block_ids
                for row in self.family_confirmations
            )
            or tuple(row.family_sha256 for row in self.family_confirmations)
            != tuple(sorted({row.family_sha256 for row in self.family_confirmations}))
        ):
            raise ValueError("E3b confirmation family coverage is not exact")
        expected_status = (
            "CONFIRMED"
            if all(row.status == "CONFIRMED" for row in self.family_confirmations)
            else "NOT_CONFIRMED"
        )
        if self.status != expected_status:
            raise ValueError(
                "E3b confirmation status differs from all-family decisions"
            )
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE3bConfirmationReceipt:
    payload: E3bConfirmationReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        manifest: FormalDownstreamEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E3bConfirmationReceipt:
        if type(self.payload) is not E3bConfirmationReceipt:
            raise TypeError("signed E3b confirmation payload has the wrong type")
        expected = reduce_e3b_confirmation_from_proofs(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E3b confirmation differs from proof reducer")
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
class E1aConfigurationEvaluation:
    configuration: tuple[tuple[str, str | int], ...]
    cell_ids: tuple[str, str]
    minimum_confidence_lower_request_rate_ratio: float
    peak_hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int

    def __post_init__(self) -> None:
        if tuple(name for name, _value in self.configuration) != (
            "parameterization",
            "rank",
            "scope",
        ):
            raise ValueError("E1a configuration fields are not canonical")
        if (
            self.cell_ids != tuple(sorted(set(self.cell_ids)))
            or len(self.cell_ids) != 2
        ):
            raise ValueError("E1a evaluation requires two verification-mode cells")
        for cell_id in self.cell_ids:
            _sha256("E1a evaluation cell", cell_id)
        if (
            type(self.minimum_confidence_lower_request_rate_ratio) is not float
            or self.minimum_confidence_lower_request_rate_ratio <= 0
        ):
            raise ValueError("E1a confidence lower request-rate ratio is invalid")
        for label, value in (
            ("peak HBM", self.peak_hbm_bytes),
            ("p99 ITL", self.p99_itl_us),
            ("exposed update", self.exposed_update_us),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"E1a {label} must be non-negative")

    @cached_property
    def configuration_sha256(self) -> str:
        return content_sha256(self.configuration)


@dataclass(frozen=True)
class E1aVerificationReceipt:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    model: str
    frozen_tts_recipe_sha256: str
    source_lightcone_recipe_sha256: str
    evaluations: tuple[E1aConfigurationEvaluation, ...]
    selected_configuration: tuple[tuple[str, str | int], ...]
    selected_dspark_recipe_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E1a verification schema 1 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("protocol", self.protocol_sha256),
            ("frozen TTS recipe", self.frozen_tts_recipe_sha256),
            ("source LightCone recipe", self.source_lightcone_recipe_sha256),
            ("selected DSpark recipe", self.selected_dspark_recipe_sha256),
        ):
            _sha256(f"E1a verification {label}", digest)
        if self.protocol_sha256 != E1A_VERIFICATION_PROTOCOL_SHA256:
            raise ValueError("E1a verification uses another reducer protocol")
        ids = tuple(row.configuration_sha256 for row in self.evaluations)
        if (
            len(self.evaluations) != 56
            or ids != tuple(sorted(set(ids)))
            or self.selected_configuration
            not in {row.configuration for row in self.evaluations}
            or type(self.model) is not str
            or not self.model
        ):
            raise ValueError("E1a evaluations/selection are not exact")
        expected_recipe = content_sha256(
            {
                "schema_version": 1,
                "kind": "lightcone_e1a_selected_dspark_recipe",
                "source_lightcone_recipe_sha256": self.source_lightcone_recipe_sha256,
                "configuration": self.selected_configuration,
                "verification_protocol_sha256": self.protocol_sha256,
            }
        )
        if self.selected_dspark_recipe_sha256 != expected_recipe:
            raise ValueError("E1a selected DSpark recipe differs from configuration")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE1aVerificationReceipt:
    payload: E1aVerificationReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        manifest: FormalDownstreamEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E1aVerificationReceipt:
        if type(self.payload) is not E1aVerificationReceipt:
            raise TypeError("signed E1a verification payload has the wrong type")
        expected = reduce_e1a_verification_from_proofs(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E1a verification differs from proof reducer")
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
class E5PowerAndAnchorReceipt:
    """Proof-derived E5 block prefix and native-p99 anchor selection."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    upstream_e1a_verification_sha256: str
    pilot_materialization_receipt_sha256: str
    pilot_coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    model: str
    frozen_tts_recipe_sha256: str
    dflash_lightcone_recipe_sha256: str
    dspark_lightcone_recipe_sha256: str
    family_power_commitments: tuple[FormalFamilyPowerCommitment, ...]
    selected_final_blocks: int
    selected_final_prefix: tuple[int, ...]
    p99_anchors: tuple[E5SelectedP99Anchor, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only E5 power/anchor schema 2 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("upstream E1a verification", self.upstream_e1a_verification_sha256),
            ("pilot materialization", self.pilot_materialization_receipt_sha256),
            ("pilot coverage", self.pilot_coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("protocol", self.protocol_sha256),
            ("frozen TTS recipe", self.frozen_tts_recipe_sha256),
            ("DFlash LightCone recipe", self.dflash_lightcone_recipe_sha256),
            ("DSpark LightCone recipe", self.dspark_lightcone_recipe_sha256),
        ):
            _sha256(f"E5 power/anchor {label}", digest)
        if self.protocol_sha256 != E5_POWER_AND_ANCHOR_PROTOCOL_SHA256:
            raise ValueError("E5 power/anchor receipt uses another reducer protocol")
        if (
            type(self.model) is not str
            or not self.model
            or len(self.family_power_commitments) != 90
            or any(
                type(row) is not FormalFamilyPowerCommitment
                or row.stage != "E5"
                or row.model != self.model
                for row in self.family_power_commitments
            )
            or tuple(row.family_sha256 for row in self.family_power_commitments)
            != tuple(
                sorted({row.family_sha256 for row in self.family_power_commitments})
            )
            or max(
                row.power_sizing.selected_final_blocks or 0
                for row in self.family_power_commitments
            )
            != self.selected_final_blocks
            or not 12 <= self.selected_final_blocks <= 20
            or self.selected_final_prefix
            != tuple(
                range(
                    PILOT_BLOCK_COUNT,
                    PILOT_BLOCK_COUNT + self.selected_final_blocks,
                )
            )
        ):
            raise ValueError("E5 power/anchor block prefix is not preregistered")
        anchor_ids = tuple(row.anchor_id for row in self.p99_anchors)
        expected_pairs = {
            (backend, topology) for backend in E5_BACKENDS for topology in E5_TOPOLOGIES
        }
        if (
            len(self.p99_anchors) != len(expected_pairs)
            or anchor_ids != tuple(sorted(set(anchor_ids)))
            or {(row.backend, row.topology) for row in self.p99_anchors}
            != expected_pairs
            or any(row.minimum_completions != 10_000 for row in self.p99_anchors)
        ):
            raise ValueError("E5 power/anchor receipt lacks the exact six anchors")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE5PowerAndAnchorReceipt:
    payload: E5PowerAndAnchorReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        pilot_materialization: StageMaterializationReceipt,
        pilot_coverage: StageCoverageReceipt,
        manifest: FormalDownstreamEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E5PowerAndAnchorReceipt:
        if type(self.payload) is not E5PowerAndAnchorReceipt:
            raise TypeError("signed E5 power/anchor payload has the wrong type")
        expected = reduce_e5_power_and_anchors_from_proofs(
            protocol_lock=protocol_lock,
            pilot_materialization=pilot_materialization,
            pilot_coverage=pilot_coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError(
                "signed E5 power/anchor receipt differs from proof reducer"
            )
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
class E5FailureCellEvidence:
    """Durable proof input for one dedicated staged failure assignment."""

    schema_version: int
    materialized_cell_id: str
    failure_execution_binding_sha256: str
    assignment_sha256: str
    serving_execution_plan_sha256: str
    proof_artifact_path: str
    proof_artifact_raw_sha256: str
    proof_artifact_semantic_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("E5 failure evidence schema is unsupported")
        for label, digest in (
            ("cell", self.materialized_cell_id),
            ("execution binding", self.failure_execution_binding_sha256),
            ("assignment", self.assignment_sha256),
            ("serving plan", self.serving_execution_plan_sha256),
            ("raw proof", self.proof_artifact_raw_sha256),
            ("semantic proof", self.proof_artifact_semantic_sha256),
        ):
            _sha256(f"E5 failure evidence {label}", digest)
        binding = CanonicalJsonProofBinding.bind(
            _absolute_path("E5 failure proof", self.proof_artifact_path)
        )
        if (
            binding.raw_sha256 != self.proof_artifact_raw_sha256
            or binding.semantic_sha256 != self.proof_artifact_semantic_sha256
        ):
            raise ValueError("E5 failure proof changed after binding")

    @classmethod
    def bind(
        cls,
        *,
        execution_binding: VerifiedFormalFailureExecutionBinding,
        proof_artifact_path: str,
    ) -> E5FailureCellEvidence:
        verified = require_verified_formal_failure_execution_binding(execution_binding)
        proof = CanonicalJsonProofBinding.bind(proof_artifact_path)
        return cls(
            schema_version=1,
            materialized_cell_id=verified.subject.materialized_cell_id,
            failure_execution_binding_sha256=verified.sha256,
            assignment_sha256=verified.subject.assignment_sha256,
            serving_execution_plan_sha256=(
                verified.subject.serving_execution_plan_sha256
            ),
            proof_artifact_path=proof.absolute_path,
            proof_artifact_raw_sha256=proof.raw_sha256,
            proof_artifact_semantic_sha256=proof.semantic_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E5FailureEvidenceManifest:
    schema_version: int
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    inventory_sha256: str
    cells: tuple[E5FailureCellEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("E5 failure manifest schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha256(f"E5 failure manifest {label}", digest)
        ids = tuple(row.materialized_cell_id for row in self.cells)
        assignments = tuple(row.assignment_sha256 for row in self.cells)
        raw_proofs = tuple(row.proof_artifact_raw_sha256 for row in self.cells)
        if (
            len(self.cells) != 264
            or any(type(row) is not E5FailureCellEvidence for row in self.cells)
            or ids != tuple(sorted(set(ids)))
            or len(assignments) != len(set(assignments))
            or len(raw_proofs) != len(set(raw_proofs))
        ):
            raise ValueError("E5 failure manifest must cover 264 unique assignments")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E5P99AnchorCompletion:
    anchor_id: str
    completed_requests: int

    def __post_init__(self) -> None:
        _sha256("E5 confirmation p99 anchor", self.anchor_id)
        if type(self.completed_requests) is not int or self.completed_requests < 10_000:
            raise ValueError(
                "E5 confirmation p99 anchor has fewer than 10,000 completions"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E5ConfirmationReceipt:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    headline_evidence_manifest_sha256: str
    failure_evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    model: str
    frozen_tts_recipe_sha256: str
    dflash_lightcone_recipe_sha256: str
    dspark_lightcone_recipe_sha256: str
    block_count: int  # final blocks only; four tuning pilots are outside this receipt
    headline_cell_count: int
    failure_cell_count: int
    family_confirmations: tuple[FormalFamilyConfirmationResult, ...]
    p99_anchor_completions: tuple[E5P99AnchorCompletion, ...]
    failure_result_sha256s: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("E5 confirmation schema is unsupported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("headline evidence", self.headline_evidence_manifest_sha256),
            ("failure evidence", self.failure_evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("protocol", self.protocol_sha256),
            ("frozen TTS recipe", self.frozen_tts_recipe_sha256),
            ("DFlash LightCone recipe", self.dflash_lightcone_recipe_sha256),
            ("DSpark LightCone recipe", self.dspark_lightcone_recipe_sha256),
        ):
            _sha256(f"E5 confirmation {label}", digest)
        if self.protocol_sha256 != E5_CONFIRMATION_PROTOCOL_SHA256:
            raise ValueError("E5 confirmation uses another reducer protocol")
        if (
            self.block_count not in range(12, 21)
            or self.headline_cell_count != 450 * self.block_count
            or self.failure_cell_count != 264
            or type(self.model) is not str
            or not self.model
            or len(self.family_confirmations) != 90
            or any(
                type(row) is not FormalFamilyConfirmationResult
                or row.stage != "E5"
                or row.model != self.model
                or len(row.final_block_ids) != self.block_count
                for row in self.family_confirmations
            )
            or tuple(row.family_sha256 for row in self.family_confirmations)
            != tuple(sorted({row.family_sha256 for row in self.family_confirmations}))
        ):
            raise ValueError("E5 confirmation count/status is not exact")
        expected_status = (
            "CONFIRMED"
            if all(row.status == "CONFIRMED" for row in self.family_confirmations)
            else "NOT_CONFIRMED"
        )
        if self.status != expected_status:
            raise ValueError("E5 status differs from all-family primary inference")
        anchor_ids = tuple(row.anchor_id for row in self.p99_anchor_completions)
        if (
            len(self.p99_anchor_completions) != 6
            or any(
                type(row) is not E5P99AnchorCompletion
                for row in self.p99_anchor_completions
            )
            or anchor_ids != tuple(sorted(set(anchor_ids)))
            or len(self.failure_result_sha256s) != 264
            or self.failure_result_sha256s
            != tuple(sorted(set(self.failure_result_sha256s)))
        ):
            raise ValueError("E5 confirmation evidence coverage is not exact")
        for digest in self.failure_result_sha256s:
            _sha256("E5 confirmation failure result", digest)
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE5ConfirmationReceipt:
    payload: E5ConfirmationReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        headline_manifest: FormalDownstreamEvidenceManifest,
        headline_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        failure_manifest: E5FailureEvidenceManifest,
        failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E5ConfirmationReceipt:
        if type(self.payload) is not E5ConfirmationReceipt:
            raise TypeError("signed E5 confirmation payload has the wrong type")
        expected = reduce_e5_confirmation_from_proofs(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            headline_manifest=headline_manifest,
            headline_execution_bindings=headline_execution_bindings,
            failure_manifest=failure_manifest,
            failure_execution_bindings=failure_execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E5 confirmation differs from proof reducer")
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


def _aggregate_request_rate(rows: list[object]) -> Fraction:
    numerator = sum(metric.output_tokens for row in rows for metric in row.metrics)
    denominator = sum(metric.latency_ns for row in rows for metric in row.metrics)
    if numerator < 1 or denominator < 1:
        raise ValueError("E3b block/role has no completed timed output")
    return Fraction(numerator * 1_000_000_000, denominator)


def _formal_slo_goodput(validated: object) -> FormalSloGoodputObservation:
    """Project one deeply validated cell into exact integer SLO-goodput."""

    timing_rows = tuple(validated.timing.requests)
    metric_rows = tuple(validated.metrics)
    timing_by_id = {row.request_id: row for row in timing_rows}
    metric_by_id = {row.request_id: row for row in metric_rows}
    if (
        not timing_by_id
        or len(timing_by_id) != len(timing_rows)
        or len(metric_by_id) != len(metric_rows)
        or set(timing_by_id) != set(metric_by_id)
    ):
        raise ValueError("formal SLO terminal/timing request coverage differs")
    evidence = tuple(
        FormalSloRequestEvidence(
            request_id=request_id,
            input_token_ids=metric_by_id[request_id].input_token_ids,
            output_token_ids=metric_by_id[request_id].output_token_ids,
            request_started_ns=timing_by_id[request_id].request_started_ns,
            request_terminal_ns=timing_by_id[request_id].request_terminal_ns,
            token_observed_ns=timing_by_id[request_id].token_observed_ns,
            eligible=True,
            completed=True,
            error=False,
        )
        for request_id in sorted(metric_by_id)
    )
    return reduce_formal_slo_goodput(evidence)


def _e3b_family_dimensions(
    dimensions: dict[str, object],
    *,
    final: bool,
) -> tuple[tuple[str, str | int], ...]:
    """Project execution lineage away from the exact E3b scientific family."""

    scientific = {
        "context": dimensions.get("context"),
        "load": dimensions.get("load"),
        "regime": dimensions.get("regime"),
        "width_panel": dimensions.get("width_panel"),
    }
    if (
        scientific["context"] not in E3B_CONTEXTS
        or scientific["load"] not in E3B_LOADS
        or scientific["regime"] not in E3B_REGIMES
        or scientific["width_panel"] not in E3B_WIDTH_PANELS
    ):
        raise ValueError("E3b scientific family dimensions are invalid")
    expected_keys = {
        "block",
        "block_phase",
        *scientific,
    }
    if final:
        expected_keys.update(
            {
                "pilot_coverage_receipt_sha256",
                "pilot_materialization_receipt_sha256",
                "signed_power_prefix_sha256",
            }
        )
    if set(dimensions) != expected_keys:
        raise ValueError("E3b cell dimensions contain foreign family/lineage fields")
    return tuple(sorted(scientific.items()))  # type: ignore[return-value]


def _e5_family_dimensions(
    dimensions: dict[str, object],
    *,
    final: bool,
    method_role: str,
) -> tuple[tuple[str, str | int | float], ...]:
    """Return one of the exact 90 backend/family headline identities."""

    backend = dimensions.get("backend_authority")
    family = dimensions.get("family")
    topology = dimensions.get("topology")
    family_id = dimensions.get("family_id")
    if backend not in E5_BACKENDS or topology not in E5_TOPOLOGIES:
        raise ValueError("E5 family backend/topology is invalid")
    variant: dict[str, str | int | float]
    if family == "closed_loop":
        concurrency = dimensions.get("concurrency")
        if concurrency not in E5_CLOSED_LOOP_CONCURRENCY:
            raise ValueError("E5 closed-loop family is invalid")
        variant = {"concurrency": concurrency}  # type: ignore[dict-item]
        expected_family_id = f"closed_loop_c{concurrency}"
    elif family == "open_loop":
        load_factor = dimensions.get("load_factor")
        if load_factor not in E5_OPEN_LOOP_LOAD_FACTORS:
            raise ValueError("E5 open-loop family is invalid")
        variant = {"load_factor": load_factor}  # type: ignore[dict-item]
        expected_family_id = f"open_loop_{load_factor}"
    elif family == "trace_or_soak":
        arrival = dimensions.get("arrival")
        if arrival not in E5_TRACE_AND_SOAK_ARRIVALS:
            raise ValueError("E5 trace/soak family is invalid")
        variant = {"arrival": arrival}  # type: ignore[dict-item]
        expected_family_id = f"trace_or_soak_{arrival}"
    elif family == "topology_cohort":
        cohort_count = dimensions.get("cohort_count")
        cohort_distribution = dimensions.get("cohort_distribution")
        if (
            cohort_count not in E5_COHORT_COUNTS
            or cohort_distribution not in E5_COHORT_DISTRIBUTIONS
        ):
            raise ValueError("E5 topology/cohort family is invalid")
        variant = {
            "cohort_count": cohort_count,  # type: ignore[dict-item]
            "cohort_distribution": cohort_distribution,  # type: ignore[dict-item]
        }
        expected_family_id = (
            f"topology_cohort_{topology}_k{cohort_count}_{cohort_distribution}"
        )
    else:
        raise ValueError("E5 headline family is not registered")
    if family_id != expected_family_id:
        raise ValueError("E5 family ID differs from its scientific dimensions")
    scientific: dict[str, str | int | float] = {
        "backend_authority": backend,  # type: ignore[dict-item]
        "family": family,
        "family_id": family_id,  # type: ignore[dict-item]
        "topology": topology,  # type: ignore[dict-item]
        **variant,
    }
    expected_keys = {
        "block",
        "block_phase",
        *scientific,
        "upstream_e1a_verification_sha256",
        "frozen_tts_recipe_sha256",
        "dflash_lightcone_recipe_sha256",
        "dspark_lightcone_recipe_sha256",
    }
    if final:
        expected_keys.update(
            {
                "pilot_coverage_receipt_sha256",
                "pilot_materialization_receipt_sha256",
                "signed_power_and_anchor_prefix_sha256",
            }
        )
    anchor_fields = {
        "p99_anchor_id",
        "p99_minimum_completions",
        "p99_selection_receipt_sha256",
    }
    present_anchor_fields = set(dimensions) & anchor_fields
    if present_anchor_fields and (
        present_anchor_fields != anchor_fields or method_role != "LightCone"
    ):
        raise ValueError("E5 p99 anchor fields are partial or attached to another role")
    expected_keys.update(present_anchor_fields)
    if set(dimensions) != expected_keys:
        raise ValueError("E5 cell dimensions contain foreign family/lineage fields")
    return tuple(sorted(scientific.items()))


def _family_power_commitment(
    *,
    stage: str,
    model: str,
    task: str,
    family_dimensions: tuple[tuple[str, str | int | float], ...],
    rows_by_block: dict[int, list[object]],
) -> FormalFamilyPowerCommitment:
    """Power one exact family from its own four paired pilot blocks."""

    family_sha256 = content_sha256(
        {
            "stage": stage,
            "model": model,
            "task": task,
            "dimensions": list(family_dimensions),
        }
    )
    observations: list[tuple[int, str, FormalSloGoodputObservation]] = []
    pilot_blocks: list[PilotBlock] = []
    for block in range(PILOT_BLOCK_COUNT):
        rows = rows_by_block.get(block, [])
        by_role = {row.cell.method_role: row for row in rows}
        if len(by_role) != len(rows) or set(by_role) != set(FORMAL_METHOD_ROLES):
            raise ValueError("formal family pilot method coverage differs")
        primary = {
            role: _formal_slo_goodput(by_role[role])
            for role in ("Static", "TTS", "LightCone")
        }
        paired = dict(require_paired_primary_goodputs(primary))
        observations.extend(
            (block, role, primary[role]) for role in ("Static", "TTS", "LightCone")
        )
        pilot_blocks.append(
            PilotBlock(
                block_id=f"{stage}:{family_sha256}:excluded_pilot:{block}",
                static_goodput=float(paired["Static"]),
                tts_goodput=float(paired["TTS"]),
                lightcone_goodput=float(paired["LightCone"]),
            )
        )
    power = preregister_power_sizing(tuple(pilot_blocks))
    if power.underpowered or power.selected_final_blocks is None:
        raise ValueError(
            f"{stage} family {family_sha256} is UNDERPOWERED at 20 final blocks"
        )
    return FormalFamilyPowerCommitment(
        schema_version=1,
        stage=stage,
        model=model,
        task=task,
        family_dimensions=family_dimensions,
        family_sha256=family_sha256,
        slo_goodput_protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        pilot_goodput_observation_sha256s=tuple(
            sorted((block, role, row.sha256) for block, role, row in observations)
        ),
        power_sizing=power,
    )


def _family_confirmation_result(
    *,
    stage: str,
    model: str,
    task: str,
    family_dimensions: tuple[tuple[str, str | int | float], ...],
    rows_by_block: dict[int, list[object]],
) -> FormalFamilyConfirmationResult:
    """Reduce primary inference without pooling independent families."""

    family_sha256 = content_sha256(
        {
            "stage": stage,
            "model": model,
            "task": task,
            "dimensions": list(family_dimensions),
        }
    )
    observations: list[tuple[str, str, FormalSloGoodputObservation]] = []
    paired: dict[str, dict[str, tuple[float, float]]] = {
        "lightcone_vs_tts": {},
        "lightcone_vs_static": {},
    }
    for block in sorted(rows_by_block):
        rows = rows_by_block[block]
        by_role = {row.cell.method_role: row for row in rows}
        if len(by_role) != len(rows) or set(by_role) != set(FORMAL_METHOD_ROLES):
            raise ValueError("formal family final method coverage differs")
        primary = {
            role: _formal_slo_goodput(by_role[role])
            for role in ("Static", "TTS", "LightCone")
        }
        paired_goodputs = dict(require_paired_primary_goodputs(primary))
        block_id = f"{stage}:final:{block - PILOT_BLOCK_COUNT:02d}"
        observations.extend(
            (block_id, role, primary[role]) for role in ("Static", "TTS", "LightCone")
        )
        paired["lightcone_vs_tts"][block_id] = (
            float(paired_goodputs["LightCone"]),
            float(paired_goodputs["TTS"]),
        )
        paired["lightcone_vs_static"][block_id] = (
            float(paired_goodputs["LightCone"]),
            float(paired_goodputs["Static"]),
        )
    contrasts = tuple(
        paired_bca_contrast(name, paired[name]) for name in PRIMARY_CONTRASTS
    )
    decisions = holm_primary_contrasts({row.name: row for row in contrasts})
    status = (
        "CONFIRMED"
        if all(
            decision.rejected and contrast.ci_lower_relative_gain > 0
            for contrast, decision in zip(contrasts, decisions, strict=True)
        )
        else "NOT_CONFIRMED"
    )
    result = FormalFamilyConfirmationResult(
        schema_version=1,
        stage=stage,
        model=model,
        task=task,
        family_dimensions=family_dimensions,
        family_sha256=family_sha256,
        slo_goodput_protocol_sha256=FORMAL_SLO_GOODPUT_PROTOCOL_SHA256,
        final_block_ids=tuple(sorted(paired["lightcone_vs_tts"])),
        final_goodput_observation_sha256s=tuple(
            sorted((block_id, role, row.sha256) for block_id, role, row in observations)
        ),
        primary_contrasts=contrasts,
        holm_decisions=decisions,
        status=status,
    )
    result.__post_init__()
    return result


def reduce_e3b_power_prefix_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E3bPowerPrefixReceipt:
    """Deep-replay 1,920 E3b pilot rows and freeze the final block prefix."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E3b power reduction requires an exact ProtocolLock")
    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("E3b power reduction requires exact pilot materialization")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E3b power reduction requires exact pilot coverage")
    if type(manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E3b power reduction requires exact proof manifest")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E3b power reduction requires sealed execution bindings")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E3b power reduction time must be non-negative")
    if (
        pilot_materialization.stage != "E3b"
        or pilot_materialization.materialization_rule
        != "e3b_exact_480_rows_x_4_excluded_pilot_blocks"
        or pilot_materialization.expected_cell_count != 480 * PILOT_BLOCK_COUNT
        or pilot_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.stage != "E3b"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != pilot_materialization.sha256
        or manifest.coverage_receipt_sha256 != pilot_coverage.sha256
        or manifest.source_authority_sha256
        != pilot_materialization.source_decision_sha256
    ):
        raise ValueError(
            "E3b pilot evidence differs from exact materialization lineage"
        )
    pilot_coverage.validate_against(pilot_materialization)
    if any(row.status != "COMPLETE" for row in pilot_coverage.dispositions):
        raise ValueError("E3b power reduction requires all-COMPLETE pilot coverage")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E3b power reduction reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_cells = {cell.cell_id for cell in pilot_materialization.cells}
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in pilot_coverage.dispositions
    }
    if (
        set(evidence_by_cell) != expected_cells
        or set(bindings_by_cell) != expected_cells
    ):
        raise ValueError("E3b pilot proof/binding coverage is not exact")
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E3b",
        )
        for cell in pilot_materialization.cells
    }
    by_block_stratum: dict[
        tuple[int, tuple[tuple[str, object], ...]], list[object]
    ] = {}
    for cell in pilot_materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if (
            type(block) is not int
            or block not in range(PILOT_BLOCK_COUNT)
            or dimensions.get("block_phase") != "excluded_pilot"
            or cell.method_role not in FORMAL_METHOD_ROLES
        ):
            raise ValueError("E3b pilot cell lies outside the four excluded blocks")
        row = validated[cell.cell_id]
        if row.safety_reasons or (
            cell.method_role in {"TTS", "L0-naive", "LightCone"}
            and row.published_updates < 1
        ):
            raise ValueError("E3b power pilot contains unsafe/inactive evidence")
        stratum = _e3b_family_dimensions(dimensions, final=False)
        by_block_stratum.setdefault((block, stratum), []).append(row)
    if len(by_block_stratum) != PILOT_BLOCK_COUNT * 96:
        raise ValueError("E3b pilot lacks exact 96-family block coverage")
    for rows in by_block_stratum.values():
        if (
            len(rows) != len(FORMAL_METHOD_ROLES)
            or {row.cell.method_role for row in rows} != set(FORMAL_METHOD_ROLES)
            or len({_request_identity(row.metrics) for row in rows}) != 1
        ):
            raise ValueError("E3b pilot methods differ in paired requests/trajectories")
    strata = {stratum for _block, stratum in by_block_stratum}
    if len(strata) != 96 or any(
        {(block, stratum) for block in range(PILOT_BLOCK_COUNT)} - set(by_block_stratum)
        for stratum in strata
    ):
        raise ValueError("E3b pilot family does not have exactly four blocks")
    commitments = []
    for stratum in sorted(strata, key=content_sha256):
        rows_by_block = {
            block: by_block_stratum[(block, stratum)]
            for block in range(PILOT_BLOCK_COUNT)
        }
        cells = tuple(row.cell for rows in rows_by_block.values() for row in rows)
        models = {cell.model for cell in cells}
        tasks = {cell.task for cell in cells}
        if (
            len(models) != 1
            or len(tasks) != 1
            or any(
                type(key) is not str or type(value) not in {str, int}
                for key, value in stratum
            )
        ):
            raise ValueError("E3b pilot family identity is ambiguous")
        commitments.append(
            _family_power_commitment(
                stage="E3b",
                model=next(iter(models)),
                task=next(iter(tasks)),
                family_dimensions=stratum,  # type: ignore[arg-type]
                rows_by_block=rows_by_block,
            )
        )
    family_commitments = tuple(sorted(commitments, key=lambda row: row.family_sha256))
    selected_final_blocks = max(
        row.power_sizing.selected_final_blocks or 0 for row in family_commitments
    )
    receipt = E3bPowerPrefixReceipt(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_coverage_receipt_sha256=pilot_coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        protocol_sha256=E3B_POWER_PREFIX_PROTOCOL_SHA256,
        family_power_commitments=family_commitments,
        selected_final_blocks=selected_final_blocks,
        selected_final_prefix=tuple(
            range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + selected_final_blocks)
        ),
    )
    receipt.__post_init__()
    return receipt


def reduce_e1a_verification_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E1aVerificationReceipt:
    """Deep-replay all 116 E1a rows and lock one DSpark configuration."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E1a verification requires an exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E1a verification requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E1a verification requires exact coverage")
    if type(manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E1a verification requires exact proof manifest")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E1a verification requires sealed execution bindings")
    if (
        type(now_ns) is not int
        or now_ns < 0
        or materialization.stage != "E1a"
        or materialization.materialization_rule
        != "58_configurations_x_2_verification_modes"
        or materialization.expected_cell_count != 116
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.stage != "E1a"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_authority_sha256 != materialization.source_decision_sha256
    ):
        raise ValueError("E1a evidence differs from exact materialization lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E1a verification requires all-COMPLETE coverage")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E1a verification reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_cells = {cell.cell_id for cell in materialization.cells}
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    if (
        set(evidence_by_cell) != expected_cells
        or set(bindings_by_cell) != expected_cells
    ):
        raise ValueError("E1a proof/binding coverage is not exact")
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E1a",
        )
        for cell in materialization.cells
    }
    by_mode: dict[str, list[object]] = {}
    static_by_mode: dict[str, object] = {}
    grouped: dict[tuple[tuple[str, str | int], ...], list[object]] = {}
    models = {cell.model for cell in materialization.cells}
    source_recipes = {
        cell.recipe_sha256
        for cell in materialization.cells
        if cell.method_role == "LightCone-candidate"
    }
    frozen_tts_recipes = {
        dict(cell.dimensions).get("frozen_tts_recipe_sha256")
        for cell in materialization.cells
    }
    if (
        len(models) != 1
        or len(source_recipes) != 1
        or None in source_recipes
        or len(frozen_tts_recipes) != 1
        or None in frozen_tts_recipes
    ):
        raise ValueError("E1a model/source recipe identity is ambiguous")
    for cell in materialization.cells:
        row = validated[cell.cell_id]
        dimensions = dict(cell.dimensions)
        mode = dimensions.get("verification_mode")
        if mode not in {"fixed_verification_budget", "native_scheduler"}:
            raise ValueError("E1a verification mode is unsupported")
        if dimensions.get("fixed_verification_budget") != (
            E1A_FIXED_VERIFICATION_BUDGET
            if mode == "fixed_verification_budget"
            else E1A_NATIVE_VERIFICATION_BUDGET
        ):
            raise ValueError("E1a fixed verification budget differs")
        by_mode.setdefault(str(mode), []).append(row)
        if cell.method_role == "Static":
            if mode in static_by_mode:
                raise ValueError("E1a repeats a Static verification anchor")
            static_by_mode[str(mode)] = row
        elif cell.method_role == "LightCone-candidate":
            configuration = (
                ("parameterization", dimensions["parameterization"]),
                ("rank", dimensions["rank"]),
                ("scope", dimensions["scope"]),
            )
            grouped.setdefault(configuration, []).append(row)
    if (
        set(by_mode) != {"fixed_verification_budget", "native_scheduler"}
        or set(static_by_mode) != set(by_mode)
        or any(len(rows) != 58 for rows in by_mode.values())
        or len(grouped) != 56
        or any(len(rows) != 2 for rows in grouped.values())
    ):
        raise ValueError("E1a configuration/mode coverage is not exact")
    for rows in by_mode.values():
        if len({_request_identity(row.metrics) for row in rows}) != 1:
            raise ValueError("E1a configurations use different request trajectories")
    evaluations = []
    for configuration, rows in grouped.items():
        if any(row.safety_reasons or row.published_updates < 1 for row in rows):
            continue
        lower = []
        for row in rows:
            mode = str(dict(row.cell.dimensions)["verification_mode"])
            lower.append(
                _paired_confidence_lower(row.metrics, static_by_mode[mode].metrics)
            )
        evaluations.append(
            E1aConfigurationEvaluation(
                configuration=configuration,
                cell_ids=tuple(sorted(row.cell.cell_id for row in rows)),  # type: ignore[arg-type]
                minimum_confidence_lower_request_rate_ratio=min(lower),
                peak_hbm_bytes=max(row.peak_hbm_bytes for row in rows),
                p99_itl_us=max(
                    max(
                        (
                            metric.p99_itl_ns.numerator
                            + metric.p99_itl_ns.denominator * 1_000
                            - 1
                        )
                        // (metric.p99_itl_ns.denominator * 1_000)
                        for metric in row.metrics
                    )
                    for row in rows
                ),
                exposed_update_us=max(row.exposed_update_us for row in rows),
            )
        )
    ordered = tuple(sorted(evaluations, key=lambda row: row.configuration_sha256))
    if len(ordered) != 56:
        raise ValueError("E1a has no complete safe evaluation for every configuration")
    winner = min(
        ordered,
        key=lambda row: (
            -row.minimum_confidence_lower_request_rate_ratio,
            row.peak_hbm_bytes,
            row.p99_itl_us,
            row.exposed_update_us,
            row.configuration_sha256,
        ),
    )
    source_recipe = next(iter(source_recipes))
    assert source_recipe is not None
    selected_recipe = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_e1a_selected_dspark_recipe",
            "source_lightcone_recipe_sha256": source_recipe,
            "configuration": winner.configuration,
            "verification_protocol_sha256": E1A_VERIFICATION_PROTOCOL_SHA256,
        }
    )
    receipt = E1aVerificationReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        protocol_sha256=E1A_VERIFICATION_PROTOCOL_SHA256,
        model=next(iter(models)),
        frozen_tts_recipe_sha256=next(iter(frozen_tts_recipes)),  # type: ignore[arg-type]
        source_lightcone_recipe_sha256=source_recipe,
        evaluations=ordered,
        selected_configuration=winner.configuration,
        selected_dspark_recipe_sha256=selected_recipe,
    )
    receipt.__post_init__()
    return receipt


def reduce_e5_power_and_anchors_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E5PowerAndAnchorReceipt:
    """Deep-replay the 1,800 excluded E5 headline pilots."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E5 power reduction requires an exact ProtocolLock")
    if type(pilot_materialization) is not StageMaterializationReceipt:
        raise TypeError("E5 power reduction requires exact pilot materialization")
    if type(pilot_coverage) is not StageCoverageReceipt:
        raise TypeError("E5 power reduction requires exact pilot coverage")
    if type(manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 power reduction requires exact proof manifest")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E5 power reduction requires sealed execution bindings")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E5 power reduction time must be non-negative")
    if (
        pilot_materialization.stage != "E5"
        or pilot_materialization.materialization_rule
        != "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks"
        or pilot_materialization.expected_cell_count != 450 * PILOT_BLOCK_COUNT
        or pilot_materialization.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.stage != "E5"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != pilot_materialization.sha256
        or manifest.coverage_receipt_sha256 != pilot_coverage.sha256
        or manifest.source_authority_sha256
        != pilot_materialization.source_decision_sha256
    ):
        raise ValueError("E5 pilot evidence differs from exact materialization lineage")
    pilot_coverage.validate_against(pilot_materialization)
    if any(row.status != "COMPLETE" for row in pilot_coverage.dispositions):
        raise ValueError("E5 power reduction requires all-COMPLETE pilot coverage")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E5 power reduction reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_cells = {cell.cell_id for cell in pilot_materialization.cells}
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in pilot_coverage.dispositions
    }
    if (
        set(evidence_by_cell) != expected_cells
        or set(bindings_by_cell) != expected_cells
    ):
        raise ValueError("E5 pilot proof/binding coverage is not exact")
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E5",
        )
        for cell in pilot_materialization.cells
    }
    by_block_stratum: dict[
        tuple[int, tuple[tuple[str, object], ...]], list[object]
    ] = {}
    anchor_groups: dict[tuple[str, str, str], list[object]] = {}
    models = {cell.model for cell in pilot_materialization.cells}
    source_values: dict[str, set[object]] = {
        name: set()
        for name in (
            "upstream_e1a_verification_sha256",
            "frozen_tts_recipe_sha256",
            "dflash_lightcone_recipe_sha256",
            "dspark_lightcone_recipe_sha256",
        )
    }
    for cell in pilot_materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if (
            type(block) is not int
            or block not in range(PILOT_BLOCK_COUNT)
            or dimensions.get("block_phase") != "excluded_pilot"
            or cell.method_role not in FORMAL_METHOD_ROLES
        ):
            raise ValueError("E5 pilot cell lies outside the four excluded blocks")
        for name, values in source_values.items():
            values.add(dimensions.get(name))
        row = validated[cell.cell_id]
        if row.safety_reasons or (
            cell.method_role in {"TTS", "L0-naive", "LightCone"}
            and row.published_updates < 1
        ):
            raise ValueError("E5 power pilot contains unsafe/inactive evidence")
        stratum = _e5_family_dimensions(
            dimensions,
            final=False,
            method_role=cell.method_role,
        )
        by_block_stratum.setdefault((block, stratum), []).append(row)
        if cell.method_role == "LightCone":
            backend = dimensions.get("backend_authority")
            topology = dimensions.get("topology")
            family_id = dimensions.get("family_id")
            if (
                backend not in E5_BACKENDS
                or topology not in E5_TOPOLOGIES
                or type(family_id) is not str
                or not family_id
            ):
                raise ValueError("E5 LightCone pilot family identity is invalid")
            anchor_groups.setdefault(
                (str(backend), str(topology), family_id), []
            ).append(row)
    if len(by_block_stratum) != PILOT_BLOCK_COUNT * 90 or any(
        len(rows) != len(FORMAL_METHOD_ROLES)
        or {row.cell.method_role for row in rows} != set(FORMAL_METHOD_ROLES)
        or len({_request_identity(row.metrics) for row in rows}) != 1
        for rows in by_block_stratum.values()
    ):
        raise ValueError("E5 pilot methods differ in paired requests/trajectories")
    if len(models) != 1 or any(
        len(values) != 1 or None in values for values in source_values.values()
    ):
        raise ValueError("E5 pilot model/source authority identity is ambiguous")
    strata = {stratum for _block, stratum in by_block_stratum}
    if len(strata) != 90 or any(
        {(block, stratum) for block in range(PILOT_BLOCK_COUNT)} - set(by_block_stratum)
        for stratum in strata
    ):
        raise ValueError("E5 pilot family does not have exactly four blocks")
    commitments = []
    for stratum in sorted(strata, key=content_sha256):
        rows_by_block = {
            block: by_block_stratum[(block, stratum)]
            for block in range(PILOT_BLOCK_COUNT)
        }
        cells = tuple(row.cell for rows in rows_by_block.values() for row in rows)
        family_models = {cell.model for cell in cells}
        family_tasks = {cell.task for cell in cells}
        if len(family_models) != 1 or len(family_tasks) != 1:
            raise ValueError("E5 pilot family model/task is ambiguous")
        commitments.append(
            _family_power_commitment(
                stage="E5",
                model=next(iter(family_models)),
                task=next(iter(family_tasks)),
                family_dimensions=stratum,  # type: ignore[arg-type]
                rows_by_block=rows_by_block,
            )
        )
    family_commitments = tuple(sorted(commitments, key=lambda row: row.family_sha256))
    selected_final_blocks = max(
        row.power_sizing.selected_final_blocks or 0 for row in family_commitments
    )
    anchors = []
    for backend in E5_BACKENDS:
        for topology in E5_TOPOLOGIES:
            eligible = {
                family_id: rows
                for (
                    row_backend,
                    row_topology,
                    family_id,
                ), rows in anchor_groups.items()
                if row_backend == backend and row_topology == topology
            }
            if not eligible or any(
                len(rows) != PILOT_BLOCK_COUNT for rows in eligible.values()
            ):
                raise ValueError("E5 p99 anchor family lacks all four excluded pilots")
            worst_family = min(
                eligible,
                key=lambda family_id: (
                    -max(
                        metric.p99_itl_ns
                        for row in eligible[family_id]
                        for metric in row.metrics
                    ),
                    family_id,
                ),
            )
            anchors.append(
                E5SelectedP99Anchor(
                    backend=backend,
                    topology=topology,
                    family_id=worst_family,
                    minimum_completions=10_000,
                )
            )
    receipt = E5PowerAndAnchorReceipt(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        upstream_e1a_verification_sha256=next(
            iter(source_values["upstream_e1a_verification_sha256"])
        ),  # type: ignore[arg-type]
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_coverage_receipt_sha256=pilot_coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        protocol_sha256=E5_POWER_AND_ANCHOR_PROTOCOL_SHA256,
        model=next(iter(models)),
        frozen_tts_recipe_sha256=next(iter(source_values["frozen_tts_recipe_sha256"])),  # type: ignore[arg-type]
        dflash_lightcone_recipe_sha256=next(
            iter(source_values["dflash_lightcone_recipe_sha256"])
        ),  # type: ignore[arg-type]
        dspark_lightcone_recipe_sha256=next(
            iter(source_values["dspark_lightcone_recipe_sha256"])
        ),  # type: ignore[arg-type]
        family_power_commitments=family_commitments,
        selected_final_blocks=selected_final_blocks,
        selected_final_prefix=tuple(
            range(
                PILOT_BLOCK_COUNT,
                PILOT_BLOCK_COUNT + selected_final_blocks,
            )
        ),
        p99_anchors=tuple(sorted(anchors, key=lambda row: row.anchor_id)),
    )
    receipt.__post_init__()
    return receipt


def reduce_e5_confirmation_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    headline_manifest: FormalDownstreamEvidenceManifest,
    headline_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    failure_manifest: E5FailureEvidenceManifest,
    failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...],
    now_ns: int,
) -> E5ConfirmationReceipt:
    """Deep-reopen every powered E5 headline and one-shot failure proof."""

    from lightcone_spec.experiments.failure_actuator import (
        validate_formal_failure_actuation_proof_artifact,
    )

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E5 confirmation requires exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E5 confirmation requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E5 confirmation requires exact coverage")
    if type(headline_manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E5 confirmation requires exact headline proof manifest")
    if type(failure_manifest) is not E5FailureEvidenceManifest:
        raise TypeError("E5 confirmation requires exact failure proof manifest")
    if type(headline_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in headline_execution_bindings
    ):
        raise TypeError("E5 confirmation requires sealed headline bindings")
    if type(failure_execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalFailureExecutionBinding
        for row in failure_execution_bindings
    ):
        raise TypeError("E5 confirmation requires sealed failure bindings")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E5 confirmation time must be positive")
    headline_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "production_slo_power_prefix"
    )
    failure_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "deterministic_failure_injection"
    )
    blocks, remainder = divmod(len(headline_cells), 450)
    if (
        materialization.stage != "E5"
        or materialization.materialization_rule
        != ("450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics")
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or remainder != 0
        or blocks not in range(12, 21)
        or len(headline_cells) != 450 * blocks
        or len(failure_cells) != 264
        or len(materialization.cells) != 450 * blocks + 264
        or headline_manifest.stage != "E5"
        or headline_manifest.protocol_lock_sha256 != protocol_lock.sha256
        or headline_manifest.materialization_receipt_sha256 != materialization.sha256
        or headline_manifest.coverage_receipt_sha256 != coverage.sha256
        or headline_manifest.source_authority_sha256
        != materialization.source_decision_sha256
        or failure_manifest.protocol_lock_sha256 != protocol_lock.sha256
        or failure_manifest.materialization_receipt_sha256 != materialization.sha256
        or failure_manifest.coverage_receipt_sha256 != coverage.sha256
        or failure_manifest.inventory_sha256 != headline_manifest.inventory_sha256
    ):
        raise ValueError("E5 confirmation evidence differs from exact lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E5 confirmation requires all-COMPLETE coverage")
    headline_block_values = {
        dict(cell.dimensions).get("block") for cell in headline_cells
    }
    if (
        any(type(block) is not int for block in headline_block_values)
        or headline_block_values
        != set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + blocks))
        or any(
            dict(cell.dimensions).get("block_phase") != "final"
            for cell in headline_cells
        )
    ):
        raise ValueError("E5 confirmation contains pilot or non-prefix headline rows")
    terminals = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    expected_headline_ids = {cell.cell_id for cell in headline_cells}
    expected_failure_ids = {cell.cell_id for cell in failure_cells}
    headline_evidence = {
        row.materialized_cell_id: row for row in headline_manifest.cells
    }
    failure_evidence = {row.materialized_cell_id: row for row in failure_manifest.cells}
    headline_bindings: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in headline_execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in headline_bindings:
            raise ValueError("E5 confirmation reuses a headline binding")
        headline_bindings[cell_id] = verified
    failure_bindings: dict[str, VerifiedFormalFailureExecutionBinding] = {}
    for binding in failure_execution_bindings:
        verified = require_verified_formal_failure_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in failure_bindings:
            raise ValueError("E5 confirmation reuses a failure binding")
        failure_bindings[cell_id] = verified
    if (
        set(headline_evidence) != expected_headline_ids
        or set(headline_bindings) != expected_headline_ids
        or set(failure_evidence) != expected_failure_ids
        or set(failure_bindings) != expected_failure_ids
    ):
        raise ValueError("E5 confirmation proof/binding coverage is not exact")
    validated_headline = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=headline_evidence[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=headline_bindings[cell.cell_id],
            coverage_terminal_sha256=terminals[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=headline_manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E5",
        )
        for cell in headline_cells
    }
    for cell in headline_cells:
        row = validated_headline[cell.cell_id]
        if row.safety_reasons or (
            cell.method_role in {"TTS", "L0-naive", "LightCone"}
            and row.published_updates < 1
        ):
            raise ValueError(
                "E5 confirmation contains unsafe/inactive headline evidence"
            )
    by_block_stratum: dict[
        tuple[int, tuple[tuple[str, object], ...]], list[object]
    ] = {}
    for cell in headline_cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        assert type(block) is int
        stratum = _e5_family_dimensions(
            dimensions,
            final=True,
            method_role=cell.method_role,
        )
        by_block_stratum.setdefault((block, stratum), []).append(
            validated_headline[cell.cell_id]
        )
    final_blocks = tuple(sorted(headline_block_values))
    if len(by_block_stratum) != len(final_blocks) * 90 or any(
        len(rows) != len(FORMAL_METHOD_ROLES)
        or {row.cell.method_role for row in rows} != set(FORMAL_METHOD_ROLES)
        or len({_request_identity(row.metrics) for row in rows}) != 1
        for rows in by_block_stratum.values()
    ):
        raise ValueError("E5 confirmation methods are not exactly family-paired")
    strata = {stratum for _block, stratum in by_block_stratum}
    if len(strata) != 90 or any(
        {(block, stratum) for block in final_blocks} - set(by_block_stratum)
        for stratum in strata
    ):
        raise ValueError("E5 confirmation family lacks every final block")
    family_results = []
    for stratum in sorted(strata, key=content_sha256):
        rows_by_block = {
            block: by_block_stratum[(block, stratum)] for block in final_blocks
        }
        cells = tuple(row.cell for rows in rows_by_block.values() for row in rows)
        family_models = {cell.model for cell in cells}
        family_tasks = {cell.task for cell in cells}
        if len(family_models) != 1 or len(family_tasks) != 1:
            raise ValueError("E5 confirmation family model/task is ambiguous")
        family_results.append(
            _family_confirmation_result(
                stage="E5",
                model=next(iter(family_models)),
                task=next(iter(family_tasks)),
                family_dimensions=stratum,  # type: ignore[arg-type]
                rows_by_block=rows_by_block,
            )
        )
    family_confirmations = tuple(
        sorted(family_results, key=lambda row: row.family_sha256)
    )
    failure_results = []
    for cell in failure_cells:
        evidence = failure_evidence[cell.cell_id]
        binding = failure_bindings[cell.cell_id]
        result = validate_formal_failure_actuation_proof_artifact(
            evidence.proof_artifact_path,
            binding=binding,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        subject = binding.subject
        dimensions = dict(cell.dimensions)
        if (
            evidence.failure_execution_binding_sha256 != binding.sha256
            or evidence.assignment_sha256 != subject.assignment_sha256
            or evidence.serving_execution_plan_sha256
            != subject.serving_execution_plan_sha256
            or result.assignment_sha256 != subject.assignment_sha256
            or result.cell_id != cell.cell_id
            or result.scenario != dimensions.get("failure")
            or result.inventory_sha256 != headline_manifest.inventory_sha256
            or result.registry_sha256 != protocol_lock.registry_sha256
            or result.correctness_only is not True
            or terminals[cell.cell_id] != result.sha256
        ):
            raise ValueError("E5 controlled failure proof differs from sealed cell")
        failure_results.append(result)
    anchors: dict[str, tuple[int, int]] = {}
    for cell in headline_cells:
        dimensions = dict(cell.dimensions)
        anchor_id = dimensions.get("p99_anchor_id")
        if anchor_id is None:
            continue
        if cell.method_role != "LightCone":
            raise ValueError("E5 p99 requirement is attached to a non-LightCone row")
        _sha256("E5 p99 anchor", anchor_id)
        minimum = dimensions.get("p99_minimum_completions")
        if type(minimum) is not int or minimum != 10_000:
            raise ValueError("E5 p99 minimum differs from selected anchor receipt")
        completed, observed_minimum = anchors.get(str(anchor_id), (0, minimum))
        if observed_minimum != minimum:
            raise ValueError("E5 p99 anchor minimum changes across blocks")
        anchors[str(anchor_id)] = (
            completed + len(validated_headline[cell.cell_id].metrics),
            minimum,
        )
    if len(anchors) != 6:
        raise ValueError("E5 confirmation lacks the exact six p99 anchors")
    anchor_completions = tuple(
        sorted(
            (
                E5P99AnchorCompletion(
                    anchor_id=anchor_id,
                    completed_requests=completed,
                )
                for anchor_id, (completed, minimum) in anchors.items()
                if completed >= minimum
            ),
            key=lambda row: row.anchor_id,
        )
    )
    if len(anchor_completions) != 6:
        raise ValueError("E5 selected p99 anchor has fewer than 10,000 completions")
    models = {cell.model for cell in headline_cells}
    frozen_tts = {
        cell.recipe_sha256
        for cell in headline_cells
        if cell.method_role in {"TTS", "L0-naive"}
    }
    dflash_recipes = {
        cell.recipe_sha256
        for cell in headline_cells
        if cell.method_role == "LightCone"
        and dict(cell.dimensions).get("backend_authority") == "DFLASH"
    }
    dspark_recipes = {
        cell.recipe_sha256
        for cell in headline_cells
        if cell.method_role == "LightCone"
        and dict(cell.dimensions).get("backend_authority") == "DSPARK"
    }
    if (
        len(models) != 1
        or len(frozen_tts) != 1
        or None in frozen_tts
        or len(dflash_recipes) != 1
        or None in dflash_recipes
        or len(dspark_recipes) != 1
        or None in dspark_recipes
    ):
        raise ValueError("E5 confirmation model/recipe identity is ambiguous")
    receipt = E5ConfirmationReceipt(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        headline_evidence_manifest_sha256=headline_manifest.sha256,
        failure_evidence_manifest_sha256=failure_manifest.sha256,
        inventory_sha256=headline_manifest.inventory_sha256,
        protocol_sha256=E5_CONFIRMATION_PROTOCOL_SHA256,
        model=next(iter(models)),
        frozen_tts_recipe_sha256=next(iter(frozen_tts)),  # type: ignore[arg-type]
        dflash_lightcone_recipe_sha256=next(iter(dflash_recipes)),  # type: ignore[arg-type]
        dspark_lightcone_recipe_sha256=next(iter(dspark_recipes)),  # type: ignore[arg-type]
        block_count=blocks,
        headline_cell_count=len(headline_cells),
        failure_cell_count=len(failure_cells),
        family_confirmations=family_confirmations,
        p99_anchor_completions=anchor_completions,
        failure_result_sha256s=tuple(sorted(row.sha256 for row in failure_results)),
        status=(
            "CONFIRMED"
            if all(row.status == "CONFIRMED" for row in family_confirmations)
            else "NOT_CONFIRMED"
        ),
    )
    receipt.__post_init__()
    return receipt


def reduce_e3b_confirmation_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E3bConfirmationReceipt:
    """Deep-replay E3b and reduce the preregistered two-member Holm family."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E3b confirmation requires an exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E3b confirmation requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E3b confirmation requires exact coverage")
    if type(manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E3b confirmation requires exact proof manifest")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E3b confirmation requires sealed execution bindings")
    if type(now_ns) is not int or now_ns < 0:
        raise ValueError("E3b confirmation time must be non-negative")
    block_values = {
        dict(cell.dimensions).get("block") for cell in materialization.cells
    }
    if (
        materialization.stage != "E3b"
        or materialization.materialization_rule
        != ("five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only")
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or any(type(value) is not int for value in block_values)
        or block_values
        != set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + len(block_values)))
        or not 12 <= len(block_values) <= 20
        or materialization.expected_cell_count != 480 * len(block_values)
        or manifest.stage != "E3b"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_authority_sha256 != materialization.source_decision_sha256
    ):
        raise ValueError("E3b confirmation evidence differs from exact lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E3b confirmation requires all-COMPLETE coverage")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E3b confirmation reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_cells = {cell.cell_id for cell in materialization.cells}
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    if (
        set(evidence_by_cell) != expected_cells
        or set(bindings_by_cell) != expected_cells
    ):
        raise ValueError("E3b confirmation proof/binding coverage is not exact")
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E3b",
        )
        for cell in materialization.cells
    }
    by_block_stratum: dict[
        tuple[int, tuple[tuple[str, object], ...]], list[object]
    ] = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions["block"]
        assert type(block) is int
        row = validated[cell.cell_id]
        if row.safety_reasons or (
            cell.method_role in {"TTS", "L0-naive", "LightCone"}
            and row.published_updates < 1
        ):
            raise ValueError("E3b confirmation contains unsafe/inactive evidence")
        if dimensions.get("block_phase") != "final":
            raise ValueError("E3b confirmation contains a non-final block")
        stratum = _e3b_family_dimensions(dimensions, final=True)
        by_block_stratum.setdefault((block, stratum), []).append(row)
    if len(by_block_stratum) != len(block_values) * 96:
        raise ValueError("E3b confirmation lacks exact block/family coverage")
    for rows in by_block_stratum.values():
        if (
            len(rows) != len(FORMAL_METHOD_ROLES)
            or {row.cell.method_role for row in rows} != set(FORMAL_METHOD_ROLES)
            or len({_request_identity(row.metrics) for row in rows}) != 1
        ):
            raise ValueError("E3b confirmation methods are not exactly paired")
    final_blocks = tuple(sorted(block_values))
    strata = {stratum for _block, stratum in by_block_stratum}
    if len(strata) != 96 or any(
        {(block, stratum) for block in final_blocks} - set(by_block_stratum)
        for stratum in strata
    ):
        raise ValueError("E3b confirmation family lacks every final block")
    family_results = []
    for stratum in sorted(strata, key=content_sha256):
        rows_by_block = {
            block: by_block_stratum[(block, stratum)] for block in final_blocks
        }
        cells = tuple(row.cell for rows in rows_by_block.values() for row in rows)
        family_models = {cell.model for cell in cells}
        family_tasks = {cell.task for cell in cells}
        if len(family_models) != 1 or len(family_tasks) != 1:
            raise ValueError("E3b confirmation family model/task is ambiguous")
        family_results.append(
            _family_confirmation_result(
                stage="E3b",
                model=next(iter(family_models)),
                task=next(iter(family_tasks)),
                family_dimensions=stratum,  # type: ignore[arg-type]
                rows_by_block=rows_by_block,
            )
        )
    family_confirmations = tuple(
        sorted(family_results, key=lambda row: row.family_sha256)
    )
    models = {cell.model for cell in materialization.cells}
    tts_recipes = {
        cell.recipe_sha256
        for cell in materialization.cells
        if cell.method_role in {"TTS", "L0-naive"}
    }
    lightcone_recipes = {
        cell.recipe_sha256
        for cell in materialization.cells
        if cell.method_role == "LightCone"
    }
    if (
        len(models) != 1
        or len(tts_recipes) != 1
        or None in tts_recipes
        or len(lightcone_recipes) != 1
        or None in lightcone_recipes
    ):
        raise ValueError("E3b confirmation model/recipe identity is ambiguous")
    tts_recipe = next(iter(tts_recipes))
    lightcone_recipe = next(iter(lightcone_recipes))
    assert tts_recipe is not None and lightcone_recipe is not None
    receipt = E3bConfirmationReceipt(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        protocol_sha256=E3B_CONFIRMATION_PROTOCOL_SHA256,
        model=next(iter(models)),
        frozen_tts_recipe_sha256=tts_recipe,
        lightcone_recipe_sha256=lightcone_recipe,
        final_block_ids=tuple(f"E3b:final:{block - 4:02d}" for block in final_blocks),
        family_confirmations=family_confirmations,
        status=(
            "CONFIRMED"
            if all(row.status == "CONFIRMED" for row in family_confirmations)
            else "NOT_CONFIRMED"
        ),
    )
    receipt.__post_init__()
    return receipt


__all__ = [
    "E1A_VERIFICATION_PROTOCOL_SHA256",
    "E3B_CONFIRMATION_PROTOCOL_SHA256",
    "E3B_POWER_PREFIX_PROTOCOL_SHA256",
    "E5_CONFIRMATION_PROTOCOL_SHA256",
    "E5_POWER_AND_ANCHOR_PROTOCOL_SHA256",
    "E1aConfigurationEvaluation",
    "E1aVerificationReceipt",
    "E3bConfirmationReceipt",
    "E3bPowerPrefixReceipt",
    "E5ConfirmationReceipt",
    "E5FailureCellEvidence",
    "E5FailureEvidenceManifest",
    "E5P99AnchorCompletion",
    "E5PowerAndAnchorReceipt",
    "FormalDownstreamCellEvidence",
    "FormalDownstreamEvidenceManifest",
    "FormalFamilyConfirmationResult",
    "FormalFamilyPowerCommitment",
    "SignedE1aVerificationReceipt",
    "SignedE3bConfirmationReceipt",
    "SignedE3bPowerPrefixReceipt",
    "SignedE5ConfirmationReceipt",
    "SignedE5PowerAndAnchorReceipt",
    "reduce_e1a_verification_from_proofs",
    "reduce_e3b_confirmation_from_proofs",
    "reduce_e3b_power_prefix_from_proofs",
    "reduce_e5_confirmation_from_proofs",
    "reduce_e5_power_and_anchors_from_proofs",
]
