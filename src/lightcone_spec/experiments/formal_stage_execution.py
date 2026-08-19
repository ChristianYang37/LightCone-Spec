"""Verifier-sealed serving bindings for signed-staged materialized cells.

This module closes the identity gap between a scientific ``MaterializedCell``
and the native terminal that later claims to implement it.  A terminal's
``materialized_cell_id`` is not authority: the exact RunConfig, recipe source,
physical GPU order, runtime qualification, execution-plan identity, and rank
configuration are rebuilt here before a private-sealed binding is returned.

Every registered serving stage has an exact source/recipe adapter.  A stage
still fails with a stable BLOCKED reason when a required signed upstream
receipt, native proof, or runtime capability is absent; the historical generic
registry mapper is never used as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from lightcone_spec.config import (
    OptimizerConfig,
    RunConfig,
    load_run_config,
    run_config_sha256,
)
from lightcone_spec.experiments.formal_protocol import (
    FormalRuntimeAuthorityManifest,
    ProtocolLock,
    SignedTtsCalibrationSeal,
    TtsCalibrationAuthority,
    content_sha256,
    reject_banned_model_identity,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.registry import build_industrial_registry
from lightcone_spec.experiments.stage_materialization import (
    E1_OPTIMIZER_ANCHORS,
    E1A_FIXED_VERIFICATION_BUDGET,
    E1A_NATIVE_VERIFICATION_BUDGET,
    E1Geometry,
    E2CandidateRecipe,
    E2RecipeGridAuthority,
    E5FailureDiagnosticAuthority,
    MaterializedCell,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.attestation import TrustedAttesterPolicy
from lightcone_spec.runtime.backend import (
    VerifiedNextNTp2Authority,
    validate_nextn_tp2_dynamic_authority_artifact,
)
from lightcone_spec.runtime.content_authorization import (
    ContentVerificationReceipt,
    DatasetContentPathBinding,
    TtsCalibrationTuningWindow,
    VerifiedDatasetContentRelease,
    VerifiedPreparedModelContentRelease,
    VerifiedReleaseWorkloadSources,
    revalidate_authorized_dataset_content_release,
)
from lightcone_spec.runtime.distributed import (
    DistributedRuntimeGpuProofArtifact,
    VerifiedDistributedRuntimeGpuProof,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.readiness import (
    NativeRuntimeGpuProofArtifact,
    VerifiedNativeRuntimeGpuProof,
    require_chronobelief_gpu_proof,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.downstream_stage_authority import (
        E5FailureEvidenceManifest,
        FormalDownstreamEvidenceManifest,
        SignedE1aVerificationReceipt,
        SignedE3bConfirmationReceipt,
        SignedE3bPowerPrefixReceipt,
        SignedE5ConfirmationReceipt,
        SignedE5PowerAndAnchorReceipt,
    )
    from lightcone_spec.experiments.e0_stage_authority import (
        E0FormalRegistryAuthorityBundle,
        E0OnlineSpecSourceAuthority,
        E0OnlineSpecTuningProofSet,
        E6ConfirmationProofBundle,
        SignedE0OnlineSpecTuningSeal,
    )
    from lightcone_spec.experiments.e2_stage_authority import (
        E2StagedRoundEvidenceManifest,
        SignedE2StagedRoundSelectionReceipt,
    )
    from lightcone_spec.experiments.e4_stage_authority import (
        E4StagedEvidenceManifest,
        SignedE4StageSelectionReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
        SignedE6ConfirmationReceipt,
        SignedE6ModelCompatibilityReceipt,
        SignedE6PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.formal_failure_execution import (
        VerifiedFormalFailureExecutionBinding,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import (
        SignedE0CompatibilityReceipt,
        StageCoverageReceipt,
    )

FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "lightcone_formal_serving_execution_binding_protocol",
        "materialized_cell": "exact_signed_stage_materialization_member",
        "configuration": "strict_RunConfig_and_recipe_authority",
        "physical_identity": (
            "inventory_gpu_uuid_order_and_canonical_runtime_topology"
        ),
        "qualification": (
            "exact_backend_topology_native_suite_union_plus_distributed_proof"
        ),
        "nextn_tp2": (
            "verifier_owned_two_model_shard_authority_joined_to_live_proof_union"
        ),
        "terminal_join": (
            "derived_execution_plan_and_rank_config_sha256_not_caller_labels"
        ),
        "content": ("durable_content_verification_receipt_and_exact_stage_members"),
        "supported_stages": (
            "E3a",
            "TTS-Cal",
            "E1",
            "E2",
            "E4",
            "E3b",
            "E1a",
            "E5",
            "E6",
            "E0",
        ),
        "downstream_source": (
            "private_sealed_binding_from_exact_typed_materializer_rebuild"
        ),
    }
)

FORMAL_SERVING_EXECUTION_REBUILD_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_serving_execution_rebuild_protocol",
        "subject": "strict_schema4_codec",
        "run_config": "path_bound_canonical_json",
        "content": "path_bound_content_verification_receipt",
        "runtime_gpu_proofs": "exact_subject_path_set_reopened_by_public_verifier",
        "nextn_tp2": "path_bound_E6_two_model_authority_input_reopened_and_joined",
        "stage_source": "private_seal_rebuilt_upstream_then_sha_bound",
        "recipe_authorities": "exact_subject_authority_sha_set",
        "registry": "durable_verification_receipt_sha_bound_when_required",
        "generic_or_naked_digest_fallback": "forbidden",
    }
)
FORMAL_SINGLE_OPERATOR_EXECUTION_BINDING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "lightcone_formal_single_operator_execution_binding_protocol",
        "trust_model": "formal_single_operator_v1",
        "current_source": (
            "path_bound_FormalSingleOperatorExecutionSource_deep_reopened"
        ),
        "configuration": (
            "RunConfig_and_GPU_assignment_derived_from_CompileLaunchManifest"
        ),
        "cell_join": (
            "exact_current_materialization_member_then_existing_formal_mapper"
        ),
        "runtime_proofs": "existing_private_verifier_reused_without_bypass",
        "e4_headline": (
            "current_e2r3_or_screen_actual_to_source_owned_config_launch_plan"
        ),
        "e4_profiler": "blocked_until_dedicated_nsys_ncu_plan",
        "durable_rebuild": ("all_verifier_inputs_path_bound_for_clean_process_restart"),
        "caller_recipe_config_port_argv_run_identity": "forbidden",
    }
)
FORMAL_SERVING_EXECUTION_RUNNER_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_serving_execution_mapper_runner",
        "module": "lightcone_spec.experiments.formal_stage_execution",
        "entrypoints": (
            "prepare_formal_serving_execution_subject",
            "verify_formal_serving_execution_binding",
        ),
        "supported_stages": (
            "E3a",
            "TTS-Cal",
            "E1",
            "E2",
            "E4",
            "E3b",
            "E1a",
            "E5",
            "E6",
            "E0",
        ),
    }
)
FORMAL_SERVING_EXECUTION_TEST_SET_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_serving_execution_mapper_acceptance_set",
        "requirements": (
            "direct_subject_construction_non_authorizing",
            "exact_recipe_and_workload_rebuild",
            "exact_canonical_topology_and_gpu_uuid_order",
            "durable_content_and_runtime_gpu_proof_reopen",
            "backend_topology_suite_union_missing_and_foreign_rejected",
            "nextn_tp2_two_model_content_shard_and_live_proof_join",
            "foreign_mapper_identity_rejected",
        ),
    }
)

CanonicalTopologyMode = Literal["tp1_dp1", "tp2_dp1", "tp1_dp2"]
FormalServingStage = Literal[
    "E3a", "TTS-Cal", "E1", "E2", "E4", "E3b", "E1a", "E5", "E6", "E0"
]
FormalServingMethod = Literal[
    "target_only",
    "static",
    "tts",
    "l0",
    "onlinespec_ogd",
    "onlinespec_opt",
    "onlinespec_ens",
]
_FORMAL_SERVING_STAGES = {
    "E3a",
    "TTS-Cal",
    "E1",
    "E2",
    "E4",
    "E3b",
    "E1a",
    "E5",
    "E6",
    "E0",
}
_DOWNSTREAM_SERVING_STAGES = {"E4", "E3b", "E1a", "E5", "E6", "E0"}


class FormalStageExecutionBlocked(RuntimeError):
    """A cell lacks a source-owned execution authority required for launch."""

    def __init__(self, reason_code: str) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("formal execution BLOCKED reason must be text")
        self.reason_code = reason_code
        super().__init__(f"formal stage execution is BLOCKED: {reason_code}")


_VERIFIED_CURRENT_E0_EAGLE3_PROOF_ROW_SEAL = object()


@dataclass(frozen=True, init=False)
class _VerifiedCurrentE0Eagle3ProofRow:
    """Private join of one current auxiliary task row and live proof path."""

    execution_source_sha256: str
    compatibility_bundle_sha256: str
    interface_receipt_sha256: str
    terminal_sha256: str
    proof_row_sha256: str
    execution_authority_sha256: str
    compatibility_authority_sha256: str
    model_selector_sha256: str
    native_gpu_receipt_sha256: str
    _construction_seal: object

    def __init__(
        self,
        *,
        execution_source_sha256: str,
        compatibility_bundle_sha256: str,
        interface_receipt_sha256: str,
        terminal_sha256: str,
        proof_row_sha256: str,
        execution_authority_sha256: str,
        compatibility_authority_sha256: str,
        model_selector_sha256: str,
        native_gpu_receipt_sha256: str,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VERIFIED_CURRENT_E0_EAGLE3_PROOF_ROW_SEAL:
            raise TypeError("current E0 EAGLE3 proof row is verifier-constructed only")
        for label, value in (
            ("execution source", execution_source_sha256),
            ("compatibility bundle", compatibility_bundle_sha256),
            ("interface receipt", interface_receipt_sha256),
            ("compatibility terminal", terminal_sha256),
            ("proof row", proof_row_sha256),
            ("execution authority", execution_authority_sha256),
            ("compatibility authority", compatibility_authority_sha256),
            ("model selector", model_selector_sha256),
            ("native GPU receipt", native_gpu_receipt_sha256),
        ):
            _require_sha256(f"current E0 EAGLE3 {label}", value)
        for name, value in (
            ("execution_source_sha256", execution_source_sha256),
            ("compatibility_bundle_sha256", compatibility_bundle_sha256),
            ("interface_receipt_sha256", interface_receipt_sha256),
            ("terminal_sha256", terminal_sha256),
            ("proof_row_sha256", proof_row_sha256),
            ("execution_authority_sha256", execution_authority_sha256),
            ("compatibility_authority_sha256", compatibility_authority_sha256),
            ("model_selector_sha256", model_selector_sha256),
            ("native_gpu_receipt_sha256", native_gpu_receipt_sha256),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)


_VERIFIED_FORMAL_STAGE_SOURCE_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedFormalStageMaterializationSource:
    """Private-sealed proof that a downstream receipt was exactly rebuilt.

    The public materializers already implement the typed signed-source DAG.
    Re-running the relevant materializer is both stricter and less fragile
    than duplicating that DAG here.  Only the stage-specific functions below
    may mint this value; a digest or a caller-authored label is never accepted
    as a substitute.
    """

    stage: str
    phase: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    source_decision_sha256: str
    typed_source_authority_sha256s: tuple[str, ...]
    _construction_seal: object

    def __init__(
        self,
        *,
        stage: str,
        phase: str,
        materialization: StageMaterializationReceipt,
        typed_source_authority_sha256s: tuple[str, ...],
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VERIFIED_FORMAL_STAGE_SOURCE_SEAL:
            raise TypeError("formal stage source binding is verifier-constructed only")
        if stage not in {"E4", "E3b", "E1a", "E5", "E6", "E0"}:
            raise ValueError("formal stage source binding names an unsupported stage")
        if materialization.stage != stage:
            raise ValueError("formal stage source binding names another receipt")
        if type(phase) is not str or not phase or phase.strip() != phase:
            raise ValueError("formal stage source phase is invalid")
        if (
            type(typed_source_authority_sha256s) is not tuple
            or not typed_source_authority_sha256s
            or typed_source_authority_sha256s
            != tuple(sorted(set(typed_source_authority_sha256s)))
        ):
            raise ValueError("formal stage typed source identities are not canonical")
        for digest in typed_source_authority_sha256s:
            _require_sha256("formal stage typed source authority", digest)
        for name, value in (
            ("stage", stage),
            ("phase", phase),
            ("protocol_lock_sha256", materialization.protocol_lock_sha256),
            ("materialization_receipt_sha256", materialization.sha256),
            ("source_decision_sha256", materialization.source_decision_sha256),
            ("typed_source_authority_sha256s", typed_source_authority_sha256s),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "verified_formal_stage_materialization_source",
                "stage": self.stage,
                "phase": self.phase,
                "protocol_lock_sha256": self.protocol_lock_sha256,
                "materialization_receipt_sha256": (self.materialization_receipt_sha256),
                "source_decision_sha256": self.source_decision_sha256,
                "typed_source_authority_sha256s": (self.typed_source_authority_sha256s),
            }
        )


def _seal_rebuilt_stage_source(
    *,
    expected: StageMaterializationReceipt,
    rebuilt: StageMaterializationReceipt,
    phase: str,
    authority_sha256s: tuple[str, ...],
) -> VerifiedFormalStageMaterializationSource:
    if type(expected) is not StageMaterializationReceipt:
        raise TypeError("formal stage source requires exact materialization")
    if rebuilt != expected:
        raise ValueError(
            "formal stage materialization differs from typed-source rebuild"
        )
    return VerifiedFormalStageMaterializationSource(
        stage=expected.stage,
        phase=phase,
        materialization=expected,
        typed_source_authority_sha256s=tuple(sorted(set(authority_sha256s))),
        _construction_seal=_VERIFIED_FORMAL_STAGE_SOURCE_SEAL,
    )


def require_verified_formal_stage_materialization_source(
    value: object,
    *,
    materialization: StageMaterializationReceipt,
) -> VerifiedFormalStageMaterializationSource:
    if (
        type(value) is not VerifiedFormalStageMaterializationSource
        or value._construction_seal is not _VERIFIED_FORMAL_STAGE_SOURCE_SEAL
        or value.stage != materialization.stage
        or value.protocol_lock_sha256 != materialization.protocol_lock_sha256
        or value.materialization_receipt_sha256 != materialization.sha256
        or value.source_decision_sha256 != materialization.source_decision_sha256
    ):
        raise TypeError("formal downstream execution requires its sealed typed source")
    return value


@dataclass(frozen=True)
class E4ScreenStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e2_final_selection: SignedE2StagedRoundSelectionReceipt
    e2_materialization: StageMaterializationReceipt
    e2_coverage: StageCoverageReceipt
    e2_source_recipes: tuple[E2CandidateRecipe, ...]
    e2_evidence_manifest: E2StagedRoundEvidenceManifest
    e2_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]


@dataclass(frozen=True)
class E4LocalStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e4_screen_selection: SignedE4StageSelectionReceipt
    screen_materialization: StageMaterializationReceipt
    screen_coverage: StageCoverageReceipt
    screen_evidence_manifest: E4StagedEvidenceManifest
    screen_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]


@dataclass(frozen=True)
class E4ProfilerStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e4_final_selection: SignedE4StageSelectionReceipt
    local_materialization: StageMaterializationReceipt
    local_coverage: StageCoverageReceipt
    local_evidence_manifest: E4StagedEvidenceManifest
    local_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]


@dataclass(frozen=True)
class E3bPilotStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e4_final_selection: SignedE4StageSelectionReceipt
    local_materialization: StageMaterializationReceipt
    local_coverage: StageCoverageReceipt
    local_evidence_manifest: E4StagedEvidenceManifest
    local_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    profiler_materialization: StageMaterializationReceipt
    profiler_coverage: StageCoverageReceipt
    tts_calibration_authority: TtsCalibrationAuthority
    signed_tts_calibration_seal: SignedTtsCalibrationSeal


@dataclass(frozen=True)
class E3bFinalStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_power_prefix: SignedE3bPowerPrefixReceipt
    pilot_materialization: StageMaterializationReceipt
    pilot_coverage: StageCoverageReceipt
    pilot_evidence_manifest: FormalDownstreamEvidenceManifest
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]


@dataclass(frozen=True)
class E1aStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e3b_confirmation: SignedE3bConfirmationReceipt
    e3b_materialization: StageMaterializationReceipt
    e3b_coverage: StageCoverageReceipt
    e3b_evidence_manifest: FormalDownstreamEvidenceManifest
    e3b_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]


@dataclass(frozen=True)
class E5PilotStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e1a_verification: SignedE1aVerificationReceipt
    e1a_materialization: StageMaterializationReceipt
    e1a_coverage: StageCoverageReceipt
    e1a_evidence_manifest: FormalDownstreamEvidenceManifest
    e1a_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest


@dataclass(frozen=True)
class E5FinalStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_power_and_anchor_prefix: SignedE5PowerAndAnchorReceipt
    pilot_materialization: StageMaterializationReceipt
    pilot_coverage: StageCoverageReceipt
    pilot_evidence_manifest: FormalDownstreamEvidenceManifest
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest
    failure_diagnostic_authority: E5FailureDiagnosticAuthority


@dataclass(frozen=True)
class E6PilotStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e5_confirmation: SignedE5ConfirmationReceipt
    e5_materialization: StageMaterializationReceipt
    e5_coverage: StageCoverageReceipt
    e5_headline_evidence_manifest: FormalDownstreamEvidenceManifest
    e5_headline_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    e5_failure_evidence_manifest: E5FailureEvidenceManifest
    e5_failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...]
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...]


@dataclass(frozen=True)
class E6FinalStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e5_confirmation: SignedE5ConfirmationReceipt
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...]
    signed_power_prefix: SignedE6PowerPrefixReceipt
    pilot_materialization: StageMaterializationReceipt
    pilot_coverage: StageCoverageReceipt
    pilot_evidence_manifest: FormalDownstreamEvidenceManifest
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]


@dataclass(frozen=True)
class E0TuningStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e6_confirmation: SignedE6ConfirmationReceipt
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle
    signed_compatibility_receipt: SignedE0CompatibilityReceipt
    onlinespec_source_authority: E0OnlineSpecSourceAuthority


@dataclass(frozen=True)
class E0PilotStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    signed_e6_confirmation: SignedE6ConfirmationReceipt
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle
    signed_compatibility_receipt: SignedE0CompatibilityReceipt
    signed_onlinespec_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...]
    onlinespec_source_authority: E0OnlineSpecSourceAuthority
    tuning_proof_set: E0OnlineSpecTuningProofSet


@dataclass(frozen=True)
class E0FinalStageSourceRebuildInputs:
    registry_verification_receipt: FormalRegistryVerificationReceipt
    authority_bundle: E0FormalRegistryAuthorityBundle


FormalStageSourceRebuildInputs = (
    E4ScreenStageSourceRebuildInputs
    | E4LocalStageSourceRebuildInputs
    | E4ProfilerStageSourceRebuildInputs
    | E3bPilotStageSourceRebuildInputs
    | E3bFinalStageSourceRebuildInputs
    | E1aStageSourceRebuildInputs
    | E5PilotStageSourceRebuildInputs
    | E5FinalStageSourceRebuildInputs
    | E6PilotStageSourceRebuildInputs
    | E6FinalStageSourceRebuildInputs
    | E0TuningStageSourceRebuildInputs
    | E0PilotStageSourceRebuildInputs
    | E0FinalStageSourceRebuildInputs
)


def _binding_sha256s(
    rows: tuple[VerifiedFormalServingExecutionBinding, ...],
) -> tuple[str, ...]:
    if type(rows) is not tuple or not rows:
        raise ValueError("formal stage source rebuild execution bindings are empty")
    values = tuple(
        require_verified_formal_serving_execution_binding(row).sha256 for row in rows
    )
    if len(values) != len(set(values)):
        raise ValueError("formal stage source rebuild reuses an execution binding")
    return values


def _failure_binding_sha256s(
    rows: tuple[VerifiedFormalFailureExecutionBinding, ...],
) -> tuple[str, ...]:
    from lightcone_spec.experiments.formal_failure_execution import (
        require_verified_formal_failure_execution_binding,
    )

    if type(rows) is not tuple or not rows:
        raise ValueError("formal stage source rebuild failure bindings are empty")
    values = tuple(
        require_verified_formal_failure_execution_binding(row).sha256 for row in rows
    )
    if len(values) != len(set(values)):
        raise ValueError("formal stage source rebuild reuses a failure binding")
    return values


def _e6_confirmation_bundle_commitment(value: object) -> str:
    from lightcone_spec.experiments.e0_stage_authority import (
        E6ConfirmationProofBundle,
    )

    if type(value) is not E6ConfirmationProofBundle:
        raise TypeError("formal stage source rebuild E6 proof bundle is not exact")
    value.__post_init__()
    return content_sha256(
        {
            "signed_model_compatibility_sha256": (
                value.signed_model_compatibility.sha256
            ),
            "compatibility_source_sha256s": tuple(
                row.sha256 for row in value.compatibility_sources
            ),
            "materialization_receipt_sha256": value.materialization.sha256,
            "coverage_receipt_sha256": value.coverage.sha256,
            "evidence_manifest_sha256": value.manifest.sha256,
            "execution_binding_sha256s": _binding_sha256s(value.execution_bindings),
        }
    )


def _e0_proof_set_commitment(value: object) -> str:
    from lightcone_spec.experiments.e0_stage_authority import (
        E0OnlineSpecTuningProofSet,
    )

    if type(value) is not E0OnlineSpecTuningProofSet:
        raise TypeError("formal stage source rebuild E0 proof set is not exact")
    value.__post_init__()
    return content_sha256(
        {
            "e6_confirmation_proof_bundle_sha256": (
                _e6_confirmation_bundle_commitment(value.e6_confirmation_proof_bundle)
            ),
            "materialization_receipt_sha256": value.materialization.sha256,
            "coverage_receipt_sha256": value.coverage.sha256,
            "evidence_manifest_sha256": value.manifest.sha256,
            "execution_binding_sha256s": _binding_sha256s(value.execution_bindings),
        }
    )


def _stage_source_input_commitment(
    value: FormalStageSourceRebuildInputs,
) -> tuple[str, str, str]:
    from lightcone_spec.experiments.downstream_stage_authority import (
        E5FailureEvidenceManifest,
        FormalDownstreamEvidenceManifest,
        SignedE1aVerificationReceipt,
        SignedE3bConfirmationReceipt,
        SignedE3bPowerPrefixReceipt,
        SignedE5ConfirmationReceipt,
        SignedE5PowerAndAnchorReceipt,
    )
    from lightcone_spec.experiments.e0_stage_authority import (
        E0FormalRegistryAuthorityBundle,
        E0OnlineSpecSourceAuthority,
        E0OnlineSpecTuningProofSet,
        E6ConfirmationProofBundle,
        SignedE0OnlineSpecTuningSeal,
    )
    from lightcone_spec.experiments.e2_stage_authority import (
        E2StagedRoundEvidenceManifest,
        SignedE2StagedRoundSelectionReceipt,
    )
    from lightcone_spec.experiments.e4_stage_authority import (
        E4StagedEvidenceManifest,
        SignedE4StageSelectionReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
        SignedE6ConfirmationReceipt,
        SignedE6ModelCompatibilityReceipt,
        SignedE6PowerPrefixReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.stage_materialization import (
        SignedE0CompatibilityReceipt,
        StageCoverageReceipt,
    )

    if type(value.registry_verification_receipt) is not (
        FormalRegistryVerificationReceipt
    ):
        raise TypeError("formal stage source rebuild registry receipt is not exact")
    registry_sha256 = value.registry_verification_receipt.sha256
    if type(value) is E4ScreenStageSourceRebuildInputs:
        if (
            type(value.signed_e2_final_selection)
            is not SignedE2StagedRoundSelectionReceipt
            or type(value.e2_materialization) is not StageMaterializationReceipt
            or type(value.e2_coverage) is not StageCoverageReceipt
            or type(value.e2_source_recipes) is not tuple
            or any(
                type(row) is not E2CandidateRecipe for row in value.e2_source_recipes
            )
            or type(value.e2_evidence_manifest) is not E2StagedRoundEvidenceManifest
        ):
            raise TypeError("E4 screen stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e2_final_selection_sha256": (
                    value.signed_e2_final_selection.sha256
                ),
                "e2_materialization_sha256": value.e2_materialization.sha256,
                "e2_coverage_sha256": value.e2_coverage.sha256,
                "e2_source_recipe_sha256s": tuple(
                    row.sha256 for row in value.e2_source_recipes
                ),
                "e2_evidence_manifest_sha256": value.e2_evidence_manifest.sha256,
                "e2_execution_binding_sha256s": _binding_sha256s(
                    value.e2_execution_bindings
                ),
            }
        )
        return "E4", "screen", commitment
    if type(value) is E4LocalStageSourceRebuildInputs:
        if (
            type(value.signed_e4_screen_selection) is not SignedE4StageSelectionReceipt
            or type(value.screen_materialization) is not StageMaterializationReceipt
            or type(value.screen_coverage) is not StageCoverageReceipt
            or type(value.screen_evidence_manifest) is not E4StagedEvidenceManifest
        ):
            raise TypeError("E4 local stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e4_screen_selection_sha256": (
                    value.signed_e4_screen_selection.sha256
                ),
                "screen_materialization_sha256": value.screen_materialization.sha256,
                "screen_coverage_sha256": value.screen_coverage.sha256,
                "screen_evidence_manifest_sha256": (
                    value.screen_evidence_manifest.sha256
                ),
                "screen_execution_binding_sha256s": _binding_sha256s(
                    value.screen_execution_bindings
                ),
            }
        )
        return "E4", "local", commitment
    if type(value) is E4ProfilerStageSourceRebuildInputs:
        if (
            type(value.signed_e4_final_selection) is not SignedE4StageSelectionReceipt
            or type(value.local_materialization) is not StageMaterializationReceipt
            or type(value.local_coverage) is not StageCoverageReceipt
            or type(value.local_evidence_manifest) is not E4StagedEvidenceManifest
        ):
            raise TypeError("E4 profiler stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e4_final_selection_sha256": (
                    value.signed_e4_final_selection.sha256
                ),
                "local_materialization_sha256": value.local_materialization.sha256,
                "local_coverage_sha256": value.local_coverage.sha256,
                "local_evidence_manifest_sha256": value.local_evidence_manifest.sha256,
                "local_execution_binding_sha256s": _binding_sha256s(
                    value.local_execution_bindings
                ),
            }
        )
        return "E4", "profiler", commitment
    if type(value) is E3bPilotStageSourceRebuildInputs:
        if (
            type(value.signed_e4_final_selection) is not SignedE4StageSelectionReceipt
            or type(value.local_materialization) is not StageMaterializationReceipt
            or type(value.local_coverage) is not StageCoverageReceipt
            or type(value.local_evidence_manifest) is not E4StagedEvidenceManifest
            or type(value.profiler_materialization) is not StageMaterializationReceipt
            or type(value.profiler_coverage) is not StageCoverageReceipt
            or type(value.tts_calibration_authority) is not TtsCalibrationAuthority
            or type(value.signed_tts_calibration_seal) is not SignedTtsCalibrationSeal
        ):
            raise TypeError("E3b pilot stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e4_final_selection_sha256": (
                    value.signed_e4_final_selection.sha256
                ),
                "local_materialization_sha256": value.local_materialization.sha256,
                "local_coverage_sha256": value.local_coverage.sha256,
                "local_evidence_manifest_sha256": value.local_evidence_manifest.sha256,
                "local_execution_binding_sha256s": _binding_sha256s(
                    value.local_execution_bindings
                ),
                "profiler_materialization_sha256": (
                    value.profiler_materialization.sha256
                ),
                "profiler_coverage_sha256": value.profiler_coverage.sha256,
                "tts_calibration_authority_sha256": (
                    value.tts_calibration_authority.sha256
                ),
                "signed_tts_calibration_seal_sha256": (
                    value.signed_tts_calibration_seal.sha256
                ),
            }
        )
        return "E3b", "excluded_pilot", commitment
    if type(value) is E3bFinalStageSourceRebuildInputs:
        if (
            type(value.signed_power_prefix) is not SignedE3bPowerPrefixReceipt
            or type(value.pilot_materialization) is not StageMaterializationReceipt
            or type(value.pilot_coverage) is not StageCoverageReceipt
            or type(value.pilot_evidence_manifest)
            is not FormalDownstreamEvidenceManifest
        ):
            raise TypeError("E3b final stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_power_prefix_sha256": value.signed_power_prefix.sha256,
                "pilot_materialization_sha256": value.pilot_materialization.sha256,
                "pilot_coverage_sha256": value.pilot_coverage.sha256,
                "pilot_evidence_manifest_sha256": (
                    value.pilot_evidence_manifest.sha256
                ),
                "pilot_execution_binding_sha256s": _binding_sha256s(
                    value.pilot_execution_bindings
                ),
            }
        )
        return "E3b", "final", commitment
    if type(value) is E1aStageSourceRebuildInputs:
        if (
            type(value.signed_e3b_confirmation) is not SignedE3bConfirmationReceipt
            or type(value.e3b_materialization) is not StageMaterializationReceipt
            or type(value.e3b_coverage) is not StageCoverageReceipt
            or type(value.e3b_evidence_manifest) is not FormalDownstreamEvidenceManifest
        ):
            raise TypeError("E1a stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e3b_confirmation_sha256": (
                    value.signed_e3b_confirmation.sha256
                ),
                "e3b_materialization_sha256": value.e3b_materialization.sha256,
                "e3b_coverage_sha256": value.e3b_coverage.sha256,
                "e3b_evidence_manifest_sha256": value.e3b_evidence_manifest.sha256,
                "e3b_execution_binding_sha256s": _binding_sha256s(
                    value.e3b_execution_bindings
                ),
            }
        )
        return "E1a", "verification", commitment
    if type(value) is E5PilotStageSourceRebuildInputs:
        if (
            type(value.signed_e1a_verification) is not SignedE1aVerificationReceipt
            or type(value.e1a_materialization) is not StageMaterializationReceipt
            or type(value.e1a_coverage) is not StageCoverageReceipt
            or type(value.e1a_evidence_manifest) is not FormalDownstreamEvidenceManifest
            or type(value.formal_runtime_authority_manifest)
            is not FormalRuntimeAuthorityManifest
        ):
            raise TypeError("E5 pilot stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e1a_verification_sha256": (
                    value.signed_e1a_verification.sha256
                ),
                "e1a_materialization_sha256": value.e1a_materialization.sha256,
                "e1a_coverage_sha256": value.e1a_coverage.sha256,
                "e1a_evidence_manifest_sha256": value.e1a_evidence_manifest.sha256,
                "e1a_execution_binding_sha256s": _binding_sha256s(
                    value.e1a_execution_bindings
                ),
                "formal_runtime_authority_manifest_sha256": (
                    value.formal_runtime_authority_manifest.sha256
                ),
            }
        )
        return "E5", "excluded_pilot", commitment
    if type(value) is E5FinalStageSourceRebuildInputs:
        if (
            type(value.signed_power_and_anchor_prefix)
            is not SignedE5PowerAndAnchorReceipt
            or type(value.pilot_materialization) is not StageMaterializationReceipt
            or type(value.pilot_coverage) is not StageCoverageReceipt
            or type(value.pilot_evidence_manifest)
            is not FormalDownstreamEvidenceManifest
            or type(value.formal_runtime_authority_manifest)
            is not FormalRuntimeAuthorityManifest
            or type(value.failure_diagnostic_authority)
            is not E5FailureDiagnosticAuthority
        ):
            raise TypeError("E5 final stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_power_and_anchor_prefix_sha256": (
                    value.signed_power_and_anchor_prefix.sha256
                ),
                "pilot_materialization_sha256": value.pilot_materialization.sha256,
                "pilot_coverage_sha256": value.pilot_coverage.sha256,
                "pilot_evidence_manifest_sha256": (
                    value.pilot_evidence_manifest.sha256
                ),
                "pilot_execution_binding_sha256s": _binding_sha256s(
                    value.pilot_execution_bindings
                ),
                "formal_runtime_authority_manifest_sha256": (
                    value.formal_runtime_authority_manifest.sha256
                ),
                "failure_diagnostic_authority_sha256": (
                    value.failure_diagnostic_authority.sha256
                ),
            }
        )
        return "E5", "final_and_one_shot_failure", commitment
    if type(value) is E6PilotStageSourceRebuildInputs:
        if (
            type(value.signed_e5_confirmation) is not SignedE5ConfirmationReceipt
            or type(value.e5_materialization) is not StageMaterializationReceipt
            or type(value.e5_coverage) is not StageCoverageReceipt
            or type(value.e5_headline_evidence_manifest)
            is not FormalDownstreamEvidenceManifest
            or type(value.e5_failure_evidence_manifest) is not E5FailureEvidenceManifest
            or type(value.signed_model_compatibility)
            is not SignedE6ModelCompatibilityReceipt
            or type(value.compatibility_sources) is not tuple
            or any(
                type(row) is not E6NextnModelAuthorityInput
                for row in value.compatibility_sources
            )
        ):
            raise TypeError("E6 pilot stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e5_confirmation_sha256": (value.signed_e5_confirmation.sha256),
                "e5_materialization_sha256": value.e5_materialization.sha256,
                "e5_coverage_sha256": value.e5_coverage.sha256,
                "e5_headline_evidence_manifest_sha256": (
                    value.e5_headline_evidence_manifest.sha256
                ),
                "e5_headline_execution_binding_sha256s": _binding_sha256s(
                    value.e5_headline_execution_bindings
                ),
                "e5_failure_evidence_manifest_sha256": (
                    value.e5_failure_evidence_manifest.sha256
                ),
                "e5_failure_execution_binding_sha256s": (
                    _failure_binding_sha256s(value.e5_failure_execution_bindings)
                ),
                "signed_model_compatibility_sha256": (
                    value.signed_model_compatibility.sha256
                ),
                "compatibility_source_sha256s": tuple(
                    row.sha256 for row in value.compatibility_sources
                ),
            }
        )
        return "E6", "excluded_pilot_and_model_preflight", commitment
    if type(value) is E6FinalStageSourceRebuildInputs:
        if (
            type(value.signed_model_compatibility)
            is not SignedE6ModelCompatibilityReceipt
            or type(value.signed_power_prefix) is not SignedE6PowerPrefixReceipt
            or type(value.compatibility_sources) is not tuple
            or any(
                type(row) is not E6NextnModelAuthorityInput
                for row in value.compatibility_sources
            )
            or type(value.pilot_materialization) is not StageMaterializationReceipt
            or type(value.pilot_coverage) is not StageCoverageReceipt
        ):
            raise TypeError("E6 final stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e5_confirmation_sha256": (value.signed_e5_confirmation.sha256),
                "signed_model_compatibility_sha256": (
                    value.signed_model_compatibility.sha256
                ),
                "compatibility_source_sha256s": tuple(
                    row.sha256 for row in value.compatibility_sources
                ),
                "signed_power_prefix_sha256": value.signed_power_prefix.sha256,
                "pilot_materialization_sha256": value.pilot_materialization.sha256,
                "pilot_coverage_sha256": value.pilot_coverage.sha256,
                "pilot_evidence_manifest_sha256": (
                    value.pilot_evidence_manifest.sha256
                ),
                "pilot_execution_binding_sha256s": _binding_sha256s(
                    value.pilot_execution_bindings
                ),
            }
        )
        return "E6", "final", commitment
    if type(value) is E0TuningStageSourceRebuildInputs:
        if (
            type(value.signed_e6_confirmation) is not SignedE6ConfirmationReceipt
            or type(value.e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle
            or type(value.signed_compatibility_receipt)
            is not SignedE0CompatibilityReceipt
            or type(value.onlinespec_source_authority)
            is not E0OnlineSpecSourceAuthority
        ):
            raise TypeError("E0 tuning stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e6_confirmation_sha256": (value.signed_e6_confirmation.sha256),
                "e6_confirmation_proof_bundle_sha256": (
                    _e6_confirmation_bundle_commitment(
                        value.e6_confirmation_proof_bundle
                    )
                ),
                "signed_compatibility_sha256": (
                    value.signed_compatibility_receipt.sha256
                ),
                "onlinespec_source_authority_sha256": (
                    value.onlinespec_source_authority.sha256
                ),
            }
        )
        return "E0", "onlinespec_tuning", commitment
    if type(value) is E0PilotStageSourceRebuildInputs:
        if (
            type(value.signed_e6_confirmation) is not SignedE6ConfirmationReceipt
            or type(value.e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle
            or type(value.signed_compatibility_receipt)
            is not SignedE0CompatibilityReceipt
            or type(value.signed_onlinespec_tuning_seals) is not tuple
            or any(
                type(row) is not SignedE0OnlineSpecTuningSeal
                for row in value.signed_onlinespec_tuning_seals
            )
            or type(value.onlinespec_source_authority)
            is not E0OnlineSpecSourceAuthority
            or type(value.tuning_proof_set) is not E0OnlineSpecTuningProofSet
        ):
            raise TypeError("E0 pilot stage source rebuild inputs are not exact")
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e6_confirmation_sha256": (value.signed_e6_confirmation.sha256),
                "e6_confirmation_proof_bundle_sha256": (
                    _e6_confirmation_bundle_commitment(
                        value.e6_confirmation_proof_bundle
                    )
                ),
                "signed_compatibility_sha256": (
                    value.signed_compatibility_receipt.sha256
                ),
                "signed_tuning_seal_sha256s": tuple(
                    row.sha256 for row in value.signed_onlinespec_tuning_seals
                ),
                "onlinespec_source_authority_sha256": (
                    value.onlinespec_source_authority.sha256
                ),
                "tuning_proof_set_sha256": _e0_proof_set_commitment(
                    value.tuning_proof_set
                ),
            }
        )
        return "E0", "excluded_pilot", commitment
    if type(value) is E0FinalStageSourceRebuildInputs:
        if type(value.authority_bundle) is not E0FormalRegistryAuthorityBundle:
            raise TypeError("E0 final stage source rebuild bundle is not exact")
        bundle = value.authority_bundle
        bundle.__post_init__()
        commitment = content_sha256(
            {
                "registry_receipt_sha256": registry_sha256,
                "signed_e6_confirmation_sha256": (bundle.signed_e6_confirmation.sha256),
                "e6_confirmation_proof_bundle_sha256": (
                    _e6_confirmation_bundle_commitment(
                        bundle.e6_confirmation_proof_bundle
                    )
                ),
                "signed_compatibility_sha256": bundle.signed_compatibility.sha256,
                "onlinespec_source_authority_sha256": (bundle.source_authority.sha256),
                "tuning_proof_set_sha256": _e0_proof_set_commitment(
                    bundle.tuning_proof_set
                ),
                "signed_tuning_seal_sha256s": tuple(
                    row.sha256 for row in bundle.signed_tuning_seals
                ),
                "pilot_proof_set_sha256": _e0_proof_set_commitment(
                    bundle.pilot_proof_set
                ),
                "signed_power_prefix_sha256": bundle.signed_power_prefix.sha256,
            }
        )
        return "E0", "final", commitment
    raise TypeError("formal stage source rebuild input kind is unsupported")


@dataclass(frozen=True)
class FormalStageSourceRebuildInput:
    schema_version: Literal[1]
    kind: Literal["formal_stage_source_rebuild_input"]
    stage: Literal["E4", "E3b", "E1a", "E5", "E6", "E0"]
    phase: Literal[
        "screen",
        "local",
        "profiler",
        "verification",
        "final",
        "final_and_one_shot_failure",
        "onlinespec_tuning",
        "excluded_pilot",
        "excluded_pilot_and_model_preflight",
    ]
    materialization_receipt_sha256: str
    source_decision_sha256: str
    registry_verification_receipt_sha256: str
    source_input_commitment_sha256: str
    expected_stage_source_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != (
            "formal_stage_source_rebuild_input"
        ):
            raise ValueError("formal stage source rebuild descriptor is unsupported")
        if (self.stage, self.phase) not in {
            ("E4", "screen"),
            ("E4", "local"),
            ("E4", "profiler"),
            ("E3b", "excluded_pilot"),
            ("E3b", "final"),
            ("E1a", "verification"),
            ("E5", "excluded_pilot"),
            ("E5", "final_and_one_shot_failure"),
            ("E6", "excluded_pilot_and_model_preflight"),
            ("E6", "final"),
            ("E0", "onlinespec_tuning"),
            ("E0", "excluded_pilot"),
            ("E0", "final"),
        }:
            raise ValueError("formal stage source rebuild phase is unsupported")
        for label, digest in (
            ("materialization", self.materialization_receipt_sha256),
            ("source decision", self.source_decision_sha256),
            ("registry receipt", self.registry_verification_receipt_sha256),
            ("source input commitment", self.source_input_commitment_sha256),
            ("expected stage source", self.expected_stage_source_sha256),
        ):
            _require_sha256(f"formal stage source rebuild {label}", digest)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "stage": self.stage,
            "phase": self.phase,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "source_decision_sha256": self.source_decision_sha256,
            "registry_verification_receipt_sha256": (
                self.registry_verification_receipt_sha256
            ),
            "source_input_commitment_sha256": (self.source_input_commitment_sha256),
            "expected_stage_source_sha256": self.expected_stage_source_sha256,
        }
        if include_sha256:
            value["rebuild_input_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_rebuild_object(
            "formal stage source rebuild descriptor",
            value,
            {*cls.__dataclass_fields__, "rebuild_input_sha256"},
        )
        declared = _require_sha256(
            "formal stage source rebuild descriptor",
            row.pop("rebuild_input_sha256"),
        )
        descriptor = cls(**row)  # type: ignore[arg-type]
        if descriptor.sha256 != declared:
            raise ValueError("formal stage source rebuild descriptor digest differs")
        return descriptor


def publish_formal_stage_source_rebuild_input(
    descriptor: FormalStageSourceRebuildInput,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish one typed source descriptor with atomic no-replace semantics."""

    if type(descriptor) is not FormalStageSourceRebuildInput:
        raise TypeError("formal stage source publisher requires an exact descriptor")
    descriptor.__post_init__()
    publish_canonical_json_no_replace(str(output_path), descriptor.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def bind_formal_stage_source_rebuild_input(
    source: VerifiedFormalStageMaterializationSource,
    *,
    materialization: StageMaterializationReceipt,
    source_inputs: FormalStageSourceRebuildInputs,
) -> FormalStageSourceRebuildInput:
    sealed = require_verified_formal_stage_materialization_source(
        source,
        materialization=materialization,
    )
    stage, phase, commitment = _stage_source_input_commitment(source_inputs)
    registry_sha256 = source_inputs.registry_verification_receipt.sha256
    if sealed.stage != stage or sealed.phase != phase:
        raise ValueError("formal stage source rebuild input phase differs")
    return FormalStageSourceRebuildInput(
        schema_version=1,
        kind="formal_stage_source_rebuild_input",
        stage=stage,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        materialization_receipt_sha256=materialization.sha256,
        source_decision_sha256=materialization.source_decision_sha256,
        registry_verification_receipt_sha256=registry_sha256,
        source_input_commitment_sha256=commitment,
        expected_stage_source_sha256=sealed.sha256,
    )


def rebuild_formal_stage_materialization_source(
    descriptor: FormalStageSourceRebuildInput,
    *,
    materialization: StageMaterializationReceipt,
    source_inputs: FormalStageSourceRebuildInputs,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    """Rebuild a downstream private source token from an exact typed DAG node."""

    if type(descriptor) is not FormalStageSourceRebuildInput:
        raise TypeError("formal stage source rebuild requires exact descriptor")
    descriptor.__post_init__()
    stage, phase, commitment = _stage_source_input_commitment(source_inputs)
    if (
        descriptor.stage != stage
        or descriptor.phase != phase
        or descriptor.materialization_receipt_sha256 != materialization.sha256
        or descriptor.source_decision_sha256 != materialization.source_decision_sha256
        or descriptor.registry_verification_receipt_sha256
        != source_inputs.registry_verification_receipt.sha256
        or descriptor.source_input_commitment_sha256 != commitment
    ):
        raise ValueError("formal stage source rebuild typed lineage differs")
    if type(source_inputs) is E4ScreenStageSourceRebuildInputs:
        rebuilt = verify_e4_screen_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e2_final_selection=source_inputs.signed_e2_final_selection,
            e2_materialization=source_inputs.e2_materialization,
            e2_coverage=source_inputs.e2_coverage,
            e2_source_recipes=source_inputs.e2_source_recipes,
            e2_evidence_manifest=source_inputs.e2_evidence_manifest,
            e2_execution_bindings=source_inputs.e2_execution_bindings,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E4LocalStageSourceRebuildInputs:
        rebuilt = verify_e4_local_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e4_screen_selection=(source_inputs.signed_e4_screen_selection),
            screen_materialization=source_inputs.screen_materialization,
            screen_coverage=source_inputs.screen_coverage,
            screen_evidence_manifest=source_inputs.screen_evidence_manifest,
            screen_execution_bindings=source_inputs.screen_execution_bindings,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E4ProfilerStageSourceRebuildInputs:
        rebuilt = verify_e4_profiler_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e4_final_selection=source_inputs.signed_e4_final_selection,
            local_materialization=source_inputs.local_materialization,
            local_coverage=source_inputs.local_coverage,
            local_evidence_manifest=source_inputs.local_evidence_manifest,
            local_execution_bindings=source_inputs.local_execution_bindings,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E3bPilotStageSourceRebuildInputs:
        rebuilt = verify_e3b_pilot_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e4_final_selection=source_inputs.signed_e4_final_selection,
            local_materialization=source_inputs.local_materialization,
            local_coverage=source_inputs.local_coverage,
            local_evidence_manifest=source_inputs.local_evidence_manifest,
            local_execution_bindings=source_inputs.local_execution_bindings,
            profiler_materialization=source_inputs.profiler_materialization,
            profiler_coverage=source_inputs.profiler_coverage,
            tts_calibration_authority=source_inputs.tts_calibration_authority,
            signed_tts_calibration_seal=(source_inputs.signed_tts_calibration_seal),
            now_ns=now_ns,
        )
    elif type(source_inputs) is E3bFinalStageSourceRebuildInputs:
        rebuilt = verify_e3b_final_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_power_prefix=source_inputs.signed_power_prefix,
            pilot_materialization=source_inputs.pilot_materialization,
            pilot_coverage=source_inputs.pilot_coverage,
            pilot_evidence_manifest=source_inputs.pilot_evidence_manifest,
            pilot_execution_bindings=source_inputs.pilot_execution_bindings,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E1aStageSourceRebuildInputs:
        rebuilt = verify_e1a_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e3b_confirmation=source_inputs.signed_e3b_confirmation,
            e3b_materialization=source_inputs.e3b_materialization,
            e3b_coverage=source_inputs.e3b_coverage,
            e3b_evidence_manifest=source_inputs.e3b_evidence_manifest,
            e3b_execution_bindings=source_inputs.e3b_execution_bindings,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E5PilotStageSourceRebuildInputs:
        rebuilt = verify_e5_pilot_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e1a_verification=source_inputs.signed_e1a_verification,
            e1a_materialization=source_inputs.e1a_materialization,
            e1a_coverage=source_inputs.e1a_coverage,
            e1a_evidence_manifest=source_inputs.e1a_evidence_manifest,
            e1a_execution_bindings=source_inputs.e1a_execution_bindings,
            formal_runtime_authority_manifest=(
                source_inputs.formal_runtime_authority_manifest
            ),
            now_ns=now_ns,
        )
    elif type(source_inputs) is E5FinalStageSourceRebuildInputs:
        rebuilt = verify_e5_final_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_power_and_anchor_prefix=(
                source_inputs.signed_power_and_anchor_prefix
            ),
            pilot_materialization=source_inputs.pilot_materialization,
            pilot_coverage=source_inputs.pilot_coverage,
            pilot_evidence_manifest=source_inputs.pilot_evidence_manifest,
            pilot_execution_bindings=source_inputs.pilot_execution_bindings,
            formal_runtime_authority_manifest=(
                source_inputs.formal_runtime_authority_manifest
            ),
            failure_diagnostic_authority=(source_inputs.failure_diagnostic_authority),
            now_ns=now_ns,
        )
    elif type(source_inputs) is E6PilotStageSourceRebuildInputs:
        rebuilt = verify_e6_pilot_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e5_confirmation=source_inputs.signed_e5_confirmation,
            e5_materialization=source_inputs.e5_materialization,
            e5_coverage=source_inputs.e5_coverage,
            e5_headline_evidence_manifest=(source_inputs.e5_headline_evidence_manifest),
            e5_headline_execution_bindings=(
                source_inputs.e5_headline_execution_bindings
            ),
            e5_failure_evidence_manifest=(source_inputs.e5_failure_evidence_manifest),
            e5_failure_execution_bindings=(source_inputs.e5_failure_execution_bindings),
            signed_model_compatibility=source_inputs.signed_model_compatibility,
            compatibility_sources=source_inputs.compatibility_sources,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E6FinalStageSourceRebuildInputs:
        rebuilt = verify_e6_final_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e5_confirmation=source_inputs.signed_e5_confirmation,
            signed_model_compatibility=source_inputs.signed_model_compatibility,
            compatibility_sources=source_inputs.compatibility_sources,
            signed_power_prefix=source_inputs.signed_power_prefix,
            pilot_materialization=source_inputs.pilot_materialization,
            pilot_coverage=source_inputs.pilot_coverage,
            pilot_evidence_manifest=source_inputs.pilot_evidence_manifest,
            pilot_execution_bindings=source_inputs.pilot_execution_bindings,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E0TuningStageSourceRebuildInputs:
        rebuilt = verify_e0_tuning_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e6_confirmation=source_inputs.signed_e6_confirmation,
            e6_confirmation_proof_bundle=(source_inputs.e6_confirmation_proof_bundle),
            signed_compatibility_receipt=(source_inputs.signed_compatibility_receipt),
            onlinespec_source_authority=(source_inputs.onlinespec_source_authority),
            now_ns=now_ns,
        )
    elif type(source_inputs) is E0PilotStageSourceRebuildInputs:
        rebuilt = verify_e0_pilot_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            signed_e6_confirmation=source_inputs.signed_e6_confirmation,
            e6_confirmation_proof_bundle=(source_inputs.e6_confirmation_proof_bundle),
            signed_compatibility_receipt=(source_inputs.signed_compatibility_receipt),
            signed_onlinespec_tuning_seals=(
                source_inputs.signed_onlinespec_tuning_seals
            ),
            onlinespec_source_authority=(source_inputs.onlinespec_source_authority),
            tuning_proof_set=source_inputs.tuning_proof_set,
            now_ns=now_ns,
        )
    elif type(source_inputs) is E0FinalStageSourceRebuildInputs:
        rebuilt = verify_e0_final_execution_source(
            materialization=materialization,
            registry_verification_receipt=(source_inputs.registry_verification_receipt),
            authority_bundle=source_inputs.authority_bundle,
            now_ns=now_ns,
        )
    else:  # pragma: no cover - exact union is closed above
        raise AssertionError("unsupported formal stage source rebuild input")
    if rebuilt.sha256 != descriptor.expected_stage_source_sha256:
        raise ValueError("formal stage source rebuild result differs")
    return rebuilt


def verify_e4_screen_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e2_final_selection: SignedE2StagedRoundSelectionReceipt,
    e2_materialization: StageMaterializationReceipt,
    e2_coverage: StageCoverageReceipt,
    e2_source_recipes: tuple[E2CandidateRecipe, ...],
    e2_evidence_manifest: E2StagedRoundEvidenceManifest,
    e2_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import (
        materialize_e4_strength2_screen,
    )

    rebuilt = materialize_e4_strength2_screen(
        registry_verification_receipt=registry_verification_receipt,
        signed_e2_final_selection=signed_e2_final_selection,
        e2_materialization=e2_materialization,
        e2_coverage=e2_coverage,
        e2_source_recipes=e2_source_recipes,
        e2_evidence_manifest=e2_evidence_manifest,
        e2_execution_bindings=e2_execution_bindings,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="screen",
        authority_sha256s=(signed_e2_final_selection.sha256,),
    )


def verify_e4_local_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e4_screen_selection: SignedE4StageSelectionReceipt,
    screen_materialization: StageMaterializationReceipt,
    screen_coverage: StageCoverageReceipt,
    screen_evidence_manifest: E4StagedEvidenceManifest,
    screen_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import (
        materialize_e4_winner_neighborhood,
    )

    rebuilt = materialize_e4_winner_neighborhood(
        registry_verification_receipt=registry_verification_receipt,
        signed_e4_screen_selection=signed_e4_screen_selection,
        screen_materialization=screen_materialization,
        screen_coverage=screen_coverage,
        screen_evidence_manifest=screen_evidence_manifest,
        screen_execution_bindings=screen_execution_bindings,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="local",
        authority_sha256s=(signed_e4_screen_selection.sha256,),
    )


def verify_e4_profiler_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e4_final_selection: SignedE4StageSelectionReceipt,
    local_materialization: StageMaterializationReceipt,
    local_coverage: StageCoverageReceipt,
    local_evidence_manifest: E4StagedEvidenceManifest,
    local_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import materialize_e4_profiler

    rebuilt = materialize_e4_profiler(
        registry_verification_receipt=registry_verification_receipt,
        signed_e4_final_selection=signed_e4_final_selection,
        local_materialization=local_materialization,
        local_coverage=local_coverage,
        local_evidence_manifest=local_evidence_manifest,
        local_execution_bindings=local_execution_bindings,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="profiler",
        authority_sha256s=(signed_e4_final_selection.sha256,),
    )


def verify_e3b_pilot_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e4_final_selection: SignedE4StageSelectionReceipt,
    local_materialization: StageMaterializationReceipt,
    local_coverage: StageCoverageReceipt,
    local_evidence_manifest: E4StagedEvidenceManifest,
    local_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    profiler_materialization: StageMaterializationReceipt,
    profiler_coverage: StageCoverageReceipt,
    tts_calibration_authority: TtsCalibrationAuthority,
    signed_tts_calibration_seal: SignedTtsCalibrationSeal,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import (
        materialize_e3b_excluded_pilots,
    )

    rebuilt = materialize_e3b_excluded_pilots(
        registry_verification_receipt=registry_verification_receipt,
        signed_e4_final_selection=signed_e4_final_selection,
        local_materialization=local_materialization,
        local_coverage=local_coverage,
        local_evidence_manifest=local_evidence_manifest,
        local_execution_bindings=local_execution_bindings,
        profiler_materialization=profiler_materialization,
        profiler_coverage=profiler_coverage,
        tts_calibration_authority=tts_calibration_authority,
        signed_tts_calibration_seal=signed_tts_calibration_seal,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="excluded_pilot",
        authority_sha256s=(
            signed_e4_final_selection.sha256,
            signed_tts_calibration_seal.sha256,
            tts_calibration_authority.sha256,
        ),
    )


def verify_e3b_final_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_power_prefix: SignedE3bPowerPrefixReceipt,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    pilot_evidence_manifest: FormalDownstreamEvidenceManifest,
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import materialize_e3b

    rebuilt = materialize_e3b(
        registry_verification_receipt=registry_verification_receipt,
        signed_power_prefix=signed_power_prefix,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        pilot_evidence_manifest=pilot_evidence_manifest,
        pilot_execution_bindings=pilot_execution_bindings,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="final",
        authority_sha256s=(signed_power_prefix.sha256,),
    )


def verify_e1a_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e3b_confirmation: SignedE3bConfirmationReceipt,
    e3b_materialization: StageMaterializationReceipt,
    e3b_coverage: StageCoverageReceipt,
    e3b_evidence_manifest: FormalDownstreamEvidenceManifest,
    e3b_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import materialize_e1a

    rebuilt = materialize_e1a(
        registry_verification_receipt=registry_verification_receipt,
        signed_e3b_confirmation=signed_e3b_confirmation,
        e3b_materialization=e3b_materialization,
        e3b_coverage=e3b_coverage,
        e3b_evidence_manifest=e3b_evidence_manifest,
        e3b_execution_bindings=e3b_execution_bindings,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="verification",
        authority_sha256s=(signed_e3b_confirmation.sha256,),
    )


def verify_e5_pilot_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e1a_verification: SignedE1aVerificationReceipt,
    e1a_materialization: StageMaterializationReceipt,
    e1a_coverage: StageCoverageReceipt,
    e1a_evidence_manifest: FormalDownstreamEvidenceManifest,
    e1a_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import (
        materialize_e5_excluded_pilots,
    )

    rebuilt = materialize_e5_excluded_pilots(
        registry_verification_receipt=registry_verification_receipt,
        signed_e1a_verification=signed_e1a_verification,
        e1a_materialization=e1a_materialization,
        e1a_coverage=e1a_coverage,
        e1a_evidence_manifest=e1a_evidence_manifest,
        e1a_execution_bindings=e1a_execution_bindings,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="excluded_pilot",
        authority_sha256s=(
            signed_e1a_verification.sha256,
            formal_runtime_authority_manifest.sha256,
        ),
    )


def verify_e5_final_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_power_and_anchor_prefix: SignedE5PowerAndAnchorReceipt,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    pilot_evidence_manifest: FormalDownstreamEvidenceManifest,
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    failure_diagnostic_authority: E5FailureDiagnosticAuthority,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import materialize_e5

    rebuilt = materialize_e5(
        registry_verification_receipt=registry_verification_receipt,
        signed_power_and_anchor_prefix=signed_power_and_anchor_prefix,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        pilot_evidence_manifest=pilot_evidence_manifest,
        pilot_execution_bindings=pilot_execution_bindings,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        failure_diagnostic_authority=failure_diagnostic_authority,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="final_and_one_shot_failure",
        authority_sha256s=(
            signed_power_and_anchor_prefix.sha256,
            formal_runtime_authority_manifest.sha256,
            failure_diagnostic_authority.sha256,
        ),
    )


def verify_e6_pilot_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e5_confirmation: SignedE5ConfirmationReceipt,
    e5_materialization: StageMaterializationReceipt,
    e5_coverage: StageCoverageReceipt,
    e5_headline_evidence_manifest: FormalDownstreamEvidenceManifest,
    e5_headline_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    e5_failure_evidence_manifest: E5FailureEvidenceManifest,
    e5_failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...],
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import (
        materialize_e6_excluded_pilots,
    )

    rebuilt = materialize_e6_excluded_pilots(
        registry_verification_receipt=registry_verification_receipt,
        signed_e5_confirmation=signed_e5_confirmation,
        e5_materialization=e5_materialization,
        e5_coverage=e5_coverage,
        e5_headline_evidence_manifest=e5_headline_evidence_manifest,
        e5_headline_execution_bindings=e5_headline_execution_bindings,
        e5_failure_evidence_manifest=e5_failure_evidence_manifest,
        e5_failure_execution_bindings=e5_failure_execution_bindings,
        signed_model_compatibility=signed_model_compatibility,
        compatibility_sources=compatibility_sources,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="excluded_pilot_and_model_preflight",
        authority_sha256s=(
            signed_e5_confirmation.sha256,
            signed_model_compatibility.sha256,
            *(row.sha256 for row in compatibility_sources),
        ),
    )


def verify_e6_final_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e5_confirmation: SignedE5ConfirmationReceipt,
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
    signed_power_prefix: SignedE6PowerPrefixReceipt,
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    pilot_evidence_manifest: FormalDownstreamEvidenceManifest,
    pilot_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    from lightcone_spec.experiments.stage_materialization import materialize_e6

    rebuilt = materialize_e6(
        registry_verification_receipt=registry_verification_receipt,
        signed_e5_confirmation=signed_e5_confirmation,
        signed_model_compatibility=signed_model_compatibility,
        compatibility_sources=compatibility_sources,
        signed_power_prefix=signed_power_prefix,
        pilot_materialization=pilot_materialization,
        pilot_coverage=pilot_coverage,
        pilot_evidence_manifest=pilot_evidence_manifest,
        pilot_execution_bindings=pilot_execution_bindings,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="final",
        authority_sha256s=(
            signed_e5_confirmation.sha256,
            signed_model_compatibility.sha256,
            signed_power_prefix.sha256,
            *(row.sha256 for row in compatibility_sources),
        ),
    )


def verify_e0_tuning_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e6_confirmation: SignedE6ConfirmationReceipt,
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle,
    signed_compatibility_receipt: SignedE0CompatibilityReceipt,
    onlinespec_source_authority: E0OnlineSpecSourceAuthority,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    """Rebuild the complete source-owned E0 tuning grid."""

    from lightcone_spec.experiments.stage_materialization import (
        SignedE0CompatibilityReceipt,
        materialize_e0_onlinespec_tuning,
    )

    if type(signed_compatibility_receipt) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 tuning execution requires signed compatibility")
    rebuilt = materialize_e0_onlinespec_tuning(
        registry_verification_receipt=registry_verification_receipt,
        signed_e6_confirmation=signed_e6_confirmation,
        e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
        signed_compatibility_receipt=signed_compatibility_receipt,
        onlinespec_source_authority=onlinespec_source_authority,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="onlinespec_tuning",
        authority_sha256s=(
            signed_e6_confirmation.sha256,
            signed_compatibility_receipt.sha256,
            onlinespec_source_authority.sha256,
        ),
    )


def verify_e0_pilot_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    signed_e6_confirmation: SignedE6ConfirmationReceipt,
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle,
    signed_compatibility_receipt: SignedE0CompatibilityReceipt,
    signed_onlinespec_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...],
    onlinespec_source_authority: E0OnlineSpecSourceAuthority,
    tuning_proof_set: E0OnlineSpecTuningProofSet,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    """Rebuild E0's exact four excluded pilot blocks."""

    from lightcone_spec.experiments.stage_materialization import (
        SignedE0CompatibilityReceipt,
        materialize_e0_excluded_pilots,
    )

    if type(signed_compatibility_receipt) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 pilot execution requires signed compatibility")
    rebuilt = materialize_e0_excluded_pilots(
        registry_verification_receipt=registry_verification_receipt,
        signed_e6_confirmation=signed_e6_confirmation,
        e6_confirmation_proof_bundle=e6_confirmation_proof_bundle,
        signed_compatibility_receipt=signed_compatibility_receipt,
        signed_onlinespec_tuning_seals=signed_onlinespec_tuning_seals,
        onlinespec_source_authority=onlinespec_source_authority,
        tuning_proof_set=tuning_proof_set,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="excluded_pilot",
        authority_sha256s=(
            signed_e6_confirmation.sha256,
            signed_compatibility_receipt.sha256,
            onlinespec_source_authority.sha256,
            *(row.sha256 for row in signed_onlinespec_tuning_seals),
        ),
    )


def verify_e0_final_execution_source(
    *,
    materialization: StageMaterializationReceipt,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    authority_bundle: E0FormalRegistryAuthorityBundle,
    now_ns: int,
) -> VerifiedFormalStageMaterializationSource:
    """Rebuild the exact powered E0 prefix from its complete typed bundle."""

    rebuilt = authority_bundle.verify_against(
        registry_verification_receipt=registry_verification_receipt,
        materialization=materialization,
        now_ns=now_ns,
    )
    return _seal_rebuilt_stage_source(
        expected=materialization,
        rebuilt=rebuilt,
        phase="final",
        authority_sha256s=(
            authority_bundle.signed_e6_confirmation.sha256,
            authority_bundle.signed_compatibility.sha256,
            authority_bundle.source_authority.sha256,
            authority_bundle.signed_power_prefix.sha256,
            *(row.sha256 for row in authority_bundle.signed_tuning_seals),
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


def _strict_rebuild_object(
    label: str,
    value: object,
    fields: set[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


@dataclass(frozen=True)
class FormalOptimizerRecipe:
    """Complete optimizer identity used by one E1 anchor.

    Storing all schema fields prevents an omitted field from silently taking a
    new ``OptimizerConfig`` default after the ProtocolLock is signed.
    """

    anchor_name: Literal["adamw", "sgdm"]
    optimizer: OptimizerConfig
    stride: int

    def __post_init__(self) -> None:
        if self.anchor_name not in E1_OPTIMIZER_ANCHORS:
            raise ValueError("E1 optimizer recipe names an unregistered anchor")
        if type(self.optimizer) is not OptimizerConfig:
            raise TypeError("E1 optimizer recipe requires exact OptimizerConfig")
        if self.optimizer.name != self.anchor_name:
            raise ValueError("E1 optimizer config differs from its anchor name")
        if type(self.stride) is not int or self.stride < 1:
            raise ValueError("E1 optimizer anchor stride must be positive")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "anchor_name": self.anchor_name,
                "optimizer": self.optimizer.model_dump(mode="json"),
                "stride": self.stride,
            }
        )


@dataclass(frozen=True)
class E1RecipeAnchorAuthority:
    """ProtocolLock-bound complete numeric authority for the two E1 anchors."""

    schema_version: int
    authority_id: str
    trainable_plan_sha256: str
    anchors: tuple[FormalOptimizerRecipe, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E1 recipe-anchor authority schema 1 is supported")
        if (
            type(self.authority_id) is not str
            or not self.authority_id
            or self.authority_id.strip() != self.authority_id
        ):
            raise ValueError("E1 recipe-anchor authority ID must be canonical text")
        _require_sha256("E1 recipe-anchor trainable plan", self.trainable_plan_sha256)
        if (
            type(self.anchors) is not tuple
            or len(self.anchors) != 2
            or any(type(row) is not FormalOptimizerRecipe for row in self.anchors)
            or tuple(row.anchor_name for row in self.anchors)
            != tuple(sorted(E1_OPTIMIZER_ANCHORS))
        ):
            raise ValueError("E1 recipe authority requires exact adamw/sgdm anchors")
        for row in self.anchors:
            row.__post_init__()

    def anchor(self, name: str) -> FormalOptimizerRecipe:
        for row in self.anchors:
            if row.anchor_name == name:
                return row
        raise ValueError("E1 optimizer anchor is absent")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "authority_id": self.authority_id,
                "trainable_plan_sha256": self.trainable_plan_sha256,
                "anchors": tuple(
                    {
                        "anchor_name": row.anchor_name,
                        "optimizer": row.optimizer.model_dump(mode="json"),
                        "stride": row.stride,
                    }
                    for row in self.anchors
                ),
            }
        )


E1_RECIPE_ANCHOR_AUTHORITY_ID = "lightcone-e1-recipe-anchor-authority-v1"
E1_RECIPE_ANCHOR_ARTIFACT_KIND = "lightcone_e1_recipe_anchor_authority_artifact"


def _source_e1_recipe_anchor_authority(
    source: CanonicalJsonProofBinding,
) -> E1RecipeAnchorAuthority:
    from lightcone_spec.adaptation.plan_authority import (
        TrainablePlanAuthorityBinding,
        trainable_plan_authority_binding_from_dict,
    )

    if type(source) is not CanonicalJsonProofBinding:
        raise TypeError("E1 recipe anchors require an exact path-bound plan source")
    rebound = CanonicalJsonProofBinding.bind(source.absolute_path)
    if rebound != source:
        raise ValueError("E1 recipe-anchor trainable-plan source changed")
    binding = trainable_plan_authority_binding_from_dict(source.reopen())
    if (
        type(binding) is not TrainablePlanAuthorityBinding
        or binding.sha256 != source.semantic_sha256
    ):
        raise ValueError("E1 recipe-anchor source is not one exact plan authority")
    result = binding.revalidate()
    if result.binding != binding or result.plan.sha256 != binding.trainable_plan_sha256:
        raise ValueError("E1 recipe-anchor trainable plan did not replay exactly")
    anchors = tuple(
        sorted(
            (
                FormalOptimizerRecipe(
                    anchor_name="adamw",
                    optimizer=OptimizerConfig(
                        name="adamw",
                        learning_rate=1e-4,
                        weight_decay=0.01,
                    ),
                    stride=10,
                ),
                FormalOptimizerRecipe(
                    anchor_name="sgdm",
                    optimizer=OptimizerConfig(
                        name="sgdm",
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        momentum=0.9,
                    ),
                    stride=10,
                ),
            ),
            key=lambda row: row.anchor_name,
        )
    )
    return E1RecipeAnchorAuthority(
        schema_version=1,
        authority_id=E1_RECIPE_ANCHOR_AUTHORITY_ID,
        trainable_plan_sha256=result.plan.sha256,
        anchors=anchors,
    )


def e1_recipe_anchor_authority_to_dict(
    value: E1RecipeAnchorAuthority,
) -> dict[str, object]:
    if type(value) is not E1RecipeAnchorAuthority:
        raise TypeError("E1 recipe-anchor codec requires an exact authority")
    value.__post_init__()
    return {
        "schema_version": value.schema_version,
        "authority_id": value.authority_id,
        "trainable_plan_sha256": value.trainable_plan_sha256,
        "anchors": [
            {
                "anchor_name": row.anchor_name,
                "optimizer": row.optimizer.model_dump(mode="json"),
                "stride": row.stride,
            }
            for row in value.anchors
        ],
        "authority_sha256": value.sha256,
    }


def e1_recipe_anchor_authority_from_dict(
    value: object,
) -> E1RecipeAnchorAuthority:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "authority_id",
        "trainable_plan_sha256",
        "anchors",
        "authority_sha256",
    }:
        raise ValueError("E1 recipe-anchor authority fields differ")
    declared = value["authority_sha256"]
    _require_sha256("E1 recipe-anchor declared authority", declared)
    raw_anchors = value["anchors"]
    if type(raw_anchors) is not list:
        raise TypeError("E1 recipe-anchor rows must be an array")
    anchors = []
    for raw in raw_anchors:
        if type(raw) is not dict or set(raw) != {
            "anchor_name",
            "optimizer",
            "stride",
        }:
            raise ValueError("E1 recipe-anchor row fields differ")
        anchors.append(
            FormalOptimizerRecipe(
                anchor_name=raw["anchor_name"],  # type: ignore[arg-type]
                optimizer=OptimizerConfig.model_validate(raw["optimizer"]),
                stride=raw["stride"],  # type: ignore[arg-type]
            )
        )
    authority = E1RecipeAnchorAuthority(
        schema_version=value["schema_version"],  # type: ignore[arg-type]
        authority_id=value["authority_id"],  # type: ignore[arg-type]
        trainable_plan_sha256=value["trainable_plan_sha256"],  # type: ignore[arg-type]
        anchors=tuple(anchors),
    )
    if authority.sha256 != declared:
        raise ValueError("E1 recipe-anchor authority digest differs from content")
    return authority


@dataclass(frozen=True)
class E1RecipeAnchorAuthorityArtifact:
    """Durable source proof for the deterministic two-anchor authority."""

    schema_version: Literal[1]
    kind: Literal["lightcone_e1_recipe_anchor_authority_artifact"]
    trainable_plan_authority_source: CanonicalJsonProofBinding
    authority: E1RecipeAnchorAuthority

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != E1_RECIPE_ANCHOR_ARTIFACT_KIND:
            raise ValueError("E1 recipe-anchor artifact schema is unsupported")
        if type(self.trainable_plan_authority_source) is not CanonicalJsonProofBinding:
            raise TypeError("E1 recipe-anchor artifact requires a bound plan source")
        if type(self.authority) is not E1RecipeAnchorAuthority:
            raise TypeError("E1 recipe-anchor artifact requires an exact authority")
        expected = _source_e1_recipe_anchor_authority(
            self.trainable_plan_authority_source
        )
        if self.authority != expected:
            raise ValueError("E1 recipe anchors differ from source-owned reduction")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "trainable_plan_authority_source": (
                self.trainable_plan_authority_source.to_dict()
            ),
            "authority": e1_recipe_anchor_authority_to_dict(self.authority),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> E1RecipeAnchorAuthorityArtifact:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "kind",
            "trainable_plan_authority_source",
            "authority",
            "artifact_sha256",
        }:
            raise ValueError("E1 recipe-anchor artifact fields differ")
        declared = value["artifact_sha256"]
        _require_sha256("E1 recipe-anchor artifact", declared)
        artifact = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            kind=value["kind"],  # type: ignore[arg-type]
            trainable_plan_authority_source=CanonicalJsonProofBinding.from_dict(
                value["trainable_plan_authority_source"]
            ),
            authority=e1_recipe_anchor_authority_from_dict(value["authority"]),
        )
        if artifact.sha256 != declared:
            raise ValueError("E1 recipe-anchor artifact digest differs from content")
        return artifact


def build_source_e1_recipe_anchor_authority_artifact(
    trainable_plan_authority_path: str | Path,
) -> E1RecipeAnchorAuthorityArtifact:
    source = CanonicalJsonProofBinding.bind(trainable_plan_authority_path)
    return E1RecipeAnchorAuthorityArtifact(
        schema_version=1,
        kind=E1_RECIPE_ANCHOR_ARTIFACT_KIND,
        trainable_plan_authority_source=source,
        authority=_source_e1_recipe_anchor_authority(source),
    )


def publish_e1_recipe_anchor_authority_artifact(
    artifact: E1RecipeAnchorAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not E1RecipeAnchorAuthorityArtifact:
        raise TypeError("E1 recipe-anchor publisher requires an exact artifact")
    artifact.__post_init__()
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    reopened = E1RecipeAnchorAuthorityArtifact.from_dict(binding.reopen())
    if reopened != artifact:
        raise RuntimeError("E1 recipe-anchor artifact changed during publication")
    return binding


def load_e1_recipe_anchor_authority_artifact(
    artifact_path: str | Path,
) -> E1RecipeAnchorAuthorityArtifact:
    before = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = E1RecipeAnchorAuthorityArtifact.from_dict(before.reopen())
    if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
        raise RuntimeError("E1 recipe-anchor artifact changed while loaded")
    return artifact


@dataclass(frozen=True)
class FormalServingExecutionSubject:
    """Deterministic, non-authorizing cell-to-runtime mapping."""

    schema_version: int
    protocol_lock_sha256: str
    formal_runtime_authority_manifest_sha256: str
    execution_mapper_authority_sha256: str
    materialization_receipt_sha256: str
    materialized_cell_id: str
    stage: FormalServingStage
    method: FormalServingMethod
    stage_source_binding_sha256: str | None
    run_config_sha256: str
    recipe_authority_sha256s: tuple[str, ...]
    workload_authority_sha256: str
    content_verification_receipt_sha256: str | None
    prepared_model_member_sha256s: tuple[str, ...]
    workload_member_sha256s: tuple[str, ...]
    inventory_sha256: str
    topology_mode: CanonicalTopologyMode
    gpu_uuids: tuple[str, ...]
    runtime_gpu_proof_artifacts: tuple[CanonicalJsonProofBinding, ...]
    execution_plan_sha256: str
    rank_config_sha256: str
    execution_identity: StageItlExecutionIdentity

    def __post_init__(self) -> None:
        if self.schema_version != 4 or self.stage not in _FORMAL_SERVING_STAGES:
            raise ValueError(
                "formal serving execution subject schema/stage is unsupported"
            )
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            (
                "formal runtime authority manifest",
                self.formal_runtime_authority_manifest_sha256,
            ),
            ("execution mapper authority", self.execution_mapper_authority_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("materialized cell", self.materialized_cell_id),
            ("run config", self.run_config_sha256),
            ("workload authority", self.workload_authority_sha256),
            ("inventory", self.inventory_sha256),
            ("execution plan", self.execution_plan_sha256),
            ("rank config", self.rank_config_sha256),
        ):
            _require_sha256(f"formal serving {label}", digest)
        if self.stage in _DOWNSTREAM_SERVING_STAGES:
            _require_sha256(
                "formal serving stage source binding",
                self.stage_source_binding_sha256,
            )
        elif self.stage_source_binding_sha256 is not None:
            raise ValueError("early serving stage cannot carry downstream source")
        if not self.recipe_authority_sha256s or self.recipe_authority_sha256s != tuple(
            sorted(set(self.recipe_authority_sha256s))
        ):
            raise ValueError("formal serving recipe authorities are not canonical")
        for digest in self.recipe_authority_sha256s:
            _require_sha256("formal serving recipe authority", digest)
        if self.content_verification_receipt_sha256 is None:
            if self.prepared_model_member_sha256s or self.workload_member_sha256s:
                raise ValueError(
                    "unverified formal content cannot carry verified members"
                )
        else:
            _require_sha256(
                "formal serving content verification receipt",
                self.content_verification_receipt_sha256,
            )
            if (
                not self.prepared_model_member_sha256s
                or self.prepared_model_member_sha256s
                != tuple(sorted(set(self.prepared_model_member_sha256s)))
                or self.workload_member_sha256s
                != tuple(sorted(set(self.workload_member_sha256s)))
            ):
                raise ValueError("formal serving verified content is not canonical")
            for digest in (
                *self.prepared_model_member_sha256s,
                *self.workload_member_sha256s,
            ):
                _require_sha256("formal serving verified content member", digest)
        if self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("formal serving topology is not canonical")
        expected_gpu_count = 1 if self.topology_mode == "tp1_dp1" else 2
        if (
            type(self.gpu_uuids) is not tuple
            or len(self.gpu_uuids) != expected_gpu_count
            or len(set(self.gpu_uuids)) != expected_gpu_count
        ):
            raise ValueError("formal serving GPU UUID coverage differs from topology")
        if (
            type(self.runtime_gpu_proof_artifacts) is not tuple
            or not self.runtime_gpu_proof_artifacts
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.runtime_gpu_proof_artifacts
            )
        ):
            raise TypeError("formal serving requires durable runtime GPU proofs")
        proof_paths = tuple(
            row.absolute_path for row in self.runtime_gpu_proof_artifacts
        )
        if len(proof_paths) != len(set(proof_paths)):
            raise ValueError("formal serving runtime GPU proof is duplicated")
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("formal serving requires an exact execution identity")
        if (
            self.execution_identity.materialized_cell_id != self.materialized_cell_id
            or self.execution_identity.inventory_sha256 != self.inventory_sha256
            or self.execution_identity.execution_plan_sha256
            != self.execution_plan_sha256
            or self.execution_identity.rank_config_sha256 != self.rank_config_sha256
            or self.execution_identity.method != self.method
        ):
            raise ValueError("formal serving execution identity differs from subject")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "formal_runtime_authority_manifest_sha256": (
                self.formal_runtime_authority_manifest_sha256
            ),
            "execution_mapper_authority_sha256": (
                self.execution_mapper_authority_sha256
            ),
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "materialized_cell_id": self.materialized_cell_id,
            "stage": self.stage,
            "method": self.method,
            "stage_source_binding_sha256": self.stage_source_binding_sha256,
            "run_config_sha256": self.run_config_sha256,
            "recipe_authority_sha256s": list(self.recipe_authority_sha256s),
            "workload_authority_sha256": self.workload_authority_sha256,
            "content_verification_receipt_sha256": (
                self.content_verification_receipt_sha256
            ),
            "prepared_model_member_sha256s": list(self.prepared_model_member_sha256s),
            "workload_member_sha256s": list(self.workload_member_sha256s),
            "inventory_sha256": self.inventory_sha256,
            "topology_mode": self.topology_mode,
            "gpu_uuids": list(self.gpu_uuids),
            "runtime_gpu_proof_artifacts": [
                row.to_dict() for row in self.runtime_gpu_proof_artifacts
            ],
            "execution_plan_sha256": self.execution_plan_sha256,
            "rank_config_sha256": self.rank_config_sha256,
            "execution_identity": self.execution_identity.to_dict(),
            "subject_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_rebuild_object(
            "formal serving execution subject",
            value,
            {*cls.__dataclass_fields__, "subject_sha256"},
        )
        declared = _require_sha256(
            "formal serving execution subject", row.pop("subject_sha256")
        )
        tuple_fields = (
            "recipe_authority_sha256s",
            "prepared_model_member_sha256s",
            "workload_member_sha256s",
            "gpu_uuids",
        )
        for field in tuple_fields:
            raw = row[field]
            if type(raw) is not list or any(type(item) is not str for item in raw):
                raise TypeError(f"formal serving execution {field} must be an array")
            row[field] = tuple(raw)
        raw_proofs = row["runtime_gpu_proof_artifacts"]
        if type(raw_proofs) is not list:
            raise TypeError("formal serving execution GPU proofs must be an array")
        row["runtime_gpu_proof_artifacts"] = tuple(
            CanonicalJsonProofBinding.from_dict(item) for item in raw_proofs
        )
        row["execution_identity"] = StageItlExecutionIdentity.from_dict(
            row["execution_identity"]
        )
        subject = cls(**row)  # type: ignore[arg-type]
        if subject.sha256 != declared:
            raise ValueError("formal serving execution subject digest differs")
        return subject


_VERIFIED_FORMAL_SERVING_EXECUTION_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedFormalServingExecutionBinding:
    """Private-sealed authority consumed by stage evidence reducers."""

    subject: FormalServingExecutionSubject
    run_config: RunConfig
    runtime_gpu_proof_sha256s: tuple[str, ...]
    verified_native_gpu_proofs: tuple[VerifiedNativeRuntimeGpuProof, ...]
    verified_distributed_gpu_proofs: tuple[VerifiedDistributedRuntimeGpuProof, ...]
    verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None
    hardware_envelope_sha256: str
    _construction_seal: object

    def __init__(
        self,
        *,
        subject: FormalServingExecutionSubject,
        run_config: RunConfig,
        runtime_gpu_proof_sha256s: tuple[str, ...],
        verified_native_gpu_proofs: tuple[VerifiedNativeRuntimeGpuProof, ...],
        verified_distributed_gpu_proofs: tuple[VerifiedDistributedRuntimeGpuProof, ...],
        verified_nextn_tp2_authority: VerifiedNextNTp2Authority | None,
        hardware_envelope_sha256: str,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _VERIFIED_FORMAL_SERVING_EXECUTION_SEAL:
            raise TypeError("formal serving binding is verifier-constructed only")
        _require_sha256("formal serving hardware envelope", hardware_envelope_sha256)
        if (
            type(runtime_gpu_proof_sha256s) is not tuple
            or not runtime_gpu_proof_sha256s
            or runtime_gpu_proof_sha256s
            != tuple(sorted(set(runtime_gpu_proof_sha256s)))
        ):
            raise ValueError("formal serving verified GPU proof set is not canonical")
        if (
            type(verified_native_gpu_proofs) is not tuple
            or any(
                type(row) is not VerifiedNativeRuntimeGpuProof
                for row in verified_native_gpu_proofs
            )
            or verified_native_gpu_proofs
            != tuple(
                sorted(
                    verified_native_gpu_proofs,
                    key=lambda row: (row.suite_id, row.sha256),
                )
            )
            or type(verified_distributed_gpu_proofs) is not tuple
            or any(
                type(row) is not VerifiedDistributedRuntimeGpuProof
                for row in verified_distributed_gpu_proofs
            )
            or verified_distributed_gpu_proofs
            != tuple(
                sorted(
                    verified_distributed_gpu_proofs,
                    key=lambda row: (row.topology_mode, row.sha256),
                )
            )
        ):
            raise TypeError("formal serving verified GPU proof tokens are not exact")
        token_sha256s = tuple(
            sorted(
                row.sha256
                for row in (
                    *verified_native_gpu_proofs,
                    *verified_distributed_gpu_proofs,
                )
            )
        )
        if token_sha256s != runtime_gpu_proof_sha256s:
            raise ValueError("formal serving proof tokens differ from proof identities")
        if (
            verified_nextn_tp2_authority is not None
            and type(verified_nextn_tp2_authority) is not VerifiedNextNTp2Authority
        ):
            raise TypeError("formal serving NEXTN TP2 authority is not verifier-owned")
        requires_nextn_tp2 = (
            run_config.model.algorithm == "NEXTN" and subject.topology_mode == "tp2_dp1"
        )
        if requires_nextn_tp2 != (verified_nextn_tp2_authority is not None):
            raise ValueError("formal serving NEXTN TP2 authority coverage is not exact")
        if verified_nextn_tp2_authority is not None:
            nextn_native = tuple(
                row for row in verified_native_gpu_proofs if row.suite_id == "nextn_tp2"
            )
            if (
                len(nextn_native) != 1
                or len(verified_distributed_gpu_proofs) != 1
                or verified_nextn_tp2_authority.native_gpu_proof_sha256
                != nextn_native[0].sha256
                or verified_nextn_tp2_authority.distributed_gpu_proof_sha256
                != verified_distributed_gpu_proofs[0].sha256
                or verified_nextn_tp2_authority.inventory_sha256
                != subject.inventory_sha256
                or verified_nextn_tp2_authority.gpu_uuids != subject.gpu_uuids
            ):
                raise ValueError(
                    "formal serving NEXTN TP2 authority differs from GPU proof union"
                )
        for name, value in (
            ("subject", subject),
            ("run_config", run_config),
            ("runtime_gpu_proof_sha256s", runtime_gpu_proof_sha256s),
            ("verified_native_gpu_proofs", verified_native_gpu_proofs),
            (
                "verified_distributed_gpu_proofs",
                verified_distributed_gpu_proofs,
            ),
            ("verified_nextn_tp2_authority", verified_nextn_tp2_authority),
            ("hardware_envelope_sha256", hardware_envelope_sha256),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "protocol_sha256": FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
                "subject_sha256": self.subject.sha256,
                "runtime_gpu_proof_sha256s": self.runtime_gpu_proof_sha256s,
                "hardware_envelope_sha256": self.hardware_envelope_sha256,
            }
        )


_FORMAL_SINGLE_OPERATOR_EXECUTION_SEAL = object()


@dataclass(frozen=True, init=False)
class FormalSingleOperatorExecutionBinding:
    """Verifier-sealed current-only execution authority.

    The scientific mapping and GPU proof validation remain owned by
    :class:`VerifiedFormalServingExecutionBinding`.  This wrapper adds the
    exact current single-operator stage source and source-owned compile launch
    identities, so an operator cannot substitute a caller-authored RunConfig,
    recipe, GPU order, port, argv, or run identity.
    """

    verified_binding: VerifiedFormalServingExecutionBinding
    execution_source: CanonicalJsonProofBinding
    execution_source_sha256: str
    compile_launch_manifest: CanonicalJsonProofBinding
    inventory_source: CanonicalJsonProofBinding
    content_verification_receipt_source: CanonicalJsonProofBinding
    runtime_authority_manifest_source: CanonicalJsonProofBinding
    tts_calibration_authority_source: CanonicalJsonProofBinding | None
    e1_recipe_anchor_authority_source: CanonicalJsonProofBinding | None
    formal_registry_verification_receipt_source: CanonicalJsonProofBinding | None
    repository_root: str | None
    _construction_seal: object

    def __init__(
        self,
        *,
        verified_binding: VerifiedFormalServingExecutionBinding,
        execution_source: CanonicalJsonProofBinding,
        execution_source_sha256: str,
        compile_launch_manifest: CanonicalJsonProofBinding,
        inventory_source: CanonicalJsonProofBinding,
        content_verification_receipt_source: CanonicalJsonProofBinding,
        runtime_authority_manifest_source: CanonicalJsonProofBinding,
        tts_calibration_authority_source: CanonicalJsonProofBinding | None,
        e1_recipe_anchor_authority_source: CanonicalJsonProofBinding | None,
        formal_registry_verification_receipt_source: CanonicalJsonProofBinding | None,
        repository_root: str | None,
        _construction_seal: object,
    ) -> None:
        if _construction_seal is not _FORMAL_SINGLE_OPERATOR_EXECUTION_SEAL:
            raise TypeError(
                "single-operator execution binding is verifier-constructed only"
            )
        if (
            type(verified_binding) is not VerifiedFormalServingExecutionBinding
            or verified_binding._construction_seal
            is not _VERIFIED_FORMAL_SERVING_EXECUTION_SEAL
        ):
            raise TypeError(
                "single-operator execution binding requires the private verifier"
            )
        _require_sha256(
            "single-operator execution source identity", execution_source_sha256
        )
        for label, value in (
            ("execution source", execution_source),
            ("compile launch", compile_launch_manifest),
            ("inventory", inventory_source),
            ("content receipt", content_verification_receipt_source),
            ("runtime authority", runtime_authority_manifest_source),
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError(f"single-operator execution {label} is not path-bound")
        for label, value in (
            ("TTS calibration authority", tts_calibration_authority_source),
            ("E1 recipe anchor", e1_recipe_anchor_authority_source),
            (
                "formal registry verification receipt",
                formal_registry_verification_receipt_source,
            ),
        ):
            if value is not None and type(value) is not CanonicalJsonProofBinding:
                raise TypeError(
                    f"single-operator execution optional {label} is not path-bound"
                )
        normalized_repository_root: str | None = None
        if repository_root is not None:
            requested_root = Path(repository_root)
            resolved_root = requested_root.resolve(strict=False)
            if (
                not requested_root.is_absolute()
                or requested_root != resolved_root
                or not resolved_root.is_dir()
                or resolved_root.is_symlink()
            ):
                raise ValueError(
                    "single-operator execution repository root is not a real "
                    "absolute directory"
                )
            normalized_repository_root = str(resolved_root)
        for name, value in (
            ("verified_binding", verified_binding),
            ("execution_source", execution_source),
            ("execution_source_sha256", execution_source_sha256),
            ("compile_launch_manifest", compile_launch_manifest),
            ("inventory_source", inventory_source),
            (
                "content_verification_receipt_source",
                content_verification_receipt_source,
            ),
            ("runtime_authority_manifest_source", runtime_authority_manifest_source),
            (
                "tts_calibration_authority_source",
                tts_calibration_authority_source,
            ),
            (
                "e1_recipe_anchor_authority_source",
                e1_recipe_anchor_authority_source,
            ),
            (
                "formal_registry_verification_receipt_source",
                formal_registry_verification_receipt_source,
            ),
            ("repository_root", normalized_repository_root),
            ("_construction_seal", _construction_seal),
        ):
            object.__setattr__(self, name, value)

    @property
    def subject(self) -> FormalServingExecutionSubject:
        return self.verified_binding.subject

    @property
    def run_config(self) -> RunConfig:
        return self.verified_binding.run_config

    @property
    def runtime_gpu_proof_sha256s(self) -> tuple[str, ...]:
        return self.verified_binding.runtime_gpu_proof_sha256s

    @property
    def verified_native_gpu_proofs(
        self,
    ) -> tuple[VerifiedNativeRuntimeGpuProof, ...]:
        return self.verified_binding.verified_native_gpu_proofs

    @property
    def verified_distributed_gpu_proofs(
        self,
    ) -> tuple[VerifiedDistributedRuntimeGpuProof, ...]:
        return self.verified_binding.verified_distributed_gpu_proofs

    @property
    def verified_nextn_tp2_authority(self) -> VerifiedNextNTp2Authority | None:
        return self.verified_binding.verified_nextn_tp2_authority

    @property
    def hardware_envelope_sha256(self) -> str:
        return self.verified_binding.hardware_envelope_sha256

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "protocol_sha256": (
                    FORMAL_SINGLE_OPERATOR_EXECUTION_BINDING_PROTOCOL_SHA256
                ),
                "verified_execution_binding_sha256": self.verified_binding.sha256,
                "execution_source": self.execution_source.to_dict(),
                "execution_source_sha256": self.execution_source_sha256,
                "compile_launch_manifest": self.compile_launch_manifest.to_dict(),
                "inventory_source": self.inventory_source.to_dict(),
                "content_verification_receipt_source": (
                    self.content_verification_receipt_source.to_dict()
                ),
                "runtime_authority_manifest_source": (
                    self.runtime_authority_manifest_source.to_dict()
                ),
                "tts_calibration_authority_source": (
                    None
                    if self.tts_calibration_authority_source is None
                    else self.tts_calibration_authority_source.to_dict()
                ),
                "e1_recipe_anchor_authority_source": (
                    None
                    if self.e1_recipe_anchor_authority_source is None
                    else self.e1_recipe_anchor_authority_source.to_dict()
                ),
                "formal_registry_verification_receipt_source": (
                    None
                    if self.formal_registry_verification_receipt_source is None
                    else self.formal_registry_verification_receipt_source.to_dict()
                ),
                "repository_root": self.repository_root,
            }
        )


FormalServingExecutionBinding = (
    VerifiedFormalServingExecutionBinding | FormalSingleOperatorExecutionBinding
)


def require_verified_formal_serving_execution_binding(
    value: object,
) -> FormalServingExecutionBinding:
    if type(value) is FormalSingleOperatorExecutionBinding:
        if (
            value._construction_seal is not _FORMAL_SINGLE_OPERATOR_EXECUTION_SEAL
            or type(value.verified_binding) is not VerifiedFormalServingExecutionBinding
            or value.verified_binding._construction_seal
            is not _VERIFIED_FORMAL_SERVING_EXECUTION_SEAL
            or type(value.subject) is not FormalServingExecutionSubject
            or type(value.run_config) is not RunConfig
        ):
            raise TypeError(
                "formal serving evidence requires a sealed execution binding"
            )
        return value
    if (
        type(value) is not VerifiedFormalServingExecutionBinding
        or value._construction_seal is not _VERIFIED_FORMAL_SERVING_EXECUTION_SEAL
        or type(value.subject) is not FormalServingExecutionSubject
        or type(value.run_config) is not RunConfig
        or type(value.verified_native_gpu_proofs) is not tuple
        or any(
            type(row) is not VerifiedNativeRuntimeGpuProof
            for row in value.verified_native_gpu_proofs
        )
        or type(value.verified_distributed_gpu_proofs) is not tuple
        or any(
            type(row) is not VerifiedDistributedRuntimeGpuProof
            for row in value.verified_distributed_gpu_proofs
        )
        or (
            value.verified_nextn_tp2_authority is not None
            and type(value.verified_nextn_tp2_authority)
            is not VerifiedNextNTp2Authority
        )
    ):
        raise TypeError("formal serving evidence requires a sealed execution binding")
    return value


@dataclass(frozen=True)
class FormalServingExecutionRebuildInput:
    """Durable public descriptor for rebuilding one private execution token.

    This value contains no private construction seal.  Its paths and exact
    authority identities are reopened by :func:`rebuild_formal_serving_execution_binding`;
    a consumer cannot turn the descriptor itself into execution authority.
    """

    schema_version: Literal[1]
    kind: Literal["formal_serving_execution_rebuild_input"]
    protocol_sha256: str
    subject: FormalServingExecutionSubject
    run_config_source: CanonicalJsonProofBinding
    content_verification_receipt_source: CanonicalJsonProofBinding
    runtime_gpu_proof_artifacts: tuple[CanonicalJsonProofBinding, ...]
    nextn_tp2_authority_input_source: CanonicalJsonProofBinding | None
    stage_source_binding_sha256: str | None
    recipe_authority_sha256s: tuple[str, ...]
    registry_verification_receipt_sha256: str | None
    execution_binding_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_serving_execution_rebuild_input"
            or self.protocol_sha256 != FORMAL_SERVING_EXECUTION_REBUILD_PROTOCOL_SHA256
        ):
            raise ValueError("formal serving rebuild descriptor is unsupported")
        if type(self.subject) is not FormalServingExecutionSubject:
            raise TypeError("formal serving rebuild subject is not exact")
        for label, binding in (
            ("run config", self.run_config_source),
            ("content receipt", self.content_verification_receipt_source),
        ):
            if type(binding) is not CanonicalJsonProofBinding:
                raise TypeError(f"formal serving rebuild {label} is not path-bound")
        if (
            type(self.runtime_gpu_proof_artifacts) is not tuple
            or not self.runtime_gpu_proof_artifacts
            or any(
                type(row) is not CanonicalJsonProofBinding
                for row in self.runtime_gpu_proof_artifacts
            )
            or self.runtime_gpu_proof_artifacts
            != self.subject.runtime_gpu_proof_artifacts
        ):
            raise ValueError("formal serving rebuild GPU proof set differs")
        requires_nextn_tp2_source = (
            self.subject.stage == "E6" and self.subject.topology_mode == "tp2_dp1"
        )
        if requires_nextn_tp2_source:
            if type(self.nextn_tp2_authority_input_source) is not (
                CanonicalJsonProofBinding
            ):
                raise TypeError(
                    "formal serving E6 rebuild lacks path-bound NEXTN TP2 authority"
                )
        elif self.nextn_tp2_authority_input_source is not None:
            raise ValueError("non-E6 rebuild carries NEXTN TP2 authority input")
        if (
            self.run_config_source.semantic_sha256 != self.subject.run_config_sha256
            or self.content_verification_receipt_source.semantic_sha256
            != self.subject.content_verification_receipt_sha256
            or self.stage_source_binding_sha256
            != self.subject.stage_source_binding_sha256
            or self.recipe_authority_sha256s != self.subject.recipe_authority_sha256s
        ):
            raise ValueError("formal serving rebuild immutable subject inputs differ")
        if self.subject.stage in {"E3a", "TTS-Cal"}:
            if self.registry_verification_receipt_sha256 is not None:
                _require_sha256(
                    "formal serving rebuild registry receipt",
                    self.registry_verification_receipt_sha256,
                )
        else:
            _require_sha256(
                "formal serving rebuild registry receipt",
                self.registry_verification_receipt_sha256,
            )
        _require_sha256(
            "formal serving rebuild execution binding",
            self.execution_binding_sha256,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "subject": self.subject.to_dict(),
            "run_config_source": self.run_config_source.to_dict(),
            "content_verification_receipt_source": (
                self.content_verification_receipt_source.to_dict()
            ),
            "runtime_gpu_proof_artifacts": [
                row.to_dict() for row in self.runtime_gpu_proof_artifacts
            ],
            "nextn_tp2_authority_input_source": (
                None
                if self.nextn_tp2_authority_input_source is None
                else self.nextn_tp2_authority_input_source.to_dict()
            ),
            "stage_source_binding_sha256": self.stage_source_binding_sha256,
            "recipe_authority_sha256s": list(self.recipe_authority_sha256s),
            "registry_verification_receipt_sha256": (
                self.registry_verification_receipt_sha256
            ),
            "execution_binding_sha256": self.execution_binding_sha256,
        }
        if include_sha256:
            value["rebuild_input_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict_rebuild_object(
            "formal serving rebuild descriptor",
            value,
            {*cls.__dataclass_fields__, "rebuild_input_sha256"},
        )
        declared = _require_sha256(
            "formal serving rebuild descriptor",
            row.pop("rebuild_input_sha256"),
        )
        raw_proofs = row.pop("runtime_gpu_proof_artifacts")
        raw_recipes = row.pop("recipe_authority_sha256s")
        if type(raw_proofs) is not list or type(raw_recipes) is not list:
            raise TypeError("formal serving rebuild collections must be arrays")
        subject = FormalServingExecutionSubject.from_dict(row.pop("subject"))
        run_config_source = CanonicalJsonProofBinding.from_dict(
            row.pop("run_config_source")
        )
        content_source = CanonicalJsonProofBinding.from_dict(
            row.pop("content_verification_receipt_source")
        )
        raw_nextn_source = row.pop("nextn_tp2_authority_input_source")
        nextn_source = (
            None
            if raw_nextn_source is None
            else CanonicalJsonProofBinding.from_dict(raw_nextn_source)
        )
        descriptor = cls(
            **row,  # type: ignore[arg-type]
            subject=subject,
            run_config_source=run_config_source,
            content_verification_receipt_source=content_source,
            runtime_gpu_proof_artifacts=tuple(
                CanonicalJsonProofBinding.from_dict(item) for item in raw_proofs
            ),
            nextn_tp2_authority_input_source=nextn_source,
            recipe_authority_sha256s=tuple(raw_recipes),
        )
        if descriptor.sha256 != declared:
            raise ValueError("formal serving rebuild descriptor digest differs")
        return descriptor


def _reopen_nextn_tp2_authority_input(
    binding: CanonicalJsonProofBinding,
) -> object:
    """Deep-open one E6 NEXTN source without importing a private seal."""

    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
    )

    if type(binding) is not CanonicalJsonProofBinding:
        raise TypeError("NEXTN TP2 rebuild source must be path-bound")
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError("NEXTN TP2 rebuild source path identity changed")
    raw = before.reopen()
    expected_fields = {
        *E6NextnModelAuthorityInput.__dataclass_fields__,
        "source_input_sha256",
    }
    if type(raw) is not dict or set(raw) != expected_fields:
        raise ValueError("NEXTN TP2 authority input fields differ from schema")
    row = dict(raw)
    declared = _require_sha256(
        "NEXTN TP2 authority input", row.pop("source_input_sha256")
    )
    source = E6NextnModelAuthorityInput(**row)  # type: ignore[arg-type]
    if (
        source.sha256 != declared
        or CanonicalJsonProofBinding.bind(binding.absolute_path) != before
    ):
        raise ValueError("NEXTN TP2 authority input changed or has another digest")
    return source


def bind_formal_serving_execution_rebuild_input(
    binding: VerifiedFormalServingExecutionBinding,
    *,
    run_config_source_path: str,
    content_verification_receipt_source_path: str,
    nextn_tp2_authority_input_source_path: str | None = None,
    registry_verification_receipt: object | None,
) -> FormalServingExecutionRebuildInput:
    """Bind durable JSON sources for a previously verifier-built token."""

    verified = require_verified_formal_serving_execution_binding(binding)
    run_config_source = CanonicalJsonProofBinding.bind(run_config_source_path)
    run_config = load_run_config(run_config_source.absolute_path)
    if run_config != verified.run_config or (
        run_config_source.semantic_sha256 != run_config_sha256(run_config)
    ):
        raise ValueError("formal serving rebuild RunConfig source differs")
    content_source = CanonicalJsonProofBinding.bind(
        content_verification_receipt_source_path
    )
    content_receipt = ContentVerificationReceipt.from_dict(content_source.reopen())
    if content_source.semantic_sha256 != content_receipt.sha256 or (
        content_receipt.sha256 != verified.subject.content_verification_receipt_sha256
    ):
        raise ValueError("formal serving rebuild content source differs")
    nextn_source = (
        None
        if nextn_tp2_authority_input_source_path is None
        else CanonicalJsonProofBinding.bind(nextn_tp2_authority_input_source_path)
    )
    if verified.verified_nextn_tp2_authority is None:
        if nextn_source is not None:
            raise ValueError("non-NEXTN rebuild carries a NEXTN authority source")
    else:
        if nextn_source is None:
            raise ValueError("NEXTN TP2 rebuild lacks its authority source")
        source = _reopen_nextn_tp2_authority_input(nextn_source)
        if (
            source.artifact_semantic_sha256
            != verified.verified_nextn_tp2_authority.artifact_sha256
            or source.model != verified.verified_nextn_tp2_authority.target_model_id
        ):
            raise ValueError("NEXTN TP2 rebuild source differs from verified authority")
    registry_sha256: str | None = None
    if registry_verification_receipt is not None:
        from lightcone_spec.experiments.formal_registry import (
            FormalRegistryVerificationReceipt,
        )

        if type(registry_verification_receipt) is not (
            FormalRegistryVerificationReceipt
        ):
            raise TypeError("formal serving rebuild registry receipt is not exact")
        registry_sha256 = registry_verification_receipt.sha256
    return FormalServingExecutionRebuildInput(
        schema_version=1,
        kind="formal_serving_execution_rebuild_input",
        protocol_sha256=FORMAL_SERVING_EXECUTION_REBUILD_PROTOCOL_SHA256,
        subject=verified.subject,
        run_config_source=run_config_source,
        content_verification_receipt_source=content_source,
        runtime_gpu_proof_artifacts=verified.subject.runtime_gpu_proof_artifacts,
        nextn_tp2_authority_input_source=nextn_source,
        stage_source_binding_sha256=(verified.subject.stage_source_binding_sha256),
        recipe_authority_sha256s=verified.subject.recipe_authority_sha256s,
        registry_verification_receipt_sha256=registry_sha256,
        execution_binding_sha256=verified.sha256,
    )


def rebuild_formal_serving_execution_binding(
    descriptor: FormalServingExecutionRebuildInput,
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    tts_authority: TtsCalibrationAuthority | None,
    signed_tts_seal: SignedTtsCalibrationSeal | None = None,
    e1_recipe_anchor_authority: E1RecipeAnchorAuthority | None = None,
    e2_recipe_grid_authority: E2RecipeGridAuthority | None = None,
    lightcone_recipe: E2CandidateRecipe | None = None,
    stage_source: VerifiedFormalStageMaterializationSource | None = None,
    now_ns: int,
    registry_verification_receipt: object | None = None,
) -> VerifiedFormalServingExecutionBinding:
    """Deep-reopen a descriptor and reconstruct the private execution token."""

    if type(descriptor) is not FormalServingExecutionRebuildInput:
        raise TypeError("formal serving rebuild requires exact typed descriptor")
    descriptor.__post_init__()
    subject = FormalServingExecutionSubject.from_dict(descriptor.subject.to_dict())
    run_config_binding = CanonicalJsonProofBinding.bind(
        descriptor.run_config_source.absolute_path
    )
    content_binding = CanonicalJsonProofBinding.bind(
        descriptor.content_verification_receipt_source.absolute_path
    )
    nextn_binding = (
        None
        if descriptor.nextn_tp2_authority_input_source is None
        else CanonicalJsonProofBinding.bind(
            descriptor.nextn_tp2_authority_input_source.absolute_path
        )
    )
    if (
        run_config_binding != descriptor.run_config_source
        or content_binding != descriptor.content_verification_receipt_source
        or nextn_binding != descriptor.nextn_tp2_authority_input_source
        or tuple(
            CanonicalJsonProofBinding.bind(row.absolute_path)
            for row in descriptor.runtime_gpu_proof_artifacts
        )
        != descriptor.runtime_gpu_proof_artifacts
    ):
        raise ValueError("formal serving rebuild path identity changed")
    run_config = load_run_config(run_config_binding.absolute_path)
    content_receipt = ContentVerificationReceipt.from_dict(content_binding.reopen())
    nextn_tp2_authority_input = (
        None
        if nextn_binding is None
        else _reopen_nextn_tp2_authority_input(nextn_binding)
    )
    if (
        run_config_sha256(run_config) != subject.run_config_sha256
        or content_receipt.sha256 != subject.content_verification_receipt_sha256
    ):
        raise ValueError("formal serving rebuild durable source differs")
    registry_sha256: str | None = None
    if registry_verification_receipt is not None:
        from lightcone_spec.experiments.formal_registry import (
            FormalRegistryVerificationReceipt,
        )

        if type(registry_verification_receipt) is not (
            FormalRegistryVerificationReceipt
        ):
            raise TypeError("formal serving rebuild registry receipt is not exact")
        registry_sha256 = registry_verification_receipt.sha256
    if registry_sha256 != descriptor.registry_verification_receipt_sha256:
        raise ValueError("formal serving rebuild registry lineage differs")
    if stage_source is not None:
        sealed_source = require_verified_formal_stage_materialization_source(
            stage_source,
            materialization=materialization,
        )
        if sealed_source.sha256 != descriptor.stage_source_binding_sha256:
            raise ValueError("formal serving rebuild stage source differs")
    elif descriptor.stage_source_binding_sha256 is not None:
        raise TypeError("formal serving rebuild lacks its typed stage source")
    rebuilt = verify_formal_serving_execution_binding(
        subject,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        run_config=run_config,
        inventory=inventory,
        tts_authority=tts_authority,
        signed_tts_seal=signed_tts_seal,
        e1_recipe_anchor_authority=e1_recipe_anchor_authority,
        e2_recipe_grid_authority=e2_recipe_grid_authority,
        lightcone_recipe=lightcone_recipe,
        stage_source=stage_source,
        now_ns=now_ns,
        content_verification_receipt=content_receipt,
        registry_verification_receipt=registry_verification_receipt,
        nextn_tp2_authority_input=nextn_tp2_authority_input,
    )
    if rebuilt.sha256 != descriptor.execution_binding_sha256:
        raise ValueError("formal serving rebuild execution binding differs")
    return rebuilt


def _method_for_cell(
    cell: MaterializedCell,
) -> FormalServingMethod:
    try:
        return {
            "Target-only": "target_only",
            "Static": "static",
            "TTS": "tts",
            "L0-naive": "l0",
            "LightCone-candidate": "l0",
            "LightCone": "l0",
            "TTS-calibration-candidate": "tts",
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
        }[cell.method_role]  # type: ignore[return-value]
    except KeyError as error:
        raise FormalStageExecutionBlocked(
            "formal_method_role_runtime_adapter_unregistered"
        ) from error


def _require_execution_mapper_authority(
    protocol_lock: ProtocolLock,
    manifest: FormalRuntimeAuthorityManifest,
) -> str:
    if type(manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("formal serving requires exact runtime authority manifest")
    if manifest.sha256 != protocol_lock.formal_runtime_authority_manifest_sha256:
        raise ValueError("formal runtime authority manifest differs from ProtocolLock")
    member = manifest.member("all_stage_execution_mapper")
    if (
        member.protocol_sha256 != FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256
        or member.runner_sha256 != FORMAL_SERVING_EXECUTION_RUNNER_SHA256
        or member.test_set_sha256 != FORMAL_SERVING_EXECUTION_TEST_SET_SHA256
    ):
        raise FormalStageExecutionBlocked(
            "formal_execution_mapper_source_identity_mismatch"
        )
    return member.sha256


def _expected_backend_algorithm(cell: MaterializedCell) -> str | None:
    return None if cell.backend == "NONE" else cell.backend


def _validate_base_run_config(
    cell: MaterializedCell,
    config: RunConfig,
    *,
    expected_method: str,
    topology_mode: CanonicalTopologyMode,
    gpu_uuids: tuple[str, ...],
) -> None:
    if type(config) is not RunConfig:
        raise TypeError("formal serving binding requires an exact RunConfig")
    expected_backend = _expected_backend_algorithm(cell)
    if (
        config.method != expected_method
        or config.model.target != cell.model
        or (expected_backend is not None and config.model.algorithm != expected_backend)
        or config.runtime.topology_mode != topology_mode
        or config.runtime.device_identity != gpu_uuids[0]
    ):
        raise ValueError("formal serving RunConfig differs from materialized cell")
    dimensions = dict(cell.dimensions)
    if "topology" in dimensions and dimensions["topology"] != topology_mode:
        raise ValueError("materialized topology differs from RunConfig")
    expected_load = dimensions.get("common_load", dimensions.get("concurrency"))
    if expected_load is not None and config.runtime.max_running_requests != int(
        expected_load
    ):
        raise ValueError("formal serving RunConfig differs from materialized load")


def _validate_tts_calibration_config(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
    authority: TtsCalibrationAuthority,
) -> tuple[str, ...]:
    if (
        authority.sha256 != protocol_lock.tts_calibration_authority_sha256
        or materialization.source_decision_sha256 != authority.sha256
        or config.adaptation is None
    ):
        raise ValueError("TTS-Cal recipe authority differs from ProtocolLock/cell")
    dimensions = dict(cell.dimensions)
    learning_rate = dimensions.get("learning_rate")
    stride = dimensions.get("stride")
    if type(learning_rate) is not float or type(stride) is not int:
        raise ValueError("TTS-Cal cell lacks exact learning-rate/stride identity")
    authority.validate_runtime_optimizer_config(config.adaptation.optimizer)
    if (
        config.adaptation.weight_update_mode != "full"
        or config.adaptation.parameter_scope != "all"
        or config.adaptation.rank is not None
        or config.adaptation.lora_alpha is not None
        or config.adaptation.stride != stride
        or config.adaptation.optimizer.learning_rate != learning_rate
        or cell.recipe_sha256
        != authority.candidate_id(learning_rate=learning_rate, stride=stride)
    ):
        raise ValueError("TTS-Cal RunConfig differs from exact numeric candidate")
    registry_cell_id = dimensions.get("registry_cell_id")
    _require_sha256("TTS-Cal registry cell", registry_cell_id)
    source_registry = build_industrial_registry()
    source = {row.cell_id: row for row in source_registry.cells_for("TTS-Cal")}.get(
        registry_cell_id
    )
    if source is None:
        raise ValueError("TTS-Cal materialized cell has no staged registry source")
    identity = source.identity
    if (
        identity.model != cell.model
        or identity.backend != cell.backend
        or identity.method != "tts"
        or identity.learning_rate != learning_rate
        or identity.block != dimensions.get("block")
        or int(identity.variant.removeprefix("tts_calibration:stride=")) != stride
    ):
        raise ValueError("TTS-Cal materialized cell differs from staged registry row")
    return (authority.sha256,)


def _validate_e3a_config(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
) -> tuple[str, ...]:
    """Rebuild one baseline-capacity cell from its signed-staged source row."""

    if (
        materialization.source_decision_sha256
        != protocol_lock.formal_workload_e3a_authorization_sha256
        or config.adaptation is not None
        or config.online_spec is not None
    ):
        raise ValueError("E3a source/config differs from ProtocolLock")
    dimensions = dict(cell.dimensions)
    registry_cell_id = dimensions.get("registry_cell_id")
    _require_sha256("E3a registry cell", registry_cell_id)
    source_registry = build_industrial_registry()
    source = {row.cell_id: row for row in source_registry.cells_for("E3a")}.get(
        registry_cell_id
    )
    if source is None:
        raise ValueError("E3a materialized cell has no staged registry source")
    identity = source.identity
    expected_method = _method_for_cell(cell)
    expected_width = identity.width
    if (
        source_registry.sha256 != protocol_lock.registry_sha256
        or identity.model != cell.model
        or identity.backend != cell.backend
        or identity.task != cell.task
        or identity.method != expected_method
        or identity.context != dimensions.get("context")
        or identity.regime != dimensions.get("regime")
        or identity.concurrency != dimensions.get("concurrency")
        or identity.width != dimensions.get("width")
        or identity.topology != "tp1_dp1"
        or config.runtime.topology_mode != "tp1_dp1"
        or config.runtime.max_running_requests != identity.concurrency
    ):
        raise ValueError("E3a RunConfig differs from staged registry row")
    if expected_method == "target_only":
        if expected_width is not None or config.runtime.speculation_enabled:
            raise ValueError("E3a Target-only cell unexpectedly enables speculation")
    elif expected_method == "static":
        if (
            expected_width is None
            or not config.runtime.speculation_enabled
            or config.runtime.speculative_num_draft_tokens != expected_width
            or config.model.draft_depth + 1 != expected_width
        ):
            raise ValueError("E3a Static verify width differs from materialized cell")
    else:  # pragma: no cover - the staged registry has only two E3a methods
        raise ValueError("E3a materialization contains an unsupported method")
    return (protocol_lock.formal_workload_e3a_authorization_sha256,)


def _validate_e1_config(
    *,
    protocol_lock: ProtocolLock,
    cell: MaterializedCell,
    config: RunConfig,
    tts_authority: TtsCalibrationAuthority,
    signed_tts_seal: SignedTtsCalibrationSeal,
    tts_seal_policy: TrustedAttesterPolicy,
    expected_tts_seal_policy_sha256: str,
    e1_recipe_anchor_authority: E1RecipeAnchorAuthority,
    now_ns: int,
) -> tuple[str, ...]:
    if (
        tts_authority.sha256 != protocol_lock.tts_calibration_authority_sha256
        or e1_recipe_anchor_authority.sha256
        != protocol_lock.e1_recipe_anchor_authority_sha256
    ):
        raise ValueError("E1 recipe authority differs from ProtocolLock")
    seal = signed_tts_seal.verify(
        authority=tts_authority,
        policy=tts_seal_policy,
        expected_policy_sha256=expected_tts_seal_policy_sha256,
        now_ns=now_ns,
    )
    if seal.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("E1 frozen TTS seal belongs to another ProtocolLock")
    dimensions = dict(cell.dimensions)
    if cell.method_role in {"Target-only", "Static"}:
        if config.adaptation is not None or cell.recipe_sha256 is not None:
            raise ValueError("E1 non-adaptive anchor allocates adaptation state")
    elif cell.method_role in {"TTS", "L0-naive"}:
        if config.adaptation is None:
            raise ValueError("E1 frozen TTS/L0 anchor lacks adaptation config")
        tts_authority.validate_runtime_optimizer_config(config.adaptation.optimizer)
        if (
            cell.recipe_sha256 != seal.selected_candidate_id
            or config.adaptation.optimizer.learning_rate != seal.selected_learning_rate
            or config.adaptation.stride != seal.selected_stride
            or config.adaptation.weight_update_mode != "full"
            or config.adaptation.parameter_scope != "all"
        ):
            raise ValueError("E1 frozen TTS/L0 RunConfig differs from signed seal")
    elif cell.method_role == "LightCone-candidate":
        if config.adaptation is None:
            raise ValueError("E1 LightCone candidate lacks adaptation config")
        anchor_name = dimensions.get("optimizer_anchor")
        if type(anchor_name) is not str:
            raise ValueError("E1 candidate lacks optimizer-anchor identity")
        anchor = e1_recipe_anchor_authority.anchor(anchor_name)
        adaptation = config.adaptation
        rank = None if dimensions.get("rank") == "none" else dimensions.get("rank")
        if (
            adaptation.optimizer != anchor.optimizer
            or adaptation.stride != anchor.stride
            or adaptation.weight_update_mode != dimensions.get("parameterization")
            or adaptation.parameter_scope != dimensions.get("scope")
            or adaptation.rank != rank
            or adaptation.lora_alpha != rank
        ):
            raise ValueError("E1 LightCone RunConfig differs from geometry/anchor")
        expected_recipe = content_sha256(
            {
                "kind": "e1_lightcone_candidate",
                "geometry": {
                    "scope": dimensions["scope"],
                    "parameterization": dimensions["parameterization"],
                    "rank": rank,
                    "alpha_over_rank": (
                        None
                        if dimensions.get("alpha_over_rank") == "none"
                        else dimensions.get("alpha_over_rank")
                    ),
                },
                "optimizer_anchor": anchor_name,
                "matched_width": dimensions["matched_width"],
                "recipe_anchor_authority_sha256": (e1_recipe_anchor_authority.sha256),
            }
        )
        # ``E1Geometry`` is a dataclass in the materializer; canonicalizing its
        # fields yields the same object as the explicit mapping above.
        if cell.recipe_sha256 != expected_recipe:
            from lightcone_spec.experiments.stage_materialization import E1Geometry

            geometry = E1Geometry(
                scope=str(dimensions["scope"]),
                parameterization=str(dimensions["parameterization"]),  # type: ignore[arg-type]
                rank=None if rank is None else int(rank),
                alpha_over_rank=(
                    None
                    if dimensions.get("alpha_over_rank") == "none"
                    else float(dimensions["alpha_over_rank"])
                ),
            )
            canonical_recipe = content_sha256(
                {
                    "kind": "e1_lightcone_candidate",
                    "geometry": geometry,
                    "optimizer_anchor": anchor_name,
                    "matched_width": dimensions["matched_width"],
                    "recipe_anchor_authority_sha256": (
                        e1_recipe_anchor_authority.sha256
                    ),
                }
            )
            if cell.recipe_sha256 != canonical_recipe:
                raise ValueError("E1 candidate recipe digest differs from authority")
    else:  # pragma: no cover - guarded by _method_for_cell
        raise ValueError("E1 method role is unsupported")
    if config.runtime.speculative_num_draft_tokens != int(dimensions["matched_width"]):
        raise ValueError("E1 RunConfig differs from selected matched width")
    return tuple(
        sorted(
            {
                tts_authority.sha256,
                signed_tts_seal.sha256,
                e1_recipe_anchor_authority.sha256,
            }
        )
    )


def _validate_e2_config(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
    tts_authority: TtsCalibrationAuthority,
    signed_tts_seal: SignedTtsCalibrationSeal,
    tts_seal_policy: TrustedAttesterPolicy,
    grid: E2RecipeGridAuthority,
    now_ns: int,
) -> tuple[str, ...]:
    """Rebuild one E2 RunConfig from complete grid and frozen TTS authority."""

    if (
        materialization.stage != "E2"
        or grid.sha256 != protocol_lock.e2_recipe_grid_authority_sha256
        or tts_authority.sha256 != protocol_lock.tts_calibration_authority_sha256
        or config.runtime.topology_mode != "tp1_dp1"
    ):
        raise ValueError("E2 recipe authority differs from ProtocolLock/stage")
    seal = signed_tts_seal.verify(
        authority=tts_authority,
        policy=tts_seal_policy,
        expected_policy_sha256=tts_seal_policy.sha256,
        now_ns=now_ns,
    )
    if seal.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("E2 frozen TTS seal belongs to another ProtocolLock")
    dimensions = dict(cell.dimensions)
    matched_width = dimensions.get("matched_width")
    common_load = dimensions.get("common_load")
    round_index = dimensions.get("round")
    if (
        type(matched_width) is not int
        or matched_width < 1
        or type(common_load) is not int
        or common_load < 1
        or type(round_index) is not int
        or round_index not in range(4)
        or config.runtime.speculative_num_draft_tokens != matched_width
        or config.runtime.max_running_requests != common_load
    ):
        raise ValueError("E2 RunConfig differs from selected width/common load/round")
    authorities = {
        grid.sha256,
        grid.optimizer_recipe_authority.sha256,
        signed_tts_seal.sha256,
        tts_authority.sha256,
    }
    if cell.method_role in {"Target-only", "Static"}:
        if config.adaptation is not None or cell.recipe_sha256 is not None:
            raise ValueError("E2 non-adaptive anchor allocates adaptation state")
    elif cell.method_role in {"TTS", "L0-naive"}:
        adaptation = config.adaptation
        if adaptation is None:
            raise ValueError("E2 frozen TTS/L0 anchor lacks adaptation config")
        tts_authority.validate_runtime_optimizer_config(adaptation.optimizer)
        if (
            cell.recipe_sha256 != seal.selected_candidate_id
            or adaptation.optimizer.learning_rate != seal.selected_learning_rate
            or adaptation.stride != seal.selected_stride
            or adaptation.weight_update_mode != "full"
            or adaptation.parameter_scope != "all"
        ):
            raise ValueError("E2 frozen TTS/L0 RunConfig differs from signed seal")
    elif cell.method_role == "LightCone-candidate":
        adaptation = config.adaptation
        if adaptation is None:
            raise ValueError("E2 LightCone candidate lacks adaptation config")
        rank_value = dimensions.get("rank")
        rank = None if rank_value == "none" else rank_value
        alpha_value = dimensions.get("alpha_over_rank")
        geometry = E1Geometry(
            scope=str(dimensions.get("scope")),
            parameterization=str(dimensions.get("parameterization")),  # type: ignore[arg-type]
            rank=None if rank is None else int(rank),
            alpha_over_rank=(None if alpha_value == "none" else float(alpha_value)),
        )
        candidate = E2CandidateRecipe(
            geometry=geometry,
            optimizer=str(dimensions.get("optimizer")),
            schedule=str(dimensions.get("schedule")),
            learning_rate=float(dimensions.get("learning_rate")),
            optimizer_recipe_authority_sha256=(grid.optimizer_recipe_authority.sha256),
        )
        numeric = grid.optimizer_recipe_authority.optimizer_recipe(candidate.optimizer)
        schedule_numeric = grid.optimizer_recipe_authority.schedule_recipe(
            candidate.schedule
        )
        expected_adaptation = grid.adaptation_config_for(
            candidate,
            canvas_tokens=matched_width,
            adaptation_group_id=f"e2:{cell.cell_id}",
            chronobelief_gpu_proof_sha256=(
                adaptation.chronobelief_gpu_proof_sha256
                if candidate.optimizer == "chronobelief"
                else None
            ),
        )
        if (
            cell.recipe_sha256 != candidate.sha256
            or dimensions.get("geometry_sha256") != geometry.sha256
            or dimensions.get("optimizer_recipe_authority_sha256")
            != grid.optimizer_recipe_authority.sha256
            or dimensions.get("optimizer_numeric_recipe_sha256") != numeric.sha256
            or dimensions.get("schedule_numeric_recipe_sha256")
            != schedule_numeric.sha256
            or dimensions.get("stride") != numeric.stride
            or adaptation != expected_adaptation
        ):
            raise ValueError("E2 LightCone RunConfig differs from complete recipe")
        authorities.update((candidate.sha256, numeric.sha256, schedule_numeric.sha256))
    else:  # pragma: no cover - guarded by the materializer
        raise ValueError("E2 method role is unsupported")
    return tuple(sorted(authorities))


def _validate_downstream_config(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    cell: MaterializedCell,
    config: RunConfig,
    source: VerifiedFormalStageMaterializationSource,
    tts_authority: TtsCalibrationAuthority | None,
    signed_tts_seal: SignedTtsCalibrationSeal | None,
    registry_verification_receipt: object | None,
    e2_recipe_grid_authority: E2RecipeGridAuthority | None,
    lightcone_recipe: E2CandidateRecipe | None,
    now_ns: int,
) -> tuple[str, ...]:
    """Bind downstream numeric runtime state to typed upstream authorities.

    The signed-source rebuild owns the cell universe and recipe digests.  This
    function additionally checks every RunConfig field that is represented by
    that universe.  Named load/width panels remain fail closed unless their
    exact numeric value is present in the config and the upstream E3a source.
    """

    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    dimensions = dict(cell.dimensions)
    authorities = {source.sha256, *source.typed_source_authority_sha256s}
    if cell.recipe_sha256 is not None:
        authorities.add(cell.recipe_sha256)
    if cell.method_role in {"Target-only", "Static"}:
        if config.adaptation is not None or config.online_spec is not None:
            raise ValueError("downstream non-adaptive role allocates adaptation state")
    elif cell.method_role in {"TTS", "L0-naive"}:
        if (
            type(tts_authority) is not TtsCalibrationAuthority
            or type(signed_tts_seal) is not SignedTtsCalibrationSeal
            or type(registry_verification_receipt)
            is not FormalRegistryVerificationReceipt
            or config.adaptation is None
            or config.online_spec is not None
        ):
            raise TypeError("downstream frozen TTS/L0 requires its exact signed recipe")
        registry_verification_receipt.revalidate(current_ns=now_ns)
        policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
        seal = signed_tts_seal.verify(
            authority=tts_authority,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
        adaptation = config.adaptation
        tts_authority.validate_runtime_optimizer_config(adaptation.optimizer)
        if (
            seal.protocol_lock_sha256 != protocol_lock.sha256
            or cell.recipe_sha256 != seal.selected_candidate_id
            or adaptation.optimizer.learning_rate != seal.selected_learning_rate
            or adaptation.stride != seal.selected_stride
            or adaptation.weight_update_mode != "full"
            or adaptation.parameter_scope != "all"
        ):
            raise ValueError("downstream frozen TTS/L0 config differs from signed seal")
        authorities.update(
            (tts_authority.sha256, signed_tts_seal.sha256, seal.selected_candidate_id)
        )
    elif cell.method_role in {"LightCone", "LightCone-candidate"}:
        adaptation = config.adaptation
        if adaptation is None or config.online_spec is not None:
            raise ValueError(
                "downstream LightCone cell lacks isolated adaptation state"
            )
        if type(e2_recipe_grid_authority) is not E2RecipeGridAuthority:
            raise TypeError("downstream LightCone requires exact E2 numeric grid")
        if e2_recipe_grid_authority.sha256 != (
            protocol_lock.e2_recipe_grid_authority_sha256
        ):
            raise ValueError("downstream LightCone numeric grid differs from lock")
        if type(lightcone_recipe) is not E2CandidateRecipe:
            raise TypeError("downstream LightCone requires its typed winning recipe")
        expected_optimizer = e2_recipe_grid_authority.optimizer_config_for(
            lightcone_recipe
        )
        if adaptation.optimizer != expected_optimizer:
            raise ValueError("downstream LightCone optimizer differs from E2 winner")
        if materialization.stage in {"E4", "E3b"} and (
            cell.recipe_sha256 != lightcone_recipe.sha256
        ):
            raise ValueError("downstream DFlash recipe differs from E2 winner")
        if materialization.stage == "E4":
            expected_profile = cell.task == "mechanism_profile_only"
            if config.runtime.telemetry_detail != (
                "profile" if expected_profile else "headline"
            ):
                raise ValueError("E4 profile/headline telemetry is not separated")
            if not expected_profile and (
                adaptation.stride != int(dimensions["update_stride"])
                or config.runtime.adaptation_microbatch_size
                != int(dimensions["microbatch"])
                or config.runtime.adaptation_publication_coalescing
                != int(dimensions["coalescing"])
                or config.runtime.adaptation_stream_priority
                != dimensions["stream_priority"]
            ):
                raise ValueError("E4 RunConfig differs from signed factor row")
        elif materialization.stage == "E1a":
            rank = None if dimensions["rank"] == "none" else int(dimensions["rank"])
            verification = str(dimensions["verification_mode"])
            if (
                adaptation.parameter_scope != dimensions["scope"]
                or adaptation.weight_update_mode != dimensions["parameterization"]
                or adaptation.rank != rank
                or adaptation.lora_alpha != rank
                or adaptation.verification_mode
                != (
                    "fixed_budget"
                    if verification == "fixed_verification_budget"
                    else "native_scheduler"
                )
                or (verification == "fixed_verification_budget")
                != (adaptation.fixed_verification_budget is not None)
                or adaptation.fixed_verification_budget
                != (
                    E1A_FIXED_VERIFICATION_BUDGET
                    if verification == "fixed_verification_budget"
                    else None
                )
                or dimensions.get("fixed_verification_budget")
                != (
                    E1A_FIXED_VERIFICATION_BUDGET
                    if verification == "fixed_verification_budget"
                    else E1A_NATIVE_VERIFICATION_BUDGET
                )
            ):
                raise ValueError("E1a RunConfig differs from exact DSpark geometry")
        elif materialization.stage == "E5":
            topology = dimensions.get("topology", "tp1_dp1")
            if config.runtime.topology_mode != topology:
                raise ValueError("E5 RunConfig differs from materialized topology")
        elif materialization.stage == "E6":
            if config.runtime.topology_mode != "tp2_dp1":
                raise ValueError("E6 NEXTN execution requires exact TP2 topology")
            for field, actual in (
                ("target_model_id", config.model.target),
                ("target_revision", config.model.target_revision),
                ("drafter_model_id", config.model.drafter),
                ("drafter_revision", config.model.drafter_revision),
            ):
                if dimensions.get(field) != actual:
                    raise ValueError(
                        "E6 RunConfig differs from compatibility authority"
                    )
        authorities.update(
            (
                e2_recipe_grid_authority.sha256,
                e2_recipe_grid_authority.optimizer_recipe_authority.sha256,
                lightcone_recipe.sha256,
            )
        )
    elif cell.method_role.startswith("OnlineSPEC-"):
        if config.adaptation is None or config.online_spec is None:
            raise ValueError("OnlineSPEC cell lacks its independent online state")
        from lightcone_spec.experiments.data import (
            DFLASH_BLOCK_SIZE,
            DFLASH_LOSS_POSITION_DECAY,
        )
        from lightcone_spec.experiments.onlinespec import onlinespec_candidates

        candidates = {
            candidate.candidate_id: candidate for candidate in onlinespec_candidates()
        }
        candidate = candidates.get(cell.recipe_sha256 or "")
        adaptation = config.adaptation
        online = config.online_spec
        if (
            candidate is None
            or candidate.method != config.method
            or dimensions.get("onlinespec_method", candidate.method) != candidate.method
            or adaptation.adaptation_group_id != f"e0:{cell.cell_id}"
            or adaptation.weight_update_mode != candidate.weight_update_mode
            or adaptation.parameter_scope != candidate.parameter_scope
            or adaptation.optimizer
            != OptimizerConfig(
                name="sgd",
                learning_rate=candidate.learning_rate,
                weight_decay=0.0,
                grad_clip=candidate.grad_clip,
            )
            or adaptation.rank != candidate.rank
            or adaptation.lora_alpha
            != (candidate.rank if candidate.weight_update_mode == "lora" else None)
            or adaptation.stride != candidate.stride
            or adaptation.canvas_tokens != DFLASH_BLOCK_SIZE
            or adaptation.loss_position_decay != DFLASH_LOSS_POSITION_DECAY
            or online.projection_radius != candidate.projection_radius
            or online.additional_learning_rates != candidate.additional_learning_rates
            or online.hedge_learning_rate != candidate.hedge_learning_rate
        ):
            raise ValueError("OnlineSPEC RunConfig differs from its registered recipe")
        authorities.add(candidate.candidate_id)
    else:  # pragma: no cover - guarded by the closed role mapper
        raise ValueError("downstream method role is unsupported")

    if materialization.stage == "E3b":
        expected_load = dimensions.get("load")
        if expected_load == "concurrency_one":
            if config.runtime.max_running_requests != 1:
                raise ValueError("E3b concurrency-one cell has another load")
        elif expected_load == "common_load":
            if (
                type(registry_verification_receipt)
                is not FormalRegistryVerificationReceipt
            ):
                raise TypeError("E3b common load requires durable E3a lineage")
            common_loads = {
                row.payload.common_load
                for row in (
                    registry_verification_receipt.cumulative_signed_e3a_staged_selections
                )
            }
            if common_loads != {config.runtime.max_running_requests}:
                raise ValueError("E3b common load differs from signed E3a selection")
        if config.runtime.speculative_num_draft_tokens not in {4, 8, 16}:
            raise ValueError("E3b width panel is outside the locked DFlash widths")
    if materialization.stage in {"E5", "E6", "E0"}:
        load = dimensions.get("load")
        if load == "concurrency_one" and config.runtime.max_running_requests != 1:
            raise ValueError("downstream concurrency-one cell has another load")
        if load == "common_slo_load" and config.runtime.max_running_requests <= 1:
            raise ValueError("downstream common-SLO load is not an executable load")
    return tuple(sorted(authorities))


def _proof_bindings(paths: tuple[str, ...]) -> tuple[CanonicalJsonProofBinding, ...]:
    if type(paths) is not tuple or not paths:
        raise ValueError("formal serving requires runtime GPU proof paths")
    rows = tuple(
        sorted(
            (
                CanonicalJsonProofBinding.bind(
                    _require_absolute_path("GPU proof", path)
                )
                for path in paths
            ),
            key=lambda row: row.absolute_path,
        )
    )
    if len({row.absolute_path for row in rows}) != len(rows):
        raise ValueError("formal serving GPU proof path is duplicated")
    return rows


def _verified_content_identity(
    *,
    receipt: ContentVerificationReceipt,
    protocol_lock: ProtocolLock,
    stage: FormalServingStage,
    cell: MaterializedCell,
    run_config: RunConfig,
    tts_authority: TtsCalibrationAuthority | None,
    now_ns: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(receipt) is not ContentVerificationReceipt:
        raise TypeError("formal serving requires an exact content receipt")
    verified_rows = receipt.revalidate_formal_scope(current_ns=now_ns)
    prepared = tuple(
        row for row in verified_rows if type(row) is VerifiedPreparedModelContentRelease
    )
    workloads = tuple(
        row for row in verified_rows if type(row) is VerifiedReleaseWorkloadSources
    )
    datasets = tuple(
        row for row in verified_rows if type(row) is VerifiedDatasetContentRelease
    )
    if stage == "E0":
        if len(verified_rows) != 2 or len(prepared) != 1 or len(datasets) != 1:
            raise ValueError(
                "E0 content requires one prepared-model and one task-native authority"
            )
        dataset = datasets[0]
        if (
            dataset.authority_domain != "e0_task_native"
            or dataset.authorization_sha256
            != protocol_lock.formal_workload_e0_authorization_sha256
            or dataset.authorization.root_manifest_sha256
            != protocol_lock.offline_release_trust_root_sha256
        ):
            raise ValueError("E0 task-native authority differs from ProtocolLock")
        path_artifacts = tuple(
            artifact
            for artifact in receipt.content_artifacts
            if (
                type(payload := artifact.load()) is dict
                and payload.get("kind") == "lightcone_dataset_content_path_binding"
                and payload.get("authority_domain") == "e0_task_native"
            )
        )
        if len(path_artifacts) != 1:
            raise ValueError("E0 content lacks one task-native path binding")
        path_binding = DatasetContentPathBinding.from_dict(path_artifacts[0].load())
        members = revalidate_authorized_dataset_content_release(
            path_binding,
            authorization=dataset,
        )
        dimensions = dict(cell.dimensions)
        required_shape_sha256 = dimensions.get("task_native_workload_sha256")
        if cell.task == "independent_onlinespec_tuning":
            deployment_task = dimensions.get("deployment_task")
            if type(deployment_task) is not str or not deployment_task:
                raise ValueError("E0 tuning cell lacks its deployment task identity")
        elif dimensions.get("load") not in {
            "concurrency_one",
            "common_slo_load",
        }:
            raise ValueError("E0 serving cell lacks an exact registered load")
        matched = tuple(
            member
            for member in members
            if member.request_shape_sha256 == required_shape_sha256
        )
        if len(matched) != 1:
            raise ValueError(
                "E0 task/load cell does not name one path-reopened request shape"
            )
        dataset_workload_sha256s = tuple(
            sorted(
                {
                    dataset.authorization_sha256,
                    content_sha256(matched[0]),
                    path_binding.sha256,
                }
            )
        )
    else:
        if datasets:
            raise ValueError("non-E0 serving content contains unused dataset authority")
        dataset_workload_sha256s = ()
    expected_authorization_count = 2
    if len(verified_rows) != expected_authorization_count:
        raise ValueError("formal serving content contains unused authorizations")
    if len(prepared) != 1:
        raise ValueError("formal serving content lacks one prepared-model authority")
    prepared_authority = prepared[0]
    if (
        prepared_authority.authorization_sha256
        != protocol_lock.prepared_model_content_authorization_sha256
        or prepared_authority.authorization.root_manifest_sha256
        != protocol_lock.offline_release_trust_root_sha256
    ):
        raise ValueError("formal serving prepared-model authority differs from lock")
    stage_members = prepared_authority.require_stage(stage)
    target = tuple(row for row in stage_members if row.role == "target")
    drafter = tuple(row for row in stage_members if row.role == "drafter")
    tokenizers = tuple(row for row in stage_members if row.role == "tokenizer")
    expected_backend = run_config.model.algorithm
    if (
        len(stage_members) != 3
        or len(target) != 1
        or len(drafter) != 1
        or len(tokenizers) != 1
        or target[0].model_id != run_config.model.target
        or target[0].revision != run_config.model.target_revision
        or drafter[0].model_id != run_config.model.drafter
        or drafter[0].revision != run_config.model.drafter_revision
        or any(row.backend != expected_backend for row in stage_members)
        or tokenizers[0].model_id
        not in {run_config.model.target, run_config.model.drafter}
        or tokenizers[0].revision
        not in {run_config.model.target_revision, run_config.model.drafter_revision}
    ):
        raise ValueError(
            "formal serving prepared-model members differ from exact RunConfig"
        )
    content_rows = receipt.content_artifacts
    for member in stage_members:
        if not any(
            row.raw_sha256 == member.snapshot_manifest_raw_sha256
            and row.semantic_sha256 == member.snapshot_manifest_semantic_sha256
            for row in content_rows
        ):
            raise ValueError("formal serving prepared snapshot was not path-reopened")
    prepared_sha256s = tuple(sorted(content_sha256(member) for member in stage_members))
    if stage == "E0":
        if workloads:
            raise ValueError("E0 cannot inherit the E3a workload authority")
        workload_sha256s = dataset_workload_sha256s
    elif stage != "TTS-Cal":
        if len(workloads) != 1:
            raise ValueError("E1 serving content lacks one workload authority")
        workload = workloads[0]
        if (
            workload.authorization_sha256
            != protocol_lock.formal_workload_e3a_authorization_sha256
            or workload.authorization.root_manifest_sha256
            != protocol_lock.offline_release_trust_root_sha256
        ):
            raise ValueError("E1 workload authority differs from ProtocolLock")
        workload_sha256s = (
            (workload.source("livecodebench_v6_hard").sha256,)
            if stage in {"E1", "E2", "E1a"}
            else tuple(
                sorted(
                    (
                        workload.source("livecodebench_v6_hard").sha256,
                        workload.source("math500_level5").sha256,
                    )
                )
            )
        )
    else:
        if len(workloads) != 1:
            raise ValueError("TTS-Cal requires one root-authorized workload authority")
        workload = workloads[0]
        if (
            workload.authorization_sha256
            != protocol_lock.formal_workload_e3a_authorization_sha256
            or workload.authorization.root_manifest_sha256
            != protocol_lock.offline_release_trust_root_sha256
        ):
            raise ValueError("TTS-Cal workload authority differs from ProtocolLock")
        if type(tts_authority) is not TtsCalibrationAuthority:
            raise TypeError("TTS-Cal content requires exact calibration authority")
        tuning = tuple(
            row
            for row in content_rows
            if row.artifact_id == "tts_calibration_tuning_window"
            and row.semantic_sha256 == tts_authority.tuning_window_sha256
        )
        if len(tuning) != 1:
            raise ValueError(
                "TTS-Cal tuning window was not path-reopened from its authority"
            )
        window = TtsCalibrationTuningWindow.from_dict(tuning[0].load())
        if window.sha256 != tts_authority.tuning_window_sha256 or any(
            row.source_descriptor_sha256 != workload.source(row.workload_id).sha256
            for row in window.entries
        ):
            raise ValueError(
                "TTS-Cal tuning requests differ from root-authorized workload rows"
            )
        workload_sha256s = tuple(
            sorted(
                {
                    tuning[0].semantic_sha256,
                    *(row.source_descriptor_sha256 for row in window.entries),
                }
            )
        )
    return prepared_sha256s, workload_sha256s


def prepare_formal_serving_execution_subject(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    materialized_cell_id: str,
    run_config: RunConfig,
    inventory: GpuInventory,
    gpu_uuids: tuple[str, ...],
    runtime_gpu_proof_artifact_paths: tuple[str, ...],
    run_id: str,
    run_nonce_sha256: str,
    attempt_id: str,
    tts_authority: TtsCalibrationAuthority | None,
    signed_tts_seal: SignedTtsCalibrationSeal | None = None,
    e1_recipe_anchor_authority: E1RecipeAnchorAuthority | None = None,
    e2_recipe_grid_authority: E2RecipeGridAuthority | None = None,
    lightcone_recipe: E2CandidateRecipe | None = None,
    stage_source: VerifiedFormalStageMaterializationSource | None = None,
    now_ns: int,
    content_verification_receipt: ContentVerificationReceipt | None = None,
    registry_verification_receipt: object | None = None,
    current_e0_eagle3_proof_row: _VerifiedCurrentE0Eagle3ProofRow | None = None,
) -> FormalServingExecutionSubject:
    """Build an exact mapping without granting execution authority."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal serving requires exact ProtocolLock")
    mapper_authority_sha256 = _require_execution_mapper_authority(
        protocol_lock, formal_runtime_authority_manifest
    )
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal serving requires exact materialization")
    if materialization.protocol_lock_sha256 != protocol_lock.sha256:
        raise ValueError("formal serving materialization belongs to another lock")
    cells = {cell.cell_id: cell for cell in materialization.cells}
    cell = cells.get(materialized_cell_id)
    if cell is None:
        raise ValueError("formal serving cell is outside materialization")
    if materialization.stage not in _FORMAL_SERVING_STAGES:
        raise FormalStageExecutionBlocked("formal_serving_stage_unregistered")
    if type(inventory) is not GpuInventory:
        raise TypeError("formal serving requires exact GPU inventory")
    for uuid in gpu_uuids:
        inventory.device(uuid)
    method = _method_for_cell(cell)
    topology_mode = run_config.runtime.topology_mode
    _validate_base_run_config(
        cell,
        run_config,
        expected_method=method,
        topology_mode=topology_mode,
        gpu_uuids=gpu_uuids,
    )
    stage_source_binding_sha256: str | None = None
    if materialization.stage == "E3a":
        recipe_authorities = _validate_e3a_config(
            protocol_lock=protocol_lock,
            materialization=materialization,
            cell=cell,
            config=run_config,
        )
        workload_authority_sha256 = (
            protocol_lock.formal_workload_e3a_authorization_sha256
        )
        dimensions = dict(cell.dimensions)
        registry_cell_id = dimensions["registry_cell_id"]
        source = {
            row.cell_id: row for row in build_industrial_registry().cells_for("E3a")
        }[registry_cell_id]
        logical = source.identity.gpu_uuids
        expected_gpus = tuple(
            inventory.devices[int(value.removeprefix("logical-rank-slot-"))].uuid
            for value in logical
        )
        if gpu_uuids != expected_gpus:
            raise ValueError("E3a physical GPU differs from staged slot assignment")
    elif materialization.stage == "TTS-Cal":
        if type(tts_authority) is not TtsCalibrationAuthority:
            raise TypeError("TTS-Cal execution requires exact calibration authority")
        recipe_authorities = _validate_tts_calibration_config(
            protocol_lock=protocol_lock,
            materialization=materialization,
            cell=cell,
            config=run_config,
            authority=tts_authority,
        )
        workload_authority_sha256 = tts_authority.tuning_window_sha256
        dimensions = dict(cell.dimensions)
        registry_cell_id = dimensions["registry_cell_id"]
        source = {
            row.cell_id: row for row in build_industrial_registry().cells_for("TTS-Cal")
        }[registry_cell_id]
        logical = source.identity.gpu_uuids
        expected_gpus = tuple(
            inventory.devices[int(value.removeprefix("logical-rank-slot-"))].uuid
            for value in logical
        )
        if gpu_uuids != expected_gpus:
            raise ValueError("TTS-Cal physical GPU differs from staged slot assignment")
    elif materialization.stage == "E1":
        from lightcone_spec.experiments.formal_registry import (
            FormalRegistryVerificationReceipt,
        )

        if signed_tts_seal is None or e1_recipe_anchor_authority is None:
            raise TypeError("E1 execution requires sealed TTS and recipe anchors")
        if type(tts_authority) is not TtsCalibrationAuthority:
            raise TypeError("E1 execution requires exact calibration authority")
        if type(registry_verification_receipt) is not (
            FormalRegistryVerificationReceipt
        ):
            raise TypeError(
                "E1 execution requires durable registry verification receipt"
            )
        manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
        if (
            registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
            or manifest.protocol_lock_sha256 != protocol_lock.sha256
            or signed_tts_seal
            not in registry_verification_receipt.cumulative_signed_tts_calibration_seals
        ):
            raise ValueError("E1 execution registry receipt lacks sealed TTS lineage")
        tts_seal_policy = registry_verification_receipt.trusted_release_policy(
            current_ns=now_ns
        )
        recipe_authorities = _validate_e1_config(
            protocol_lock=protocol_lock,
            cell=cell,
            config=run_config,
            tts_authority=tts_authority,
            signed_tts_seal=signed_tts_seal,
            tts_seal_policy=tts_seal_policy,
            expected_tts_seal_policy_sha256=tts_seal_policy.sha256,
            e1_recipe_anchor_authority=e1_recipe_anchor_authority,
            now_ns=now_ns,
        )
        workload_authority_sha256 = (
            protocol_lock.formal_workload_e3a_authorization_sha256
        )
    elif materialization.stage == "E2":
        from lightcone_spec.experiments.formal_registry import (
            FormalRegistryVerificationReceipt,
        )

        if type(tts_authority) is not TtsCalibrationAuthority:
            raise TypeError("E2 execution requires exact calibration authority")
        if signed_tts_seal is None:
            raise TypeError("E2 execution requires the signed frozen TTS seal")
        if type(e2_recipe_grid_authority) is not E2RecipeGridAuthority:
            raise TypeError("E2 execution requires complete numeric recipe authority")
        if type(registry_verification_receipt) is not (
            FormalRegistryVerificationReceipt
        ):
            raise TypeError(
                "E2 execution requires durable registry verification receipt"
            )
        manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
        materialization_ids = {
            row.materialization_receipt_sha256 for row in manifest.materializations
        }
        source_authority_ids = {
            row.signed_authority_sha256 for row in manifest.source_authorities
        }
        source_signed_ids = {
            row.sha256
            for row in (
                *registry_verification_receipt.cumulative_signed_e1_survivor_selections,
                *registry_verification_receipt.cumulative_signed_e2_staged_selections,
            )
        }
        if (
            registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
            or manifest.protocol_lock_sha256 != protocol_lock.sha256
            or materialization.sha256 not in materialization_ids
            or signed_tts_seal
            not in registry_verification_receipt.cumulative_signed_tts_calibration_seals
            or materialization.source_decision_sha256 not in source_signed_ids
            or materialization.source_decision_sha256 not in source_authority_ids
        ):
            raise ValueError("E2 registry receipt lacks exact staged source lineage")
        tts_seal_policy = registry_verification_receipt.trusted_release_policy(
            current_ns=now_ns
        )
        recipe_authorities = _validate_e2_config(
            protocol_lock=protocol_lock,
            materialization=materialization,
            cell=cell,
            config=run_config,
            tts_authority=tts_authority,
            signed_tts_seal=signed_tts_seal,
            tts_seal_policy=tts_seal_policy,
            grid=e2_recipe_grid_authority,
            now_ns=now_ns,
        )
        workload_authority_sha256 = (
            protocol_lock.formal_workload_e3a_authorization_sha256
        )
    else:
        assert materialization.stage in _DOWNSTREAM_SERVING_STAGES
        verified_source = require_verified_formal_stage_materialization_source(
            stage_source,
            materialization=materialization,
        )
        stage_source_binding_sha256 = verified_source.sha256
        recipe_authorities = _validate_downstream_config(
            protocol_lock=protocol_lock,
            materialization=materialization,
            cell=cell,
            config=run_config,
            source=verified_source,
            tts_authority=tts_authority,
            signed_tts_seal=signed_tts_seal,
            registry_verification_receipt=registry_verification_receipt,
            e2_recipe_grid_authority=e2_recipe_grid_authority,
            lightcone_recipe=lightcone_recipe,
            now_ns=now_ns,
        )
        workload_authority_sha256 = (
            protocol_lock.formal_workload_e0_authorization_sha256
            if materialization.stage == "E0"
            else protocol_lock.formal_workload_e3a_authorization_sha256
        )
    if content_verification_receipt is None:
        content_receipt_sha256 = None
        prepared_model_member_sha256s: tuple[str, ...] = ()
        workload_member_sha256s: tuple[str, ...] = ()
    else:
        prepared_model_member_sha256s, workload_member_sha256s = (
            _verified_content_identity(
                receipt=content_verification_receipt,
                protocol_lock=protocol_lock,
                stage=materialization.stage,  # type: ignore[arg-type]
                cell=cell,
                run_config=run_config,
                tts_authority=tts_authority,
                now_ns=now_ns,
            )
        )
        content_receipt_sha256 = content_verification_receipt.sha256
    proofs = _proof_bindings(runtime_gpu_proof_artifact_paths)
    if (
        materialization.stage == "E0"
        and cell.backend == "EAGLE3"
        and run_config.adaptation is not None
    ):
        if stage_source_binding_sha256 is None:
            raise AssertionError("E0 EAGLE3 lost its sealed stage source")
        eagle = run_config.adaptation
        if current_e0_eagle3_proof_row is None:
            # Preserve the legacy signed-source mapping.  Current trusted
            # single-operator execution supplies the stronger path-replayed
            # task row below instead of treating stage/interface digests as
            # runtime proof identities.
            dimensions = dict(cell.dimensions)
            if (
                eagle.eagle3_e0_execution_authority_sha256
                != stage_source_binding_sha256
                or eagle.eagle3_compatibility_authority_sha256
                != dimensions.get("signed_e0_compatibility_sha256")
                or eagle.eagle3_model_selector_sha256
                != dimensions.get("interface_sha256")
                or eagle.eagle3_native_gpu_proof_sha256
                not in {proof.semantic_sha256 for proof in proofs}
            ):
                raise ValueError(
                    "E0 EAGLE3 RunConfig lacks exact compatibility/GPU authority"
                )
        else:
            current = current_e0_eagle3_proof_row
            if (
                type(current) is not _VerifiedCurrentE0Eagle3ProofRow
                or current._construction_seal
                is not _VERIFIED_CURRENT_E0_EAGLE3_PROOF_ROW_SEAL
                or eagle.eagle3_e0_execution_authority_sha256
                != current.execution_authority_sha256
                or eagle.eagle3_compatibility_authority_sha256
                != current.compatibility_authority_sha256
                or eagle.eagle3_model_selector_sha256 != current.model_selector_sha256
                or eagle.eagle3_native_gpu_proof_sha256
                != current.native_gpu_receipt_sha256
            ):
                raise ValueError(
                    "current E0 EAGLE3 RunConfig differs from its task proof row"
                )
            recipe_authorities = tuple(
                sorted(
                    {
                        *recipe_authorities,
                        current.compatibility_bundle_sha256,
                        current.interface_receipt_sha256,
                        current.terminal_sha256,
                        current.proof_row_sha256,
                    }
                )
            )
    elif current_e0_eagle3_proof_row is not None:
        raise ValueError("non-adaptive E0 EAGLE3 cell carries a current proof row")
    config_sha256 = run_config_sha256(run_config)
    execution_plan_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_serving_execution_plan_identity",
            "protocol_sha256": FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256,
            "protocol_lock_sha256": protocol_lock.sha256,
            "formal_runtime_authority_manifest_sha256": (
                formal_runtime_authority_manifest.sha256
            ),
            "execution_mapper_authority_sha256": mapper_authority_sha256,
            "materialization_receipt_sha256": materialization.sha256,
            "materialized_cell_id": cell.cell_id,
            "stage_source_binding_sha256": stage_source_binding_sha256,
            "run_config_sha256": config_sha256,
            "recipe_authority_sha256s": recipe_authorities,
            "workload_authority_sha256": workload_authority_sha256,
            "content_verification_receipt_sha256": content_receipt_sha256,
            "prepared_model_member_sha256s": prepared_model_member_sha256s,
            "workload_member_sha256s": workload_member_sha256s,
            "inventory_sha256": inventory.sha256,
            "topology_mode": topology_mode,
            "gpu_uuids": gpu_uuids,
            "runtime_gpu_proof_artifacts": tuple(
                (row.raw_sha256, row.semantic_sha256) for row in proofs
            ),
        }
    )
    rank_config_sha256 = content_sha256(
        {
            "schema_version": 1,
            "kind": "lightcone_formal_serving_rank_config_set",
            "execution_plan_sha256": execution_plan_sha256,
            "run_config_sha256": config_sha256,
            "topology_mode": topology_mode,
            "gpu_uuids": gpu_uuids,
            "tensor_parallel_size": run_config.runtime.tensor_parallel_size,
            "data_parallel_size": run_config.runtime.data_parallel_size,
            "router_identity": run_config.runtime.router_identity,
            "process_group_backend": run_config.runtime.process_group_backend,
        }
    )
    execution_identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=cell.cell_id,
        inventory_sha256=inventory.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=rank_config_sha256,
        run_id=run_id,
        run_nonce_sha256=run_nonce_sha256,
        attempt_id=attempt_id,
        method=method,
    )
    return FormalServingExecutionSubject(
        schema_version=4,
        protocol_lock_sha256=protocol_lock.sha256,
        formal_runtime_authority_manifest_sha256=(
            formal_runtime_authority_manifest.sha256
        ),
        execution_mapper_authority_sha256=mapper_authority_sha256,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        stage=materialization.stage,  # type: ignore[arg-type]
        method=method,
        stage_source_binding_sha256=stage_source_binding_sha256,
        run_config_sha256=config_sha256,
        recipe_authority_sha256s=recipe_authorities,
        workload_authority_sha256=workload_authority_sha256,
        content_verification_receipt_sha256=content_receipt_sha256,
        prepared_model_member_sha256s=prepared_model_member_sha256s,
        workload_member_sha256s=workload_member_sha256s,
        inventory_sha256=inventory.sha256,
        topology_mode=topology_mode,
        gpu_uuids=gpu_uuids,
        runtime_gpu_proof_artifacts=proofs,
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=rank_config_sha256,
        execution_identity=execution_identity,
    )


def _expected_native_gpu_proof_suites(
    *,
    algorithm: str,
    topology: CanonicalTopologyMode,
    chronobelief: bool,
) -> frozenset[str]:
    """Return the closed backend/topology-native qualification union."""

    if topology == "tp1_dp1":
        backend_suite = {
            "DFLASH": None,
            "DSPARK": "dspark_tp1",
            "NEXTN": "nextn_tp1",
            "EAGLE3": "eagle3_tp1",
        }.get(algorithm, "unsupported")
        if backend_suite == "unsupported":
            raise ValueError("formal serving backend lacks a TP1 GPU proof policy")
        suites = {"native_hot_path_tp1"}
        if backend_suite is not None:
            suites.add(backend_suite)
        if chronobelief:
            suites.add("chronobelief_gpu_parity")
        return frozenset(suites)
    if chronobelief:
        raise ValueError(
            "distributed ChronoBelief lacks an exact registered parity suite"
        )
    distributed_suite = {
        ("DFLASH", "tp2_dp1"): None,
        ("DFLASH", "tp1_dp2"): None,
        ("DSPARK", "tp2_dp1"): "dspark_tp2",
        ("DSPARK", "tp1_dp2"): "dspark_dp2",
        ("NEXTN", "tp2_dp1"): "nextn_tp2",
    }.get((algorithm, topology), "unsupported")
    if distributed_suite == "unsupported":
        raise ValueError(
            "formal serving backend/topology lacks an exact GPU proof policy"
        )
    return frozenset(() if distributed_suite is None else (distributed_suite,))


def _revalidate_gpu_proofs(
    subject: FormalServingExecutionSubject,
    *,
    protocol_lock: ProtocolLock,
    run_config: RunConfig,
    now_ns: int,
) -> tuple[
    tuple[str, ...],
    str,
    tuple[VerifiedNativeRuntimeGpuProof, ...],
    tuple[VerifiedDistributedRuntimeGpuProof, ...],
]:
    native: list[VerifiedNativeRuntimeGpuProof] = []
    distributed: list[VerifiedDistributedRuntimeGpuProof] = []
    native_receipt_sha256s: dict[str, str] = {}
    roots: set[str] = set()
    for binding in subject.runtime_gpu_proof_artifacts:
        raw = binding.reopen()
        if type(raw) is not dict:
            raise TypeError("runtime GPU proof artifact must be an object")
        kind = raw.get("kind")
        if kind == "lightcone_native_runtime_gpu_proof_artifact":
            artifact = NativeRuntimeGpuProofArtifact.from_dict(raw)
            verified = artifact.revalidate(now_ns=now_ns)
            if type(verified) is not VerifiedNativeRuntimeGpuProof:
                raise TypeError("native GPU proof verifier returned a foreign token")
            native.append(verified)
            if verified.suite_id in native_receipt_sha256s:
                raise ValueError("formal serving repeats a native GPU proof suite")
            native_receipt_sha256s[verified.suite_id] = verified.receipt_sha256
            roots.add(
                artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
            )
        elif kind == "lightcone_distributed_runtime_gpu_proof_artifact":
            artifact = DistributedRuntimeGpuProofArtifact.from_dict(raw)
            verified = artifact.revalidate(now_ns=now_ns)
            if type(verified) is not VerifiedDistributedRuntimeGpuProof:
                raise TypeError(
                    "distributed GPU proof verifier returned a foreign token"
                )
            distributed.append(verified)
            roots.add(
                artifact.control_attestation.deployment_policy_authorization.root_manifest_sha256
            )
        else:
            raise ValueError("formal serving runtime GPU proof kind is unsupported")
    if roots != {protocol_lock.offline_release_trust_root_sha256}:
        raise ValueError("runtime GPU proof uses another offline release root")
    expected_source = protocol_lock.native_runtime_qualification_source_identity_sha256
    all_proofs = (*native, *distributed)
    if any(
        row.source_identity_sha256 != expected_source
        or row.inventory_sha256 != subject.inventory_sha256
        for row in all_proofs
    ):
        raise ValueError("runtime GPU proof differs from lock/inventory")
    hardware = {row.hardware_envelope_sha256 for row in all_proofs}
    if len(hardware) != 1:
        raise ValueError("runtime GPU proofs use different hardware envelopes")
    topology = subject.topology_mode
    algorithm = run_config.model.algorithm
    adaptation = run_config.adaptation
    chronobelief = (
        adaptation is not None and adaptation.optimizer.name == "chronobelief"
    )
    expected_suites = _expected_native_gpu_proof_suites(
        algorithm=algorithm,
        topology=topology,
        chronobelief=chronobelief,
    )
    if topology == "tp1_dp1":
        if distributed:
            raise ValueError("TP1 formal serving cannot consume a distributed proof")
    else:
        if (
            len(distributed) != 1
            or distributed[0].topology_mode != topology
            or distributed[0].gpu_uuids != subject.gpu_uuids
        ):
            raise ValueError("distributed formal serving proof coverage is not exact")
    native_by_suite = {row.suite_id: row for row in native}
    if (
        len(native_by_suite) != len(native)
        or set(native_by_suite) != expected_suites
        or any(
            row.topology_mode != topology or row.gpu_uuids != subject.gpu_uuids
            for row in native
        )
    ):
        raise ValueError("formal serving native GPU proof coverage is not exact")
    if "native_hot_path_tp1" in expected_suites and (
        native_by_suite["native_hot_path_tp1"].backend_capabilities
        != ("graph_hot_path", "native_itl")
    ):
        raise ValueError("formal serving native hot-path capabilities differ")
    if "chronobelief_gpu_parity" in expected_suites:
        assert adaptation is not None
        assert adaptation.chronobelief_release_capability_sha256 is not None
        assert adaptation.chronobelief_gpu_proof_sha256 is not None
        require_chronobelief_gpu_proof(
            claimed_source_capability_sha256=(
                adaptation.chronobelief_release_capability_sha256
            ),
            claimed_gpu_proof_sha256=adaptation.chronobelief_gpu_proof_sha256,
            verified_gpu_proof=native_by_suite["chronobelief_gpu_parity"],
            expected_source_identity_sha256=expected_source,
            expected_inventory_sha256=subject.inventory_sha256,
            expected_gpu_uuids=subject.gpu_uuids,
        )
    elif adaptation is not None and any(
        value is not None
        for value in (
            adaptation.chronobelief_release_capability_sha256,
            adaptation.chronobelief_gpu_proof_sha256,
        )
    ):
        raise ValueError("non-ChronoBelief execution carries Chrono GPU authority")
    if algorithm == "EAGLE3" and adaptation is not None:
        if adaptation.eagle3_native_gpu_proof_sha256 is None:
            raise ValueError("EAGLE3 formal execution lacks native GPU authority")
        if (
            native_receipt_sha256s.get("eagle3_tp1")
            != adaptation.eagle3_native_gpu_proof_sha256
        ):
            raise ValueError("EAGLE3 execution claims another native GPU proof")
    proof_ids = tuple(sorted(row.sha256 for row in all_proofs))
    if len(proof_ids) != len(set(proof_ids)):
        raise ValueError("formal serving reuses a runtime GPU proof")
    return (
        proof_ids,
        next(iter(hardware)),
        tuple(sorted(native, key=lambda row: (row.suite_id, row.sha256))),
        tuple(sorted(distributed, key=lambda row: (row.topology_mode, row.sha256))),
    )


def _revalidate_nextn_tp2_authority(
    *,
    source_input: object | None,
    subject: FormalServingExecutionSubject,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    run_config: RunConfig,
    content_verification_receipt: ContentVerificationReceipt,
    native_gpu_proofs: tuple[VerifiedNativeRuntimeGpuProof, ...],
    distributed_gpu_proofs: tuple[VerifiedDistributedRuntimeGpuProof, ...],
    now_ns: int,
) -> VerifiedNextNTp2Authority | None:
    """Join E6's two-model source DAG with the exact live proof union."""

    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
    )

    requires_authority = (
        run_config.model.algorithm == "NEXTN" and subject.topology_mode == "tp2_dp1"
    )
    if not requires_authority:
        if source_input is not None:
            raise ValueError("non-NEXTN execution carries a NEXTN TP2 authority")
        return None
    if materialization.stage != "E6" or type(source_input) is not (
        E6NextnModelAuthorityInput
    ):
        raise TypeError("E6 NEXTN TP2 execution requires its exact authority input")
    source_input.__post_init__()
    cells = {
        cell.cell_id: cell
        for cell in materialization.cells
        if cell.task != "immutable_metadata_interface_and_fit_preflight"
    }
    cell = cells.get(subject.materialized_cell_id)
    if cell is None or cell.backend != "NEXTN" or cell.model != source_input.model:
        raise ValueError("NEXTN TP2 authority input names another materialized cell")
    dimensions = dict(cell.dimensions)
    if (
        source_input.model != run_config.model.target
        or source_input.target_member_id != dimensions.get("target_member_id")
        or source_input.drafter_member_id != dimensions.get("drafter_member_id")
        or source_input.expected_interface_sha256 != dimensions.get("interface_sha256")
        or source_input.expected_topology_sha256
        != dimensions.get("topology_authority_sha256")
        or source_input.expected_source_adapter_version
        != dimensions.get("source_adapter_version")
    ):
        raise ValueError("NEXTN TP2 authority input differs from materialized lineage")
    verified = validate_nextn_tp2_dynamic_authority_artifact(
        source_input.artifact_path,
        expected_inventory_sha256=subject.inventory_sha256,
        expected_registry_sha256=protocol_lock.registry_sha256,
        expected_root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        expected_interface_sha256=source_input.expected_interface_sha256,
        expected_topology_sha256=source_input.expected_topology_sha256,
        expected_source_adapter_version=source_input.expected_source_adapter_version,
        expected_target_member_id=source_input.target_member_id,
        expected_drafter_member_id=source_input.drafter_member_id,
        now_ns=now_ns,
    )
    if type(verified) is not VerifiedNextNTp2Authority:
        raise TypeError("NEXTN TP2 verifier returned a foreign authority token")
    nextn_native = tuple(
        row for row in native_gpu_proofs if row.suite_id == "nextn_tp2"
    )
    if (
        len(nextn_native) != 1
        or len(distributed_gpu_proofs) != 1
        or verified.sha256 != dimensions.get("e6_verified_authority_sha256")
        or verified.native_gpu_proof_sha256 != nextn_native[0].sha256
        or verified.native_gpu_proof_sha256 != dimensions.get("native_gpu_proof_sha256")
        or verified.distributed_gpu_proof_sha256 != distributed_gpu_proofs[0].sha256
        or verified.distributed_gpu_proof_sha256
        != dimensions.get("distributed_gpu_proof_sha256")
        or verified.content_verification_receipt_sha256
        != content_verification_receipt.sha256
        or verified.content_verification_receipt_sha256
        != dimensions.get("content_verification_receipt_sha256")
        or verified.gpu_uuids != subject.gpu_uuids
    ):
        raise ValueError("NEXTN TP2 authority differs from live proof/content union")
    return verified


def verify_formal_serving_execution_binding(
    subject: FormalServingExecutionSubject,
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    run_config: RunConfig,
    inventory: GpuInventory,
    tts_authority: TtsCalibrationAuthority | None,
    signed_tts_seal: SignedTtsCalibrationSeal | None = None,
    e1_recipe_anchor_authority: E1RecipeAnchorAuthority | None = None,
    e2_recipe_grid_authority: E2RecipeGridAuthority | None = None,
    lightcone_recipe: E2CandidateRecipe | None = None,
    stage_source: VerifiedFormalStageMaterializationSource | None = None,
    now_ns: int,
    content_verification_receipt: ContentVerificationReceipt | None = None,
    registry_verification_receipt: object | None = None,
    nextn_tp2_authority_input: object | None = None,
    current_e0_eagle3_proof_row: _VerifiedCurrentE0Eagle3ProofRow | None = None,
) -> VerifiedFormalServingExecutionBinding:
    """Deep-open runtime proofs and return the only execution-authorizing value."""

    if type(subject) is not FormalServingExecutionSubject:
        raise TypeError("formal serving verifier requires an exact subject")
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal serving verifier requires exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal serving verifier requires exact materialization")
    expected = prepare_formal_serving_execution_subject(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        materialization=materialization,
        materialized_cell_id=subject.materialized_cell_id,
        run_config=run_config,
        inventory=inventory,
        gpu_uuids=subject.gpu_uuids,
        runtime_gpu_proof_artifact_paths=tuple(
            row.absolute_path for row in subject.runtime_gpu_proof_artifacts
        ),
        run_id=subject.execution_identity.run_id,
        run_nonce_sha256=subject.execution_identity.run_nonce_sha256,
        attempt_id=subject.execution_identity.attempt_id,
        tts_authority=tts_authority,
        signed_tts_seal=signed_tts_seal,
        e1_recipe_anchor_authority=e1_recipe_anchor_authority,
        e2_recipe_grid_authority=e2_recipe_grid_authority,
        lightcone_recipe=lightcone_recipe,
        stage_source=stage_source,
        now_ns=now_ns,
        content_verification_receipt=content_verification_receipt,
        registry_verification_receipt=registry_verification_receipt,
        current_e0_eagle3_proof_row=current_e0_eagle3_proof_row,
    )
    if subject != expected:
        raise ValueError(
            "formal serving subject was not deterministically rebuilt from authorities"
        )
    if content_verification_receipt is None:
        raise FormalStageExecutionBlocked(
            "durable_content_verification_receipt_missing"
        )
    (
        proof_ids,
        hardware_envelope_sha256,
        verified_native_gpu_proofs,
        verified_distributed_gpu_proofs,
    ) = _revalidate_gpu_proofs(
        subject,
        protocol_lock=protocol_lock,
        run_config=run_config,
        now_ns=now_ns,
    )
    verified_nextn_tp2_authority = _revalidate_nextn_tp2_authority(
        source_input=nextn_tp2_authority_input,
        subject=subject,
        protocol_lock=protocol_lock,
        materialization=materialization,
        run_config=run_config,
        content_verification_receipt=content_verification_receipt,
        native_gpu_proofs=verified_native_gpu_proofs,
        distributed_gpu_proofs=verified_distributed_gpu_proofs,
        now_ns=now_ns,
    )
    return VerifiedFormalServingExecutionBinding(
        subject=subject,
        run_config=run_config,
        runtime_gpu_proof_sha256s=proof_ids,
        verified_native_gpu_proofs=verified_native_gpu_proofs,
        verified_distributed_gpu_proofs=verified_distributed_gpu_proofs,
        verified_nextn_tp2_authority=verified_nextn_tp2_authority,
        hardware_envelope_sha256=hardware_envelope_sha256,
        _construction_seal=_VERIFIED_FORMAL_SERVING_EXECUTION_SEAL,
    )


def _revalidate_current_e0_eagle3_proof_row(
    *,
    source: object,
    cell: MaterializedCell,
    launch: object,
    run_config: RunConfig,
    runtime_gpu_proof_artifact_paths: tuple[str, ...],
) -> _VerifiedCurrentE0Eagle3ProofRow | None:
    """Join current E0 auxiliary evidence to one adaptive serving config.

    This is deliberately separate from the legacy signed-stage mapper.  The
    current trusted workflow must reopen the trusted 12/36/108 publication,
    the exact interface receipt and terminal, the task proof row, and the
    native artifact whose nested receipt that row names.
    """

    from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
        e0_eagle3_runtime_authority_for_task,
        e0_eagle3_runtime_proof_row_for_task,
        load_e0_compatibility_probe_terminal,
        load_e0_prepared_model_backend_interface_receipt,
        revalidate_trusted_e0_compatibility_bundle_value,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorExecutionSource,
    )
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest
    from lightcone_spec.runtime.readiness import NativeRuntimeGpuProofArtifact

    adaptive_eagle3 = (
        run_config.model.algorithm == "EAGLE3" and run_config.adaptation is not None
    )
    if not adaptive_eagle3:
        return None
    if (
        type(source) is not FormalSingleOperatorExecutionSource
        or source.stage != "E0"
        or cell.stage != "E0"
        or cell.backend != "EAGLE3"
        or type(launch) is not CompileLaunchManifest
    ):
        raise ValueError("current E0 EAGLE3 proof-row scope differs")
    auxiliary = source.auxiliary_source_binding("e0_compatibility")
    publication = revalidate_trusted_e0_compatibility_bundle_value(
        auxiliary.reopen(label="current E0 EAGLE3 compatibility auxiliary")
    )
    decisions = tuple(
        row
        for row in publication.compatibility.decisions
        if (row.model, row.backend, row.task) == (cell.model, "EAGLE3", cell.task)
    )
    if len(decisions) != 1 or decisions[0].disposition != "VALID":
        raise ValueError("current E0 EAGLE3 cell lacks a VALID compatibility row")

    interface_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.interface_receipts
        if (
            (raw := binding.reopen()).get("model") == cell.model
            and raw.get("backend") == "EAGLE3"
        )
    )
    terminal_bindings = tuple(
        binding
        for binding in publication.evidence_manifest.probe_terminals
        if (
            (raw := binding.reopen()).get("model") == cell.model
            and raw.get("backend") == "EAGLE3"
            and raw.get("task") == cell.task
        )
    )
    if len(interface_bindings) != 1 or len(terminal_bindings) != 1:
        raise ValueError("current E0 EAGLE3 path coverage is not unique")
    interface = load_e0_prepared_model_backend_interface_receipt(
        interface_bindings[0].absolute_path
    )
    terminal = load_e0_compatibility_probe_terminal(terminal_bindings[0].absolute_path)
    proof_row = e0_eagle3_runtime_proof_row_for_task(
        interface,
        task=cell.task,
        terminal=terminal,
    )
    claims = e0_eagle3_runtime_authority_for_task(
        interface,
        task=cell.task,
        terminal=terminal,
    )
    adaptation = run_config.adaptation
    assert adaptation is not None
    if any(getattr(adaptation, name) != value for name, value in claims.items()):
        raise ValueError("current E0 EAGLE3 adaptive claims differ from proof row")
    if (
        interface.schema_version not in {2, 3}
        or interface.support_status != "READY"
        or interface.compile_launch_manifest is None
        or terminal.schema_version != interface.schema_version
        or terminal.disposition != "VALID"
        or terminal.interface_receipt_sha256 != interface.sha256
        or terminal.eagle3_runtime_proof_row_sha256 != proof_row.sha256
        or interface.target_model_id != run_config.model.target
        or interface.target_revision != run_config.model.target_revision
        or interface.drafter_model_id != run_config.model.drafter
        or interface.drafter_revision != run_config.model.drafter_revision
        or interface.tokenizer_model_id != launch.tokenizer_model_id
        or interface.tokenizer_revision != launch.tokenizer_revision
        or interface.target_member_sha256 != launch.target_content_member_id
        or interface.drafter_member_sha256 != launch.drafter_content_member_id
        or interface.tokenizer_member_sha256 != launch.tokenizer_content_member_id
    ):
        raise ValueError("current E0 EAGLE3 interface/config identity differs")

    execution = proof_row.execution_authority.reopen()
    if (
        type(execution) is not dict
        or execution.get("task") != cell.task
        or execution.get("target_revision") != run_config.model.target_revision
        or execution.get("drafter_revision") != run_config.model.drafter_revision
        or execution.get("interface_sha256") != decisions[0].interface_sha256
        or execution.get("inventory_sha256") != launch.inventory_sha256
        or tuple(execution.get("gpu_uuids", ())) != launch.gpu_uuids
    ):
        raise ValueError("current E0 EAGLE3 execution authority scope differs")

    if interface.schema_version == 2:
        native_matches = []
        for path in runtime_gpu_proof_artifact_paths:
            binding = CanonicalJsonProofBinding.bind(path)
            raw = binding.reopen()
            if type(raw) is not dict or raw.get("kind") != (
                "lightcone_native_runtime_gpu_proof_artifact"
            ):
                continue
            artifact = NativeRuntimeGpuProofArtifact.from_dict(raw)
            if artifact.receipt == proof_row.native_gpu_proof:
                native_matches.append((binding, artifact))
        if len(native_matches) != 1:
            raise ValueError(
                "current E0 EAGLE3 task row lacks one exact native proof artifact"
            )
        if native_matches[0][1].receipt.semantic_sha256 != (
            proof_row.native_gpu_proof_sha256
        ):
            raise ValueError("current E0 EAGLE3 native receipt identity differs")
    else:
        native = proof_row.native_gpu_proof.reopen()
        if (
            type(native) is not dict
            or native.get("kind")
            != "trusted_single_operator_e0_eagle3_postprobe_native_gpu_proof"
            or native.get("inventory_sha256") != launch.inventory_sha256
            or tuple(native.get("gpu_uuids", ())) != launch.gpu_uuids
        ):
            raise ValueError("fresh E0 EAGLE3 native probe scope differs")
    return _VerifiedCurrentE0Eagle3ProofRow(
        execution_source_sha256=source.sha256,
        compatibility_bundle_sha256=publication.bundle["bundle_sha256"],
        interface_receipt_sha256=interface.sha256,
        terminal_sha256=terminal.sha256,
        proof_row_sha256=proof_row.sha256,
        execution_authority_sha256=proof_row.execution_authority_sha256,
        compatibility_authority_sha256=proof_row.compatibility_authority_sha256,
        model_selector_sha256=proof_row.model_selector_sha256,
        native_gpu_receipt_sha256=proof_row.native_gpu_proof_sha256,
        _construction_seal=_VERIFIED_CURRENT_E0_EAGLE3_PROOF_ROW_SEAL,
    )


def verify_formal_single_operator_execution_binding(
    *,
    execution_source_path: str | Path,
    materialized_cell_id: str,
    formal_runtime_authority_manifest_path: str | Path,
    compile_launch_manifest_path: str | Path,
    inventory_path: str | Path,
    content_verification_receipt_path: str | Path,
    runtime_gpu_proof_artifact_paths: tuple[str, ...],
    tts_calibration_authority_artifact_path: str | Path | None = None,
    e1_recipe_anchor_authority_artifact_path: str | Path | None = None,
    formal_registry_verification_receipt_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    now_ns: int | None = None,
) -> FormalSingleOperatorExecutionBinding:
    """Build the trusted current-only execution binding from durable sources.

    The API intentionally has no RunConfig, recipe digest, method, topology,
    GPU UUID, port, argv, run ID, nonce, or attempt arguments.  Those values are
    reconstructed from the exact current materialization and the source-owned
    :class:`CompileLaunchManifest`; the existing formal serving verifier then
    performs the scientific and GPU-proof joins.

    E3a and TTS-Cal are directly reachable.  E1/E2 additionally require the
    exact durable registry receipt (and E1's path-bound anchor artifact).
    E4 screen/local must use the source-owned current-stage launch mapper and
    an explicit clean repository root; E4 profiler and later stages remain
    fail-closed rather than falling back to the generic downstream mapper.
    """

    import time

    from lightcone_spec.experiments.formal_method_authority import (
        load_tts_calibration_authority_artifact,
    )
    from lightcone_spec.experiments.formal_registry import (
        formal_runtime_authority_manifest_from_dict,
        protocol_lock_from_dict,
        stage_materialization_receipt_from_dict,
    )
    from lightcone_spec.experiments.formal_registry_layers import (
        load_formal_registry_verification_receipt_path,
    )
    from lightcone_spec.experiments.formal_single_operator_stages import (
        load_formal_single_operator_execution_source,
    )
    from lightcone_spec.experiments.stage_materialization import (
        default_e2_recipe_grid_authority,
    )
    from lightcone_spec.runtime.compile_runner import CompileLaunchManifest

    replay_ns = time.time_ns() if now_ns is None else now_ns
    if type(replay_ns) is not int or replay_ns < 1:
        raise ValueError("single-operator execution verification time is invalid")

    execution_source_binding = CanonicalJsonProofBinding.bind(execution_source_path)
    source = load_formal_single_operator_execution_source(
        execution_source_binding.absolute_path
    )
    protocol_lock = protocol_lock_from_dict(
        source.protocol_lock_source.reopen(
            label="single-operator execution ProtocolLock"
        )
    )
    materialization = stage_materialization_receipt_from_dict(
        source.materialization_source.reopen(
            label="single-operator execution materialization"
        )
    )
    cells = {cell.cell_id: cell for cell in materialization.cells}
    cell = cells.get(materialized_cell_id)
    if cell is None:
        raise ValueError("single-operator execution cell is outside materialization")
    if (
        source.stage != materialization.stage
        or source.materialization_sha256 != materialization.sha256
        or source.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("single-operator current source differs from cell lineage")
    e4_headline = (
        materialization.stage == "E4"
        and source.node in {"e4_screen", "e4_local"}
        and cell.task
        in {
            "mechanism_strength2_screen_headline",
            "winner_neighborhood_local_factorial_headline",
        }
    )
    if materialization.stage in _DOWNSTREAM_SERVING_STAGES and not e4_headline:
        raise FormalStageExecutionBlocked(
            "current_single_operator_downstream_mapper_not_implemented"
        )
    if e4_headline and repository_root is None:
        raise FormalStageExecutionBlocked(
            "current_single_operator_e4_clean_repository_root_missing"
        )

    runtime_binding = CanonicalJsonProofBinding.bind(
        formal_runtime_authority_manifest_path
    )
    runtime_manifest = formal_runtime_authority_manifest_from_dict(
        runtime_binding.reopen()
    )
    if (
        runtime_manifest.sha256 != source.runtime_authority_manifest_sha256
        or runtime_manifest.sha256
        != protocol_lock.formal_runtime_authority_manifest_sha256
    ):
        raise ValueError(
            "single-operator runtime authority differs from current source"
        )

    inventory_binding = CanonicalJsonProofBinding.bind(inventory_path)
    inventory = GpuInventory.from_dict(inventory_binding.reopen())
    if inventory_binding.semantic_sha256 != inventory.sha256:
        raise ValueError("single-operator inventory canonical identity differs")

    content_binding = CanonicalJsonProofBinding.bind(content_verification_receipt_path)
    content_receipt = ContentVerificationReceipt.from_dict(content_binding.reopen())
    if content_binding.semantic_sha256 != content_receipt.sha256:
        raise ValueError("single-operator content receipt canonical identity differs")
    content_receipt.revalidate(current_ns=replay_ns)

    launch_binding = CanonicalJsonProofBinding.bind(compile_launch_manifest_path)
    launch = CompileLaunchManifest.load(launch_binding.absolute_path)
    if launch_binding.semantic_sha256 != launch.sha256:
        raise ValueError("single-operator compile launch canonical identity differs")
    run_config = load_run_config(launch.run_config_path)
    config_sha256 = run_config_sha256(run_config)
    for gpu_uuid in launch.gpu_uuids:
        inventory.device(gpu_uuid)
    if (
        launch.run_config_semantic_sha256 != config_sha256
        or launch.inventory_sha256 != inventory.sha256
        or launch.target_model_id != run_config.model.target
        or launch.target_revision != run_config.model.target_revision
        or launch.sampling_profile_sha256 != run_config.runtime.sampling_profile_sha256
    ):
        raise ValueError("single-operator compile launch differs from config/inventory")
    if run_config.method == "target_only":
        if launch.drafter_model_id is not None:
            raise ValueError("Target-only single-operator launch includes a drafter")
    elif (
        launch.drafter_model_id != run_config.model.drafter
        or launch.drafter_revision != run_config.model.drafter_revision
    ):
        raise ValueError("single-operator drafter launch differs from RunConfig")

    current_e0_eagle3_proof_row = _revalidate_current_e0_eagle3_proof_row(
        source=source,
        cell=cell,
        launch=launch,
        run_config=run_config,
        runtime_gpu_proof_artifact_paths=runtime_gpu_proof_artifact_paths,
    )

    e4_context = None
    stage_source = None
    lightcone_recipe = None
    if e4_headline:
        from lightcone_spec.experiments.formal_single_operator_e4_execution import (
            FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256,
            revalidate_formal_single_operator_e4_compile_launch,
        )

        assert repository_root is not None
        e4_context = revalidate_formal_single_operator_e4_compile_launch(
            execution_source_path=execution_source_binding.absolute_path,
            materialized_cell_id=cell.cell_id,
            repository_root=repository_root,
            inventory_path=inventory_binding.absolute_path,
            compile_launch_manifest_path=launch_binding.absolute_path,
        )
        if (
            e4_context.execution_source != source
            or e4_context.materialization != materialization
            or e4_context.cell != cell
            or e4_context.launch != launch
            or e4_context.run_config != run_config
        ):
            raise ValueError("single-operator E4 mapper replay differs")
        lightcone_recipe = e4_context.lightcone_recipe
        stage_source = _seal_rebuilt_stage_source(
            expected=materialization,
            rebuilt=e4_context.materialization,
            phase=source.phase,
            authority_sha256s=(
                FORMAL_SINGLE_OPERATOR_E4_EXECUTION_PROTOCOL_SHA256,
                source.sha256,
                e4_context.predecessor.artifact.sha256,
                e4_context.predecessor.decision.sha256,
                lightcone_recipe.sha256,
            ),
        )

    tts_authority: TtsCalibrationAuthority | None = None
    if tts_calibration_authority_artifact_path is not None:
        tts_artifact = load_tts_calibration_authority_artifact(
            tts_calibration_authority_artifact_path
        )
        tts_authority = tts_artifact.authority
        if tts_authority.sha256 != protocol_lock.tts_calibration_authority_sha256:
            raise ValueError("single-operator TTS authority differs from ProtocolLock")

    e1_anchor: E1RecipeAnchorAuthority | None = None
    if e1_recipe_anchor_authority_artifact_path is not None:
        e1_anchor = load_e1_recipe_anchor_authority_artifact(
            e1_recipe_anchor_authority_artifact_path
        ).authority
        if e1_anchor.sha256 != protocol_lock.e1_recipe_anchor_authority_sha256:
            raise ValueError("single-operator E1 anchor differs from ProtocolLock")

    registry_receipt: object | None = None
    signed_tts_seal: SignedTtsCalibrationSeal | None = None
    if formal_registry_verification_receipt_path is not None:
        registry_receipt = load_formal_registry_verification_receipt_path(
            formal_registry_verification_receipt_path,
            now_ns=replay_ns,
        )
        candidates = tuple(
            seal
            for seal in registry_receipt.cumulative_signed_tts_calibration_seals
            if seal.payload.protocol_lock_sha256 == protocol_lock.sha256
            and seal.payload.authority_sha256
            == protocol_lock.tts_calibration_authority_sha256
        )
        if len(candidates) != 1:
            raise ValueError("single-operator registry lacks one exact frozen TTS seal")
        signed_tts_seal = candidates[0]

    if materialization.stage == "TTS-Cal" and tts_authority is None:
        raise TypeError("TTS-Cal single-operator execution lacks TTS authority")
    if materialization.stage == "E1" and (
        tts_authority is None
        or e1_anchor is None
        or signed_tts_seal is None
        or registry_receipt is None
    ):
        raise TypeError("E1 single-operator execution lacks sealed recipe sources")
    if materialization.stage == "E2" and (
        tts_authority is None or signed_tts_seal is None or registry_receipt is None
    ):
        raise TypeError("E2 single-operator execution lacks sealed recipe sources")

    identity_seed = content_sha256(
        {
            "schema_version": 1,
            "kind": "formal_single_operator_execution_identity_seed",
            "protocol_sha256": (
                FORMAL_SINGLE_OPERATOR_EXECUTION_BINDING_PROTOCOL_SHA256
            ),
            "execution_source_sha256": source.sha256,
            "materialized_cell_id": cell.cell_id,
            "compile_launch_manifest_sha256": launch.sha256,
            "attempt_index": 0,
        }
    )
    run_id = f"formal-single-{source.node}-{cell.cell_id[:16]}-{identity_seed[:16]}"
    run_nonce_sha256 = content_sha256(
        {
            "kind": "formal_single_operator_run_nonce",
            "identity_seed_sha256": identity_seed,
        }
    )
    attempt_id = "attempt-0000"

    grid = (
        default_e2_recipe_grid_authority()
        if materialization.stage in {"E2", "E4"}
        else None
    )
    if (
        grid is not None
        and grid.sha256 != protocol_lock.e2_recipe_grid_authority_sha256
    ):
        raise ValueError("single-operator E2 grid differs from ProtocolLock")
    subject = prepare_formal_serving_execution_subject(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        materialized_cell_id=cell.cell_id,
        run_config=run_config,
        inventory=inventory,
        gpu_uuids=launch.gpu_uuids,
        runtime_gpu_proof_artifact_paths=runtime_gpu_proof_artifact_paths,
        run_id=run_id,
        run_nonce_sha256=run_nonce_sha256,
        attempt_id=attempt_id,
        tts_authority=tts_authority,
        signed_tts_seal=signed_tts_seal,
        e1_recipe_anchor_authority=e1_anchor,
        e2_recipe_grid_authority=grid,
        lightcone_recipe=lightcone_recipe,
        stage_source=stage_source,
        now_ns=replay_ns,
        content_verification_receipt=content_receipt,
        registry_verification_receipt=registry_receipt,
        current_e0_eagle3_proof_row=current_e0_eagle3_proof_row,
    )
    verified = verify_formal_serving_execution_binding(
        subject,
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        run_config=run_config,
        inventory=inventory,
        tts_authority=tts_authority,
        signed_tts_seal=signed_tts_seal,
        e1_recipe_anchor_authority=e1_anchor,
        e2_recipe_grid_authority=grid,
        lightcone_recipe=lightcone_recipe,
        stage_source=stage_source,
        now_ns=replay_ns,
        content_verification_receipt=content_receipt,
        registry_verification_receipt=registry_receipt,
        current_e0_eagle3_proof_row=current_e0_eagle3_proof_row,
    )
    optional_sources = tuple(
        CanonicalJsonProofBinding.bind(path)
        for path in (
            tts_calibration_authority_artifact_path,
            e1_recipe_anchor_authority_artifact_path,
            formal_registry_verification_receipt_path,
        )
        if path is not None
    )
    for before in (
        execution_source_binding,
        runtime_binding,
        inventory_binding,
        content_binding,
        launch_binding,
        *optional_sources,
    ):
        if CanonicalJsonProofBinding.bind(before.absolute_path) != before:
            raise RuntimeError(
                "single-operator execution source changed while verified"
            )
    return FormalSingleOperatorExecutionBinding(
        verified_binding=verified,
        execution_source=execution_source_binding,
        execution_source_sha256=source.sha256,
        compile_launch_manifest=launch_binding,
        inventory_source=inventory_binding,
        content_verification_receipt_source=content_binding,
        runtime_authority_manifest_source=runtime_binding,
        tts_calibration_authority_source=(
            None
            if tts_calibration_authority_artifact_path is None
            else CanonicalJsonProofBinding.bind(tts_calibration_authority_artifact_path)
        ),
        e1_recipe_anchor_authority_source=(
            None
            if e1_recipe_anchor_authority_artifact_path is None
            else CanonicalJsonProofBinding.bind(
                e1_recipe_anchor_authority_artifact_path
            )
        ),
        formal_registry_verification_receipt_source=(
            None
            if formal_registry_verification_receipt_path is None
            else CanonicalJsonProofBinding.bind(
                formal_registry_verification_receipt_path
            )
        ),
        repository_root=(
            None
            if repository_root is None
            else str(Path(repository_root).resolve(strict=False))
        ),
        _construction_seal=_FORMAL_SINGLE_OPERATOR_EXECUTION_SEAL,
    )


__all__ = [
    "E1_RECIPE_ANCHOR_ARTIFACT_KIND",
    "E1_RECIPE_ANCHOR_AUTHORITY_ID",
    "FORMAL_SERVING_EXECUTION_BINDING_PROTOCOL_SHA256",
    "FORMAL_SERVING_EXECUTION_REBUILD_PROTOCOL_SHA256",
    "FORMAL_SERVING_EXECUTION_RUNNER_SHA256",
    "FORMAL_SERVING_EXECUTION_TEST_SET_SHA256",
    "FORMAL_SINGLE_OPERATOR_EXECUTION_BINDING_PROTOCOL_SHA256",
    "E0FinalStageSourceRebuildInputs",
    "E0PilotStageSourceRebuildInputs",
    "E0TuningStageSourceRebuildInputs",
    "E1RecipeAnchorAuthority",
    "E1RecipeAnchorAuthorityArtifact",
    "E1aStageSourceRebuildInputs",
    "E3bFinalStageSourceRebuildInputs",
    "E3bPilotStageSourceRebuildInputs",
    "E4LocalStageSourceRebuildInputs",
    "E4ProfilerStageSourceRebuildInputs",
    "E4ScreenStageSourceRebuildInputs",
    "E5FinalStageSourceRebuildInputs",
    "E5PilotStageSourceRebuildInputs",
    "E6FinalStageSourceRebuildInputs",
    "E6PilotStageSourceRebuildInputs",
    "FormalOptimizerRecipe",
    "FormalServingExecutionBinding",
    "FormalServingExecutionRebuildInput",
    "FormalServingExecutionSubject",
    "FormalSingleOperatorExecutionBinding",
    "FormalStageExecutionBlocked",
    "FormalStageSourceRebuildInput",
    "VerifiedFormalServingExecutionBinding",
    "VerifiedFormalStageMaterializationSource",
    "bind_formal_serving_execution_rebuild_input",
    "bind_formal_stage_source_rebuild_input",
    "build_source_e1_recipe_anchor_authority_artifact",
    "e1_recipe_anchor_authority_from_dict",
    "e1_recipe_anchor_authority_to_dict",
    "load_e1_recipe_anchor_authority_artifact",
    "prepare_formal_serving_execution_subject",
    "publish_e1_recipe_anchor_authority_artifact",
    "publish_formal_stage_source_rebuild_input",
    "rebuild_formal_serving_execution_binding",
    "rebuild_formal_stage_materialization_source",
    "require_verified_formal_serving_execution_binding",
    "require_verified_formal_stage_materialization_source",
    "verify_e0_final_execution_source",
    "verify_e0_pilot_execution_source",
    "verify_e0_tuning_execution_source",
    "verify_e1a_execution_source",
    "verify_e3b_final_execution_source",
    "verify_e3b_pilot_execution_source",
    "verify_e4_local_execution_source",
    "verify_e4_profiler_execution_source",
    "verify_e4_screen_execution_source",
    "verify_e5_final_execution_source",
    "verify_e5_pilot_execution_source",
    "verify_e6_final_execution_source",
    "verify_e6_pilot_execution_source",
    "verify_formal_serving_execution_binding",
    "verify_formal_single_operator_execution_binding",
]
