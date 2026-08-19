"""Typed, signed decisions that authorize downstream stage materialization.

These receipts are deliberately separate from the materialized cell matrix.
They bind a source-owned reducer that can be reopened from local raw evidence;
a caller-authored digest or a duplicate JSON summary is never sufficient to
select a downstream width, load, model, or candidate set.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cached_property

from lightcone_spec.experiments.e1_stage_authority import (
    E1StagedParetoEvidenceManifest,
    reduce_e1_staged_pareto_from_proofs,
)
from lightcone_spec.experiments.e3a_stage_authority import (
    E3aStagedSelectionArtifact,
    SignedE3aStagedSelectionReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.industrial_analysis import (
    RawE2StageEvidenceManifest,
    reduce_e2_stage_from_raw,
    validate_raw_evidence_manifest_sidecars,
)
from lightcone_spec.experiments.planning import (
    E2CandidateIdentity,
    E2StageReductionArtifact,
)
from lightcone_spec.experiments.registry import (
    ExperimentReceipt,
    ExperimentRegistry,
)
from lightcone_spec.experiments.selection_authority import (
    E1ParetoReductionAuthority,
    E3aSelectionReductionAuthority,
)
from lightcone_spec.experiments.stage_materialization import (
    E2_OPTIMIZERS,
    E2_SCHEDULES,
    E1Geometry,
    E2CandidateRecipe,
    E2RecipeGridAuthority,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)


def _require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lower-case SHA-256")
    return value


def _require_text(name: str, value: object) -> str:
    if type(value) is not str or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be exact non-empty single-line text")
    return value


def _require_complete_coverage(
    coverage: StageCoverageReceipt,
    materialization: StageMaterializationReceipt,
) -> None:
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("stage selection requires all-COMPLETE upstream coverage")


def _require_signed_staged_source_registry(
    registry: ExperimentRegistry,
    *,
    expected_registry_sha256: str,
    stage: str,
) -> None:
    if type(registry) is not ExperimentRegistry:
        raise TypeError(f"{stage} source authority requires an exact registry")
    _require_sha256(f"{stage} expected registry", expected_registry_sha256)
    if (
        registry.materialization_mode != "signed_staged"
        or registry.sha256 != expected_registry_sha256
    ):
        raise ValueError(
            f"legacy eager {stage} authority is diagnostic and non-authorizing"
        )


@dataclass(frozen=True)
class E3aSelectionReceipt:
    """The exact E3a width/load decision reopened from raw capacity evidence."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    e3a_materialization_receipt_sha256: str
    e3a_coverage_receipt_sha256: str
    e3a_workload_authority_sha256: str
    reduction_authority_sha256: str
    source_selection_sha256: str
    model: str
    matched_width: int
    common_load: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only E3a selection receipt schema 1 is supported")
        for name in (
            "protocol_lock_sha256",
            "registry_sha256",
            "e3a_materialization_receipt_sha256",
            "e3a_coverage_receipt_sha256",
            "e3a_workload_authority_sha256",
            "reduction_authority_sha256",
            "source_selection_sha256",
        ):
            _require_sha256(f"E3a selection {name}", getattr(self, name))
        _require_text("E3a selection model", self.model)
        if type(self.matched_width) is not int or self.matched_width < 1:
            raise ValueError("E3a selected width must be a positive integer")
        if type(self.common_load) is not int or self.common_load < 1:
            raise ValueError("E3a selected common load must be a positive integer")
        reject_banned_model_identity(self)

    def validate_sources(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        reduction_authority: E3aSelectionReductionAuthority,
    ) -> None:
        """Deep-reopen the reducer and bind every selected scalar to it."""

        if type(protocol_lock) is not ProtocolLock:
            raise TypeError("E3a selection requires an exact ProtocolLock")
        if type(materialization) is not StageMaterializationReceipt:
            raise TypeError("E3a selection requires an exact materialization")
        if type(coverage) is not StageCoverageReceipt:
            raise TypeError("E3a selection requires an exact coverage receipt")
        if type(reduction_authority) is not E3aSelectionReductionAuthority:
            raise TypeError("E3a selection requires its exact reduction authority")
        if materialization.stage != "E3a":
            raise ValueError("E3a selection cannot consume another stage")
        _require_complete_coverage(coverage, materialization)
        selection = reduction_authority.revalidate()
        models = {cell.model for cell in materialization.cells}
        if len(models) != 1:
            raise ValueError("E3a selection requires one exact model identity")
        if (
            self.protocol_lock_sha256 != protocol_lock.sha256
            or self.registry_sha256 != protocol_lock.registry_sha256
            or selection.registry_sha256 != protocol_lock.registry_sha256
            or self.e3a_materialization_receipt_sha256 != materialization.sha256
            or self.e3a_coverage_receipt_sha256 != coverage.sha256
            or self.e3a_workload_authority_sha256
            != materialization.source_decision_sha256
            or self.reduction_authority_sha256 != reduction_authority.sha256
            or self.source_selection_sha256 != selection.sha256
            or self.model != next(iter(models))
            or self.matched_width != selection.width
            or self.common_load != selection.concurrency
        ):
            raise ValueError("E3a selection receipt differs from reopened evidence")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


def reduce_e3a_stage_selection_receipt(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    reduction_authority: E3aSelectionReductionAuthority,
) -> E3aSelectionReceipt:
    """Create the signable E3a decision only after a complete raw replay."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E3a selection requires an exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E3a selection requires an exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E3a selection requires an exact coverage receipt")
    if type(reduction_authority) is not E3aSelectionReductionAuthority:
        raise TypeError("E3a selection requires its exact reduction authority")
    _require_complete_coverage(coverage, materialization)
    selection = reduction_authority.revalidate()
    models = {cell.model for cell in materialization.cells}
    if len(models) != 1:
        raise ValueError("E3a selection requires one exact model identity")
    receipt = E3aSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        e3a_materialization_receipt_sha256=materialization.sha256,
        e3a_coverage_receipt_sha256=coverage.sha256,
        e3a_workload_authority_sha256=materialization.source_decision_sha256,
        reduction_authority_sha256=reduction_authority.sha256,
        source_selection_sha256=selection.sha256,
        model=next(iter(models)),
        matched_width=selection.width,
        common_load=selection.concurrency,
    )
    receipt.validate_sources(
        protocol_lock=protocol_lock,
        materialization=materialization,
        coverage=coverage,
        reduction_authority=reduction_authority,
    )
    return receipt


@dataclass(frozen=True)
class SignedE3aSelectionReceipt:
    payload: E3aSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        reduction_authority: E3aSelectionReductionAuthority,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int | None = None,
    ) -> E3aSelectionReceipt:
        if type(self.payload) is not E3aSelectionReceipt:
            raise TypeError("signed E3a selection payload has the wrong type")
        self.payload.validate_sources(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            reduction_authority=reduction_authority,
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


def _materialized_e1_geometries(
    materialization: StageMaterializationReceipt,
) -> tuple[E1Geometry, ...]:
    candidates = tuple(
        cell
        for cell in materialization.cells
        if cell.method_role == "LightCone-candidate"
    )
    geometries = {
        E1Geometry(
            scope=str(dict(cell.dimensions)["scope"]),
            parameterization=str(dict(cell.dimensions)["parameterization"]),  # type: ignore[arg-type]
            rank=(
                None
                if dict(cell.dimensions)["rank"] == "none"
                else int(dict(cell.dimensions)["rank"])
            ),
            alpha_over_rank=(
                None
                if dict(cell.dimensions)["alpha_over_rank"] == "none"
                else float(dict(cell.dimensions)["alpha_over_rank"])
            ),
        )
        for cell in candidates
    }
    if len(candidates) != 2 * len(geometries):
        raise ValueError("E1 materialization does not contain two anchors per geometry")
    return tuple(sorted(geometries, key=lambda row: row.sha256))


@dataclass(frozen=True)
class E1SurvivorSelectionReceipt:
    """Staged-native E1 Pareto survivors used to derive E2 round zero."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    e1_materialization_receipt_sha256: str
    e1_coverage_receipt_sha256: str
    e3a_selection_receipt_sha256: str
    staged_pareto_evidence_manifest_sha256: str
    staged_pareto_artifact_sha256: str
    inventory_sha256: str
    model: str
    frozen_tts_recipe_sha256: str
    surviving_geometries: tuple[E1Geometry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("only E1 survivor selection schema 2 is supported")
        for name in (
            "protocol_lock_sha256",
            "registry_sha256",
            "e1_materialization_receipt_sha256",
            "e1_coverage_receipt_sha256",
            "e3a_selection_receipt_sha256",
            "staged_pareto_evidence_manifest_sha256",
            "staged_pareto_artifact_sha256",
            "inventory_sha256",
            "frozen_tts_recipe_sha256",
        ):
            _require_sha256(f"E1 survivor {name}", getattr(self, name))
        _require_text("E1 survivor model", self.model)
        if (
            type(self.surviving_geometries) is not tuple
            or not self.surviving_geometries
            or any(type(row) is not E1Geometry for row in self.surviving_geometries)
            or tuple(row.sha256 for row in self.surviving_geometries)
            != tuple(sorted({row.sha256 for row in self.surviving_geometries}))
        ):
            raise ValueError("E1 survivor geometries are not sorted unique typed rows")
        reject_banned_model_identity(self)

    def validate_sources(
        self,
        *,
        protocol_lock: ProtocolLock,
        e1_materialization: StageMaterializationReceipt,
        e1_coverage: StageCoverageReceipt,
        e3a_selection_artifact: E3aStagedSelectionArtifact,
        signed_e3a_selection: SignedE3aStagedSelectionReceipt,
        e3a_policy: TrustedAttesterPolicy,
        expected_e3a_policy_sha256: str,
        pareto_evidence_manifest: E1StagedParetoEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        now_ns: int,
    ) -> None:
        if type(protocol_lock) is not ProtocolLock:
            raise TypeError("E1 survivor selection requires an exact ProtocolLock")
        if e1_materialization.stage != "E1":
            raise ValueError("E1 survivor selection cannot consume another stage")
        if type(pareto_evidence_manifest) is not E1StagedParetoEvidenceManifest:
            raise TypeError("E1 survivor selection requires staged-native evidence")
        _require_complete_coverage(e1_coverage, e1_materialization)
        e3a_selection = signed_e3a_selection.verify(
            artifact=e3a_selection_artifact,
            policy=e3a_policy,
            expected_policy_sha256=expected_e3a_policy_sha256,
            now_ns=now_ns,
        )
        pareto = reduce_e1_staged_pareto_from_proofs(
            protocol_lock=protocol_lock,
            materialization=e1_materialization,
            coverage=e1_coverage,
            e3a_selection_receipt_sha256=signed_e3a_selection.sha256,
            manifest=pareto_evidence_manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        materialized_geometries = _materialized_e1_geometries(e1_materialization)
        surviving = tuple(
            sorted(
                (
                    E1Geometry(
                        scope=row.scope,
                        parameterization=row.parameterization,  # type: ignore[arg-type]
                        rank=row.rank,
                        alpha_over_rank=row.alpha_over_rank,
                    )
                    for row in pareto.surviving_geometries
                ),
                key=lambda row: row.sha256,
            )
        )
        models = {cell.model for cell in e1_materialization.cells}
        tts_recipes = {
            cell.recipe_sha256
            for cell in e1_materialization.cells
            if cell.method_role == "TTS"
        }
        if len(models) != 1 or len(tts_recipes) != 1 or None in tts_recipes:
            raise ValueError("E1 survivor source has ambiguous model or TTS recipe")
        if not set(surviving) <= set(materialized_geometries):
            raise ValueError("E1 Pareto survivor is foreign to the materialized grid")
        if (
            self.protocol_lock_sha256 != protocol_lock.sha256
            or self.registry_sha256 != protocol_lock.registry_sha256
            or self.e1_materialization_receipt_sha256 != e1_materialization.sha256
            or self.e1_coverage_receipt_sha256 != e1_coverage.sha256
            or self.e3a_selection_receipt_sha256 != signed_e3a_selection.sha256
            or e1_materialization.source_decision_sha256 != signed_e3a_selection.sha256
            or e3a_selection.protocol_lock_sha256 != protocol_lock.sha256
            or e3a_selection.selection_artifact_sha256 != e3a_selection_artifact.sha256
            or self.staged_pareto_evidence_manifest_sha256
            != pareto_evidence_manifest.sha256
            or self.staged_pareto_artifact_sha256 != pareto.sha256
            or self.inventory_sha256 != pareto.inventory_sha256
            or pareto.e3a_selection_receipt_sha256 != signed_e3a_selection.sha256
            or self.model != next(iter(models))
            or self.frozen_tts_recipe_sha256 != next(iter(tts_recipes))
            or self.surviving_geometries != surviving
        ):
            raise ValueError(
                "E1 survivor receipt differs from staged-native Pareto evidence"
            )
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE1SurvivorSelectionReceipt:
    payload: E1SurvivorSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        e1_materialization: StageMaterializationReceipt,
        e1_coverage: StageCoverageReceipt,
        e3a_selection_artifact: E3aStagedSelectionArtifact,
        signed_e3a_selection: SignedE3aStagedSelectionReceipt,
        e3a_policy: TrustedAttesterPolicy,
        expected_e3a_policy_sha256: str,
        pareto_evidence_manifest: E1StagedParetoEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E1SurvivorSelectionReceipt:
        if type(self.payload) is not E1SurvivorSelectionReceipt:
            raise TypeError("signed E1 survivor payload has the wrong type")
        self.payload.validate_sources(
            protocol_lock=protocol_lock,
            e1_materialization=e1_materialization,
            e1_coverage=e1_coverage,
            e3a_selection_artifact=e3a_selection_artifact,
            signed_e3a_selection=signed_e3a_selection,
            e3a_policy=e3a_policy,
            expected_e3a_policy_sha256=expected_e3a_policy_sha256,
            pareto_evidence_manifest=pareto_evidence_manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
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


def reduce_e1_survivor_selection_receipt(
    *,
    registry_verification_receipt: object,
    protocol_lock: ProtocolLock,
    e1_materialization: StageMaterializationReceipt,
    e1_coverage: StageCoverageReceipt,
    pareto_evidence_manifest: E1StagedParetoEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E1SurvivorSelectionReceipt:
    """Reduce E1 only by deep-replaying the exact 67 controlled proof rows."""

    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError(
            "E1 survivor reduction requires durable registry verification receipt"
        )
    registry_manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or registry_manifest.protocol_lock_sha256 != protocol_lock.sha256
        or e1_materialization.sha256
        not in {
            row.materialization_receipt_sha256
            for row in registry_manifest.materializations
        }
    ):
        raise ValueError("E1 survivor registry receipt lacks exact E1 materialization")
    e3a_policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    e3a_artifacts = (
        registry_verification_receipt.cumulative_e3a_staged_selection_artifacts
    )
    signed_e3a_selections = (
        registry_verification_receipt.cumulative_signed_e3a_staged_selections
    )
    if len(e3a_artifacts) != 1 or len(signed_e3a_selections) != 1:
        raise ValueError("E1 survivor registry receipt lacks staged E3a source")
    e3a_selection_artifact = e3a_artifacts[0]
    signed_e3a_selection = signed_e3a_selections[0]
    signed_e3a_selection.verify(
        artifact=e3a_selection_artifact,
        policy=e3a_policy,
        expected_policy_sha256=e3a_policy.sha256,
        now_ns=now_ns,
    )
    _require_complete_coverage(e1_coverage, e1_materialization)
    if type(pareto_evidence_manifest) is not E1StagedParetoEvidenceManifest:
        raise TypeError("E1 survivor selection requires staged-native evidence")
    pareto = reduce_e1_staged_pareto_from_proofs(
        protocol_lock=protocol_lock,
        materialization=e1_materialization,
        coverage=e1_coverage,
        e3a_selection_receipt_sha256=signed_e3a_selection.sha256,
        manifest=pareto_evidence_manifest,
        execution_bindings=execution_bindings,
        now_ns=now_ns,
    )
    surviving = tuple(
        sorted(
            (
                E1Geometry(
                    scope=row.scope,
                    parameterization=row.parameterization,  # type: ignore[arg-type]
                    rank=row.rank,
                    alpha_over_rank=row.alpha_over_rank,
                )
                for row in pareto.surviving_geometries
            ),
            key=lambda row: row.sha256,
        )
    )
    models = {cell.model for cell in e1_materialization.cells}
    tts_recipes = {
        cell.recipe_sha256
        for cell in e1_materialization.cells
        if cell.method_role == "TTS"
    }
    if len(models) != 1 or len(tts_recipes) != 1 or None in tts_recipes:
        raise ValueError("E1 survivor source has ambiguous model or TTS recipe")
    if pareto.e3a_selection_receipt_sha256 != signed_e3a_selection.sha256:
        raise ValueError("E1 Pareto authority differs from sealed E3a selection")
    recipe = next(iter(tts_recipes))
    assert recipe is not None
    receipt = E1SurvivorSelectionReceipt(
        schema_version=2,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        e1_materialization_receipt_sha256=e1_materialization.sha256,
        e1_coverage_receipt_sha256=e1_coverage.sha256,
        e3a_selection_receipt_sha256=signed_e3a_selection.sha256,
        staged_pareto_evidence_manifest_sha256=pareto_evidence_manifest.sha256,
        staged_pareto_artifact_sha256=pareto.sha256,
        inventory_sha256=pareto.inventory_sha256,
        model=next(iter(models)),
        frozen_tts_recipe_sha256=recipe,
        surviving_geometries=surviving,
    )
    receipt.validate_sources(
        protocol_lock=protocol_lock,
        e1_materialization=e1_materialization,
        e1_coverage=e1_coverage,
        e3a_selection_artifact=e3a_selection_artifact,
        signed_e3a_selection=signed_e3a_selection,
        e3a_policy=e3a_policy,
        expected_e3a_policy_sha256=e3a_policy.sha256,
        pareto_evidence_manifest=pareto_evidence_manifest,
        execution_bindings=execution_bindings,
        now_ns=now_ns,
    )
    return receipt


@dataclass(frozen=True)
class E2StageReductionAuthority:
    """Path-bearing replay of one exact E2 successive-halving round."""

    schema_version: int
    registry: ExperimentRegistry
    e1_receipt: ExperimentReceipt
    e1_pareto_authority: E1ParetoReductionAuthority
    manifest: RawE2StageEvidenceManifest
    hardware_envelope: HardwareEnvelope
    inventory: GpuInventory
    prior_authority: E2StageReductionAuthority | None
    reduction_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 reduction authority schema 1 is supported")
        if type(self.registry) is not ExperimentRegistry:
            raise TypeError("E2 reduction authority requires an exact registry")
        _require_signed_staged_source_registry(
            self.registry,
            expected_registry_sha256=self.registry.sha256,
            stage="E2 reduction",
        )
        if type(self.e1_receipt) is not ExperimentReceipt:
            raise TypeError("E2 reduction authority requires an exact E1 receipt")
        if type(self.e1_pareto_authority) is not E1ParetoReductionAuthority:
            raise TypeError("E2 reduction authority requires path-bound E1 Pareto")
        if type(self.manifest) is not RawE2StageEvidenceManifest:
            raise TypeError("E2 reduction authority requires its exact raw manifest")
        if type(self.hardware_envelope) is not HardwareEnvelope:
            raise TypeError("E2 reduction authority requires exact hardware")
        if type(self.inventory) is not GpuInventory:
            raise TypeError("E2 reduction authority requires exact inventory")
        if self.manifest.stage_index == 0:
            if self.prior_authority is not None:
                raise ValueError("E2 round zero cannot bind a prior reduction")
        elif (
            type(self.prior_authority) is not E2StageReductionAuthority
            or self.prior_authority.manifest.stage_index
            != self.manifest.stage_index - 1
        ):
            raise ValueError("E2 reduction authority lacks its exact prior round")
        _require_sha256("E2 reduction authority digest", self.reduction_sha256)

    def revalidate(self) -> E2StageReductionArtifact:
        validate_raw_evidence_manifest_sidecars(self.manifest)
        pareto = self.e1_pareto_authority.revalidate()
        prior = (
            None if self.prior_authority is None else self.prior_authority.revalidate()
        )
        outputs = {row.name: row.content_sha256 for row in self.e1_receipt.outputs}
        if (
            self.e1_receipt.experiment != "E1"
            or outputs.get("dflash_pareto_set") != pareto.sha256
        ):
            raise ValueError("E2 reduction E1 receipt does not bind reopened Pareto")
        reduction = reduce_e2_stage_from_raw(
            registry=self.registry,
            e1_receipt=self.e1_receipt,
            pareto=pareto,
            stage_index=self.manifest.stage_index,
            cells=self.manifest.cells,
            hardware_envelope=self.hardware_envelope,
            inventory=self.inventory,
            prior_stage_reduction=prior,
            confirmation_data_visible=False,
        )
        if reduction.sha256 != self.reduction_sha256:
            raise RuntimeError("E2 reduction authority changed after binding")
        return reduction

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": "e2_stage_reduction_authority",
                "registry_sha256": self.registry.sha256,
                "e1_receipt_sha256": self.e1_receipt.sha256,
                "e1_pareto_authority_sha256": self.e1_pareto_authority.sha256,
                "raw_manifest_sha256": self.manifest.sha256,
                "hardware_envelope_sha256": content_sha256(self.hardware_envelope),
                "inventory_sha256": self.inventory.sha256,
                "inventory_source_receipt_sha256": (
                    self.inventory.source_receipt_sha256
                ),
                "prior_authority_sha256": (
                    None
                    if self.prior_authority is None
                    else self.prior_authority.sha256
                ),
                "reduction_sha256": self.reduction_sha256,
            }
        )


def bind_e2_stage_reduction_authority(
    *,
    registry: ExperimentRegistry,
    e1_receipt: ExperimentReceipt,
    e1_pareto_authority: E1ParetoReductionAuthority,
    manifest: RawE2StageEvidenceManifest,
    hardware_envelope: HardwareEnvelope,
    inventory: GpuInventory,
    prior_authority: E2StageReductionAuthority | None,
) -> E2StageReductionAuthority:
    """Bind one E2 raw round only after replaying its full prior prefix."""

    validate_raw_evidence_manifest_sidecars(manifest)
    pareto = e1_pareto_authority.revalidate()
    prior = None if prior_authority is None else prior_authority.revalidate()
    reduction = reduce_e2_stage_from_raw(
        registry=registry,
        e1_receipt=e1_receipt,
        pareto=pareto,
        stage_index=manifest.stage_index,
        cells=manifest.cells,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        prior_stage_reduction=prior,
        confirmation_data_visible=False,
    )
    authority = E2StageReductionAuthority(
        schema_version=1,
        registry=registry,
        e1_receipt=e1_receipt,
        e1_pareto_authority=e1_pareto_authority,
        manifest=manifest,
        hardware_envelope=hardware_envelope,
        inventory=inventory,
        prior_authority=prior_authority,
        reduction_sha256=reduction.sha256,
    )
    authority.revalidate()
    return authority


def _candidate_recipe_map(
    authority: E2StageReductionAuthority,
    *,
    grid: E2RecipeGridAuthority,
) -> dict[str, E2CandidateRecipe]:
    rows: dict[str, E2CandidateRecipe] = {}
    for cell in authority.registry.cells_for("E2"):
        if cell.identity.method not in {"tts", "l0"}:
            continue
        try:
            identity = E2CandidateIdentity.from_cell(cell)
        except ValueError:
            continue
        geometry = E1Geometry(
            scope=identity.scope,
            parameterization=identity.parameterization,  # type: ignore[arg-type]
            rank=identity.rank,
            alpha_over_rank=identity.alpha_over_rank,
        )
        recipe = E2CandidateRecipe(
            geometry=geometry,
            optimizer=identity.optimizer,
            schedule=identity.schedule,
            learning_rate=identity.learning_rate,
            optimizer_recipe_authority_sha256=(grid.optimizer_recipe_authority.sha256),
        )
        if identity.learning_rate not in grid.rates(
            optimizer=identity.optimizer,
            parameterization=identity.parameterization,
        ):
            raise ValueError("E2 raw candidate lies outside signed recipe grid")
        previous = rows.setdefault(identity.sha256, recipe)
        if previous != recipe:
            raise ValueError("E2 raw candidate identity maps ambiguously")
    return rows


@dataclass(frozen=True)
class E2RoundSelectionReceipt:
    """Exact survivors from raw E2 round ``k`` used to materialize ``k+1``."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    source_materialization_receipt_sha256: str
    source_coverage_receipt_sha256: str
    reduction_authority_sha256: str
    reduction_sha256: str
    source_round_index: int
    next_round_index: int
    survivor_recipes: tuple[E2CandidateRecipe, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 round selection schema 1 is supported")
        for name in (
            "protocol_lock_sha256",
            "registry_sha256",
            "source_materialization_receipt_sha256",
            "source_coverage_receipt_sha256",
            "reduction_authority_sha256",
            "reduction_sha256",
        ):
            _require_sha256(f"E2 round selection {name}", getattr(self, name))
        if (
            self.source_round_index not in range(3)
            or self.next_round_index != self.source_round_index + 1
        ):
            raise ValueError("E2 round selection must advance exactly one round")
        if type(self.survivor_recipes) is not tuple or tuple(
            row.sha256 for row in self.survivor_recipes
        ) != tuple(sorted({row.sha256 for row in self.survivor_recipes})):
            raise ValueError("E2 survivor recipes are not sorted unique")
        reject_banned_model_identity(self)

    def validate_sources(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        authority: E2StageReductionAuthority,
        grid: E2RecipeGridAuthority,
    ) -> None:
        if materialization.stage != "E2":
            raise ValueError("E2 round selection cannot consume another stage")
        _require_signed_staged_source_registry(
            authority.registry,
            expected_registry_sha256=protocol_lock.registry_sha256,
            stage="E2 reduction",
        )
        _require_complete_coverage(coverage, materialization)
        source_rounds = {
            dict(cell.dimensions).get("round") for cell in materialization.cells
        }
        if source_rounds != {self.source_round_index}:
            raise ValueError("E2 round selection source round differs")
        reduction = authority.revalidate()
        survivor = reduction.survivor_receipt
        if (
            survivor.status != "SURVIVORS"
            or survivor.stage_index != self.source_round_index
        ):
            raise ValueError("E2 raw reduction does not authorize a next round")
        mapping = _candidate_recipe_map(authority, grid=grid)
        try:
            survivors = tuple(
                sorted(
                    (
                        mapping[candidate_id]
                        for candidate_id in survivor.survivor_candidate_ids
                    ),
                    key=lambda row: row.sha256,
                )
            )
            source_recipes = {
                mapping[candidate_id] for candidate_id in survivor.source_candidate_ids
            }
        except KeyError as error:
            raise ValueError("E2 raw survivor is foreign to the signed grid") from error
        materialized_recipe_sha256s = {
            cell.recipe_sha256
            for cell in materialization.cells
            if cell.method_role == "LightCone-candidate"
        }
        if {row.sha256 for row in source_recipes} != materialized_recipe_sha256s:
            raise ValueError("E2 raw reduction source differs from materialization")
        expected_count = max(math.ceil(len(source_recipes) / 4), 21)
        expected_families = {
            (optimizer, schedule)
            for optimizer in E2_OPTIMIZERS
            for schedule in E2_SCHEDULES
        }
        if (
            len(survivors) != expected_count
            or {(row.optimizer, row.schedule) for row in survivors} != expected_families
        ):
            raise ValueError("E2 raw survivors violate quarter/family-floor protocol")
        if (
            self.protocol_lock_sha256 != protocol_lock.sha256
            or self.registry_sha256 != protocol_lock.registry_sha256
            or protocol_lock.e2_recipe_grid_authority_sha256 != grid.sha256
            or self.source_materialization_receipt_sha256 != materialization.sha256
            or self.source_coverage_receipt_sha256 != coverage.sha256
            or self.reduction_authority_sha256 != authority.sha256
            or self.reduction_sha256 != reduction.sha256
            or self.survivor_recipes != survivors
        ):
            raise ValueError("E2 survivor receipt differs from reopened raw reduction")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE2RoundSelectionReceipt:
    payload: E2RoundSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        authority: E2StageReductionAuthority,
        grid: E2RecipeGridAuthority,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E2RoundSelectionReceipt:
        if type(self.payload) is not E2RoundSelectionReceipt:
            raise TypeError("signed E2 round selection payload has the wrong type")
        self.payload.validate_sources(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            authority=authority,
            grid=grid,
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


__all__ = [
    "E1SurvivorSelectionReceipt",
    "E2RoundSelectionReceipt",
    "E2StageReductionAuthority",
    "E3aSelectionReceipt",
    "SignedE1SurvivorSelectionReceipt",
    "SignedE2RoundSelectionReceipt",
    "SignedE3aSelectionReceipt",
    "bind_e2_stage_reduction_authority",
    "reduce_e1_survivor_selection_receipt",
    "reduce_e3a_stage_selection_receipt",
]
