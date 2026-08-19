"""Path-bound NEXTN compatibility authority for the two-model E6 panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cached_property
from pathlib import Path

from lightcone_spec.experiments.downstream_stage_authority import (
    FormalDownstreamEvidenceManifest,
)
from lightcone_spec.experiments.e1_stage_authority import (
    _request_identity,
    _validated_cell,
)
from lightcone_spec.experiments.formal_protocol import (
    E6_MODELS,
    FORMAL_METHOD_ROLES,
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.stage_materialization import (
    E6_CONTEXTS,
    E6_TASKS,
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
from lightcone_spec.runtime.backend import (
    VerifiedNextNTp2Authority,
    validate_nextn_tp2_dynamic_authority_artifact,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e6_two_model_nextn_compatibility_protocol",
        "models": E6_MODELS,
        "backend": "NEXTN",
        "topology": "tp2_dp1",
        "authority": (
            "prepared_snapshot_shards_plus_nextn_tp2_native_and_distributed_gpu_proofs"
        ),
        "disposition": "both_models_mandatory_no_skip_or_na",
    }
)
E6_POWER_PREFIX_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e6_proof_derived_power_prefix_protocol",
        "pilot_blocks": 4,
        "pilot_disposition": "excluded_tuning_only",
        "model_preflights": 2,
        "pilot_serving_cells": 60 * 4,
        "panel": {
            "models": E6_MODELS,
            "roles": FORMAL_METHOD_ROLES,
            "tasks": E6_TASKS,
            "contexts": E6_CONTEXTS,
        },
        "pairing": "exact_block_model_task_context_request_and_token_trajectory",
        "power": "preregister_power_sizing_3pct_80pct_holm_first_threshold",
        "final_prefix": "first_N_of_12_to_20",
    }
)
E6_CONFIRMATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e6_proof_derived_confirmation_protocol",
        "model_preflights_executed_once_in_excluded_pilot": 2,
        "model_preflights_repeated_in_final": 0,
        "serving_cells_per_block": 60,
        "models": E6_MODELS,
        "primary_family": PRIMARY_CONTRASTS,
        "coverage": "sealed_serving_bindings_plus_path_bound_result_and_itl_proofs",
        "result": "external_validity_confirmation_without_recipe_reselection",
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


def _text(label: str, value: object) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _absolute_path(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"{label} must be absolute and resolved")
    return value


@dataclass(frozen=True)
class E6NextnModelAuthorityInput:
    """Immutable reopening instructions for one prepared two-model proof DAG."""

    schema_version: int
    model: str
    target_member_id: str
    drafter_member_id: str
    artifact_path: str
    artifact_raw_sha256: str
    artifact_semantic_sha256: str
    expected_interface_sha256: str
    expected_topology_sha256: str
    expected_source_adapter_version: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.model not in E6_MODELS:
            raise ValueError("E6 NEXTN input schema/model is unsupported")
        _text("E6 NEXTN target member", self.target_member_id)
        _text("E6 NEXTN drafter member", self.drafter_member_id)
        if self.target_member_id == self.drafter_member_id:
            raise ValueError("E6 NEXTN target/drafter members must differ")
        for label, digest in (
            ("artifact raw", self.artifact_raw_sha256),
            ("artifact semantic", self.artifact_semantic_sha256),
            ("interface", self.expected_interface_sha256),
            ("topology", self.expected_topology_sha256),
        ):
            _sha256(f"E6 NEXTN {label}", digest)
        if (
            type(self.expected_source_adapter_version) is not int
            or self.expected_source_adapter_version < 0
        ):
            raise ValueError("E6 NEXTN source adapter version is invalid")
        binding = CanonicalJsonProofBinding.bind(
            _absolute_path("E6 NEXTN authority artifact", self.artifact_path)
        )
        if (
            binding.raw_sha256 != self.artifact_raw_sha256
            or binding.semantic_sha256 != self.artifact_semantic_sha256
        ):
            raise ValueError("E6 NEXTN authority artifact changed after binding")
        reject_banned_model_identity(self)

    @classmethod
    def bind(
        cls,
        *,
        model: str,
        target_member_id: str,
        drafter_member_id: str,
        artifact_path: str,
        expected_interface_sha256: str,
        expected_topology_sha256: str,
        expected_source_adapter_version: int,
    ) -> E6NextnModelAuthorityInput:
        binding = CanonicalJsonProofBinding.bind(artifact_path)
        return cls(
            schema_version=1,
            model=model,
            target_member_id=target_member_id,
            drafter_member_id=drafter_member_id,
            artifact_path=binding.absolute_path,
            artifact_raw_sha256=binding.raw_sha256,
            artifact_semantic_sha256=binding.semantic_sha256,
            expected_interface_sha256=expected_interface_sha256,
            expected_topology_sha256=expected_topology_sha256,
            expected_source_adapter_version=expected_source_adapter_version,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E6NextnModelCompatibility:
    model: str
    source_input_sha256: str
    dynamic_artifact_sha256: str
    verified_authority_sha256: str
    interface_sha256: str
    target_member_id: str
    drafter_member_id: str
    target_model_id: str
    drafter_model_id: str
    target_revision: str
    drafter_revision: str
    target_shard_manifest_sha256: str
    drafter_shard_manifest_sha256: str
    topology_sha256: str
    source_adapter_version: int
    native_gpu_proof_sha256: str
    distributed_gpu_proof_sha256: str
    content_verification_receipt_sha256: str
    inventory_sha256: str
    gpu_uuids: tuple[str, str]

    def __post_init__(self) -> None:
        if self.model not in E6_MODELS:
            raise ValueError("E6 compatibility model is outside the exact panel")
        _text("E6 compatibility target member", self.target_member_id)
        _text("E6 compatibility drafter member", self.drafter_member_id)
        _text("E6 compatibility target model", self.target_model_id)
        _text("E6 compatibility drafter model", self.drafter_model_id)
        _text("E6 compatibility target revision", self.target_revision)
        _text("E6 compatibility drafter revision", self.drafter_revision)
        for label, digest in (
            ("source input", self.source_input_sha256),
            ("dynamic artifact", self.dynamic_artifact_sha256),
            ("verified authority", self.verified_authority_sha256),
            ("interface", self.interface_sha256),
            ("target shards", self.target_shard_manifest_sha256),
            ("drafter shards", self.drafter_shard_manifest_sha256),
            ("topology", self.topology_sha256),
            ("native GPU proof", self.native_gpu_proof_sha256),
            ("distributed GPU proof", self.distributed_gpu_proof_sha256),
            ("content receipt", self.content_verification_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha256(f"E6 compatibility {label}", digest)
        if (
            self.target_model_id != self.model
            or self.target_model_id == self.drafter_model_id
            or type(self.source_adapter_version) is not int
            or self.source_adapter_version < 0
            or len(self.gpu_uuids) != 2
            or len(set(self.gpu_uuids)) != 2
        ):
            raise ValueError("E6 compatibility adapter/GPU identity is invalid")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E6ModelCompatibilityReceipt:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    release_root_manifest_sha256: str
    protocol_sha256: str
    models: tuple[E6NextnModelCompatibility, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E6 model-compatibility schema 1 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("release root", self.release_root_manifest_sha256),
            ("protocol", self.protocol_sha256),
        ):
            _sha256(f"E6 compatibility {label}", digest)
        if self.protocol_sha256 != E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256:
            raise ValueError("E6 compatibility uses another reducer protocol")
        if (
            type(self.models) is not tuple
            or any(type(row) is not E6NextnModelCompatibility for row in self.models)
            or tuple(row.model for row in self.models) != E6_MODELS
            or len({row.source_input_sha256 for row in self.models}) != len(E6_MODELS)
            or len({row.dynamic_artifact_sha256 for row in self.models})
            != len(E6_MODELS)
            or len({row.inventory_sha256 for row in self.models}) != 1
            or len({row.content_verification_receipt_sha256 for row in self.models})
            != 1
        ):
            raise ValueError("E6 compatibility must cover both exact models once")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE6ModelCompatibilityReceipt:
    payload: E6ModelCompatibilityReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        sources: tuple[E6NextnModelAuthorityInput, ...],
        expected_inventory_sha256: str,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E6ModelCompatibilityReceipt:
        if type(self.payload) is not E6ModelCompatibilityReceipt:
            raise TypeError("signed E6 compatibility payload has the wrong type")
        expected = reduce_e6_model_compatibility_from_proofs(
            protocol_lock=protocol_lock,
            sources=sources,
            expected_inventory_sha256=expected_inventory_sha256,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E6 compatibility differs from proof reducer")
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
class E6PowerPrefixReceipt:
    """Exact 12--20 final prefix selected from the four excluded E6 pilots."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    upstream_e5_confirmation_sha256: str
    signed_model_compatibility_sha256: str
    pilot_materialization_receipt_sha256: str
    pilot_coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    frozen_tts_recipe_sha256: str
    lightcone_recipe_sha256: str
    power_sizing: PowerSizingPlan
    selected_final_blocks: int
    selected_final_prefix: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E6 power-prefix schema 1 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("upstream E5 confirmation", self.upstream_e5_confirmation_sha256),
            ("signed model compatibility", self.signed_model_compatibility_sha256),
            ("pilot materialization", self.pilot_materialization_receipt_sha256),
            ("pilot coverage", self.pilot_coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("protocol", self.protocol_sha256),
            ("frozen TTS recipe", self.frozen_tts_recipe_sha256),
            ("LightCone recipe", self.lightcone_recipe_sha256),
        ):
            _sha256(f"E6 power {label}", digest)
        if self.protocol_sha256 != E6_POWER_PREFIX_PROTOCOL_SHA256:
            raise ValueError("E6 power prefix uses another reducer protocol")
        if (
            self.power_sizing.status != "READY"
            or self.power_sizing.selected_final_blocks != self.selected_final_blocks
            or not 12 <= self.selected_final_blocks <= 20
            or self.selected_final_prefix
            != tuple(
                range(
                    PILOT_BLOCK_COUNT,
                    PILOT_BLOCK_COUNT + self.selected_final_blocks,
                )
            )
        ):
            raise ValueError("E6 power prefix differs from preregistered sizing")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE6PowerPrefixReceipt:
    payload: E6PowerPrefixReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
        compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
        pilot_materialization: StageMaterializationReceipt,
        pilot_coverage: StageCoverageReceipt,
        manifest: FormalDownstreamEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E6PowerPrefixReceipt:
        if type(self.payload) is not E6PowerPrefixReceipt:
            raise TypeError("signed E6 power-prefix payload has the wrong type")
        expected = reduce_e6_power_prefix_from_proofs(
            protocol_lock=protocol_lock,
            signed_model_compatibility=signed_model_compatibility,
            compatibility_sources=compatibility_sources,
            pilot_materialization=pilot_materialization,
            pilot_coverage=pilot_coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E6 power prefix differs from proof reducer")
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
class E6ConfirmationReceipt:
    """Proof-derived two-model E6 external-validity decision consumed by E0."""

    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    protocol_sha256: str
    upstream_e5_confirmation_sha256: str
    signed_model_compatibility_sha256: str
    frozen_tts_recipe_sha256: str
    lightcone_recipe_sha256: str
    models: tuple[str, str]
    final_block_ids: tuple[str, ...]
    primary_contrasts: tuple[PairedBcaContrast, ...]
    holm_decisions: tuple[MultiplicityDecision, ...]
    status: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E6 confirmation schema 1 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("protocol", self.protocol_sha256),
            ("upstream E5 confirmation", self.upstream_e5_confirmation_sha256),
            ("signed model compatibility", self.signed_model_compatibility_sha256),
            ("frozen TTS recipe", self.frozen_tts_recipe_sha256),
            ("LightCone recipe", self.lightcone_recipe_sha256),
        ):
            _sha256(f"E6 confirmation {label}", digest)
        if self.protocol_sha256 != E6_CONFIRMATION_PROTOCOL_SHA256:
            raise ValueError("E6 confirmation uses another reducer protocol")
        if (
            self.models != E6_MODELS
            or self.final_block_ids != tuple(sorted(set(self.final_block_ids)))
            or not 12 <= len(self.final_block_ids) <= 20
            or tuple(row.name for row in self.primary_contrasts) != PRIMARY_CONTRASTS
            or tuple(row.name for row in self.holm_decisions) != PRIMARY_CONTRASTS
        ):
            raise ValueError("E6 confirmation panel/family coverage is not exact")
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
            raise ValueError("E6 confirmation status differs from Holm/CI decisions")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE6ConfirmationReceipt:
    payload: E6ConfirmationReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
        compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        manifest: FormalDownstreamEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E6ConfirmationReceipt:
        if type(self.payload) is not E6ConfirmationReceipt:
            raise TypeError("signed E6 confirmation payload has the wrong type")
        expected = reduce_e6_confirmation_from_proofs(
            protocol_lock=protocol_lock,
            signed_model_compatibility=signed_model_compatibility,
            compatibility_sources=compatibility_sources,
            materialization=materialization,
            coverage=coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E6 confirmation differs from proof reducer")
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


def _compatibility_row(
    source: E6NextnModelAuthorityInput,
    verified: VerifiedNextNTp2Authority,
) -> E6NextnModelCompatibility:
    if verified.target_model_id != source.model:
        raise ValueError("E6 NEXTN proof target model differs from panel identity")
    return E6NextnModelCompatibility(
        model=source.model,
        source_input_sha256=source.sha256,
        dynamic_artifact_sha256=verified.artifact_sha256,
        verified_authority_sha256=verified.sha256,
        interface_sha256=verified.interface_sha256,
        target_member_id=source.target_member_id,
        drafter_member_id=source.drafter_member_id,
        target_model_id=verified.target_model_id,
        drafter_model_id=verified.drafter_model_id,
        target_revision=verified.target_revision,
        drafter_revision=verified.drafter_revision,
        target_shard_manifest_sha256=verified.target_shard_manifest_sha256,
        drafter_shard_manifest_sha256=verified.drafter_shard_manifest_sha256,
        topology_sha256=verified.topology_sha256,
        source_adapter_version=verified.source_adapter_version,
        native_gpu_proof_sha256=verified.native_gpu_proof_sha256,
        distributed_gpu_proof_sha256=verified.distributed_gpu_proof_sha256,
        content_verification_receipt_sha256=(
            verified.content_verification_receipt_sha256
        ),
        inventory_sha256=verified.inventory_sha256,
        gpu_uuids=verified.gpu_uuids,
    )


def reduce_e6_model_compatibility_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    sources: tuple[E6NextnModelAuthorityInput, ...],
    expected_inventory_sha256: str,
    now_ns: int,
) -> E6ModelCompatibilityReceipt:
    """Deep-reopen exactly two prepared NEXTN TP2 authority DAGs."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E6 compatibility requires an exact ProtocolLock")
    _sha256("E6 compatibility expected inventory", expected_inventory_sha256)
    if (
        type(sources) is not tuple
        or any(type(row) is not E6NextnModelAuthorityInput for row in sources)
        or tuple(row.model for row in sources) != E6_MODELS
        or len({row.sha256 for row in sources}) != len(E6_MODELS)
    ):
        raise ValueError("E6 compatibility sources must cover both models exactly")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E6 compatibility verification time is invalid")
    rows = []
    for source in sources:
        source.__post_init__()
        verified = validate_nextn_tp2_dynamic_authority_artifact(
            source.artifact_path,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            expected_interface_sha256=source.expected_interface_sha256,
            expected_topology_sha256=source.expected_topology_sha256,
            expected_source_adapter_version=source.expected_source_adapter_version,
            expected_target_member_id=source.target_member_id,
            expected_drafter_member_id=source.drafter_member_id,
            now_ns=now_ns,
        )
        rows.append(_compatibility_row(source, verified))
    receipt = E6ModelCompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        release_root_manifest_sha256=protocol_lock.offline_release_trust_root_sha256,
        protocol_sha256=E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256,
        models=tuple(rows),
    )
    receipt.__post_init__()
    return receipt


def _aggregate_request_rate(rows: list[object]) -> Fraction:
    numerator = sum(metric.output_tokens for row in rows for metric in row.metrics)
    denominator = sum(metric.latency_ns for row in rows for metric in row.metrics)
    if numerator < 1 or denominator < 1:
        raise ValueError("E6 block/role has no completed timed output")
    return Fraction(numerator * 1_000_000_000, denominator)


def _validated_e6_serving_universe(
    *,
    protocol_lock: ProtocolLock,
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
    expected_materialization_rule: str,
    expected_blocks: range,
    expected_block_start: int,
    expect_model_preflights: bool,
) -> tuple[
    E6ModelCompatibilityReceipt,
    tuple[object, ...],
    dict[str, object],
    set[int],
]:
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E6 reducer requires an exact ProtocolLock")
    if type(signed_model_compatibility) is not SignedE6ModelCompatibilityReceipt:
        raise TypeError("E6 reducer requires signed model compatibility")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E6 reducer requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E6 reducer requires exact coverage")
    if type(manifest) is not FormalDownstreamEvidenceManifest:
        raise TypeError("E6 reducer requires exact proof manifest")
    if type(execution_bindings) is not tuple or any(
        type(row) is not VerifiedFormalServingExecutionBinding
        for row in execution_bindings
    ):
        raise TypeError("E6 reducer requires sealed execution bindings")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E6 reducer time must be positive")
    if type(expect_model_preflights) is not bool:
        raise TypeError("E6 preflight expectation must be boolean")
    if (
        materialization.stage != "E6"
        or materialization.materialization_rule != expected_materialization_rule
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.stage != "E6"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_authority_sha256 != materialization.source_decision_sha256
    ):
        raise ValueError("E6 evidence differs from exact materialization lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E6 reducer requires all-COMPLETE coverage")
    compatibility = signed_model_compatibility.verify(
        protocol_lock=protocol_lock,
        sources=compatibility_sources,
        expected_inventory_sha256=manifest.inventory_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    compatibility_by_model = {row.model: row for row in compatibility.models}
    if set(compatibility_by_model) != set(E6_MODELS):
        raise ValueError("E6 compatibility omits an exact target model")
    preflight_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "immutable_metadata_interface_and_fit_preflight"
    )
    serving_cells = tuple(
        cell for cell in materialization.cells if cell.task in E6_TASKS
    )
    expected_preflight_count = len(E6_MODELS) if expect_model_preflights else 0
    if len(preflight_cells) != expected_preflight_count or len(serving_cells) + len(
        preflight_cells
    ) != len(materialization.cells):
        raise ValueError("E6 materialization lacks exact preflight/serving partition")
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    seen_preflight_models: set[str] = set()
    for cell in preflight_cells:
        row = compatibility_by_model.get(cell.model)
        dimensions = dict(cell.dimensions)
        if (
            row is None
            or cell.method_role != "Target-only"
            or cell.backend != "NEXTN"
            or cell.publication_policy != "none"
            or cell.recipe_sha256 is not None
            or dimensions.get("topology") != "tp2_dp1"
            or dimensions.get("e6_model_compatibility_row_sha256") != row.sha256
            or dimensions.get("e6_verified_authority_sha256")
            != row.verified_authority_sha256
            or dimensions.get("target_member_id") != row.target_member_id
            or dimensions.get("drafter_member_id") != row.drafter_member_id
            or dimensions.get("interface_sha256") != row.interface_sha256
            or terminal_by_cell[cell.cell_id] != row.verified_authority_sha256
            or cell.model in seen_preflight_models
        ):
            raise ValueError("E6 model preflight differs from compatibility authority")
        seen_preflight_models.add(cell.model)
    expected_preflight_models = set(E6_MODELS) if expect_model_preflights else set()
    if seen_preflight_models != expected_preflight_models:
        raise ValueError("E6 model preflight coverage is not exact")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E6 reducer reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_serving_ids = {cell.cell_id for cell in serving_cells}
    if (
        set(evidence_by_cell) != expected_serving_ids
        or set(bindings_by_cell) != expected_serving_ids
    ):
        raise ValueError("E6 serving proof/binding coverage is not exact")
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E6",
        )
        for cell in serving_cells
    }
    block_values = {dict(cell.dimensions).get("block") for cell in serving_cells}
    if (
        any(type(block) is not int for block in block_values)
        or block_values
        != set(range(expected_block_start, expected_block_start + len(block_values)))
        or len(block_values) not in expected_blocks
        or len(serving_cells) != 60 * len(block_values)
        or len(materialization.cells)
        != expected_preflight_count + 60 * len(block_values)
    ):
        raise ValueError("E6 block/cardinality identity is not exact")
    for cell in serving_cells:
        row = compatibility_by_model[cell.model]
        dimensions = dict(cell.dimensions)
        if (
            dimensions.get("e6_model_compatibility_row_sha256") != row.sha256
            or dimensions.get("e6_verified_authority_sha256")
            != row.verified_authority_sha256
            or dimensions.get("target_member_id") != row.target_member_id
            or dimensions.get("drafter_member_id") != row.drafter_member_id
            or dimensions.get("interface_sha256") != row.interface_sha256
            or dimensions.get("topology") != "tp2_dp1"
        ):
            raise ValueError("E6 serving cell differs from model compatibility")
        result = validated[cell.cell_id]
        if result.safety_reasons or (
            cell.method_role in {"TTS", "L0-naive", "LightCone"}
            and result.published_updates < 1
        ):
            raise ValueError("E6 contains unsafe or inactive serving evidence")
    return compatibility, serving_cells, validated, block_values  # type: ignore[return-value]


def reduce_e6_power_prefix_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
    pilot_materialization: StageMaterializationReceipt,
    pilot_coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> E6PowerPrefixReceipt:
    """Deep-reopen both NEXTN preflights and all 240 excluded pilot rows."""

    _compatibility, serving_cells, validated, block_values = (
        _validated_e6_serving_universe(
            protocol_lock=protocol_lock,
            signed_model_compatibility=signed_model_compatibility,
            compatibility_sources=compatibility_sources,
            materialization=pilot_materialization,
            coverage=pilot_coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
            expected_materialization_rule=(
                "e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks"
            ),
            expected_blocks=range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + 1),
            expected_block_start=0,
            expect_model_preflights=True,
        )
    )
    if block_values != set(range(PILOT_BLOCK_COUNT)):
        raise ValueError("E6 power reducer requires exactly four excluded pilots")
    by_block_role: dict[tuple[int, str], list[object]] = {}
    by_block_stratum: dict[
        tuple[int, tuple[tuple[str, object], ...]], list[object]
    ] = {}
    source_values: dict[str, set[object]] = {
        key: set()
        for key in (
            "upstream_e5_confirmation_sha256",
            "signed_e6_model_compatibility_sha256",
            "frozen_tts_recipe_sha256",
            "lightcone_recipe_sha256",
        )
    }
    for cell in serving_cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if (
            type(block) is not int
            or block not in range(PILOT_BLOCK_COUNT)
            or dimensions.get("block_phase") != "excluded_pilot"
            or cell.method_role not in FORMAL_METHOD_ROLES
        ):
            raise ValueError("E6 pilot cell lies outside the excluded prefix")
        for key, values in source_values.items():
            values.add(dimensions.get(key))
        row = validated[cell.cell_id]
        by_block_role.setdefault((block, cell.method_role), []).append(row)
        stratum = tuple(
            sorted(
                (key, value)
                for key, value in dimensions.items()
                if key not in {"block", "block_phase", "tts_l0_pair_id"}
            )
        )
        by_block_stratum.setdefault((block, stratum), []).append(row)
    if set(by_block_role) != {
        (block, role)
        for block in range(PILOT_BLOCK_COUNT)
        for role in FORMAL_METHOD_ROLES
    } or any(len(rows) != 12 for rows in by_block_role.values()):
        raise ValueError("E6 pilots lack exact 12-stratum role coverage")
    if len(by_block_stratum) != PILOT_BLOCK_COUNT * 12 or any(
        len(rows) != len(FORMAL_METHOD_ROLES)
        or len({_request_identity(row.metrics) for row in rows}) != 1
        for rows in by_block_stratum.values()
    ):
        raise ValueError("E6 pilot methods differ in paired requests/trajectories")
    if any(len(values) != 1 or None in values for values in source_values.values()):
        raise ValueError("E6 pilot source lineage is ambiguous")
    pilot_blocks = tuple(
        PilotBlock(
            block_id=f"E6:excluded_pilot:{block}",
            static_goodput=float(
                _aggregate_request_rate(by_block_role[(block, "Static")])
            ),
            tts_goodput=float(_aggregate_request_rate(by_block_role[(block, "TTS")])),
            lightcone_goodput=float(
                _aggregate_request_rate(by_block_role[(block, "LightCone")])
            ),
        )
        for block in range(PILOT_BLOCK_COUNT)
    )
    power = preregister_power_sizing(pilot_blocks)
    if power.underpowered or power.selected_final_blocks is None:
        raise ValueError("E6 excluded pilots are UNDERPOWERED at 20 final blocks")
    receipt = E6PowerPrefixReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        upstream_e5_confirmation_sha256=next(
            iter(source_values["upstream_e5_confirmation_sha256"])
        ),  # type: ignore[arg-type]
        signed_model_compatibility_sha256=next(
            iter(source_values["signed_e6_model_compatibility_sha256"])
        ),  # type: ignore[arg-type]
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_coverage_receipt_sha256=pilot_coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        protocol_sha256=E6_POWER_PREFIX_PROTOCOL_SHA256,
        frozen_tts_recipe_sha256=next(iter(source_values["frozen_tts_recipe_sha256"])),  # type: ignore[arg-type]
        lightcone_recipe_sha256=next(iter(source_values["lightcone_recipe_sha256"])),  # type: ignore[arg-type]
        power_sizing=power,
        selected_final_blocks=power.selected_final_blocks,
        selected_final_prefix=tuple(
            range(
                PILOT_BLOCK_COUNT,
                PILOT_BLOCK_COUNT + power.selected_final_blocks,
            )
        ),
    )
    receipt.__post_init__()
    return receipt


def reduce_e6_confirmation_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    signed_model_compatibility: SignedE6ModelCompatibilityReceipt,
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...],
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    manifest: FormalDownstreamEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> E6ConfirmationReceipt:
    """Deep-reopen the complete two-model E6 panel and its Holm family."""

    _compatibility, serving_cells, validated, block_values = (
        _validated_e6_serving_universe(
            protocol_lock=protocol_lock,
            signed_model_compatibility=signed_model_compatibility,
            compatibility_sources=compatibility_sources,
            materialization=materialization,
            coverage=coverage,
            manifest=manifest,
            execution_bindings=execution_bindings,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
            expected_materialization_rule=(
                "60_final_rows_per_block_reusing_global_model_preflights"
            ),
            expected_blocks=range(12, 21),
            expected_block_start=PILOT_BLOCK_COUNT,
            expect_model_preflights=False,
        )
    )
    by_block_role: dict[tuple[int, str], list[object]] = {}
    by_block_stratum: dict[
        tuple[int, tuple[tuple[str, object], ...]], list[object]
    ] = {}
    source_values: dict[str, set[object]] = {
        key: set()
        for key in (
            "upstream_e5_confirmation_sha256",
            "signed_e6_model_compatibility_sha256",
            "frozen_tts_recipe_sha256",
            "lightcone_recipe_sha256",
        )
    }
    for cell in serving_cells:
        dimensions = dict(cell.dimensions)
        block = dimensions["block"]
        assert type(block) is int
        for key, values in source_values.items():
            values.add(dimensions.get(key))
        row = validated[cell.cell_id]
        by_block_role.setdefault((block, cell.method_role), []).append(row)
        stratum = tuple(
            sorted(
                (key, value)
                for key, value in dimensions.items()
                if key not in {"block", "block_phase", "tts_l0_pair_id"}
            )
        )
        by_block_stratum.setdefault((block, stratum), []).append(row)
    if set(by_block_role) != {
        (block, role) for block in block_values for role in FORMAL_METHOD_ROLES
    } or any(len(rows) != 12 for rows in by_block_role.values()):
        raise ValueError("E6 confirmation lacks exact block/role strata")
    if len(by_block_stratum) != len(block_values) * 12 or any(
        len(rows) != len(FORMAL_METHOD_ROLES)
        or len({_request_identity(row.metrics) for row in rows}) != 1
        for rows in by_block_stratum.values()
    ):
        raise ValueError("E6 confirmation methods are not exactly paired")
    if any(len(values) != 1 or None in values for values in source_values.values()):
        raise ValueError("E6 confirmation source lineage is ambiguous")
    final_blocks = tuple(sorted(block_values))
    paired: dict[str, dict[str, tuple[float, float]]] = {
        "lightcone_vs_tts": {},
        "lightcone_vs_static": {},
    }
    for block in final_blocks:
        rates = {
            role: float(_aggregate_request_rate(by_block_role[(block, role)]))
            for role in ("Static", "TTS", "LightCone")
        }
        block_id = f"E6:final:{block - PILOT_BLOCK_COUNT:02d}"
        paired["lightcone_vs_tts"][block_id] = (
            rates["LightCone"],
            rates["TTS"],
        )
        paired["lightcone_vs_static"][block_id] = (
            rates["LightCone"],
            rates["Static"],
        )
    contrasts = tuple(
        paired_bca_contrast(name, paired[name]) for name in PRIMARY_CONTRASTS
    )
    decisions = holm_primary_contrasts({row.name: row for row in contrasts})
    models = tuple(sorted({cell.model for cell in serving_cells}))
    if models != tuple(sorted(E6_MODELS)):
        raise ValueError("E6 confirmation model coverage is not exact")
    receipt = E6ConfirmationReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        protocol_sha256=E6_CONFIRMATION_PROTOCOL_SHA256,
        upstream_e5_confirmation_sha256=next(
            iter(source_values["upstream_e5_confirmation_sha256"])
        ),  # type: ignore[arg-type]
        signed_model_compatibility_sha256=next(
            iter(source_values["signed_e6_model_compatibility_sha256"])
        ),  # type: ignore[arg-type]
        frozen_tts_recipe_sha256=next(iter(source_values["frozen_tts_recipe_sha256"])),  # type: ignore[arg-type]
        lightcone_recipe_sha256=next(iter(source_values["lightcone_recipe_sha256"])),  # type: ignore[arg-type]
        models=E6_MODELS,
        final_block_ids=tuple(
            f"E6:final:{block - PILOT_BLOCK_COUNT:02d}" for block in final_blocks
        ),
        primary_contrasts=contrasts,
        holm_decisions=decisions,
        status=(
            "CONFIRMED"
            if all(
                decision.rejected and contrast.ci_lower_relative_gain > 0
                for contrast, decision in zip(contrasts, decisions, strict=True)
            )
            else "NOT_CONFIRMED"
        ),
    )
    receipt.__post_init__()
    return receipt


__all__ = [
    "E6_CONFIRMATION_PROTOCOL_SHA256",
    "E6_MODEL_COMPATIBILITY_PROTOCOL_SHA256",
    "E6_POWER_PREFIX_PROTOCOL_SHA256",
    "E6ConfirmationReceipt",
    "E6ModelCompatibilityReceipt",
    "E6NextnModelAuthorityInput",
    "E6NextnModelCompatibility",
    "E6PowerPrefixReceipt",
    "SignedE6ConfirmationReceipt",
    "SignedE6ModelCompatibilityReceipt",
    "SignedE6PowerPrefixReceipt",
    "reduce_e6_confirmation_from_proofs",
    "reduce_e6_model_compatibility_from_proofs",
    "reduce_e6_power_prefix_from_proofs",
]
