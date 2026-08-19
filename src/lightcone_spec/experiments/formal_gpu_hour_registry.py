"""Durable formal-registry consumption of lifecycle-derived GPU hours.

The lifecycle reducer deliberately keeps verifier-sealed execution bindings in
memory.  This module is the durable boundary: it binds the immutable source
manifest into the append-only formal registry prefix, verifies the signed
schema-2 envelope, reopens every first-party lifecycle proof, and atomically
reserves the envelope and dynamic-control challenges.  A study total is
published only from such durable receipts; a signed scalar or the historical
schema-1 envelope is never accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, Self

from lightcone_spec.experiments.formal_protocol import (
    FORMAL_STAGE_DAG,
    FormalRuntimeAuthorityManifest,
    content_sha256,
)
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
    formal_runtime_authority_manifest_from_dict,
    formal_runtime_authority_manifest_to_dict,
    signed_stage_gpu_hour_from_dict,
    signed_stage_gpu_hour_to_dict,
    stage_materialization_receipt_from_dict,
)
from lightcone_spec.experiments.formal_registry_layers import (
    FORMAL_REGISTRY_LAYER_ARTIFACT_KIND,
    load_formal_registry_verification_receipt_path,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    LifecycleGpuHourSourceManifest,
    PreflightGpuHourSourceManifest,
    ProspectiveGpuHourSourceManifest,
    StagedProspectiveGpuHourSourceManifest,
    revalidate_persisted_preflight_gpu_hour_source_manifest,
    revalidate_persisted_prospective_gpu_hour_source_manifest,
    revalidate_persisted_stage_gpu_hour_source_manifest,
    revalidate_persisted_staged_prospective_gpu_hour_source_manifest,
    verify_registered_prospective_gpu_hour_authority,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.stage_materialization import (
    SignedStageGpuHourEnvelope,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
    ChallengeReplayStore,
    ControlArtifactAttestation,
    control_challenge_reservation_sha256,
    verify_and_reserve_release_control_artifact_attestations,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 4,
        "kind": "lightcone_formal_gpu_hour_registry_protocol",
        "inputs": (
            "append_only_formal_registry_verification_receipt",
            "signed_schema2_stage_gpu_hour_envelope",
            "path_bound_execution_lifecycle_and_first_party_proofs",
            "path_bound_excluded_pilot_materialization_and_registered_power_wrapper",
            "exact_two_gpu_inventory_and_runtime_authority_manifest",
            "root_authorized_rank_aggregate_control",
        ),
        "registry_layer": "proof_carrying_schema5_only",
        "atomic_replay": "deployment_control_and_signed_envelope_challenge",
        "aggregation": (
            "complete_cell_and_materialization_coverage_without_execution_"
            "lifecycle_raw_control_nonce_or_replay_reuse"
        ),
        "legacy_scalar_envelope": "forbidden",
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


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _load_proof_carrying_registry_layer(
    source: CanonicalJsonProofBinding,
    *,
    now_ns: int,
) -> FormalRegistryVerificationReceipt:
    """Reject a recursively serialized receipt at a scientific budget boundary."""

    observed = CanonicalJsonProofBinding.bind(source.absolute_path)
    if observed != source:
        raise ValueError("formal GPU-hour registry layer path identity changed")
    value = observed.reopen()
    if value.get("kind") != FORMAL_REGISTRY_LAYER_ARTIFACT_KIND:
        raise ValueError(
            "formal GPU-hour verification requires a proof-carrying schema-5 "
            "registry layer"
        )
    if CanonicalJsonProofBinding.bind(source.absolute_path) != observed:
        raise RuntimeError("formal GPU-hour registry layer changed while reopened")
    return load_formal_registry_verification_receipt_path(
        observed.absolute_path,
        now_ns=now_ns,
    )


def _registered_materialization(
    receipt: FormalRegistryVerificationReceipt,
    materialization_sha256: str,
) -> StageMaterializationReceipt:
    materializations = tuple(
        row.payload
        for row in receipt.cumulative_signed_materializations
        if row.payload.sha256 == materialization_sha256
    )
    if len(materializations) != 1:
        raise ValueError("formal GPU-hour registry materialization is not exact")
    return materializations[0]


def _materialization_and_coverage(
    receipt: FormalRegistryVerificationReceipt,
    materialization_sha256: str,
) -> tuple[StageMaterializationReceipt, StageCoverageReceipt]:
    materialization = _registered_materialization(receipt, materialization_sha256)
    coverage = tuple(
        row.payload
        for row in receipt.cumulative_signed_coverage
        if row.payload.materialization_receipt_sha256 == materialization_sha256
    )
    if len(coverage) != 1:
        raise ValueError("formal GPU-hour registry lineage is not exact")
    covered = coverage[0]
    covered.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in covered.dispositions):
        raise ValueError("formal GPU-hour registry requires all-COMPLETE coverage")
    return materialization, covered


def _covered_materialization_sha256s(
    receipt: FormalStageGpuHourVerificationReceipt,
    source: (
        LifecycleGpuHourSourceManifest
        | PreflightGpuHourSourceManifest
        | ProspectiveGpuHourSourceManifest
        | StagedProspectiveGpuHourSourceManifest
    ),
) -> tuple[str, ...]:
    """Return every registry materialization charged by one stage envelope.

    A powered downstream envelope deliberately includes the four excluded
    pilot blocks and the projected final prefix once.  Treating its pilot
    materialization as a second required budget would either double-charge the
    pilot lifecycle proofs or make complete aggregation impossible.
    """

    covered = [receipt.materialization_receipt_sha256]
    if type(source) is ProspectiveGpuHourSourceManifest:
        binding = receipt.prospective_pilot_materialization
        if binding is None:
            raise ValueError("prospective GPU-hour receipt lacks pilot materialization")
        if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
            raise ValueError("prospective pilot materialization binding changed")
        pilot = stage_materialization_receipt_from_dict(binding.reopen())
        if pilot.sha256 != source.pilot_materialization_receipt_sha256:
            raise ValueError("prospective pilot materialization identity differs")
        # Excluded pilot materializations are signed power/source lineage, not
        # append-only main-registry rows.  Their measured cost is already
        # included by the prospective source manifest, while this coverage
        # tuple deliberately names only the registered final materialization.
    if len(covered) != len(set(covered)):
        raise ValueError(
            "formal stage GPU-hour source double-charges a materialization"
        )
    return tuple(sorted(covered))


def _control_lineage_sha256(
    *,
    registry_receipt_sha256: str,
    signed_envelope_sha256: str,
    source_manifest: CanonicalJsonProofBinding,
    runtime_authority_manifest_sha256: str,
    inventory_sha256: str,
    prospective_pilot_materialization: CanonicalJsonProofBinding | None = None,
) -> str:
    lineage: dict[str, object] = {
        "schema_version": 1,
        "kind": "formal_stage_gpu_hour_registry_lineage",
        "registry_verification_receipt_sha256": registry_receipt_sha256,
        "signed_stage_gpu_hour_envelope_sha256": signed_envelope_sha256,
        "source_manifest": source_manifest.to_dict(),
        "runtime_authority_manifest_sha256": runtime_authority_manifest_sha256,
        "inventory_sha256": inventory_sha256,
    }
    if prospective_pilot_materialization is not None:
        lineage["prospective_pilot_materialization"] = (
            prospective_pilot_materialization.to_dict()
        )
    return content_sha256(lineage)


@dataclass(frozen=True)
class FormalStageGpuHourVerificationReceipt:
    """One replay-bound, source-reopened stage budget attached to a registry."""

    schema_version: Literal[3]
    kind: Literal["lightcone_formal_stage_gpu_hour_verification_receipt"]
    verified_ns: int
    stage: str
    registry_receipt_source: CanonicalJsonProofBinding
    signed_envelope: SignedStageGpuHourEnvelope
    source_manifest: CanonicalJsonProofBinding
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest
    inventory: GpuInventory
    prospective_pilot_materialization: CanonicalJsonProofBinding | None
    control_attestation: ControlArtifactAttestation
    reservation: ChallengeReplayReservationBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 3
            or self.kind != "lightcone_formal_stage_gpu_hour_verification_receipt"
        ):
            raise ValueError("formal stage GPU-hour receipt schema is unsupported")
        if type(self.verified_ns) is not int or self.verified_ns < 1:
            raise ValueError("formal stage GPU-hour verification time is invalid")
        if self.stage not in FORMAL_STAGE_DAG:
            raise ValueError("formal stage GPU-hour receipt names an unknown stage")
        for label, value, expected in (
            (
                "registry source",
                self.registry_receipt_source,
                CanonicalJsonProofBinding,
            ),
            ("signed envelope", self.signed_envelope, SignedStageGpuHourEnvelope),
            ("source manifest", self.source_manifest, CanonicalJsonProofBinding),
            (
                "runtime authority",
                self.formal_runtime_authority_manifest,
                FormalRuntimeAuthorityManifest,
            ),
            ("inventory", self.inventory, GpuInventory),
            ("control", self.control_attestation, ControlArtifactAttestation),
            ("reservation", self.reservation, ChallengeReplayReservationBinding),
        ):
            if type(value) is not expected:
                raise TypeError(f"formal stage GPU-hour {label} is not exact")
        if (
            self.prospective_pilot_materialization is not None
            and type(self.prospective_pilot_materialization)
            is not CanonicalJsonProofBinding
        ):
            raise TypeError(
                "formal stage GPU-hour prospective pilot materialization is not exact"
            )
        if self.reservation.reserved_ns != self.verified_ns:
            raise ValueError("formal stage GPU-hour reservation time differs")
        if self.signed_envelope.payload.schema_version != 2:
            raise ValueError("formal registry rejects legacy GPU-hour envelopes")
        if self.signed_envelope.payload.materialization_receipt_sha256 != (
            self.materialization_receipt_sha256
        ):
            raise ValueError("formal stage GPU-hour materialization differs")

    @property
    def materialization_receipt_sha256(self) -> str:
        return self.signed_envelope.payload.materialization_receipt_sha256

    @property
    def registry_receipt(self) -> FormalRegistryVerificationReceipt:
        """Deep-rebuild the compact registry layer at the acceptance time."""

        return _load_proof_carrying_registry_layer(
            self.registry_receipt_source,
            now_ns=self.verified_ns,
        )

    @property
    def control_lineage_sha256(self) -> str:
        return _control_lineage_sha256(
            registry_receipt_sha256=self.registry_receipt.sha256,
            signed_envelope_sha256=self.signed_envelope.sha256,
            source_manifest=self.source_manifest,
            runtime_authority_manifest_sha256=(
                self.formal_runtime_authority_manifest.sha256
            ),
            inventory_sha256=self.inventory.sha256,
            prospective_pilot_materialization=(self.prospective_pilot_materialization),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def revalidate(
        self, *, current_ns: int
    ) -> (
        LifecycleGpuHourSourceManifest
        | PreflightGpuHourSourceManifest
        | ProspectiveGpuHourSourceManifest
        | StagedProspectiveGpuHourSourceManifest
    ):
        self.__post_init__()
        if type(current_ns) is not int or current_ns < self.verified_ns:
            raise ValueError("formal stage GPU-hour revalidation precedes acceptance")
        manifest = self.registry_receipt.revalidate(current_ns=current_ns)
        protocol_lock = self.registry_receipt.signed_protocol_lock.payload
        if (
            manifest.inventory_sha256 != self.inventory.sha256
            or protocol_lock.formal_runtime_authority_manifest_sha256
            != self.formal_runtime_authority_manifest.sha256
            or self.signed_envelope.payload.protocol_lock_sha256 != protocol_lock.sha256
        ):
            raise ValueError("formal stage GPU-hour immutable root differs")
        materialization = _registered_materialization(
            self.registry_receipt, self.materialization_receipt_sha256
        )
        if materialization.stage != self.stage:
            raise ValueError("formal stage GPU-hour stage differs from materialization")
        policy = self.registry_receipt.trusted_release_policy(current_ns=current_ns)
        envelope = self.signed_envelope.verify(
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=self.verified_ns,
        )
        raw_source = self.source_manifest.reopen()
        if type(raw_source) is not dict:
            raise TypeError("formal stage GPU-hour source must be an object")
        if raw_source.get("kind") == "lifecycle_gpu_hour_source_manifest":
            if self.prospective_pilot_materialization is not None:
                raise ValueError(
                    "lifecycle GPU-hour source has prospective pilot input"
                )
            materialization, _coverage = _materialization_and_coverage(
                self.registry_receipt, self.materialization_receipt_sha256
            )
            source = revalidate_persisted_stage_gpu_hour_source_manifest(
                self.source_manifest.absolute_path,
                envelope=envelope,
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=(
                    self.formal_runtime_authority_manifest
                ),
                materialization=materialization,
                inventory=self.inventory,
                now_ns=current_ns,
            )
        elif raw_source.get("kind") == "preflight_gpu_hour_source_manifest":
            if self.prospective_pilot_materialization is not None:
                raise ValueError(
                    "preflight GPU-hour source has prospective pilot input"
                )
            materialization, coverage = _materialization_and_coverage(
                self.registry_receipt, self.materialization_receipt_sha256
            )
            if self.stage != "preflight":
                raise ValueError("preflight GPU-hour source was relabelled")
            source = revalidate_persisted_preflight_gpu_hour_source_manifest(
                self.source_manifest.absolute_path,
                envelope=envelope,
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=(
                    self.formal_runtime_authority_manifest
                ),
                materialization=materialization,
                stage_coverage=coverage,
                inventory=self.inventory,
                now_ns=current_ns,
            )
        elif raw_source.get("kind") == "prospective_gpu_hour_source_manifest":
            if self.prospective_pilot_materialization is None:
                raise ValueError(
                    "prospective GPU-hour source lacks pilot materialization"
                )
            pilot_binding = CanonicalJsonProofBinding.bind(
                self.prospective_pilot_materialization.absolute_path
            )
            if pilot_binding != self.prospective_pilot_materialization:
                raise ValueError("prospective pilot materialization binding changed")
            pilot_materialization = stage_materialization_receipt_from_dict(
                pilot_binding.reopen()
            )
            authority = verify_registered_prospective_gpu_hour_authority(
                registry_receipt=self.registry_receipt,
                pilot_materialization=pilot_materialization,
                final_materialization=materialization,
                current_ns=current_ns,
            )
            source = revalidate_persisted_prospective_gpu_hour_source_manifest(
                self.source_manifest.absolute_path,
                envelope=envelope,
                authority=authority,
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=(
                    self.formal_runtime_authority_manifest
                ),
                pilot_materialization=pilot_materialization,
                final_materialization=materialization,
                inventory=self.inventory,
                now_ns=current_ns,
            )
        elif raw_source.get("kind") == "staged_prospective_gpu_hour_source_manifest":
            if self.prospective_pilot_materialization is not None:
                raise ValueError(
                    "staged prospective GPU-hour source has a pilot materialization"
                )
            source = revalidate_persisted_staged_prospective_gpu_hour_source_manifest(
                self.source_manifest.absolute_path,
                envelope=envelope,
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=(
                    self.formal_runtime_authority_manifest
                ),
                materialization=materialization,
                inventory=self.inventory,
                now_ns=current_ns,
            )
        else:
            raise ValueError("formal stage GPU-hour source kind is unsupported")
        if (
            CanonicalJsonProofBinding.bind(self.source_manifest.absolute_path)
            != self.source_manifest
            or source.sha256 != envelope.signed_pilot_receipt_sha256
            or source.hardware_envelope_sha256
            != self.control_attestation.hardware_envelope_sha256
        ):
            raise ValueError("formal stage GPU-hour durable source differs")
        control = self.control_attestation
        subject = control.subject
        if (
            subject.artifact_type != "rank_aggregate"
            or subject.artifact_sha256 != self.signed_envelope.sha256
            or subject.protocol_sha256 != FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256
            or subject.registry_sha256 != protocol_lock.registry_sha256
            or subject.lineage_sha256 != self.control_lineage_sha256
        ):
            raise ValueError("formal stage GPU-hour control subject differs")
        verified = verify_release_control_artifact_attestation(
            control,
            expected_inventory_sha256=self.inventory.sha256,
            now_ns=self.verified_ns,
            consumed_challenge_sha256s=(),
        )
        expected_reservation = control_challenge_reservation_sha256(
            (verified,),
            additional_challenge_sha256s=(self.signed_envelope.challenge.sha256,),
            reserved_ns=self.verified_ns,
        )
        reserved = self.reservation.revalidate()
        if (
            expected_reservation != self.reservation.reservation_sha256
            or self.signed_envelope.challenge.sha256 not in reserved
            or verified.challenge_sha256 not in reserved
            or verified.deployment_policy_challenge_sha256 not in reserved
        ):
            raise ValueError("formal stage GPU-hour replay reservation differs")
        return source

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "verified_ns": self.verified_ns,
            "stage": self.stage,
            "registry_receipt_source": self.registry_receipt_source.to_dict(),
            "signed_envelope": signed_stage_gpu_hour_to_dict(self.signed_envelope),
            "source_manifest": self.source_manifest.to_dict(),
            "formal_runtime_authority_manifest": (
                formal_runtime_authority_manifest_to_dict(
                    self.formal_runtime_authority_manifest
                )
            ),
            "inventory": self.inventory.to_dict(),
            "prospective_pilot_materialization": (
                None
                if self.prospective_pilot_materialization is None
                else self.prospective_pilot_materialization.to_dict()
            ),
            "control_attestation": self.control_attestation.to_dict(),
            "reservation": self.reservation.to_dict(),
            "receipt_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage GPU-hour verification receipt",
            value,
            {
                "schema_version",
                "kind",
                "verified_ns",
                "stage",
                "registry_receipt_source",
                "signed_envelope",
                "source_manifest",
                "formal_runtime_authority_manifest",
                "inventory",
                "prospective_pilot_materialization",
                "control_attestation",
                "reservation",
                "receipt_sha256",
            },
        )
        declared = _sha256("formal stage GPU-hour receipt", row.pop("receipt_sha256"))
        receipt = cls(
            schema_version=row["schema_version"],
            kind=row["kind"],
            verified_ns=row["verified_ns"],
            stage=row["stage"],
            registry_receipt_source=CanonicalJsonProofBinding.from_dict(
                row["registry_receipt_source"]
            ),
            signed_envelope=signed_stage_gpu_hour_from_dict(row["signed_envelope"]),
            source_manifest=CanonicalJsonProofBinding.from_dict(row["source_manifest"]),
            formal_runtime_authority_manifest=(
                formal_runtime_authority_manifest_from_dict(
                    row["formal_runtime_authority_manifest"]
                )
            ),
            inventory=GpuInventory.from_dict(row["inventory"]),
            prospective_pilot_materialization=(
                None
                if row["prospective_pilot_materialization"] is None
                else CanonicalJsonProofBinding.from_dict(
                    row["prospective_pilot_materialization"]
                )
            ),
            control_attestation=ControlArtifactAttestation.from_dict(
                row["control_attestation"]
            ),
            reservation=ChallengeReplayReservationBinding.from_dict(row["reservation"]),
        )
        if receipt.sha256 != declared:
            raise ValueError("formal stage GPU-hour receipt digest differs")
        return receipt


def reserve_formal_stage_gpu_hour_verification_receipt(
    *,
    registry_receipt: FormalRegistryVerificationReceipt,
    registry_receipt_path: str,
    signed_envelope: SignedStageGpuHourEnvelope,
    source_manifest_path: str,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    inventory: GpuInventory,
    control_attestation: ControlArtifactAttestation,
    replay_store: ChallengeReplayStore,
    now_ns: int,
    prospective_pilot_materialization_path: str | None = None,
) -> FormalStageGpuHourVerificationReceipt:
    """Verify all sources and reserve the stage-budget authority once."""

    if type(registry_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal GPU-hour reservation requires registry verification")
    registry_receipt_source = CanonicalJsonProofBinding.bind(registry_receipt_path)
    rebuilt_registry_receipt = _load_proof_carrying_registry_layer(
        registry_receipt_source,
        now_ns=now_ns,
    )
    if rebuilt_registry_receipt != registry_receipt:
        raise ValueError("formal GPU-hour registry source reconstructs another receipt")
    if type(signed_envelope) is not SignedStageGpuHourEnvelope:
        raise TypeError("formal GPU-hour reservation requires a signed envelope")
    if type(inventory) is not GpuInventory or len(inventory.devices) != 2:
        raise ValueError("formal GPU-hour reservation requires exact two-GPU inventory")
    manifest = registry_receipt.revalidate(current_ns=now_ns)
    if manifest.inventory_sha256 != inventory.sha256:
        raise ValueError("formal GPU-hour inventory differs from registry")
    materialization = _registered_materialization(
        registry_receipt,
        signed_envelope.payload.materialization_receipt_sha256,
    )
    policy = registry_receipt.trusted_release_policy(current_ns=now_ns)
    envelope = signed_envelope.verify(
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    if envelope.schema_version != 2:
        raise ValueError("formal registry rejects legacy GPU-hour envelopes")
    protocol_lock = registry_receipt.signed_protocol_lock.payload
    source_binding = CanonicalJsonProofBinding.bind(source_manifest_path)
    prospective_pilot_binding = (
        None
        if prospective_pilot_materialization_path is None
        else CanonicalJsonProofBinding.bind(prospective_pilot_materialization_path)
    )
    raw_source = source_binding.reopen()
    if type(raw_source) is not dict:
        raise TypeError("formal GPU-hour source must be an object")
    if raw_source.get("kind") == "lifecycle_gpu_hour_source_manifest":
        if prospective_pilot_binding is not None:
            raise ValueError("lifecycle GPU-hour source has prospective pilot input")
        materialization, _coverage = _materialization_and_coverage(
            registry_receipt,
            signed_envelope.payload.materialization_receipt_sha256,
        )
        source = revalidate_persisted_stage_gpu_hour_source_manifest(
            source_binding.absolute_path,
            envelope=envelope,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    elif raw_source.get("kind") == "preflight_gpu_hour_source_manifest":
        if prospective_pilot_binding is not None:
            raise ValueError("preflight GPU-hour source has prospective pilot input")
        materialization, coverage = _materialization_and_coverage(
            registry_receipt,
            signed_envelope.payload.materialization_receipt_sha256,
        )
        if materialization.stage != "preflight":
            raise ValueError("preflight GPU-hour source was relabelled")
        source = revalidate_persisted_preflight_gpu_hour_source_manifest(
            source_binding.absolute_path,
            envelope=envelope,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            materialization=materialization,
            stage_coverage=coverage,
            inventory=inventory,
            now_ns=now_ns,
        )
    elif raw_source.get("kind") == "prospective_gpu_hour_source_manifest":
        if prospective_pilot_binding is None:
            raise ValueError("prospective GPU-hour source lacks pilot materialization")
        pilot_materialization = stage_materialization_receipt_from_dict(
            prospective_pilot_binding.reopen()
        )
        authority = verify_registered_prospective_gpu_hour_authority(
            registry_receipt=registry_receipt,
            pilot_materialization=pilot_materialization,
            final_materialization=materialization,
            current_ns=now_ns,
        )
        source = revalidate_persisted_prospective_gpu_hour_source_manifest(
            source_binding.absolute_path,
            envelope=envelope,
            authority=authority,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            pilot_materialization=pilot_materialization,
            final_materialization=materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    elif raw_source.get("kind") == "staged_prospective_gpu_hour_source_manifest":
        if prospective_pilot_binding is not None:
            raise ValueError(
                "staged prospective GPU-hour source has a pilot materialization"
            )
        source = revalidate_persisted_staged_prospective_gpu_hour_source_manifest(
            source_binding.absolute_path,
            envelope=envelope,
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=formal_runtime_authority_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    else:
        raise ValueError("formal GPU-hour source kind is unsupported")
    lineage = _control_lineage_sha256(
        registry_receipt_sha256=registry_receipt.sha256,
        signed_envelope_sha256=signed_envelope.sha256,
        source_manifest=source_binding,
        runtime_authority_manifest_sha256=formal_runtime_authority_manifest.sha256,
        inventory_sha256=inventory.sha256,
        prospective_pilot_materialization=prospective_pilot_binding,
    )
    subject = control_attestation.subject
    if (
        subject.artifact_type != "rank_aggregate"
        or subject.artifact_sha256 != signed_envelope.sha256
        or subject.protocol_sha256 != FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256
        or subject.registry_sha256 != protocol_lock.registry_sha256
        or subject.lineage_sha256 != lineage
        or control_attestation.hardware_envelope_sha256
        != source.hardware_envelope_sha256
    ):
        raise ValueError("formal stage GPU-hour control subject differs")
    verified = verify_and_reserve_release_control_artifact_attestations(
        (control_attestation,),
        expected_inventory_sha256=inventory.sha256,
        now_ns=now_ns,
        replay_store=replay_store,
        additional_challenge_sha256s=(signed_envelope.challenge.sha256,),
    )
    reservation_sha256 = control_challenge_reservation_sha256(
        verified,
        additional_challenge_sha256s=(signed_envelope.challenge.sha256,),
        reserved_ns=now_ns,
    )
    receipt = FormalStageGpuHourVerificationReceipt(
        schema_version=3,
        kind="lightcone_formal_stage_gpu_hour_verification_receipt",
        verified_ns=now_ns,
        stage=materialization.stage,
        registry_receipt_source=registry_receipt_source,
        signed_envelope=signed_envelope,
        source_manifest=source_binding,
        formal_runtime_authority_manifest=formal_runtime_authority_manifest,
        inventory=inventory,
        prospective_pilot_materialization=prospective_pilot_binding,
        control_attestation=control_attestation,
        reservation=replay_store.bind_reservation(reservation_sha256),
    )
    receipt.revalidate(current_ns=now_ns)
    return receipt


@dataclass(frozen=True)
class FormalStudyGpuHourEstimate:
    """Aggregate of durable stage receipts for one exact registry prefix."""

    schema_version: Literal[2]
    kind: Literal["lightcone_formal_study_gpu_hour_estimate"]
    status: Literal["PARTIAL", "COMPLETE"]
    registry_receipt_sha256: str
    stage_receipt_sha256s: tuple[str, ...]
    covered_materialization_sha256s: tuple[str, ...]
    missing_materialization_sha256s: tuple[str, ...]
    covered_materialized_cell_ids: tuple[str, ...]
    compute_gpu_hours: float
    reserved_gpu_hours: float
    estimated_wall_hours: float
    retry_reserve_gpu_hours: float
    profile_reserve_gpu_hours: float
    evidence_reserve_gpu_hours: float
    derivation_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != 2
            or self.kind != "lightcone_formal_study_gpu_hour_estimate"
            or self.status not in {"PARTIAL", "COMPLETE"}
        ):
            raise ValueError("formal study GPU-hour estimate schema is unsupported")
        _sha256("formal study registry receipt", self.registry_receipt_sha256)
        for label, values in (
            ("stage receipts", self.stage_receipt_sha256s),
            ("covered materializations", self.covered_materialization_sha256s),
            ("missing materializations", self.missing_materialization_sha256s),
            ("covered materialized cells", self.covered_materialized_cell_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"formal study {label} are not canonical")
            for digest in values:
                _sha256(f"formal study {label}", digest)
        if set(self.covered_materialization_sha256s) & set(
            self.missing_materialization_sha256s
        ):
            raise ValueError("formal study materialization coverage overlaps")
        expected_status = (
            "COMPLETE" if not self.missing_materialization_sha256s else "PARTIAL"
        )
        if self.status != expected_status:
            raise ValueError("formal study GPU-hour status differs from coverage")
        numeric = (
            self.compute_gpu_hours,
            self.reserved_gpu_hours,
            self.estimated_wall_hours,
            self.retry_reserve_gpu_hours,
            self.profile_reserve_gpu_hours,
            self.evidence_reserve_gpu_hours,
        )
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0
            for value in numeric
        ):
            raise ValueError("formal study GPU-hour totals must be finite floats")
        expected_derivation = content_sha256(
            {
                "registry_receipt_sha256": self.registry_receipt_sha256,
                "stage_receipt_sha256s": self.stage_receipt_sha256s,
                "covered_materialization_sha256s": (
                    self.covered_materialization_sha256s
                ),
                "missing_materialization_sha256s": (
                    self.missing_materialization_sha256s
                ),
                "covered_materialized_cell_ids": (self.covered_materialized_cell_ids),
                "compute_gpu_hours": self.compute_gpu_hours,
                "reserved_gpu_hours": self.reserved_gpu_hours,
                "estimated_wall_hours": self.estimated_wall_hours,
                "retry_reserve_gpu_hours": self.retry_reserve_gpu_hours,
                "profile_reserve_gpu_hours": self.profile_reserve_gpu_hours,
                "evidence_reserve_gpu_hours": self.evidence_reserve_gpu_hours,
            }
        )
        if self.derivation_sha256 != expected_derivation:
            raise ValueError("formal study GPU-hour derivation differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "status": self.status,
            "registry_receipt_sha256": self.registry_receipt_sha256,
            "stage_receipt_sha256s": list(self.stage_receipt_sha256s),
            "covered_materialization_sha256s": list(
                self.covered_materialization_sha256s
            ),
            "missing_materialization_sha256s": list(
                self.missing_materialization_sha256s
            ),
            "covered_materialized_cell_ids": list(self.covered_materialized_cell_ids),
            "compute_gpu_hours": self.compute_gpu_hours,
            "reserved_gpu_hours": self.reserved_gpu_hours,
            "estimated_wall_hours": self.estimated_wall_hours,
            "retry_reserve_gpu_hours": self.retry_reserve_gpu_hours,
            "profile_reserve_gpu_hours": self.profile_reserve_gpu_hours,
            "evidence_reserve_gpu_hours": self.evidence_reserve_gpu_hours,
            "derivation_sha256": self.derivation_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal study GPU-hour estimate",
            value,
            set(cls.__dataclass_fields__),
        )
        for field in (
            "stage_receipt_sha256s",
            "covered_materialization_sha256s",
            "missing_materialization_sha256s",
            "covered_materialized_cell_ids",
        ):
            raw = row[field]
            if type(raw) is not list or any(type(item) is not str for item in raw):
                raise TypeError(f"formal study GPU-hour {field} must be a list")
            row[field] = tuple(raw)
        return cls(**row)


_FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS = (
    "materialized cell",
    "actual source materialized cell",
    "source manifest raw",
    "source manifest semantic",
    "source manifest path",
    "pilot materialization",
    "pilot materialization raw",
    "pilot materialization semantic",
    "pilot materialization path",
    "power authority",
    "power challenge",
    "prospective mapping",
    "execution identity",
    "proof raw",
    "proof semantic",
    "proof path",
    "proof payload",
    "raw timing",
    "live run receipt",
    "native result proof",
    "run binding",
    "native run nonce",
    "native challenge nonce",
    "control",
    "replay reservation",
    "replay raw",
    "replay path",
    "replay challenge",
)


def _reserve_unique_evidence_identities(
    observed: dict[str, set[str]],
    incoming: dict[str, set[str]],
) -> None:
    expected = set(_FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS)
    if set(observed) != expected or set(incoming) != expected:
        raise ValueError("formal study evidence identity domains differ")
    for label in _FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS:
        values = incoming[label]
        if any(type(value) is not str or not value for value in values):
            raise ValueError(f"formal study {label} evidence is invalid")
        if observed[label] & values:
            raise ValueError(f"formal study reuses {label} evidence")
    for label in _FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS:
        observed[label].update(incoming[label])


def _require_identity_count(
    values: set[str],
    *,
    expected: int,
    label: str,
) -> None:
    if len(values) != expected:
        raise ValueError(f"formal stage repeats {label} evidence")


def _formal_stage_evidence_identities(
    receipt: FormalStageGpuHourVerificationReceipt,
    source: (
        LifecycleGpuHourSourceManifest
        | PreflightGpuHourSourceManifest
        | ProspectiveGpuHourSourceManifest
        | StagedProspectiveGpuHourSourceManifest
    ),
) -> dict[str, set[str]]:
    """Normalize serving and non-serving proof DAGs into reuse domains.

    Execution, lifecycle/preflight, and stage wrappers deliberately share the
    same control, reservation, and challenge domains.  This prevents a later
    stage from laundering one challenge through a differently named wrapper.
    """

    identities: dict[str, set[str]] = {
        label: set() for label in _FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS
    }
    identities["source manifest raw"].add(receipt.source_manifest.raw_sha256)
    identities["source manifest semantic"].add(receipt.source_manifest.semantic_sha256)
    identities["source manifest path"].add(receipt.source_manifest.absolute_path)
    identities["control"].add(receipt.control_attestation.sha256)
    identities["replay reservation"].add(receipt.reservation.reservation_sha256)
    identities["replay raw"].add(receipt.reservation.raw_sha256)
    identities["replay path"].add(receipt.reservation.path)
    stage_challenges = tuple(receipt.reservation.challenge_sha256s)
    identities["replay challenge"].update(stage_challenges)
    if len(stage_challenges) != len(set(stage_challenges)):
        raise ValueError("formal stage repeats replay challenge evidence")

    if type(source) is LifecycleGpuHourSourceManifest:
        rows = source.observations
        identities["materialized cell"].update(row.materialized_cell_id for row in rows)
        _require_identity_count(
            identities["materialized cell"],
            expected=len(rows),
            label="materialized cell",
        )
        identities["execution identity"].update(
            row.execution_binding_sha256 for row in rows
        )
        for row in rows:
            identities["proof raw"].update(
                (row.execution_proof.raw_sha256, row.lifecycle_proof.raw_sha256)
            )
            identities["proof semantic"].update(
                (
                    row.execution_proof.semantic_sha256,
                    row.lifecycle_proof.semantic_sha256,
                )
            )
            identities["proof path"].update(
                (
                    row.execution_proof.absolute_path,
                    row.lifecycle_proof.absolute_path,
                )
            )
            identities["proof payload"].update(
                (
                    row.execution_proof_payload_sha256,
                    row.verified_lifecycle_proof_sha256,
                )
            )
            identities["raw timing"].add(row.raw_timing_sha256)
            identities["live run receipt"].add(row.live_run_receipt_sha256)
            identities["native result proof"].add(row.native_result_proof_sha256)
            identities["run binding"].add(row.run_binding_sha256)
            identities["native run nonce"].add(row.native_run_binding.run_nonce_sha256)
            identities["native challenge nonce"].add(
                row.native_run_binding.challenge_nonce_sha256
            )
            identities["control"].update(
                (
                    row.execution_control_envelope_sha256,
                    row.control_envelope_sha256,
                )
            )
            for reservation in (
                row.execution_replay_reservation,
                row.lifecycle_replay_reservation,
            ):
                identities["replay reservation"].add(reservation.reservation_sha256)
                identities["replay raw"].add(reservation.raw_sha256)
                identities["replay path"].add(reservation.path)
                challenges = tuple(reservation.challenge_sha256s)
                if len(challenges) != len(set(challenges)):
                    raise ValueError("formal stage repeats replay challenge evidence")
                identities["replay challenge"].update(challenges)
        expected = len(rows)
        for label, count in (
            ("execution identity", expected),
            ("proof raw", 2 * expected),
            ("proof semantic", 2 * expected),
            ("proof path", 2 * expected),
            ("proof payload", 2 * expected),
            ("raw timing", expected),
            ("live run receipt", expected),
            ("native result proof", expected),
            ("run binding", expected),
            ("native run nonce", expected),
            ("native challenge nonce", expected),
            ("control", 2 * expected + 1),
            ("replay reservation", 2 * expected + 1),
            ("replay raw", 2 * expected + 1),
            ("replay path", 2 * expected + 1),
        ):
            _require_identity_count(identities[label], expected=count, label=label)
        challenge_count = len(stage_challenges) + sum(
            len(row.execution_replay_reservation.challenge_sha256s)
            + len(row.lifecycle_replay_reservation.challenge_sha256s)
            for row in rows
        )
    elif type(source) is PreflightGpuHourSourceManifest:
        rows = source.observations
        identities["materialized cell"].update(row.materialized_cell_id for row in rows)
        _require_identity_count(
            identities["materialized cell"],
            expected=len(rows),
            label="materialized cell",
        )
        identities["execution identity"].update(
            row.execution_identity_sha256 for row in rows
        )
        for row in rows:
            identities["proof raw"].add(row.timing_proof.raw_sha256)
            identities["proof semantic"].add(row.timing_proof.semantic_sha256)
            identities["proof path"].add(row.timing_proof.absolute_path)
            identities["proof payload"].add(row.timing_authority_sha256)
            identities["control"].add(row.control_envelope_sha256)
            reservation = row.replay_reservation
            identities["replay reservation"].add(reservation.reservation_sha256)
            identities["replay raw"].add(reservation.raw_sha256)
            identities["replay path"].add(reservation.path)
            challenges = tuple(reservation.challenge_sha256s)
            if len(challenges) != len(set(challenges)):
                raise ValueError("formal stage repeats replay challenge evidence")
            identities["replay challenge"].update(challenges)
        expected = len(rows)
        for label, count in (
            ("execution identity", expected),
            ("proof raw", expected),
            ("proof semantic", expected),
            ("proof path", expected),
            ("proof payload", expected),
            ("control", expected + 1),
            ("replay reservation", expected + 1),
            ("replay raw", expected + 1),
            ("replay path", expected + 1),
        ):
            _require_identity_count(identities[label], expected=count, label=label)
        challenge_count = len(stage_challenges) + sum(
            len(row.replay_reservation.challenge_sha256s) for row in rows
        )
    elif type(source) is StagedProspectiveGpuHourSourceManifest:
        materialization = _registered_materialization(
            receipt.registry_receipt,
            receipt.materialization_receipt_sha256,
        )
        identities["materialized cell"].update(
            cell.cell_id for cell in materialization.cells
        )
        _require_identity_count(
            identities["materialized cell"],
            expected=len(materialization.cells),
            label="materialized cell",
        )
        if source.completed_source_manifest is None:
            raise ValueError("READY staged GPU-hour source lacks completed evidence")
        nested_binding = source.completed_source_manifest
        if (
            CanonicalJsonProofBinding.bind(nested_binding.absolute_path)
            != nested_binding
        ):
            raise ValueError("staged prospective nested source binding changed")
        identities["source manifest raw"].add(nested_binding.raw_sha256)
        identities["source manifest semantic"].add(nested_binding.semantic_sha256)
        identities["source manifest path"].add(nested_binding.absolute_path)
        for label in (
            "source manifest raw",
            "source manifest semantic",
            "source manifest path",
        ):
            _require_identity_count(identities[label], expected=2, label=label)
        identities["prospective mapping"].add(source.mapping_sha256)
        nested_source = LifecycleGpuHourSourceManifest.from_dict(
            nested_binding.reopen()
        )
        rows = nested_source.observations
        identities["actual source materialized cell"].update(
            row.materialized_cell_id for row in rows
        )
        _require_identity_count(
            identities["actual source materialized cell"],
            expected=len(rows),
            label="actual source materialized cell",
        )
        identities["execution identity"].update(
            row.execution_binding_sha256 for row in rows
        )
        for row in rows:
            identities["proof raw"].update(
                (row.execution_proof.raw_sha256, row.lifecycle_proof.raw_sha256)
            )
            identities["proof semantic"].update(
                (
                    row.execution_proof.semantic_sha256,
                    row.lifecycle_proof.semantic_sha256,
                )
            )
            identities["proof path"].update(
                (
                    row.execution_proof.absolute_path,
                    row.lifecycle_proof.absolute_path,
                )
            )
            identities["proof payload"].update(
                (
                    row.execution_proof_payload_sha256,
                    row.verified_lifecycle_proof_sha256,
                )
            )
            identities["raw timing"].add(row.raw_timing_sha256)
            identities["live run receipt"].add(row.live_run_receipt_sha256)
            identities["native result proof"].add(row.native_result_proof_sha256)
            identities["run binding"].add(row.run_binding_sha256)
            identities["native run nonce"].add(row.native_run_binding.run_nonce_sha256)
            identities["native challenge nonce"].add(
                row.native_run_binding.challenge_nonce_sha256
            )
            identities["control"].update(
                (
                    row.execution_control_envelope_sha256,
                    row.control_envelope_sha256,
                )
            )
            for reservation in (
                row.execution_replay_reservation,
                row.lifecycle_replay_reservation,
            ):
                identities["replay reservation"].add(reservation.reservation_sha256)
                identities["replay raw"].add(reservation.raw_sha256)
                identities["replay path"].add(reservation.path)
                challenges = tuple(reservation.challenge_sha256s)
                if len(challenges) != len(set(challenges)):
                    raise ValueError("formal stage repeats replay challenge evidence")
                identities["replay challenge"].update(challenges)
        expected = len(rows)
        for label, count in (
            ("execution identity", expected),
            ("proof raw", 2 * expected),
            ("proof semantic", 2 * expected),
            ("proof path", 2 * expected),
            ("proof payload", 2 * expected),
            ("raw timing", expected),
            ("live run receipt", expected),
            ("native result proof", expected),
            ("run binding", expected),
            ("native run nonce", expected),
            ("native challenge nonce", expected),
            ("control", 2 * expected + 1),
            ("replay reservation", 2 * expected + 1),
            ("replay raw", 2 * expected + 1),
            ("replay path", 2 * expected + 1),
        ):
            _require_identity_count(identities[label], expected=count, label=label)
        challenge_count = len(stage_challenges) + sum(
            len(row.execution_replay_reservation.challenge_sha256s)
            + len(row.lifecycle_replay_reservation.challenge_sha256s)
            for row in rows
        )
    elif type(source) is ProspectiveGpuHourSourceManifest:
        materialization = _registered_materialization(
            receipt.registry_receipt,
            receipt.materialization_receipt_sha256,
        )
        identities["materialized cell"].update(
            cell.cell_id for cell in materialization.cells
        )
        _require_identity_count(
            identities["materialized cell"],
            expected=len(materialization.cells),
            label="materialized cell",
        )
        pilot_materialization = receipt.prospective_pilot_materialization
        if pilot_materialization is None:
            raise ValueError("prospective GPU-hour receipt lacks pilot materialization")
        pilot_receipt = stage_materialization_receipt_from_dict(
            pilot_materialization.reopen()
        )
        if pilot_receipt.sha256 != source.pilot_materialization_receipt_sha256:
            raise ValueError("prospective pilot materialization identity differs")
        identities["materialized cell"].update(
            cell.cell_id for cell in pilot_receipt.cells
        )
        _require_identity_count(
            identities["materialized cell"],
            expected=len(materialization.cells) + len(pilot_receipt.cells),
            label="materialized cell",
        )
        identities["pilot materialization"].add(
            source.pilot_materialization_receipt_sha256
        )
        identities["pilot materialization raw"].add(pilot_materialization.raw_sha256)
        identities["pilot materialization semantic"].add(
            pilot_materialization.semantic_sha256
        )
        identities["pilot materialization path"].add(
            pilot_materialization.absolute_path
        )
        identities["power authority"].add(source.signed_power_authority_sha256)
        identities["power challenge"].add(source.signed_power_challenge_sha256)
        identities["prospective mapping"].add(source.mapping_sha256)
        identities["replay challenge"].add(source.signed_power_challenge_sha256)

        nested_bindings = [source.pilot_source_manifest]
        if source.one_shot_source_manifest is not None:
            nested_bindings.append(source.one_shot_source_manifest)
        nested_sources: list[LifecycleGpuHourSourceManifest] = []
        for binding in nested_bindings:
            if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
                raise ValueError("prospective nested source binding changed")
            identities["source manifest raw"].add(binding.raw_sha256)
            identities["source manifest semantic"].add(binding.semantic_sha256)
            identities["source manifest path"].add(binding.absolute_path)
            nested_sources.append(
                LifecycleGpuHourSourceManifest.from_dict(binding.reopen())
            )
        expected_source_count = 1 + len(nested_bindings)
        for label in (
            "source manifest raw",
            "source manifest semantic",
            "source manifest path",
        ):
            _require_identity_count(
                identities[label],
                expected=expected_source_count,
                label=label,
            )
        rows = tuple(
            row
            for nested_source in nested_sources
            for row in nested_source.observations
        )
        identities["actual source materialized cell"].update(
            row.materialized_cell_id for row in rows
        )
        _require_identity_count(
            identities["actual source materialized cell"],
            expected=len(rows),
            label="actual source materialized cell",
        )
        identities["execution identity"].update(
            row.execution_binding_sha256 for row in rows
        )
        for row in rows:
            identities["proof raw"].update(
                (row.execution_proof.raw_sha256, row.lifecycle_proof.raw_sha256)
            )
            identities["proof semantic"].update(
                (
                    row.execution_proof.semantic_sha256,
                    row.lifecycle_proof.semantic_sha256,
                )
            )
            identities["proof path"].update(
                (
                    row.execution_proof.absolute_path,
                    row.lifecycle_proof.absolute_path,
                )
            )
            identities["proof payload"].update(
                (
                    row.execution_proof_payload_sha256,
                    row.verified_lifecycle_proof_sha256,
                )
            )
            identities["raw timing"].add(row.raw_timing_sha256)
            identities["live run receipt"].add(row.live_run_receipt_sha256)
            identities["native result proof"].add(row.native_result_proof_sha256)
            identities["run binding"].add(row.run_binding_sha256)
            identities["native run nonce"].add(row.native_run_binding.run_nonce_sha256)
            identities["native challenge nonce"].add(
                row.native_run_binding.challenge_nonce_sha256
            )
            identities["control"].update(
                (
                    row.execution_control_envelope_sha256,
                    row.control_envelope_sha256,
                )
            )
            for reservation in (
                row.execution_replay_reservation,
                row.lifecycle_replay_reservation,
            ):
                identities["replay reservation"].add(reservation.reservation_sha256)
                identities["replay raw"].add(reservation.raw_sha256)
                identities["replay path"].add(reservation.path)
                challenges = tuple(reservation.challenge_sha256s)
                if len(challenges) != len(set(challenges)):
                    raise ValueError("formal stage repeats replay challenge evidence")
                identities["replay challenge"].update(challenges)
        expected = len(rows)
        for label, count in (
            ("execution identity", expected),
            ("proof raw", 2 * expected),
            ("proof semantic", 2 * expected),
            ("proof path", 2 * expected),
            ("proof payload", 2 * expected),
            ("raw timing", expected),
            ("live run receipt", expected),
            ("native result proof", expected),
            ("run binding", expected),
            ("native run nonce", expected),
            ("native challenge nonce", expected),
            ("control", 2 * expected + 1),
            ("replay reservation", 2 * expected + 1),
            ("replay raw", 2 * expected + 1),
            ("replay path", 2 * expected + 1),
        ):
            _require_identity_count(identities[label], expected=count, label=label)
        challenge_count = (
            len(stage_challenges)
            + 1
            + sum(
                len(row.execution_replay_reservation.challenge_sha256s)
                + len(row.lifecycle_replay_reservation.challenge_sha256s)
                for row in rows
            )
        )
    else:  # pragma: no cover - the public receipt revalidator closes this union
        raise TypeError("formal stage GPU-hour source kind is unsupported")

    _require_identity_count(
        identities["replay challenge"],
        expected=challenge_count,
        label="replay challenge",
    )
    return identities


def aggregate_formal_study_gpu_hours(
    *,
    registry_receipt: FormalRegistryVerificationReceipt,
    stage_receipts: tuple[FormalStageGpuHourVerificationReceipt, ...],
    current_ns: int,
    require_complete: bool = True,
) -> FormalStudyGpuHourEstimate:
    """Aggregate exact durable stage budgets without reusing lifecycle proof."""

    if type(registry_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal study GPU-hour aggregation requires registry receipt")
    if type(stage_receipts) is not tuple or any(
        type(row) is not FormalStageGpuHourVerificationReceipt for row in stage_receipts
    ):
        raise TypeError("formal study GPU-hour aggregation requires exact receipts")
    registry_receipt.revalidate(current_ns=current_ns)
    allowed_registry_receipts: set[str] = set()
    cursor: FormalRegistryVerificationReceipt | None = registry_receipt
    while cursor is not None:
        allowed_registry_receipts.add(cursor.sha256)
        cursor = cursor.prior_receipt
    materialization_ids = tuple(
        sorted(
            row.payload.sha256
            for row in registry_receipt.cumulative_signed_materializations
        )
    )
    by_materialization: dict[str, FormalStageGpuHourVerificationReceipt] = {}
    reused_identity_sets: dict[str, set[str]] = {
        label: set() for label in _FORMAL_STUDY_EVIDENCE_IDENTITY_LABELS
    }

    for receipt in stage_receipts:
        source = receipt.revalidate(current_ns=current_ns)
        if receipt.registry_receipt.sha256 not in allowed_registry_receipts:
            raise ValueError("formal stage GPU-hour receipt is outside registry prefix")
        covered_by_receipt = _covered_materialization_sha256s(receipt, source)
        if set(covered_by_receipt) & set(by_materialization):
            raise ValueError("formal study repeats a materialization budget")
        identity_values = _formal_stage_evidence_identities(receipt, source)
        _reserve_unique_evidence_identities(reused_identity_sets, identity_values)
        by_materialization.update(
            (materialization_sha, receipt) for materialization_sha in covered_by_receipt
        )
    extras = set(by_materialization) - set(materialization_ids)
    if extras:
        raise ValueError(
            "formal study GPU-hour receipt covers a foreign materialization"
        )
    covered = tuple(sorted(by_materialization))
    missing = tuple(sorted(set(materialization_ids) - set(covered)))
    covered_cells = tuple(sorted(reused_identity_sets["materialized cell"]))
    if require_complete and missing:
        raise ValueError("formal study GPU-hour coverage is incomplete")
    estimates = tuple(
        receipt.signed_envelope.payload.estimate
        for receipt in sorted(stage_receipts, key=lambda row: row.sha256)
    )

    def total(field: str) -> float:
        values = tuple(getattr(row, field) for row in estimates)
        if any(type(value) is not float for value in values):
            raise ValueError("formal study GPU-hour estimate is not AVAILABLE")
        return float(math.fsum(values))

    totals = {
        field: total(field)
        for field in (
            "compute_gpu_hours",
            "reserved_gpu_hours",
            "estimated_wall_hours",
            "retry_reserve_gpu_hours",
            "profile_reserve_gpu_hours",
            "evidence_reserve_gpu_hours",
        )
    }
    derivation_values = {
        "registry_receipt_sha256": registry_receipt.sha256,
        "stage_receipt_sha256s": tuple(sorted(row.sha256 for row in stage_receipts)),
        "covered_materialization_sha256s": covered,
        "missing_materialization_sha256s": missing,
        "covered_materialized_cell_ids": covered_cells,
        **totals,
    }
    return FormalStudyGpuHourEstimate(
        schema_version=2,
        kind="lightcone_formal_study_gpu_hour_estimate",
        status="COMPLETE" if not missing else "PARTIAL",
        derivation_sha256=content_sha256(derivation_values),
        **derivation_values,
    )


__all__ = [
    "FORMAL_GPU_HOUR_REGISTRY_PROTOCOL_SHA256",
    "FormalStageGpuHourVerificationReceipt",
    "FormalStudyGpuHourEstimate",
    "aggregate_formal_study_gpu_hours",
    "reserve_formal_stage_gpu_hour_verification_receipt",
]
