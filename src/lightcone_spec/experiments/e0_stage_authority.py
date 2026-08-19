"""Source-owned tuning and power authorities for the formal E0 breadth stage.

E0 is intentionally not materialized from caller supplied recipes or a block
count.  Every compatible model/backend/task row first reopens the registered
OnlineSPEC source checkout and an exact tuning-only execution universe.  The
three independently selected OnlineSPEC recipes are signed per compatibility
row.  Four disjoint pilot blocks are then reopened to derive the only signed
12--20 block prefix accepted by the main registry.

The helpers in this module do not make E0 executable by themselves.  They
consume verifier-sealed serving bindings; if the runtime/content mapper cannot
produce such a binding for a model/backend/method row, reduction fails closed.
"""

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
from lightcone_spec.experiments.e6_stage_authority import (
    E6NextnModelAuthorityInput,
    SignedE6ConfirmationReceipt,
    SignedE6ModelCompatibilityReceipt,
)
from lightcone_spec.experiments.formal_protocol import (
    E0_METHOD_ROLES,
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.onlinespec import (
    ONLINE_SPEC_COMMIT,
    ONLINE_SPEC_METHODS,
    ONLINE_SPEC_SOURCE_AUDIT_SHA256,
    ONLINE_SPEC_TREE,
    OnlineSpecCandidate,
    onlinespec_candidates,
    verify_onlinespec_source_checkout,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_BACKENDS,
    E0_LOADS,
    E0_MODELS,
    E0_TASKS,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    SignedE0CompatibilityReceipt,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.statistics import (
    PILOT_BLOCK_COUNT,
    PilotBlock,
    PowerSizingPlan,
    preregister_power_sizing,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)

E0_ONLINESPEC_ROLES = E0_METHOD_ROLES[-3:]
E0_ONLINESPEC_TUNING_RULE = (
    "e0_full_registered_onlinespec_grid_per_valid_combination_tuning_only"
)
E0_EXCLUDED_PILOT_RULE = (
    "e0_exact_16_rows_per_valid_combination_x_4_excluded_pilot_blocks"
)
E0_FINAL_MATERIALIZATION_RULE = (
    "valid_compatibilities_x_8_roles_x_2_loads_x_final_only_powered_prefix"
)


def _role_for_method(method: str, *, candidate: bool = False) -> str:
    try:
        role = {
            "onlinespec_ogd": "OnlineSPEC-OGD",
            "onlinespec_opt": "OnlineSPEC-OPT",
            "onlinespec_ens": "OnlineSPEC-ENS",
        }[method]
    except KeyError as error:
        raise ValueError("E0 OnlineSPEC method is outside the exact panel") from error
    return f"{role}-candidate" if candidate else role


def _method_for_role(role: str) -> str:
    normalized = role.removesuffix("-candidate")
    try:
        return {
            "OnlineSPEC-OGD": "onlinespec_ogd",
            "OnlineSPEC-OPT": "onlinespec_opt",
            "OnlineSPEC-ENS": "onlinespec_ens",
            # Schema-v1 read aliases. New materializations use OPT/ENS.
            "OnlineSPEC-Optimistic-OGD": "onlinespec_opt",
            "OnlineSPEC-Hedge": "onlinespec_ens",
        }[normalized]
    except KeyError as error:
        raise ValueError("E0 OnlineSPEC role is outside the exact panel") from error


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _absolute_directory(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an absolute directory")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or not path.is_dir():
        raise ValueError(f"{label} must be an existing resolved directory")
    return value


def _absolute_file(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an absolute file")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(f"{label} must be a resolved regular file")
    return value


@dataclass(frozen=True)
class E0OnlineSpecSourceAuthority:
    """Reopening instructions for the exact clean upstream OnlineSPEC tree."""

    schema_version: int
    checkout_path: str
    audit_path: str
    source_audit_sha256: str
    commit: str
    tree: str
    verification_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "only E0 OnlineSPEC source authority schema 1 is supported"
            )
        _absolute_directory("E0 OnlineSPEC checkout", self.checkout_path)
        _absolute_file("E0 OnlineSPEC audit", self.audit_path)
        for label, digest in (
            ("source audit", self.source_audit_sha256),
            ("verification", self.verification_sha256),
        ):
            _sha256(f"E0 OnlineSPEC {label}", digest)
        if (
            self.source_audit_sha256 != ONLINE_SPEC_SOURCE_AUDIT_SHA256
            or self.commit != ONLINE_SPEC_COMMIT
            or self.tree != ONLINE_SPEC_TREE
        ):
            raise ValueError("E0 OnlineSPEC source identity is not the registered tree")
        self.revalidate()

    @classmethod
    def bind(
        cls, *, checkout_path: str, audit_path: str
    ) -> E0OnlineSpecSourceAuthority:
        checkout = str(Path(checkout_path).resolve())
        audit = str(Path(audit_path).resolve())
        verified = verify_onlinespec_source_checkout(checkout, audit)
        return cls(
            schema_version=1,
            checkout_path=checkout,
            audit_path=audit,
            source_audit_sha256=ONLINE_SPEC_SOURCE_AUDIT_SHA256,
            commit=ONLINE_SPEC_COMMIT,
            tree=ONLINE_SPEC_TREE,
            verification_sha256=content_sha256(verified),
        )

    def revalidate(self) -> dict[str, object]:
        verified = verify_onlinespec_source_checkout(
            self.checkout_path,
            self.audit_path,
            expected_audit_sha256=self.source_audit_sha256,
        )
        if (
            verified.get("commit") != self.commit
            or verified.get("tree") != self.tree
            or content_sha256(verified) != self.verification_sha256
        ):
            raise ValueError("E0 OnlineSPEC source verification changed")
        return verified

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E6ConfirmationProofBundle:
    """Exact proof inputs needed to deep-open the signed E6 predecessor."""

    signed_model_compatibility: SignedE6ModelCompatibilityReceipt
    compatibility_sources: tuple[E6NextnModelAuthorityInput, ...]
    materialization: StageMaterializationReceipt
    coverage: StageCoverageReceipt
    manifest: FormalDownstreamEvidenceManifest
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]

    def __post_init__(self) -> None:
        if type(self.signed_model_compatibility) is not (
            SignedE6ModelCompatibilityReceipt
        ):
            raise TypeError("E0 E6 bundle requires signed model compatibility")
        if type(self.compatibility_sources) is not tuple or any(
            type(row) is not E6NextnModelAuthorityInput
            for row in self.compatibility_sources
        ):
            raise TypeError("E0 E6 bundle requires exact compatibility sources")
        if type(self.materialization) is not StageMaterializationReceipt:
            raise TypeError("E0 E6 bundle requires exact materialization")
        if type(self.coverage) is not StageCoverageReceipt:
            raise TypeError("E0 E6 bundle requires exact coverage")
        if type(self.manifest) is not FormalDownstreamEvidenceManifest:
            raise TypeError("E0 E6 bundle requires exact evidence manifest")
        if type(self.execution_bindings) is not tuple or any(
            type(row) is not VerifiedFormalServingExecutionBinding
            for row in self.execution_bindings
        ):
            raise TypeError("E0 E6 bundle requires sealed execution bindings")

    def verify(
        self,
        signed_confirmation: SignedE6ConfirmationReceipt,
        *,
        protocol_lock: ProtocolLock,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ):
        return signed_confirmation.verify(
            protocol_lock=protocol_lock,
            signed_model_compatibility=self.signed_model_compatibility,
            compatibility_sources=self.compatibility_sources,
            materialization=self.materialization,
            coverage=self.coverage,
            manifest=self.manifest,
            execution_bindings=self.execution_bindings,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )


@dataclass(frozen=True)
class E0OnlineSpecTuningProofSet:
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle
    materialization: StageMaterializationReceipt
    coverage: StageCoverageReceipt
    manifest: FormalDownstreamEvidenceManifest
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]

    def __post_init__(self) -> None:
        if type(self.e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle:
            raise TypeError("E0 proof set requires exact E6 confirmation proofs")
        if type(self.materialization) is not StageMaterializationReceipt:
            raise TypeError("E0 tuning proof set requires exact materialization")
        if type(self.coverage) is not StageCoverageReceipt:
            raise TypeError("E0 tuning proof set requires exact coverage")
        if type(self.manifest) is not FormalDownstreamEvidenceManifest:
            raise TypeError("E0 tuning proof set requires exact evidence manifest")
        if type(self.execution_bindings) is not tuple or any(
            type(row) is not VerifiedFormalServingExecutionBinding
            for row in self.execution_bindings
        ):
            raise TypeError("E0 tuning proof set requires sealed execution bindings")


@dataclass(frozen=True)
class E0OnlineSpecSelectedRecipe:
    method_role: str
    candidate_id: str
    selected_cell_id: str

    def __post_init__(self) -> None:
        if self.method_role not in E0_ONLINESPEC_ROLES:
            raise ValueError("E0 selected recipe names another method role")
        _sha256("E0 selected OnlineSPEC candidate", self.candidate_id)
        _sha256("E0 selected tuning cell", self.selected_cell_id)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E0OnlineSpecTuningSeal:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    upstream_e6_confirmation_sha256: str
    signed_compatibility_sha256: str
    onlinespec_source_authority_sha256: str
    tuning_materialization_receipt_sha256: str
    tuning_coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    decision_id: str
    model: str
    backend: str
    task: str
    interface_sha256: str
    task_native_workload_sha256: str
    selected_recipes: tuple[E0OnlineSpecSelectedRecipe, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E0 OnlineSPEC tuning seal schema 1 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
            ("compatibility", self.signed_compatibility_sha256),
            ("source authority", self.onlinespec_source_authority_sha256),
            ("materialization", self.tuning_materialization_receipt_sha256),
            ("coverage", self.tuning_coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
            ("decision", self.decision_id),
            ("interface", self.interface_sha256),
            ("task-native workload", self.task_native_workload_sha256),
        ):
            _sha256(f"E0 tuning {label}", digest)
        if (
            self.model not in E0_MODELS
            or self.backend not in E0_BACKENDS
            or self.task not in E0_TASKS
            or type(self.selected_recipes) is not tuple
            or any(
                type(row) is not E0OnlineSpecSelectedRecipe
                for row in self.selected_recipes
            )
            or tuple(row.method_role for row in self.selected_recipes)
            != E0_ONLINESPEC_ROLES
            or len({row.candidate_id for row in self.selected_recipes}) != 3
            or len({row.selected_cell_id for row in self.selected_recipes}) != 3
        ):
            raise ValueError("E0 tuning seal recipe/model panel is not exact")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE0OnlineSpecTuningSeal:
    payload: E0OnlineSpecTuningSeal
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        signed_e6_confirmation: SignedE6ConfirmationReceipt,
        signed_compatibility: SignedE0CompatibilityReceipt,
        source_authority: E0OnlineSpecSourceAuthority,
        proof_set: E0OnlineSpecTuningProofSet,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E0OnlineSpecTuningSeal:
        if type(self.payload) is not E0OnlineSpecTuningSeal:
            raise TypeError("signed E0 tuning payload has the wrong type")
        expected = reduce_e0_onlinespec_tuning_seals_from_proofs(
            protocol_lock=protocol_lock,
            signed_e6_confirmation=signed_e6_confirmation,
            signed_compatibility=signed_compatibility,
            source_authority=source_authority,
            proof_set=proof_set,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        expected_by_decision = {row.decision_id: row for row in expected}
        if self.payload != expected_by_decision.get(self.payload.decision_id):
            raise ValueError("signed E0 tuning seal differs from proof reducer")
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
class E0PowerPrefixReceipt:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    upstream_e6_confirmation_sha256: str
    signed_compatibility_sha256: str
    signed_tuning_seal_sha256s: tuple[str, ...]
    pilot_materialization_receipt_sha256: str
    pilot_coverage_receipt_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    power_sizing: PowerSizingPlan
    selected_final_blocks: int
    selected_final_prefix: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E0 power-prefix schema 1 is supported")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("E6 confirmation", self.upstream_e6_confirmation_sha256),
            ("compatibility", self.signed_compatibility_sha256),
            ("pilot materialization", self.pilot_materialization_receipt_sha256),
            ("pilot coverage", self.pilot_coverage_receipt_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _sha256(f"E0 power {label}", digest)
        if (
            type(self.signed_tuning_seal_sha256s) is not tuple
            or not self.signed_tuning_seal_sha256s
            or self.signed_tuning_seal_sha256s
            != tuple(sorted(set(self.signed_tuning_seal_sha256s)))
        ):
            raise ValueError("E0 power tuning-seal set is not canonical")
        for digest in self.signed_tuning_seal_sha256s:
            _sha256("E0 power signed tuning seal", digest)
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
            raise ValueError("E0 power prefix differs from preregistered sizing")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE0PowerPrefixReceipt:
    payload: E0PowerPrefixReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        signed_e6_confirmation: SignedE6ConfirmationReceipt,
        signed_compatibility: SignedE0CompatibilityReceipt,
        signed_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...],
        source_authority: E0OnlineSpecSourceAuthority,
        tuning_proof_set: E0OnlineSpecTuningProofSet,
        pilot_proof_set: E0OnlineSpecTuningProofSet,
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E0PowerPrefixReceipt:
        if type(self.payload) is not E0PowerPrefixReceipt:
            raise TypeError("signed E0 power-prefix payload has the wrong type")
        expected = reduce_e0_power_prefix_from_proofs(
            protocol_lock=protocol_lock,
            signed_e6_confirmation=signed_e6_confirmation,
            signed_compatibility=signed_compatibility,
            signed_tuning_seals=signed_tuning_seals,
            source_authority=source_authority,
            tuning_proof_set=tuning_proof_set,
            pilot_proof_set=pilot_proof_set,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E0 power prefix differs from proof reducer")
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
class E0FormalRegistryAuthorityBundle:
    """All proof-bearing E0 sources required for one durable registry append.

    The bundle itself is not persisted as a shortcut: the registry persists the
    source checkout authority and signed compatibility/tuning/power artifacts.
    During the append this object re-runs the public materializer, which in turn
    reopens the prior durable registry, E6 confirmation, complete tuning grid,
    and all four excluded pilot blocks.
    """

    signed_e6_confirmation: SignedE6ConfirmationReceipt
    e6_confirmation_proof_bundle: E6ConfirmationProofBundle
    signed_compatibility: SignedE0CompatibilityReceipt
    source_authority: E0OnlineSpecSourceAuthority
    tuning_proof_set: E0OnlineSpecTuningProofSet
    signed_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...]
    pilot_proof_set: E0OnlineSpecTuningProofSet
    signed_power_prefix: SignedE0PowerPrefixReceipt

    def __post_init__(self) -> None:
        if type(self.signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
            raise TypeError("E0 registry bundle requires signed E6 confirmation")
        if type(self.e6_confirmation_proof_bundle) is not E6ConfirmationProofBundle:
            raise TypeError("E0 registry bundle requires exact E6 proofs")
        if type(self.signed_compatibility) is not SignedE0CompatibilityReceipt:
            raise TypeError("E0 registry bundle requires signed compatibility")
        if type(self.source_authority) is not E0OnlineSpecSourceAuthority:
            raise TypeError("E0 registry bundle requires OnlineSPEC source authority")
        if (
            type(self.tuning_proof_set) is not E0OnlineSpecTuningProofSet
            or type(self.pilot_proof_set) is not E0OnlineSpecTuningProofSet
        ):
            raise TypeError("E0 registry bundle requires exact tuning/pilot proofs")
        if (
            self.tuning_proof_set.e6_confirmation_proof_bundle
            != self.e6_confirmation_proof_bundle
            or self.pilot_proof_set.e6_confirmation_proof_bundle
            != self.e6_confirmation_proof_bundle
        ):
            raise ValueError(
                "E0 registry bundle proof sets reopen another E6 authority"
            )
        if type(self.signed_tuning_seals) is not tuple or any(
            type(row) is not SignedE0OnlineSpecTuningSeal
            for row in self.signed_tuning_seals
        ):
            raise TypeError("E0 registry bundle requires exact signed tuning seals")
        if type(self.signed_power_prefix) is not SignedE0PowerPrefixReceipt:
            raise TypeError("E0 registry bundle requires signed power prefix")

    def verify_against(
        self,
        *,
        registry_verification_receipt: object,
        materialization: StageMaterializationReceipt,
        now_ns: int,
    ) -> StageMaterializationReceipt:
        from lightcone_spec.experiments.stage_materialization import (
            materialize_e0_from_signed_compatibility,
        )

        if type(materialization) is not StageMaterializationReceipt:
            raise TypeError("E0 registry bundle requires exact main materialization")
        expected = materialize_e0_from_signed_compatibility(
            registry_verification_receipt=registry_verification_receipt,
            signed_e6_confirmation=self.signed_e6_confirmation,
            e6_confirmation_proof_bundle=self.e6_confirmation_proof_bundle,
            signed_compatibility_receipt=self.signed_compatibility,
            signed_onlinespec_tuning_seals=self.signed_tuning_seals,
            onlinespec_source_authority=self.source_authority,
            tuning_proof_set=self.tuning_proof_set,
            signed_power_prefix=self.signed_power_prefix,
            pilot_proof_set=self.pilot_proof_set,
            now_ns=now_ns,
        )
        if expected != materialization:
            raise ValueError("E0 main materialization differs from deep source replay")
        return expected


def _compatibility(
    signed: SignedE0CompatibilityReceipt,
    *,
    protocol_lock: ProtocolLock,
    upstream_e6_materialization_sha256: str,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> E0CompatibilityReceipt:
    if type(signed) is not SignedE0CompatibilityReceipt:
        raise TypeError("E0 requires an exact signed compatibility receipt")
    receipt = signed.verify(
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    if (
        receipt.protocol_lock_sha256 != protocol_lock.sha256
        or receipt.upstream_e6_receipt_sha256 != upstream_e6_materialization_sha256
    ):
        raise ValueError("E0 compatibility differs from ProtocolLock/E6 lineage")
    return receipt


def _valid_decisions(
    compatibility: E0CompatibilityReceipt,
) -> tuple[E0CompatibilityDecision, ...]:
    rows = tuple(row for row in compatibility.decisions if row.disposition == "VALID")
    if not rows:
        raise ValueError("E0 has no proof-backed compatible combination")
    return rows


def _candidate_index() -> tuple[OnlineSpecCandidate, ...]:
    rows = tuple(sorted(onlinespec_candidates(), key=lambda row: row.candidate_id))
    if (
        not rows
        or {row.method for row in rows} != set(ONLINE_SPEC_METHODS)
        or len({row.candidate_id for row in rows}) != len(rows)
    ):
        raise AssertionError("registered OnlineSPEC candidate grid is not exact")
    return rows


def _aggregate_request_rate(rows: list[object]) -> Fraction:
    numerator = sum(metric.output_tokens for row in rows for metric in row.metrics)
    denominator = sum(metric.latency_ns for row in rows for metric in row.metrics)
    if numerator < 1 or denominator < 1:
        raise ValueError("E0 proof group has no completed timed output")
    return Fraction(numerator * 1_000_000_000, denominator)


def reduce_e0_onlinespec_tuning_seals_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    signed_e6_confirmation: SignedE6ConfirmationReceipt,
    signed_compatibility: SignedE0CompatibilityReceipt,
    source_authority: E0OnlineSpecSourceAuthority,
    proof_set: E0OnlineSpecTuningProofSet,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> tuple[E0OnlineSpecTuningSeal, ...]:
    """Deep-open the complete registered candidate grid for every VALID row."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E0 tuning reducer requires an exact ProtocolLock")
    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 tuning reducer requires a signed E6 confirmation")
    if type(source_authority) is not E0OnlineSpecSourceAuthority:
        raise TypeError("E0 tuning reducer requires source checkout authority")
    if type(proof_set) is not E0OnlineSpecTuningProofSet:
        raise TypeError("E0 tuning reducer requires an exact proof set")
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("E0 tuning reducer time must be positive")
    source_authority.revalidate()
    materialization = proof_set.materialization
    coverage = proof_set.coverage
    manifest = proof_set.manifest
    e6 = proof_set.e6_confirmation_proof_bundle.verify(
        signed_e6_confirmation,
        protocol_lock=protocol_lock,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    if (
        e6.protocol_lock_sha256 != protocol_lock.sha256
        or e6.registry_sha256 != protocol_lock.registry_sha256
        or e6.status != "CONFIRMED"
    ):
        raise ValueError("E0 tuning cannot start before confirmed E6")
    compatibility = _compatibility(
        signed_compatibility,
        protocol_lock=protocol_lock,
        upstream_e6_materialization_sha256=e6.materialization_receipt_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    expected_source = content_sha256(
        {
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility.sha256,
            "e0_onlinespec_source_authority_sha256": source_authority.sha256,
        }
    )
    if (
        materialization.stage != "E0"
        or materialization.materialization_rule != E0_ONLINESPEC_TUNING_RULE
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.upstream_receipt_sha256s
        != (e6.materialization_receipt_sha256,)
        or materialization.source_decision_sha256 != expected_source
        or manifest.stage != "E0"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_authority_sha256 != expected_source
    ):
        raise ValueError("E0 tuning proof lineage differs from exact authorities")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E0 tuning requires all-COMPLETE coverage")

    candidates = _candidate_index()
    candidate_ids = {row.candidate_id for row in candidates}
    candidate_by_id = {row.candidate_id: row for row in candidates}
    valid = _valid_decisions(compatibility)
    expected_cell_count = len(valid) * (len(candidates) + 3)
    if len(materialization.cells) != expected_cell_count:
        raise ValueError("E0 tuning candidate-grid cardinality is not exact")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in proof_set.execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E0 tuning reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_ids = {cell.cell_id for cell in materialization.cells}
    if set(evidence_by_cell) != expected_ids or set(bindings_by_cell) != expected_ids:
        raise ValueError("E0 tuning proof/binding coverage is not exact")
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E0",
        )
        for cell in materialization.cells
    }
    cells_by_decision: dict[str, list[object]] = {}
    valid_by_id = {row.decision_id: row for row in valid}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        decision = valid_by_id.get(dimensions.get("compatibility_decision_id"))
        if (
            decision is None
            or cell.model != decision.model
            or cell.backend != decision.backend
            or dimensions.get("deployment_task") != decision.task
            or dimensions.get("interface_sha256") != decision.interface_sha256
            or dimensions.get("task_native_workload_sha256")
            != decision.task_native_workload_sha256
            or dimensions.get("signed_e0_compatibility_sha256")
            != signed_compatibility.sha256
            or dimensions.get("e0_onlinespec_source_authority_sha256")
            != source_authority.sha256
            or cell.task != "independent_onlinespec_tuning"
        ):
            raise ValueError("E0 tuning cell differs from its compatibility row")
        cells_by_decision.setdefault(decision.decision_id, []).append(cell)
    if set(cells_by_decision) != set(valid_by_id):
        raise ValueError("E0 tuning omits a VALID compatibility row")

    seals: list[E0OnlineSpecTuningSeal] = []
    for decision in valid:
        cells = cells_by_decision[decision.decision_id]
        anchors = tuple(
            cell for cell in cells if cell.method_role in {"Static", "TTS", "L0-naive"}
        )
        candidate_cells = tuple(
            cell for cell in cells if cell.method_role.endswith("-candidate")
        )
        observed_candidates = {
            dict(cell.dimensions).get("candidate_id") for cell in candidate_cells
        }
        if (
            len(anchors) != 3
            or {cell.method_role for cell in anchors} != {"Static", "TTS", "L0-naive"}
            or len(candidate_cells) != len(candidates)
            or observed_candidates != candidate_ids
            or len({cell.cell_id for cell in candidate_cells}) != len(candidates)
        ):
            raise ValueError("E0 tuning row lacks exact anchors/candidate grid")
        static = next(cell for cell in anchors if cell.method_role == "Static")
        static_requests = _request_identity(validated[static.cell_id].metrics)
        if not static_requests:
            raise ValueError("E0 tuning Static anchor has no completed requests")
        selected: list[E0OnlineSpecSelectedRecipe] = []
        for role in E0_ONLINESPEC_ROLES:
            role_rows = tuple(
                cell
                for cell in candidate_cells
                if cell.method_role == f"{role}-candidate"
            )
            method = _method_for_role(role)
            expected_method_candidates = {
                row.candidate_id for row in candidates if row.method == method
            }
            if {
                cell.recipe_sha256 for cell in role_rows
            } != expected_method_candidates or any(
                dict(cell.dimensions).get("candidate_id") != cell.recipe_sha256
                or candidate_by_id[cell.recipe_sha256].method != method
                for cell in role_rows
            ):
                raise ValueError("E0 tuning method candidate coverage is not exact")
            eligible = []
            for cell in role_rows:
                result = validated[cell.cell_id]
                if _request_identity(result.metrics) != static_requests:
                    raise ValueError("E0 tuning candidate is not paired to Static")
                if not result.safety_reasons and result.published_updates > 0:
                    eligible.append((cell, result))
            if not eligible:
                raise ValueError(f"E0 tuning has no safe candidate for {role}")
            winner_cell, _winner = min(
                eligible,
                key=lambda item: (
                    -_aggregate_request_rate([item[1]]),
                    item[1].peak_hbm_bytes,
                    item[1].exposed_update_us,
                    item[0].recipe_sha256,
                ),
            )
            assert winner_cell.recipe_sha256 is not None
            selected.append(
                E0OnlineSpecSelectedRecipe(
                    method_role=role,
                    candidate_id=winner_cell.recipe_sha256,
                    selected_cell_id=winner_cell.cell_id,
                )
            )
        seals.append(
            E0OnlineSpecTuningSeal(
                schema_version=1,
                protocol_lock_sha256=protocol_lock.sha256,
                registry_sha256=protocol_lock.registry_sha256,
                upstream_e6_confirmation_sha256=signed_e6_confirmation.sha256,
                signed_compatibility_sha256=signed_compatibility.sha256,
                onlinespec_source_authority_sha256=source_authority.sha256,
                tuning_materialization_receipt_sha256=materialization.sha256,
                tuning_coverage_receipt_sha256=coverage.sha256,
                evidence_manifest_sha256=manifest.sha256,
                inventory_sha256=manifest.inventory_sha256,
                decision_id=decision.decision_id,
                model=decision.model,
                backend=decision.backend,
                task=decision.task,
                interface_sha256=decision.interface_sha256,
                task_native_workload_sha256=decision.task_native_workload_sha256,
                selected_recipes=tuple(selected),
            )
        )
    result = tuple(sorted(seals, key=lambda row: row.decision_id))
    if tuple(row.decision_id for row in result) != tuple(
        sorted(row.decision_id for row in valid)
    ):
        raise AssertionError("E0 tuning seal order changed")
    return result


def _verify_tuning_seal_set(
    *,
    protocol_lock: ProtocolLock,
    signed_e6_confirmation: SignedE6ConfirmationReceipt,
    signed_compatibility: SignedE0CompatibilityReceipt,
    signed_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...],
    source_authority: E0OnlineSpecSourceAuthority,
    tuning_proof_set: E0OnlineSpecTuningProofSet,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> tuple[E0OnlineSpecTuningSeal, ...]:
    if type(signed_tuning_seals) is not tuple or any(
        type(row) is not SignedE0OnlineSpecTuningSeal for row in signed_tuning_seals
    ):
        raise TypeError("E0 requires exact signed OnlineSPEC tuning seals")
    compatibility = signed_compatibility.payload
    valid = _valid_decisions(compatibility)
    if len(signed_tuning_seals) != len(valid) or tuple(
        row.payload.decision_id for row in signed_tuning_seals
    ) != tuple(sorted(row.decision_id for row in valid)):
        raise ValueError("E0 signed tuning seals do not cover every VALID row")
    expected_payloads = reduce_e0_onlinespec_tuning_seals_from_proofs(
        protocol_lock=protocol_lock,
        signed_e6_confirmation=signed_e6_confirmation,
        signed_compatibility=signed_compatibility,
        source_authority=source_authority,
        proof_set=tuning_proof_set,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    payloads = []
    for signed in signed_tuning_seals:
        payload = signed.payload
        payload.__post_init__()
        expected = next(
            (
                row
                for row in expected_payloads
                if row.decision_id == payload.decision_id
            ),
            None,
        )
        if payload != expected:
            raise ValueError("E0 tuning seal differs from deep proof reducer")
        verify_signed_payload(
            payload,
            payload_sha256=signed.payload_sha256,
            challenge=signed.challenge,
            attestation=signed.attestation,
            policy=policy,
            expected_policy_sha256=expected_policy_sha256,
            now_ns=now_ns,
        )
        if (
            payload.protocol_lock_sha256 != protocol_lock.sha256
            or payload.registry_sha256 != protocol_lock.registry_sha256
            or payload.upstream_e6_confirmation_sha256 != signed_e6_confirmation.sha256
            or payload.signed_compatibility_sha256 != signed_compatibility.sha256
        ):
            raise ValueError("E0 tuning seal changes its typed source lineage")
        payloads.append(payload)
    return tuple(payloads)


def reduce_e0_power_prefix_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    signed_e6_confirmation: SignedE6ConfirmationReceipt,
    signed_compatibility: SignedE0CompatibilityReceipt,
    signed_tuning_seals: tuple[SignedE0OnlineSpecTuningSeal, ...],
    source_authority: E0OnlineSpecSourceAuthority,
    tuning_proof_set: E0OnlineSpecTuningProofSet,
    pilot_proof_set: E0OnlineSpecTuningProofSet,
    policy: TrustedAttesterPolicy,
    expected_policy_sha256: str,
    now_ns: int,
) -> E0PowerPrefixReceipt:
    """Deep-open all ``64V`` excluded pilot rows and derive the final prefix."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E0 power reducer requires an exact ProtocolLock")
    if type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt:
        raise TypeError("E0 power reducer requires a signed E6 confirmation")
    if type(pilot_proof_set) is not E0OnlineSpecTuningProofSet:
        raise TypeError("E0 power reducer requires an exact pilot proof set")
    if (
        pilot_proof_set.e6_confirmation_proof_bundle
        != tuning_proof_set.e6_confirmation_proof_bundle
    ):
        raise ValueError("E0 tuning and pilot proofs reopen different E6 authorities")
    e6 = signed_e6_confirmation.payload
    compatibility = _compatibility(
        signed_compatibility,
        protocol_lock=protocol_lock,
        upstream_e6_materialization_sha256=e6.materialization_receipt_sha256,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    seals = _verify_tuning_seal_set(
        protocol_lock=protocol_lock,
        signed_e6_confirmation=signed_e6_confirmation,
        signed_compatibility=signed_compatibility,
        signed_tuning_seals=signed_tuning_seals,
        source_authority=source_authority,
        tuning_proof_set=tuning_proof_set,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        now_ns=now_ns,
    )
    valid = _valid_decisions(compatibility)
    materialization = pilot_proof_set.materialization
    coverage = pilot_proof_set.coverage
    manifest = pilot_proof_set.manifest
    expected_source = content_sha256(
        {
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility.sha256,
            "signed_e0_tuning_seal_sha256s": tuple(
                sorted(row.sha256 for row in signed_tuning_seals)
            ),
        }
    )
    if (
        materialization.stage != "E0"
        or materialization.materialization_rule != E0_EXCLUDED_PILOT_RULE
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.upstream_receipt_sha256s
        != (e6.materialization_receipt_sha256,)
        or materialization.source_decision_sha256 != expected_source
        or len(materialization.cells) != 16 * len(valid) * PILOT_BLOCK_COUNT
        or manifest.stage != "E0"
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_authority_sha256 != expected_source
    ):
        raise ValueError("E0 pilot proof lineage/cardinality is not exact")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E0 power reducer requires all-COMPLETE pilots")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in pilot_proof_set.execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E0 pilots reuse an execution binding")
        bindings_by_cell[cell_id] = verified
    expected_ids = {cell.cell_id for cell in materialization.cells}
    if set(evidence_by_cell) != expected_ids or set(bindings_by_cell) != expected_ids:
        raise ValueError("E0 pilot proof/binding coverage is not exact")
    terminal_by_cell = {
        row.cell_id: row.terminal_receipt_sha256 for row in coverage.dispositions
    }
    validated = {
        cell.cell_id: _validated_cell(
            cell=cell,
            evidence=evidence_by_cell[cell.cell_id],  # type: ignore[arg-type]
            execution_binding=bindings_by_cell[cell.cell_id],
            coverage_terminal_sha256=terminal_by_cell[cell.cell_id],  # type: ignore[arg-type]
            protocol_lock=protocol_lock,
            inventory_sha256=manifest.inventory_sha256,
            now_ns=now_ns,
            expected_stage="E0",
        )
        for cell in materialization.cells
    }
    valid_by_id = {row.decision_id: row for row in valid}
    seals_by_id = {row.decision_id: row for row in seals}
    by_block_role: dict[tuple[int, str], list[object]] = {}
    by_stratum: dict[tuple[int, str, str], list[object]] = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        decision_id = dimensions.get("compatibility_decision_id")
        load = dimensions.get("load")
        decision = valid_by_id.get(decision_id)
        seal = seals_by_id.get(decision_id)
        if (
            type(block) is not int
            or block not in range(PILOT_BLOCK_COUNT)
            or dimensions.get("block_phase") != "excluded_pilot"
            or load not in E0_LOADS
            or decision is None
            or seal is None
            or cell.model != decision.model
            or cell.backend != decision.backend
            or cell.task != decision.task
            or cell.method_role not in E0_METHOD_ROLES
            or dimensions.get("signed_e0_tuning_seal_sha256")
            != signed_tuning_seals[
                tuple(row.decision_id for row in seals).index(decision_id)
            ].sha256
        ):
            raise ValueError("E0 pilot cell differs from typed sources")
        result = validated[cell.cell_id]
        if result.safety_reasons or (
            cell.method_role not in {"Target-only", "Static"}
            and result.published_updates < 1
        ):
            raise ValueError("E0 pilot contains unsafe or inactive evidence")
        by_block_role.setdefault((block, cell.method_role), []).append(result)
        by_stratum.setdefault((block, decision_id, load), []).append(result)
    expected_strata = {
        (block, decision.decision_id, load)
        for block in range(PILOT_BLOCK_COUNT)
        for decision in valid
        for load in E0_LOADS
    }
    if set(by_stratum) != expected_strata or any(
        len(rows) != len(E0_METHOD_ROLES)
        or len({_request_identity(row.metrics) for row in rows}) != 1
        for rows in by_stratum.values()
    ):
        raise ValueError("E0 pilot roles are not exactly request-paired")
    if set(by_block_role) != {
        (block, role) for block in range(PILOT_BLOCK_COUNT) for role in E0_METHOD_ROLES
    } or any(len(rows) != 2 * len(valid) for rows in by_block_role.values()):
        raise ValueError("E0 pilot block/role coverage is not exact")
    pilot_blocks = tuple(
        PilotBlock(
            block_id=f"E0:excluded_pilot:{block}",
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
        raise ValueError("E0 excluded pilots are UNDERPOWERED at 20 final blocks")
    receipt = E0PowerPrefixReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        upstream_e6_confirmation_sha256=signed_e6_confirmation.sha256,
        signed_compatibility_sha256=signed_compatibility.sha256,
        signed_tuning_seal_sha256s=tuple(
            sorted(row.sha256 for row in signed_tuning_seals)
        ),
        pilot_materialization_receipt_sha256=materialization.sha256,
        pilot_coverage_receipt_sha256=coverage.sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
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


__all__ = [
    "E0_EXCLUDED_PILOT_RULE",
    "E0_FINAL_MATERIALIZATION_RULE",
    "E0_ONLINESPEC_ROLES",
    "E0_ONLINESPEC_TUNING_RULE",
    "E0FormalRegistryAuthorityBundle",
    "E0OnlineSpecSelectedRecipe",
    "E0OnlineSpecSourceAuthority",
    "E0OnlineSpecTuningProofSet",
    "E0OnlineSpecTuningSeal",
    "E0PowerPrefixReceipt",
    "E6ConfirmationProofBundle",
    "SignedE0OnlineSpecTuningSeal",
    "SignedE0PowerPrefixReceipt",
    "reduce_e0_onlinespec_tuning_seals_from_proofs",
    "reduce_e0_power_prefix_from_proofs",
]
