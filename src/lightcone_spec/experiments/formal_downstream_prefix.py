"""Current-only append-only proof roots for the post-E4 formal DAG.

The historical E0 authority artifact recursively contains every future stage.
It is therefore not a valid authority for materializing an earlier downstream
stage.  This module starts the replacement chain at the exact E4-profiler to
E3b excluded-pilot boundary.  A materialization proof binds only the durable
registry state at that boundary and its immediate completed predecessor.

Later downstream phases extend this closed union with one completed-prefix
link at a time.  Unsupported phases fail before a payload can be signed; no
aggregate or caller-authored receipt fallback is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.downstream_stage_authority import (
    E1aVerificationReceipt,
    E3bConfirmationReceipt,
    E3bPowerPrefixReceipt,
    FormalDownstreamCellEvidence,
    FormalDownstreamEvidenceManifest,
    SignedE1aVerificationReceipt,
    SignedE3bConfirmationReceipt,
    SignedE3bPowerPrefixReceipt,
    reduce_e1a_verification_from_proofs,
    reduce_e3b_confirmation_from_proofs,
    reduce_e3b_power_prefix_from_proofs,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    FormalStageCoverageEvidenceShard,
    FormalStageCoverageProofArtifact,
    FormalStageCoverageRebuiltContext,
)
from lightcone_spec.experiments.formal_stage_execution import (
    E1aStageSourceRebuildInputs,
    E3bFinalStageSourceRebuildInputs,
    E3bPilotStageSourceRebuildInputs,
    E5PilotStageSourceRebuildInputs,
)
from lightcone_spec.experiments.formal_stage_prefix import (
    RebuiltFormalStagePrefix,
    load_and_rebuild_formal_stage_prefix,
)
from lightcone_spec.experiments.stage_materialization import (
    SignedStageMaterializationReceipt,
    StageMaterializationReceipt,
    materialize_e1a,
    materialize_e3b,
    materialize_e3b_excluded_pilots,
    materialize_e5_excluded_pilots,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_DOWNSTREAM_MATERIALIZATION_PROOF_KIND = (
    "lightcone_formal_downstream_materialization_proof_artifact"
)
FORMAL_DOWNSTREAM_MATERIALIZATION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_downstream_materialization_protocol",
        "authority": (
            "proof_carrying_schema5_registry_plus_exact_immediate_completed_"
            "predecessor_without_future_e0_aggregate"
        ),
        "first_transition": "e4_profiler_complete_to_e3b_excluded_pilot",
        "pilot_disposition": "excluded_tuning_only",
        "future_fallback": False,
    }
)
FORMAL_DOWNSTREAM_REDUCTION_PROOF_KIND = (
    "lightcone_formal_downstream_reduction_proof_artifact"
)
FORMAL_DOWNSTREAM_REDUCTION_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_downstream_reduction_protocol",
        "authority": (
            "current_materialization_proof_plus_portable_coverage_graph_plus_"
            "proof_derived_execution_manifest"
        ),
        "first_reducer": "e3b_four_excluded_pilots_to_power_prefix",
        "caller_summary": False,
        "future_aggregate": False,
    }
)
FORMAL_DOWNSTREAM_COMPLETED_PREFIX_KIND = (
    "lightcone_formal_downstream_completed_prefix_artifact"
)
FORMAL_DOWNSTREAM_COMPLETED_PREFIX_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_downstream_completed_prefix_protocol",
        "authority": (
            "proof_reduced_current_node_plus_offline_signed_proof_wrapper_"
            "without_embedded_or_future_nodes"
        ),
        "signature_replay": "deep_source_validation_and_exact_reducer_compare",
        "future_aggregate": False,
    }
)
FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_KIND = (
    "lightcone_formal_downstream_pilot_precoverage_artifact"
)
FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_downstream_pilot_precoverage_protocol",
        "authority": (
            "current_only_materialization_reducer_plus_proof_replayed_offline_"
            "signed_excluded_pilot_without_main_registry_insertion"
        ),
        "main_registry_rule": "tuning_only_pilots_remain_excluded",
        "coverage_before_signature": False,
        "future_aggregate": False,
    }
)

FormalDownstreamMaterializationPhase = Literal[
    "e3b_pilot",
    "e3b_final",
    "e1a_verification",
    "e5_pilot",
    "e5_final",
    "e6_pilot",
    "e6_final",
    "e0_tuning",
    "e0_pilot",
    "e0_final",
]

FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER: tuple[
    FormalDownstreamMaterializationPhase, ...
] = (
    "e3b_pilot",
    "e3b_final",
    "e1a_verification",
    "e5_pilot",
    "e5_final",
    "e6_pilot",
    "e6_final",
    "e0_tuning",
    "e0_pilot",
    "e0_final",
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _one(rows: tuple[object, ...], *, label: str, predicate) -> object:
    selected = tuple(row for row in rows if predicate(row))
    if len(selected) != 1:
        raise ValueError(f"downstream materialization requires one exact {label}")
    return selected[0]


@dataclass(frozen=True)
class FormalDownstreamMaterializationProofArtifact:
    """One current-only typed predecessor proof for a downstream materialization."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_downstream_materialization_proof_artifact"]
    protocol_sha256: str
    phase: FormalDownstreamMaterializationPhase
    registry_layer_source: CanonicalJsonProofBinding
    immediate_predecessor_source: CanonicalJsonProofBinding
    expected_materialization_sha256: str
    created_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_DOWNSTREAM_MATERIALIZATION_PROOF_KIND
            or self.protocol_sha256 != FORMAL_DOWNSTREAM_MATERIALIZATION_PROTOCOL_SHA256
            or self.phase not in FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER
        ):
            raise ValueError("formal downstream materialization proof identity differs")
        if any(
            type(row) is not CanonicalJsonProofBinding
            for row in (
                self.registry_layer_source,
                self.immediate_predecessor_source,
            )
        ):
            raise TypeError(
                "formal downstream materialization sources are not path-bound"
            )
        if (
            self.registry_layer_source.absolute_path
            == self.immediate_predecessor_source.absolute_path
        ):
            raise ValueError("formal downstream materialization reuses a source path")
        _sha256(
            "formal downstream expected materialization",
            self.expected_materialization_sha256,
        )
        if type(self.created_ns) is not int or self.created_ns < 1:
            raise ValueError("formal downstream materialization proof time is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "phase": self.phase,
            "registry_layer_source": self.registry_layer_source.to_dict(),
            "immediate_predecessor_source": (
                self.immediate_predecessor_source.to_dict()
            ),
            "expected_materialization_sha256": (self.expected_materialization_sha256),
            "created_ns": self.created_ns,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal downstream materialization proof fields differ")
        row = dict(value)
        declared = _sha256(
            "formal downstream materialization proof",
            row.pop("artifact_sha256"),
        )
        for name in ("registry_layer_source", "immediate_predecessor_source"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal downstream materialization proof digest differs")
        return artifact


@dataclass(frozen=True)
class RebuiltFormalDownstreamMaterialization:
    artifact: FormalDownstreamMaterializationProofArtifact
    registry_verification_receipt: FormalRegistryVerificationReceipt
    immediate_predecessor: (
        RebuiltFormalStagePrefix | RebuiltFormalDownstreamCompletedPrefix
    )
    stage_source_inputs: (
        E3bPilotStageSourceRebuildInputs
        | E3bFinalStageSourceRebuildInputs
        | E1aStageSourceRebuildInputs
        | E5PilotStageSourceRebuildInputs
    )
    materialization: StageMaterializationReceipt


@dataclass(frozen=True)
class FormalDownstreamReductionProofArtifact:
    """Current-only reducer proof for one completed downstream phase."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_downstream_reduction_proof_artifact"]
    protocol_sha256: str
    phase: FormalDownstreamMaterializationPhase
    materialization_proof_source: CanonicalJsonProofBinding
    portable_coverage_proof_source: CanonicalJsonProofBinding
    expected_reduction_sha256: str
    created_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_DOWNSTREAM_REDUCTION_PROOF_KIND
            or self.protocol_sha256 != FORMAL_DOWNSTREAM_REDUCTION_PROTOCOL_SHA256
            or self.phase not in FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER
        ):
            raise ValueError("formal downstream reduction proof identity differs")
        if any(
            type(row) is not CanonicalJsonProofBinding
            for row in (
                self.materialization_proof_source,
                self.portable_coverage_proof_source,
            )
        ):
            raise TypeError("formal downstream reduction sources are not path-bound")
        if (
            self.materialization_proof_source.absolute_path
            == self.portable_coverage_proof_source.absolute_path
        ):
            raise ValueError("formal downstream reduction reuses a source path")
        _sha256("formal downstream reduction", self.expected_reduction_sha256)
        if type(self.created_ns) is not int or self.created_ns < 1:
            raise ValueError("formal downstream reduction proof time is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "phase": self.phase,
            "materialization_proof_source": (
                self.materialization_proof_source.to_dict()
            ),
            "portable_coverage_proof_source": (
                self.portable_coverage_proof_source.to_dict()
            ),
            "expected_reduction_sha256": self.expected_reduction_sha256,
            "created_ns": self.created_ns,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal downstream reduction proof fields differ")
        row = dict(value)
        declared = _sha256(
            "formal downstream reduction proof", row.pop("artifact_sha256")
        )
        for name in (
            "materialization_proof_source",
            "portable_coverage_proof_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal downstream reduction proof digest differs")
        return artifact


@dataclass(frozen=True)
class FormalDownstreamCompletedPrefixArtifact:
    """One completed current node used as the next materializer predecessor."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_downstream_completed_prefix_artifact"]
    protocol_sha256: str
    phase: FormalDownstreamMaterializationPhase
    reduction_proof_source: CanonicalJsonProofBinding
    signed_result_source: CanonicalJsonProofBinding
    expected_signed_result_sha256: str
    created_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_DOWNSTREAM_COMPLETED_PREFIX_KIND
            or self.protocol_sha256
            != FORMAL_DOWNSTREAM_COMPLETED_PREFIX_PROTOCOL_SHA256
            or self.phase not in FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER
        ):
            raise ValueError("formal downstream completed-prefix identity differs")
        if any(
            type(row) is not CanonicalJsonProofBinding
            for row in (self.reduction_proof_source, self.signed_result_source)
        ):
            raise TypeError("formal downstream completed-prefix sources are not bound")
        if (
            self.reduction_proof_source.absolute_path
            == self.signed_result_source.absolute_path
        ):
            raise ValueError("formal downstream completed prefix reuses a source path")
        _sha256(
            "formal downstream completed signed result",
            self.expected_signed_result_sha256,
        )
        if type(self.created_ns) is not int or self.created_ns < 1:
            raise ValueError("formal downstream completed-prefix time is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "phase": self.phase,
            "reduction_proof_source": self.reduction_proof_source.to_dict(),
            "signed_result_source": self.signed_result_source.to_dict(),
            "expected_signed_result_sha256": self.expected_signed_result_sha256,
            "created_ns": self.created_ns,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal downstream completed-prefix fields differ")
        row = dict(value)
        declared = _sha256(
            "formal downstream completed prefix", row.pop("artifact_sha256")
        )
        for name in ("reduction_proof_source", "signed_result_source"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal downstream completed-prefix digest differs")
        return artifact


@dataclass(frozen=True)
class FormalDownstreamPilotPrecoverageArtifact:
    """Signed excluded-pilot materialization outside the main registry."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_downstream_pilot_precoverage_artifact"]
    protocol_sha256: str
    phase: Literal["e3b_pilot", "e5_pilot", "e6_pilot", "e0_pilot"]
    materialization_proof_source: CanonicalJsonProofBinding
    signed_materialization_source: CanonicalJsonProofBinding
    expected_signed_materialization_sha256: str
    created_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_KIND
            or self.protocol_sha256
            != FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_PROTOCOL_SHA256
            or self.phase not in {"e3b_pilot", "e5_pilot", "e6_pilot", "e0_pilot"}
        ):
            raise ValueError("formal downstream pilot precoverage identity differs")
        if any(
            type(row) is not CanonicalJsonProofBinding
            for row in (
                self.materialization_proof_source,
                self.signed_materialization_source,
            )
        ):
            raise TypeError("formal downstream pilot precoverage sources are not bound")
        if (
            self.materialization_proof_source.absolute_path
            == self.signed_materialization_source.absolute_path
        ):
            raise ValueError("formal downstream pilot precoverage reuses a source path")
        _sha256(
            "formal downstream signed pilot materialization",
            self.expected_signed_materialization_sha256,
        )
        if type(self.created_ns) is not int or self.created_ns < 1:
            raise ValueError("formal downstream pilot precoverage time is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "phase": self.phase,
            "materialization_proof_source": (
                self.materialization_proof_source.to_dict()
            ),
            "signed_materialization_source": (
                self.signed_materialization_source.to_dict()
            ),
            "expected_signed_materialization_sha256": (
                self.expected_signed_materialization_sha256
            ),
            "created_ns": self.created_ns,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal downstream pilot precoverage fields differ")
        row = dict(value)
        declared = _sha256(
            "formal downstream pilot precoverage", row.pop("artifact_sha256")
        )
        for name in (
            "materialization_proof_source",
            "signed_materialization_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal downstream pilot precoverage digest differs")
        return artifact


@dataclass(frozen=True)
class RebuiltFormalDownstreamReduction:
    artifact: FormalDownstreamReductionProofArtifact
    materialization: RebuiltFormalDownstreamMaterialization
    coverage_context: FormalStageCoverageRebuiltContext
    evidence_manifest: FormalDownstreamEvidenceManifest
    reduction: E3bPowerPrefixReceipt | E3bConfirmationReceipt | E1aVerificationReceipt


@dataclass(frozen=True)
class RebuiltFormalDownstreamCompletedPrefix:
    artifact_binding: CanonicalJsonProofBinding
    artifact: FormalDownstreamCompletedPrefixArtifact
    reduction: RebuiltFormalDownstreamReduction
    signed_result: (
        SignedE3bPowerPrefixReceipt
        | SignedE3bConfirmationReceipt
        | SignedE1aVerificationReceipt
    )


@dataclass(frozen=True)
class RebuiltFormalDownstreamPilotPrecoverage:
    artifact_binding: CanonicalJsonProofBinding
    artifact: FormalDownstreamPilotPrecoverageArtifact
    materialization: RebuiltFormalDownstreamMaterialization
    signed_materialization: SignedStageMaterializationReceipt


def _load_registry_layer(
    source: CanonicalJsonProofBinding,
    *,
    now_ns: int,
) -> FormalRegistryVerificationReceipt:
    from lightcone_spec.experiments.formal_registry_layers import (
        load_formal_registry_verification_receipt_path,
    )

    before = CanonicalJsonProofBinding.bind(source.absolute_path)
    if before != source:
        raise ValueError("formal downstream registry layer identity changed")
    receipt = load_formal_registry_verification_receipt_path(
        before.absolute_path,
        now_ns=now_ns,
    )
    if CanonicalJsonProofBinding.bind(source.absolute_path) != before:
        raise RuntimeError("formal downstream registry layer changed while rebuilt")
    return receipt


def _materialize_e3b_pilot(
    *,
    receipt: FormalRegistryVerificationReceipt,
    predecessor_source: CanonicalJsonProofBinding,
    now_ns: int,
) -> tuple[
    RebuiltFormalStagePrefix,
    E3bPilotStageSourceRebuildInputs,
    StageMaterializationReceipt,
]:
    before = CanonicalJsonProofBinding.bind(predecessor_source.absolute_path)
    if before != predecessor_source:
        raise ValueError("formal E3b predecessor path identity changed")
    predecessor = load_and_rebuild_formal_stage_prefix(
        before.absolute_path,
        now_ns=now_ns,
    )
    if CanonicalJsonProofBinding.bind(predecessor_source.absolute_path) != before:
        raise RuntimeError("formal E3b predecessor changed while rebuilt")
    if predecessor.artifact.phase != "e4_profiler":
        raise ValueError("formal E3b pilot predecessor is not E4 profiler")
    if predecessor.artifact_binding not in (
        receipt.cumulative_formal_stage_prefix_artifacts
    ):
        raise ValueError("formal E3b pilot registry lacks its completed predecessor")
    if (
        receipt.signed_protocol_lock.payload
        != predecessor.registry_verification_receipt.signed_protocol_lock.payload
        or receipt.inventory_sha256
        != predecessor.registry_verification_receipt.inventory_sha256
    ):
        raise ValueError("formal E3b pilot changes the immutable registry root")
    if (
        receipt.appended_formal_stage_prefix_artifacts
        != (predecessor.artifact_binding,)
        or len(receipt.appended_signed_coverage) != 1
        or receipt.appended_signed_coverage[0].payload != predecessor.coverage
        or receipt.appended_signed_materializations
        or receipt.appended_e4_staged_evidence_manifests
        or receipt.appended_signed_e4_stage_selections
        or receipt.appended_signed_e3b_power_prefixes
        or receipt.appended_signed_e5_power_and_anchor_prefixes
        or receipt.appended_signed_e6_power_prefixes
        or receipt.appended_e0_onlinespec_source_authorities
        or receipt.appended_signed_e0_compatibilities
        or receipt.appended_signed_e0_onlinespec_tuning_seals
        or receipt.appended_signed_e0_power_prefixes
    ):
        raise ValueError(
            "formal E3b pilot registry is not the exact completed-profiler transition"
        )
    if any(
        row.payload.stage in {"E3b", "E1a", "E5", "E6", "E0"}
        for row in receipt.cumulative_signed_materializations
    ) or any(
        row.payload.stage in {"E3b", "E1a", "E5", "E6", "E0"}
        for row in receipt.cumulative_signed_coverage
    ):
        raise ValueError("formal E3b pilot registry contains current or future rows")
    local = predecessor.prior
    if local is None or local.artifact.phase != "e4_local":
        raise ValueError("formal E3b pilot lacks its immediate E4 local prefix")
    signed_e4 = _one(
        receipt.cumulative_signed_e4_stage_selections,
        label="signed E4 local selection",
        predicate=lambda row: (
            row.payload.phase == "local"
            and row.payload.materialization_receipt_sha256
            == local.materialization.sha256
            and row.payload.coverage_receipt_sha256 == local.coverage.sha256
        ),
    )
    tts = _one(
        receipt.cumulative_tts_calibration_authorities,
        label="frozen TTS calibration authority",
        predicate=lambda _row: True,
    )
    signed_tts = _one(
        receipt.cumulative_signed_tts_calibration_seals,
        label="signed frozen TTS seal",
        predicate=lambda _row: True,
    )
    if local.evidence_manifest is None:
        raise ValueError("formal E3b pilot lacks proof-derived E4 local evidence")
    source_inputs = E3bPilotStageSourceRebuildInputs(
        registry_verification_receipt=receipt,
        signed_e4_final_selection=signed_e4,  # type: ignore[arg-type]
        local_materialization=local.materialization,
        local_coverage=local.coverage,
        local_evidence_manifest=local.evidence_manifest,  # type: ignore[arg-type]
        local_execution_bindings=local.execution_bindings,
        profiler_materialization=predecessor.materialization,
        profiler_coverage=predecessor.coverage,
        tts_calibration_authority=tts,  # type: ignore[arg-type]
        signed_tts_calibration_seal=signed_tts,  # type: ignore[arg-type]
    )
    materialization = materialize_e3b_excluded_pilots(
        registry_verification_receipt=receipt,
        signed_e4_final_selection=signed_e4,
        local_materialization=local.materialization,
        local_coverage=local.coverage,
        local_evidence_manifest=local.evidence_manifest,
        local_execution_bindings=local.execution_bindings,
        profiler_materialization=predecessor.materialization,
        profiler_coverage=predecessor.coverage,
        tts_calibration_authority=tts,
        signed_tts_calibration_seal=signed_tts,
        now_ns=now_ns,
    )
    return predecessor, source_inputs, materialization


def _materialize_e3b_final(
    *,
    receipt: FormalRegistryVerificationReceipt,
    predecessor_source: CanonicalJsonProofBinding,
    now_ns: int,
) -> tuple[
    RebuiltFormalDownstreamCompletedPrefix,
    E3bFinalStageSourceRebuildInputs,
    StageMaterializationReceipt,
]:
    predecessor = rebuild_formal_downstream_completed_prefix(
        predecessor_source.absolute_path,
        now_ns=now_ns,
    )
    if predecessor.artifact.phase != "e3b_pilot":
        raise ValueError("formal E3b final predecessor is not the pilot power prefix")
    pilot = predecessor.reduction
    if (
        receipt.signed_protocol_lock.payload != pilot.coverage_context.protocol_lock
        or receipt.inventory_sha256 != pilot.coverage_context.inventory.sha256
    ):
        raise ValueError("formal E3b final changes the immutable registry root")
    if (
        any(
            row.payload.stage in {"E3b", "E1a", "E5", "E6", "E0"}
            for row in receipt.cumulative_signed_materializations
        )
        or any(
            row.payload.stage in {"E3b", "E1a", "E5", "E6", "E0"}
            for row in receipt.cumulative_signed_coverage
        )
        or any(
            (
                receipt.cumulative_signed_e3b_power_prefixes,
                receipt.cumulative_signed_e5_power_and_anchor_prefixes,
                receipt.cumulative_signed_e6_power_prefixes,
                receipt.cumulative_e0_onlinespec_source_authorities,
                receipt.cumulative_signed_e0_compatibilities,
                receipt.cumulative_signed_e0_onlinespec_tuning_seals,
                receipt.cumulative_signed_e0_power_prefixes,
            )
        )
    ):
        raise ValueError("formal E3b final registry contains current or future rows")
    source_inputs = E3bFinalStageSourceRebuildInputs(
        registry_verification_receipt=receipt,
        signed_power_prefix=predecessor.signed_result,
        pilot_materialization=pilot.coverage_context.materialization,
        pilot_coverage=pilot.coverage_context.coverage,
        pilot_evidence_manifest=pilot.evidence_manifest,
        pilot_execution_bindings=pilot.coverage_context.execution_bindings,
    )
    materialization = materialize_e3b(
        registry_verification_receipt=receipt,
        signed_power_prefix=predecessor.signed_result,
        pilot_materialization=pilot.coverage_context.materialization,
        pilot_coverage=pilot.coverage_context.coverage,
        pilot_evidence_manifest=pilot.evidence_manifest,
        pilot_execution_bindings=pilot.coverage_context.execution_bindings,
        now_ns=now_ns,
    )
    return predecessor, source_inputs, materialization


def _materialize_e1a_verification(
    *,
    receipt: FormalRegistryVerificationReceipt,
    predecessor_source: CanonicalJsonProofBinding,
    now_ns: int,
) -> tuple[
    RebuiltFormalDownstreamCompletedPrefix,
    E1aStageSourceRebuildInputs,
    StageMaterializationReceipt,
]:
    predecessor = rebuild_formal_downstream_completed_prefix(
        predecessor_source.absolute_path,
        now_ns=now_ns,
    )
    if predecessor.artifact.phase != "e3b_final":
        raise ValueError("formal E1a predecessor is not the E3b confirmation")
    final = predecessor.reduction
    if (
        receipt.signed_protocol_lock.payload != final.coverage_context.protocol_lock
        or receipt.inventory_sha256 != final.coverage_context.inventory.sha256
        or not any(
            row.payload == final.coverage_context.materialization
            for row in receipt.cumulative_signed_materializations
        )
        or not any(
            row.payload == final.coverage_context.coverage
            for row in receipt.cumulative_signed_coverage
        )
    ):
        raise ValueError("formal E1a registry lacks the completed E3b final node")
    if (
        any(
            row.payload.stage in {"E1a", "E5", "E6", "E0"}
            for row in receipt.cumulative_signed_materializations
        )
        or any(
            row.payload.stage in {"E1a", "E5", "E6", "E0"}
            for row in receipt.cumulative_signed_coverage
        )
        or any(
            (
                receipt.cumulative_signed_e5_power_and_anchor_prefixes,
                receipt.cumulative_signed_e6_power_prefixes,
                receipt.cumulative_e0_onlinespec_source_authorities,
                receipt.cumulative_signed_e0_compatibilities,
                receipt.cumulative_signed_e0_onlinespec_tuning_seals,
                receipt.cumulative_signed_e0_power_prefixes,
            )
        )
    ):
        raise ValueError("formal E1a registry contains current or future rows")
    assert type(predecessor.signed_result) is SignedE3bConfirmationReceipt
    source_inputs = E1aStageSourceRebuildInputs(
        registry_verification_receipt=receipt,
        signed_e3b_confirmation=predecessor.signed_result,
        e3b_materialization=final.coverage_context.materialization,
        e3b_coverage=final.coverage_context.coverage,
        e3b_evidence_manifest=final.evidence_manifest,
        e3b_execution_bindings=final.coverage_context.execution_bindings,
    )
    materialization = materialize_e1a(
        registry_verification_receipt=receipt,
        signed_e3b_confirmation=predecessor.signed_result,
        e3b_materialization=final.coverage_context.materialization,
        e3b_coverage=final.coverage_context.coverage,
        e3b_evidence_manifest=final.evidence_manifest,
        e3b_execution_bindings=final.coverage_context.execution_bindings,
        now_ns=now_ns,
    )
    return predecessor, source_inputs, materialization


def _materialize_e5_pilot(
    *,
    receipt: FormalRegistryVerificationReceipt,
    predecessor_source: CanonicalJsonProofBinding,
    now_ns: int,
) -> tuple[
    RebuiltFormalDownstreamCompletedPrefix,
    E5PilotStageSourceRebuildInputs,
    StageMaterializationReceipt,
]:
    predecessor = rebuild_formal_downstream_completed_prefix(
        predecessor_source.absolute_path,
        now_ns=now_ns,
    )
    if predecessor.artifact.phase != "e1a_verification":
        raise ValueError("formal E5 pilot predecessor is not E1a verification")
    e1a = predecessor.reduction
    if (
        receipt.signed_protocol_lock.payload != e1a.coverage_context.protocol_lock
        or receipt.inventory_sha256 != e1a.coverage_context.inventory.sha256
        or not any(
            row.payload == e1a.coverage_context.materialization
            for row in receipt.cumulative_signed_materializations
        )
        or not any(
            row.payload == e1a.coverage_context.coverage
            for row in receipt.cumulative_signed_coverage
        )
    ):
        raise ValueError("formal E5 pilot registry lacks the completed E1a node")
    if (
        any(
            row.payload.stage in {"E5", "E6", "E0"}
            for row in receipt.cumulative_signed_materializations
        )
        or any(
            row.payload.stage in {"E5", "E6", "E0"}
            for row in receipt.cumulative_signed_coverage
        )
        or any(
            (
                receipt.cumulative_signed_e5_power_and_anchor_prefixes,
                receipt.cumulative_signed_e6_power_prefixes,
                receipt.cumulative_e0_onlinespec_source_authorities,
                receipt.cumulative_signed_e0_compatibilities,
                receipt.cumulative_signed_e0_onlinespec_tuning_seals,
                receipt.cumulative_signed_e0_power_prefixes,
            )
        )
    ):
        raise ValueError("formal E5 pilot registry contains current or future rows")
    assert type(predecessor.signed_result) is SignedE1aVerificationReceipt
    runtime_manifest = e1a.coverage_context.formal_runtime_authority_manifest
    source_inputs = E5PilotStageSourceRebuildInputs(
        registry_verification_receipt=receipt,
        signed_e1a_verification=predecessor.signed_result,
        e1a_materialization=e1a.coverage_context.materialization,
        e1a_coverage=e1a.coverage_context.coverage,
        e1a_evidence_manifest=e1a.evidence_manifest,
        e1a_execution_bindings=e1a.coverage_context.execution_bindings,
        formal_runtime_authority_manifest=runtime_manifest,
    )
    materialization = materialize_e5_excluded_pilots(
        registry_verification_receipt=receipt,
        signed_e1a_verification=predecessor.signed_result,
        e1a_materialization=e1a.coverage_context.materialization,
        e1a_coverage=e1a.coverage_context.coverage,
        e1a_evidence_manifest=e1a.evidence_manifest,
        e1a_execution_bindings=e1a.coverage_context.execution_bindings,
        formal_runtime_authority_manifest=runtime_manifest,
        now_ns=now_ns,
    )
    return predecessor, source_inputs, materialization


def _derive_formal_downstream_materialization(
    artifact: FormalDownstreamMaterializationProofArtifact,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamMaterialization:
    if type(now_ns) is not int or now_ns < artifact.created_ns:
        raise ValueError("formal downstream validation time predates its proof")
    receipt = _load_registry_layer(artifact.registry_layer_source, now_ns=now_ns)
    if artifact.phase not in {
        "e3b_pilot",
        "e3b_final",
        "e1a_verification",
        "e5_pilot",
    }:
        raise ValueError(
            "formal downstream phase is BLOCKED until its immediate completed "
            "prefix adapter is registered"
        )
    if artifact.phase == "e3b_pilot":
        predecessor, source_inputs, materialization = _materialize_e3b_pilot(
            receipt=receipt,
            predecessor_source=artifact.immediate_predecessor_source,
            now_ns=now_ns,
        )
    elif artifact.phase == "e3b_final":
        predecessor, source_inputs, materialization = _materialize_e3b_final(
            receipt=receipt,
            predecessor_source=artifact.immediate_predecessor_source,
            now_ns=now_ns,
        )
    elif artifact.phase == "e1a_verification":
        predecessor, source_inputs, materialization = _materialize_e1a_verification(
            receipt=receipt,
            predecessor_source=artifact.immediate_predecessor_source,
            now_ns=now_ns,
        )
    else:
        predecessor, source_inputs, materialization = _materialize_e5_pilot(
            receipt=receipt,
            predecessor_source=artifact.immediate_predecessor_source,
            now_ns=now_ns,
        )
    if materialization.sha256 != artifact.expected_materialization_sha256:
        raise ValueError("formal downstream materialization differs from proof")
    return RebuiltFormalDownstreamMaterialization(
        artifact=artifact,
        registry_verification_receipt=receipt,
        immediate_predecessor=predecessor,
        stage_source_inputs=source_inputs,
        materialization=materialization,
    )


def build_formal_downstream_materialization_proof_artifact(
    *,
    phase: FormalDownstreamMaterializationPhase,
    registry_layer_path: str | Path,
    immediate_predecessor_path: str | Path,
    now_ns: int,
) -> FormalDownstreamMaterializationProofArtifact:
    """Derive and bind one current-only downstream materialization proof."""

    if phase not in {
        "e3b_pilot",
        "e3b_final",
        "e1a_verification",
        "e5_pilot",
    }:
        raise ValueError(
            "formal downstream phase is BLOCKED until its immediate completed "
            "prefix adapter is registered"
        )
    registry_source = CanonicalJsonProofBinding.bind(registry_layer_path)
    predecessor_source = CanonicalJsonProofBinding.bind(immediate_predecessor_path)
    receipt = _load_registry_layer(registry_source, now_ns=now_ns)
    if phase == "e3b_pilot":
        _predecessor, _source_inputs, materialization = _materialize_e3b_pilot(
            receipt=receipt,
            predecessor_source=predecessor_source,
            now_ns=now_ns,
        )
    elif phase == "e3b_final":
        _predecessor, _source_inputs, materialization = _materialize_e3b_final(
            receipt=receipt,
            predecessor_source=predecessor_source,
            now_ns=now_ns,
        )
    elif phase == "e1a_verification":
        _predecessor, _source_inputs, materialization = _materialize_e1a_verification(
            receipt=receipt,
            predecessor_source=predecessor_source,
            now_ns=now_ns,
        )
    else:
        _predecessor, _source_inputs, materialization = _materialize_e5_pilot(
            receipt=receipt,
            predecessor_source=predecessor_source,
            now_ns=now_ns,
        )
    artifact = FormalDownstreamMaterializationProofArtifact(
        schema_version=1,
        kind=FORMAL_DOWNSTREAM_MATERIALIZATION_PROOF_KIND,
        protocol_sha256=FORMAL_DOWNSTREAM_MATERIALIZATION_PROTOCOL_SHA256,
        phase=phase,
        registry_layer_source=registry_source,
        immediate_predecessor_source=predecessor_source,
        expected_materialization_sha256=materialization.sha256,
        created_ns=now_ns,
    )
    rebuilt = _derive_formal_downstream_materialization(artifact, now_ns=now_ns)
    if rebuilt.materialization != materialization:
        raise RuntimeError("formal downstream materialization changed while bound")
    return artifact


def publish_formal_downstream_materialization_proof_artifact(
    artifact: FormalDownstreamMaterializationProofArtifact,
    output_path: str | Path,
    *,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Deep-replay before publishing one bounded no-replace proof root."""

    if type(artifact) is not FormalDownstreamMaterializationProofArtifact:
        raise TypeError("formal downstream publisher requires an exact artifact")
    _derive_formal_downstream_materialization(artifact, now_ns=now_ns)
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def rebuild_formal_downstream_materialization_proof(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamMaterialization:
    """Deep-open and source-rederive one downstream materialization."""

    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalDownstreamMaterializationProofArtifact.from_dict(binding.reopen())
    rebuilt = _derive_formal_downstream_materialization(artifact, now_ns=now_ns)
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("formal downstream proof changed while rebuilt")
    return rebuilt


def _validate_formal_downstream_pilot_precoverage_sources(
    artifact: FormalDownstreamPilotPrecoverageArtifact,
    *,
    now_ns: int,
) -> tuple[
    RebuiltFormalDownstreamMaterialization,
    SignedStageMaterializationReceipt,
]:
    """Replay the materializer and its offline signature before coverage.

    Excluded pilots are deliberately absent from the semantic experiment
    registry.  This proof root is the narrow bridge between the completed
    predecessor registry and pilot coverage: it proves that the exact
    source-derived pilot receipt was signed, without inserting tuning-only
    rows into the main registry or permitting coverage to precede signing.
    """

    from lightcone_spec.runtime.scientific_signing import (
        rebuild_scientific_signed_proof_wrapper,
    )

    if type(now_ns) is not int or now_ns < artifact.created_ns:
        raise ValueError("formal downstream pilot precoverage time predates proof")
    if artifact.phase not in {"e3b_pilot", "e5_pilot"}:
        raise ValueError(
            "formal downstream pilot precoverage is BLOCKED until its "
            "phase-specific materializer adapter is registered"
        )
    materialization = rebuild_formal_downstream_materialization_proof(
        artifact.materialization_proof_source.absolute_path,
        now_ns=now_ns,
    )
    if materialization.artifact.phase != artifact.phase:
        raise ValueError("formal downstream pilot precoverage phase differs")
    signed = rebuild_scientific_signed_proof_wrapper(
        artifact.signed_materialization_source.absolute_path,
        now_ns=now_ns,
    )
    if type(signed) is not SignedStageMaterializationReceipt:
        raise TypeError("formal downstream pilot materialization signature differs")
    if (
        signed.payload != materialization.materialization
        or signed.sha256 != artifact.expected_signed_materialization_sha256
    ):
        raise ValueError("formal downstream signed pilot materialization differs")
    policy = materialization.registry_verification_receipt.trusted_release_policy(
        current_ns=now_ns
    )
    signed.verify(
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    return materialization, signed


def build_formal_downstream_pilot_precoverage_artifact(
    *,
    phase: Literal["e3b_pilot", "e5_pilot", "e6_pilot", "e0_pilot"],
    materialization_proof_path: str | Path,
    signed_materialization_path: str | Path,
    now_ns: int,
) -> FormalDownstreamPilotPrecoverageArtifact:
    """Bind the exact proof-replayed pilot materialization signature."""

    if phase not in {"e3b_pilot", "e5_pilot"}:
        raise ValueError(
            "formal downstream pilot precoverage is BLOCKED until its "
            "phase-specific materializer adapter is registered"
        )
    materialization_source = CanonicalJsonProofBinding.bind(materialization_proof_path)
    signed_source = CanonicalJsonProofBinding.bind(signed_materialization_path)
    from lightcone_spec.runtime.scientific_signing import (
        rebuild_scientific_signed_proof_wrapper,
    )

    materialization = rebuild_formal_downstream_materialization_proof(
        materialization_source.absolute_path,
        now_ns=now_ns,
    )
    signed = rebuild_scientific_signed_proof_wrapper(
        signed_source.absolute_path,
        now_ns=now_ns,
    )
    if (
        materialization.artifact.phase != phase
        or type(signed) is not SignedStageMaterializationReceipt
        or signed.payload != materialization.materialization
    ):
        raise ValueError("formal downstream pilot precoverage sources differ")
    artifact = FormalDownstreamPilotPrecoverageArtifact(
        schema_version=1,
        kind=FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_KIND,
        protocol_sha256=FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_PROTOCOL_SHA256,
        phase=phase,
        materialization_proof_source=materialization_source,
        signed_materialization_source=signed_source,
        expected_signed_materialization_sha256=signed.sha256,
        created_ns=now_ns,
    )
    _validate_formal_downstream_pilot_precoverage_sources(
        artifact,
        now_ns=now_ns,
    )
    return artifact


def publish_formal_downstream_pilot_precoverage_artifact(
    artifact: FormalDownstreamPilotPrecoverageArtifact,
    output_path: str | Path,
    *,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Deep-replay, then atomically publish, one pilot precoverage root."""

    if type(artifact) is not FormalDownstreamPilotPrecoverageArtifact:
        raise TypeError("formal pilot precoverage publisher requires exact input")
    _validate_formal_downstream_pilot_precoverage_sources(
        artifact,
        now_ns=now_ns,
    )
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def rebuild_formal_downstream_pilot_precoverage(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamPilotPrecoverage:
    """Deep-open the signed excluded-pilot bridge used by coverage."""

    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalDownstreamPilotPrecoverageArtifact.from_dict(binding.reopen())
    materialization, signed = _validate_formal_downstream_pilot_precoverage_sources(
        artifact,
        now_ns=now_ns,
    )
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("formal downstream pilot precoverage changed while rebuilt")
    return RebuiltFormalDownstreamPilotPrecoverage(
        artifact_binding=binding,
        artifact=artifact,
        materialization=materialization,
        signed_materialization=signed,
    )


def rebuild_formal_downstream_evidence_manifest(
    portable_coverage_proof_path: str | Path,
    *,
    context: FormalStageCoverageRebuiltContext,
) -> FormalDownstreamEvidenceManifest:
    """Derive one downstream reducer manifest from its portable coverage graph."""

    from lightcone_spec.experiments.formal_stage_coverage_portable import (
        FormalPortableStageCoverageProofArtifact,
    )

    if type(context) is not FormalStageCoverageRebuiltContext:
        raise TypeError("formal downstream evidence requires rebuilt coverage context")
    portable_binding = CanonicalJsonProofBinding.bind(portable_coverage_proof_path)
    portable = FormalPortableStageCoverageProofArtifact.from_dict(
        portable_binding.reopen()
    )
    low_level_binding = CanonicalJsonProofBinding.bind(
        portable.coverage_proof_source.absolute_path
    )
    if low_level_binding != portable.coverage_proof_source:
        raise ValueError("formal downstream low-level coverage identity changed")
    proof = FormalStageCoverageProofArtifact.from_dict(low_level_binding.reopen())
    if (
        proof.stage != context.materialization.stage
        or proof.materialization_receipt_sha256 != context.materialization.sha256
        or proof.coverage_receipt_sha256 != context.coverage.sha256
        or portable.materialization_receipt_sha256 != context.materialization.sha256
        or portable.coverage_receipt_sha256 != context.coverage.sha256
    ):
        raise ValueError("formal downstream evidence coverage lineage differs")
    shards: list[FormalStageCoverageEvidenceShard] = []
    for source in proof.evidence_shard_sources:
        before = CanonicalJsonProofBinding.bind(source.absolute_path)
        if before != source:
            raise ValueError("formal downstream evidence shard identity changed")
        shard = FormalStageCoverageEvidenceShard.from_dict(before.reopen())
        if CanonicalJsonProofBinding.bind(source.absolute_path) != before:
            raise RuntimeError("formal downstream evidence shard changed while read")
        shards.append(shard)
    if (
        not shards
        or tuple(row.shard_index for row in shards) != tuple(range(len(shards)))
        or any(row.shard_count != len(shards) for row in shards)
    ):
        raise ValueError("formal downstream evidence shard sequence is incomplete")
    evidence_cells = tuple(cell for shard in shards for cell in shard.cells)
    bindings = {
        row.subject.materialized_cell_id: row for row in context.execution_bindings
    }
    if len(bindings) != len(context.execution_bindings) or set(bindings) != {
        row.materialized_cell_id for row in evidence_cells
    }:
        raise ValueError("formal downstream evidence/binding coverage differs")
    manifest = FormalDownstreamEvidenceManifest(
        schema_version=1,
        stage=context.materialization.stage,
        protocol_lock_sha256=context.protocol_lock.sha256,
        materialization_receipt_sha256=context.materialization.sha256,
        coverage_receipt_sha256=context.coverage.sha256,
        source_authority_sha256=context.materialization.source_decision_sha256,
        inventory_sha256=context.inventory.sha256,
        cells=tuple(
            FormalDownstreamCellEvidence.bind(
                stage=context.materialization.stage,
                execution_binding=bindings[row.materialized_cell_id],
                native_result_proof_path=row.native_result_proof.absolute_path,
                stage_itl_proof_path=row.stage_itl_proof.absolute_path,
            )
            for row in evidence_cells
        ),
    )
    manifest.__post_init__()
    if (
        CanonicalJsonProofBinding.bind(portable_binding.absolute_path)
        != portable_binding
    ):
        raise RuntimeError("formal downstream portable coverage changed while rebuilt")
    return manifest


def _derive_formal_downstream_reduction(
    artifact: FormalDownstreamReductionProofArtifact,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamReduction:
    from lightcone_spec.experiments.formal_stage_coverage_portable import (
        revalidate_portable_formal_stage_coverage_proof_artifact,
    )

    if type(now_ns) is not int or now_ns < artifact.created_ns:
        raise ValueError("formal downstream reduction time predates its proof")
    if artifact.phase not in {"e3b_pilot", "e3b_final", "e1a_verification"}:
        raise ValueError(
            "formal downstream reducer is BLOCKED until its phase-specific "
            "proof adapter is registered"
        )
    materialization = rebuild_formal_downstream_materialization_proof(
        artifact.materialization_proof_source.absolute_path,
        now_ns=now_ns,
    )
    if materialization.artifact.phase != artifact.phase:
        raise ValueError("formal downstream reducer materialization phase differs")
    coverage = revalidate_portable_formal_stage_coverage_proof_artifact(
        artifact.portable_coverage_proof_source.absolute_path,
        now_ns=now_ns,
    )
    if (
        coverage.materialization != materialization.materialization
        or coverage.protocol_lock
        != materialization.registry_verification_receipt.signed_protocol_lock.payload
        or coverage.inventory.sha256
        != materialization.registry_verification_receipt.inventory_sha256
        or coverage.stage_source is None
    ):
        raise ValueError("formal downstream reducer portable coverage lineage differs")
    manifest = rebuild_formal_downstream_evidence_manifest(
        artifact.portable_coverage_proof_source.absolute_path,
        context=coverage,
    )
    if artifact.phase == "e3b_pilot":
        reduction = reduce_e3b_power_prefix_from_proofs(
            protocol_lock=coverage.protocol_lock,
            pilot_materialization=coverage.materialization,
            pilot_coverage=coverage.coverage,
            manifest=manifest,
            execution_bindings=coverage.execution_bindings,
            now_ns=now_ns,
        )
    elif artifact.phase == "e3b_final":
        reduction = reduce_e3b_confirmation_from_proofs(
            protocol_lock=coverage.protocol_lock,
            materialization=coverage.materialization,
            coverage=coverage.coverage,
            manifest=manifest,
            execution_bindings=coverage.execution_bindings,
            now_ns=now_ns,
        )
    else:
        reduction = reduce_e1a_verification_from_proofs(
            protocol_lock=coverage.protocol_lock,
            materialization=coverage.materialization,
            coverage=coverage.coverage,
            manifest=manifest,
            execution_bindings=coverage.execution_bindings,
            now_ns=now_ns,
        )
    if reduction.sha256 != artifact.expected_reduction_sha256:
        raise ValueError("formal downstream reduction differs from proof")
    return RebuiltFormalDownstreamReduction(
        artifact=artifact,
        materialization=materialization,
        coverage_context=coverage,
        evidence_manifest=manifest,
        reduction=reduction,
    )


def build_formal_downstream_reduction_proof_artifact(
    *,
    phase: FormalDownstreamMaterializationPhase,
    materialization_proof_path: str | Path,
    portable_coverage_proof_path: str | Path,
    now_ns: int,
) -> FormalDownstreamReductionProofArtifact:
    """Reduce current proof sources, then bind the exact reducer output."""

    if phase not in {"e3b_pilot", "e3b_final", "e1a_verification"}:
        raise ValueError(
            "formal downstream reducer is BLOCKED until its phase-specific "
            "proof adapter is registered"
        )
    materialization_source = CanonicalJsonProofBinding.bind(materialization_proof_path)
    coverage_source = CanonicalJsonProofBinding.bind(portable_coverage_proof_path)
    materialization = rebuild_formal_downstream_materialization_proof(
        materialization_source.absolute_path,
        now_ns=now_ns,
    )
    if materialization.artifact.phase != phase:
        raise ValueError(
            "formal downstream reduction phase differs from materialization"
        )
    from lightcone_spec.experiments.formal_stage_coverage_portable import (
        revalidate_portable_formal_stage_coverage_proof_artifact,
    )

    coverage = revalidate_portable_formal_stage_coverage_proof_artifact(
        coverage_source.absolute_path,
        now_ns=now_ns,
    )
    if coverage.materialization != materialization.materialization:
        raise ValueError("formal downstream reduction coverage names another phase")
    manifest = rebuild_formal_downstream_evidence_manifest(
        coverage_source.absolute_path,
        context=coverage,
    )
    if phase == "e3b_pilot":
        reduction = reduce_e3b_power_prefix_from_proofs(
            protocol_lock=coverage.protocol_lock,
            pilot_materialization=coverage.materialization,
            pilot_coverage=coverage.coverage,
            manifest=manifest,
            execution_bindings=coverage.execution_bindings,
            now_ns=now_ns,
        )
    elif phase == "e3b_final":
        reduction = reduce_e3b_confirmation_from_proofs(
            protocol_lock=coverage.protocol_lock,
            materialization=coverage.materialization,
            coverage=coverage.coverage,
            manifest=manifest,
            execution_bindings=coverage.execution_bindings,
            now_ns=now_ns,
        )
    else:
        reduction = reduce_e1a_verification_from_proofs(
            protocol_lock=coverage.protocol_lock,
            materialization=coverage.materialization,
            coverage=coverage.coverage,
            manifest=manifest,
            execution_bindings=coverage.execution_bindings,
            now_ns=now_ns,
        )
    artifact = FormalDownstreamReductionProofArtifact(
        schema_version=1,
        kind=FORMAL_DOWNSTREAM_REDUCTION_PROOF_KIND,
        protocol_sha256=FORMAL_DOWNSTREAM_REDUCTION_PROTOCOL_SHA256,
        phase=phase,
        materialization_proof_source=materialization_source,
        portable_coverage_proof_source=coverage_source,
        expected_reduction_sha256=reduction.sha256,
        created_ns=now_ns,
    )
    rebuilt = _derive_formal_downstream_reduction(artifact, now_ns=now_ns)
    if rebuilt.reduction != reduction:
        raise RuntimeError("formal downstream reduction changed while bound")
    return artifact


def publish_formal_downstream_reduction_proof_artifact(
    artifact: FormalDownstreamReductionProofArtifact,
    output_path: str | Path,
    *,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Deep-reduce before atomically publishing one result proof root."""

    if type(artifact) is not FormalDownstreamReductionProofArtifact:
        raise TypeError("formal downstream reduction publisher requires exact input")
    _derive_formal_downstream_reduction(artifact, now_ns=now_ns)
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def rebuild_formal_downstream_reduction_proof(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamReduction:
    """Deep-open one current-only downstream reducer proof."""

    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalDownstreamReductionProofArtifact.from_dict(binding.reopen())
    rebuilt = _derive_formal_downstream_reduction(artifact, now_ns=now_ns)
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("formal downstream reduction proof changed while rebuilt")
    return rebuilt


def _derive_formal_downstream_completed_prefix(
    artifact_binding: CanonicalJsonProofBinding,
    artifact: FormalDownstreamCompletedPrefixArtifact,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamCompletedPrefix:
    from lightcone_spec.runtime.scientific_signing import (
        rebuild_scientific_signed_proof_wrapper,
    )

    if type(now_ns) is not int or now_ns < artifact.created_ns:
        raise ValueError("formal downstream completed-prefix time predates its proof")
    if artifact.phase not in {"e3b_pilot", "e3b_final", "e1a_verification"}:
        raise ValueError(
            "formal downstream completed prefix is BLOCKED until its phase-specific "
            "signed reducer adapter is registered"
        )
    reduction = rebuild_formal_downstream_reduction_proof(
        artifact.reduction_proof_source.absolute_path,
        now_ns=now_ns,
    )
    if reduction.artifact.phase != artifact.phase:
        raise ValueError("formal downstream completed-prefix reducer phase differs")
    signed = rebuild_scientific_signed_proof_wrapper(
        artifact.signed_result_source.absolute_path,
        now_ns=now_ns,
    )
    expected_type = {
        "e3b_pilot": SignedE3bPowerPrefixReceipt,
        "e3b_final": SignedE3bConfirmationReceipt,
        "e1a_verification": SignedE1aVerificationReceipt,
    }[artifact.phase]
    if type(signed) is not expected_type:
        raise TypeError("formal E3b completed prefix signed result type differs")
    if (
        signed.payload != reduction.reduction
        or signed.sha256 != artifact.expected_signed_result_sha256
    ):
        raise ValueError("formal downstream completed-prefix signed payload differs")
    receipt = reduction.materialization.registry_verification_receipt
    policy = receipt.trusted_release_policy(current_ns=now_ns)
    if type(signed) is SignedE3bPowerPrefixReceipt:
        signed.verify(
            protocol_lock=reduction.coverage_context.protocol_lock,
            pilot_materialization=reduction.coverage_context.materialization,
            pilot_coverage=reduction.coverage_context.coverage,
            manifest=reduction.evidence_manifest,
            execution_bindings=reduction.coverage_context.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    elif type(signed) is SignedE3bConfirmationReceipt:
        signed.verify(
            protocol_lock=reduction.coverage_context.protocol_lock,
            materialization=reduction.coverage_context.materialization,
            coverage=reduction.coverage_context.coverage,
            manifest=reduction.evidence_manifest,
            execution_bindings=reduction.coverage_context.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    else:
        assert type(signed) is SignedE1aVerificationReceipt
        signed.verify(
            protocol_lock=reduction.coverage_context.protocol_lock,
            materialization=reduction.coverage_context.materialization,
            coverage=reduction.coverage_context.coverage,
            manifest=reduction.evidence_manifest,
            execution_bindings=reduction.coverage_context.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    return RebuiltFormalDownstreamCompletedPrefix(
        artifact_binding=artifact_binding,
        artifact=artifact,
        reduction=reduction,
        signed_result=signed,
    )


def build_formal_downstream_completed_prefix_artifact(
    *,
    phase: FormalDownstreamMaterializationPhase,
    reduction_proof_path: str | Path,
    signed_result_path: str | Path,
    now_ns: int,
) -> FormalDownstreamCompletedPrefixArtifact:
    """Bind one signed current result after reducer and signature replay."""

    from lightcone_spec.runtime.scientific_signing import (
        rebuild_scientific_signed_proof_wrapper,
    )

    if phase not in {"e3b_pilot", "e3b_final", "e1a_verification"}:
        raise ValueError(
            "formal downstream completed prefix is BLOCKED until its phase-specific "
            "signed reducer adapter is registered"
        )
    reduction_source = CanonicalJsonProofBinding.bind(reduction_proof_path)
    signed_source = CanonicalJsonProofBinding.bind(signed_result_path)
    reduction = rebuild_formal_downstream_reduction_proof(
        reduction_source.absolute_path,
        now_ns=now_ns,
    )
    signed = rebuild_scientific_signed_proof_wrapper(
        signed_source.absolute_path,
        now_ns=now_ns,
    )
    expected_type = {
        "e3b_pilot": SignedE3bPowerPrefixReceipt,
        "e3b_final": SignedE3bConfirmationReceipt,
        "e1a_verification": SignedE1aVerificationReceipt,
    }[phase]
    if (
        reduction.artifact.phase != phase
        or type(signed) is not expected_type
        or signed.payload != reduction.reduction
    ):
        raise ValueError("formal downstream completed-prefix sources differ")
    artifact = FormalDownstreamCompletedPrefixArtifact(
        schema_version=1,
        kind=FORMAL_DOWNSTREAM_COMPLETED_PREFIX_KIND,
        protocol_sha256=FORMAL_DOWNSTREAM_COMPLETED_PREFIX_PROTOCOL_SHA256,
        phase=phase,
        reduction_proof_source=reduction_source,
        signed_result_source=signed_source,
        expected_signed_result_sha256=signed.sha256,
        created_ns=now_ns,
    )
    policy = (
        reduction.materialization.registry_verification_receipt.trusted_release_policy(
            current_ns=now_ns
        )
    )
    if type(signed) is SignedE3bPowerPrefixReceipt:
        signed.verify(
            protocol_lock=reduction.coverage_context.protocol_lock,
            pilot_materialization=reduction.coverage_context.materialization,
            pilot_coverage=reduction.coverage_context.coverage,
            manifest=reduction.evidence_manifest,
            execution_bindings=reduction.coverage_context.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    elif type(signed) is SignedE3bConfirmationReceipt:
        signed.verify(
            protocol_lock=reduction.coverage_context.protocol_lock,
            materialization=reduction.coverage_context.materialization,
            coverage=reduction.coverage_context.coverage,
            manifest=reduction.evidence_manifest,
            execution_bindings=reduction.coverage_context.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    else:
        assert type(signed) is SignedE1aVerificationReceipt
        signed.verify(
            protocol_lock=reduction.coverage_context.protocol_lock,
            materialization=reduction.coverage_context.materialization,
            coverage=reduction.coverage_context.coverage,
            manifest=reduction.evidence_manifest,
            execution_bindings=reduction.coverage_context.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    return artifact


def publish_formal_downstream_completed_prefix_artifact(
    artifact: FormalDownstreamCompletedPrefixArtifact,
    output_path: str | Path,
    *,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Publish, then deep-replay, one immutable completed-prefix link."""

    if type(artifact) is not FormalDownstreamCompletedPrefixArtifact:
        raise TypeError("formal downstream prefix publisher requires exact input")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    binding = CanonicalJsonProofBinding.bind(output_path)
    _derive_formal_downstream_completed_prefix(
        binding,
        artifact,
        now_ns=now_ns,
    )
    return binding


def rebuild_formal_downstream_completed_prefix(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> RebuiltFormalDownstreamCompletedPrefix:
    """Deep-open one append-only completed node and its signed reducer proof."""

    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalDownstreamCompletedPrefixArtifact.from_dict(binding.reopen())
    rebuilt = _derive_formal_downstream_completed_prefix(
        binding,
        artifact,
        now_ns=now_ns,
    )
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != binding:
        raise RuntimeError("formal downstream completed prefix changed while rebuilt")
    return rebuilt


__all__ = (
    "FORMAL_DOWNSTREAM_COMPLETED_PREFIX_KIND",
    "FORMAL_DOWNSTREAM_COMPLETED_PREFIX_PROTOCOL_SHA256",
    "FORMAL_DOWNSTREAM_MATERIALIZATION_ORDER",
    "FORMAL_DOWNSTREAM_MATERIALIZATION_PROOF_KIND",
    "FORMAL_DOWNSTREAM_MATERIALIZATION_PROTOCOL_SHA256",
    "FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_KIND",
    "FORMAL_DOWNSTREAM_PILOT_PRECOVERAGE_PROTOCOL_SHA256",
    "FORMAL_DOWNSTREAM_REDUCTION_PROOF_KIND",
    "FORMAL_DOWNSTREAM_REDUCTION_PROTOCOL_SHA256",
    "FormalDownstreamCompletedPrefixArtifact",
    "FormalDownstreamMaterializationPhase",
    "FormalDownstreamMaterializationProofArtifact",
    "FormalDownstreamPilotPrecoverageArtifact",
    "FormalDownstreamReductionProofArtifact",
    "RebuiltFormalDownstreamCompletedPrefix",
    "RebuiltFormalDownstreamMaterialization",
    "RebuiltFormalDownstreamPilotPrecoverage",
    "RebuiltFormalDownstreamReduction",
    "build_formal_downstream_completed_prefix_artifact",
    "build_formal_downstream_materialization_proof_artifact",
    "build_formal_downstream_pilot_precoverage_artifact",
    "build_formal_downstream_reduction_proof_artifact",
    "publish_formal_downstream_completed_prefix_artifact",
    "publish_formal_downstream_materialization_proof_artifact",
    "publish_formal_downstream_pilot_precoverage_artifact",
    "publish_formal_downstream_reduction_proof_artifact",
    "rebuild_formal_downstream_completed_prefix",
    "rebuild_formal_downstream_evidence_manifest",
    "rebuild_formal_downstream_materialization_proof",
    "rebuild_formal_downstream_pilot_precoverage",
    "rebuild_formal_downstream_reduction_proof",
)
