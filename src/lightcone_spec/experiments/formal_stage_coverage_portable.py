"""Caller-free replay wrapper for formal E1/E2/E4 coverage proofs.

The low-level coverage artifact deliberately stores only reducer inputs.  A
downstream verifier must also recover the frozen recipe and, for E4, the exact
typed predecessor source.  Those values must not be supplied as trusted
Python objects by the caller.  This module binds the coverage proof to one
proof-carrying registry layer and the exact predecessor prefix, then rebuilds
all verifier-private values from those durable sources.

TTS-Cal is included as the initial branch.  Its reducer inputs are
self-contained, but its signed materialization and ProtocolLock must still be
present in an exact proof-carrying pre-coverage registry layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.e2_stage_authority import (
    E2StagedRoundEvidenceManifest,
)
from lightcone_spec.experiments.e4_stage_authority import E4StagedEvidenceManifest
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry_layers import (
    load_formal_registry_verification_receipt_path,
    validate_formal_precoverage_registry_state,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
    FormalStageCoverageProofArtifact,
    FormalStageCoverageRebuiltContext,
    rebuild_formal_stage_coverage_context,
)
from lightcone_spec.experiments.formal_stage_execution import (
    E4LocalStageSourceRebuildInputs,
    E4ProfilerStageSourceRebuildInputs,
    E4ScreenStageSourceRebuildInputs,
    load_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.formal_stage_prefix import (
    RebuiltFormalStagePrefix,
    load_and_rebuild_formal_stage_prefix,
)
from lightcone_spec.experiments.stage_materialization import (
    default_e2_recipe_grid_authority,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_PORTABLE_STAGE_COVERAGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 5,
        "kind": "lightcone_formal_portable_stage_coverage_protocol",
        "coverage_reducer_protocol_sha256": FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
        "scope": (
            "e3a_plus_tts_cal_plus_e1_e2_e4_closed_precoverage_prefix_"
            "plus_current_only_downstream_materialization_proof"
        ),
        "authority": (
            "proof_carrying_registry_layer_plus_exact_predecessor_prefix_"
            "without_caller_verified_tokens_or_reducer_kwargs"
        ),
        "transport": "relocatable_closed_transitive_binding_graph",
        "readiness_gates": {
            "tts_calibration": "exact_288_source_owned_execution_artifact",
            "e3b_excluded_pilot": "signed_pilot_materialization_precoverage",
        },
    }
)

_PHASE_BY_STAGE = {
    ("E3a", "capacity"): "e3a_capacity",
    ("TTS-Cal", "calibration"): "tts_calibration",
    ("E1", "selection"): "e1_selection",
    ("E2", "round0"): "e2_round0",
    ("E2", "round1"): "e2_round1",
    ("E2", "round2"): "e2_round2",
    ("E2", "round3"): "e2_round3",
    ("E4", "screen"): "e4_screen",
    ("E4", "local"): "e4_local",
    ("E4", "profiler"): "e4_profiler",
    ("E3b", "excluded_pilot"): "e3b_pilot",
}
_PRIOR_BY_PHASE = {
    "e3a_capacity": None,
    "e1_selection": None,
    "e2_round0": "e1_selection",
    "e2_round1": "e2_round0",
    "e2_round2": "e2_round1",
    "e2_round3": "e2_round2",
    "e4_screen": "e2_round3",
    "e4_local": "e4_screen",
    "e4_profiler": "e4_local",
}

_TTS_RAW_EXECUTION_BLOCKED = (
    "portable TTS-Cal coverage is BLOCKED until the exact-288 source-owned "
    "execution artifact adapter is registered"
)


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


@dataclass(frozen=True)
class FormalPortableStageCoverageProofArtifact:
    """Compact root of a caller-free, relocatable coverage replay graph."""

    schema_version: Literal[3]
    kind: Literal["formal_portable_stage_coverage_proof_artifact"]
    protocol_sha256: str
    stage: Literal["E3a", "TTS-Cal", "E1", "E2", "E4", "E3b"]
    phase: str
    coverage_receipt_sha256: str
    materialization_receipt_sha256: str
    registry_verification_receipt_sha256: str
    coverage_proof_source: CanonicalJsonProofBinding
    registry_layer_source: CanonicalJsonProofBinding
    prior_prefix_source: CanonicalJsonProofBinding | None
    e1_recipe_anchor_authority_source: CanonicalJsonProofBinding | None
    downstream_pilot_precoverage_source: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 3
            or self.kind != "formal_portable_stage_coverage_proof_artifact"
            or self.protocol_sha256 != FORMAL_PORTABLE_STAGE_COVERAGE_PROTOCOL_SHA256
            or (self.stage, self.phase) not in _PHASE_BY_STAGE
        ):
            raise ValueError("portable stage coverage proof identity differs")
        _sha256("portable coverage receipt", self.coverage_receipt_sha256)
        _sha256(
            "portable coverage materialization",
            self.materialization_receipt_sha256,
        )
        if type(self.coverage_proof_source) is not CanonicalJsonProofBinding:
            raise TypeError("portable coverage proof source is not path-bound")
        _sha256(
            "portable coverage registry receipt",
            self.registry_verification_receipt_sha256,
        )
        if type(self.registry_layer_source) is not CanonicalJsonProofBinding:
            raise TypeError("portable coverage registry is not path-bound")
        tts_branch = self.stage == "TTS-Cal"
        downstream_branch = self.stage == "E3b"
        if tts_branch:
            if (
                self.prior_prefix_source is not None
                or self.e1_recipe_anchor_authority_source is not None
                or self.downstream_pilot_precoverage_source is not None
            ):
                raise ValueError("portable TTS coverage carries foreign authorities")
        elif downstream_branch:
            if (
                self.phase != "excluded_pilot"
                or self.prior_prefix_source is not None
                or self.e1_recipe_anchor_authority_source is not None
                or type(self.downstream_pilot_precoverage_source)
                is not CanonicalJsonProofBinding
            ):
                raise ValueError(
                    "portable downstream coverage source union is not exact"
                )
        else:
            phase = _PHASE_BY_STAGE[(self.stage, self.phase)]
            expected_prior = _PRIOR_BY_PHASE[phase]
            if (self.prior_prefix_source is None) != (expected_prior is None):
                raise ValueError("portable coverage predecessor union differs")
            if (
                self.prior_prefix_source is not None
                and type(self.prior_prefix_source) is not CanonicalJsonProofBinding
            ):
                raise TypeError("portable coverage predecessor is not path-bound")
            requires_anchor = phase == "e1_selection"
            if (self.e1_recipe_anchor_authority_source is None) == requires_anchor:
                raise ValueError("portable coverage E1 anchor union differs")
            if (
                self.e1_recipe_anchor_authority_source is not None
                and type(self.e1_recipe_anchor_authority_source)
                is not CanonicalJsonProofBinding
            ):
                raise TypeError("portable coverage E1 anchor is not path-bound")
            if self.downstream_pilot_precoverage_source is not None:
                raise ValueError("early portable coverage carries downstream authority")
        paths = tuple(
            row.absolute_path
            for row in (
                self.coverage_proof_source,
                self.registry_layer_source,
                self.prior_prefix_source,
                self.e1_recipe_anchor_authority_source,
                self.downstream_pilot_precoverage_source,
            )
            if row is not None
        )
        if len(paths) != len(set(paths)):
            raise ValueError("portable coverage reuses a top-level source path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        def optional(value: CanonicalJsonProofBinding | None) -> object:
            return None if value is None else value.to_dict()

        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "phase": self.phase,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "registry_verification_receipt_sha256": (
                self.registry_verification_receipt_sha256
            ),
            "coverage_proof_source": self.coverage_proof_source.to_dict(),
            "registry_layer_source": self.registry_layer_source.to_dict(),
            "prior_prefix_source": optional(self.prior_prefix_source),
            "e1_recipe_anchor_authority_source": optional(
                self.e1_recipe_anchor_authority_source
            ),
            "downstream_pilot_precoverage_source": optional(
                self.downstream_pilot_precoverage_source
            ),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("portable stage coverage proof fields differ")
        row = dict(value)
        declared = _sha256("portable stage coverage proof", row.pop("artifact_sha256"))
        row["coverage_proof_source"] = CanonicalJsonProofBinding.from_dict(
            row["coverage_proof_source"]
        )
        row["registry_layer_source"] = CanonicalJsonProofBinding.from_dict(
            row["registry_layer_source"]
        )
        for name in (
            "prior_prefix_source",
            "e1_recipe_anchor_authority_source",
            "downstream_pilot_precoverage_source",
        ):
            if row[name] is not None:
                row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("portable stage coverage proof digest differs")
        return artifact


def _one(rows: tuple[object, ...], *, label: str, predicate) -> object:
    matches = tuple(row for row in rows if predicate(row))
    if len(matches) != 1:
        raise ValueError(f"portable coverage requires exactly one {label}")
    return matches[0]


def _receipt_chain_sha256s(receipt: object) -> frozenset[str]:
    values: set[str] = set()
    current = receipt
    while current is not None:
        digest = getattr(current, "sha256", None)
        _sha256("portable coverage registry prefix", digest)
        if digest in values:
            raise ValueError("portable coverage registry chain contains a cycle")
        values.add(digest)
        current = getattr(current, "prior_receipt", None)
    return frozenset(values)


def _require_exact_precoverage_registry_state(
    *,
    receipt,
    coverage_proof: FormalStageCoverageProofArtifact,
    prior: RebuiltFormalStagePrefix | None,
) -> None:
    """Reject retrospective/future registry prefixes as coverage authority."""

    materializations = tuple(
        row.payload
        for row in receipt.cumulative_signed_materializations
        if row.payload.sha256 == coverage_proof.materialization_receipt_sha256
    )
    if len(materializations) != 1:
        raise ValueError(
            "portable coverage registry is not the immediate materialization prefix"
        )
    predecessor_sha256 = (
        None if prior is None else prior.artifact_binding.semantic_sha256
    )
    validate_formal_precoverage_registry_state(
        receipt,
        stage=coverage_proof.stage,
        phase=coverage_proof.phase,
        materialization=materializations[0],
        immediate_predecessor_prefix_sha256=predecessor_sha256,
    )
    current_results = (
        *receipt.cumulative_signed_e1_survivor_selections,
        *receipt.cumulative_signed_e2_staged_selections,
        *receipt.cumulative_signed_e4_stage_selections,
    )
    if any(
        getattr(row.payload, "materialization_receipt_sha256", None)
        == coverage_proof.materialization_receipt_sha256
        for row in current_results
    ):
        raise ValueError("portable coverage registry contains a current/future result")
    prefixes = receipt.cumulative_formal_stage_prefix_artifacts
    if prior is None:
        if prefixes:
            raise ValueError(
                "portable initial coverage registry contains a future prefix"
            )
    elif not prefixes or prefixes[-1] != prior.artifact_binding:
        raise ValueError("portable coverage registry is not the immediate prior prefix")
    if coverage_proof.stage == "TTS-Cal" and (
        receipt.cumulative_tts_calibration_authorities
        or receipt.cumulative_signed_tts_calibration_seals
    ):
        raise ValueError("portable TTS coverage registry contains a future seal")


def _e4_stage_source_inputs(
    phase: str,
    *,
    receipt,
    prior: RebuiltFormalStagePrefix,
):
    if phase == "e4_screen":
        signed = _one(
            receipt.cumulative_signed_e2_staged_selections,
            label="E2 final selection",
            predicate=lambda row: (
                row.payload.round_index == 3
                and row.payload.materialization_receipt_sha256
                == prior.materialization.sha256
            ),
        )
        if type(prior.evidence_manifest) is not E2StagedRoundEvidenceManifest:
            raise TypeError("portable E4 screen predecessor evidence differs")
        return E4ScreenStageSourceRebuildInputs(
            registry_verification_receipt=receipt,
            signed_e2_final_selection=signed,
            e2_materialization=prior.materialization,
            e2_coverage=prior.coverage,
            e2_source_recipes=prior.source_recipes,
            e2_evidence_manifest=prior.evidence_manifest,
            e2_execution_bindings=prior.execution_bindings,
        )
    if phase == "e4_local":
        signed = _one(
            receipt.cumulative_signed_e4_stage_selections,
            label="E4 screen selection",
            predicate=lambda row: (
                row.payload.phase == "screen"
                and row.payload.materialization_receipt_sha256
                == prior.materialization.sha256
            ),
        )
        if type(prior.evidence_manifest) is not E4StagedEvidenceManifest:
            raise TypeError("portable E4 local predecessor evidence differs")
        return E4LocalStageSourceRebuildInputs(
            registry_verification_receipt=receipt,
            signed_e4_screen_selection=signed,
            screen_materialization=prior.materialization,
            screen_coverage=prior.coverage,
            screen_evidence_manifest=prior.evidence_manifest,
            screen_execution_bindings=prior.execution_bindings,
        )
    if phase == "e4_profiler":
        signed = _one(
            receipt.cumulative_signed_e4_stage_selections,
            label="E4 local selection",
            predicate=lambda row: (
                row.payload.phase == "local"
                and row.payload.materialization_receipt_sha256
                == prior.materialization.sha256
            ),
        )
        if type(prior.evidence_manifest) is not E4StagedEvidenceManifest:
            raise TypeError("portable E4 profiler predecessor evidence differs")
        return E4ProfilerStageSourceRebuildInputs(
            registry_verification_receipt=receipt,
            signed_e4_final_selection=signed,
            local_materialization=prior.materialization,
            local_coverage=prior.coverage,
            local_evidence_manifest=prior.evidence_manifest,
            local_execution_bindings=prior.execution_bindings,
        )
    return None


def _rebuild_portable_stage_coverage(
    artifact: FormalPortableStageCoverageProofArtifact,
    *,
    now_ns: int,
) -> FormalStageCoverageRebuiltContext:
    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("portable coverage validation time is invalid")
    coverage_binding = CanonicalJsonProofBinding.bind(
        artifact.coverage_proof_source.absolute_path
    )
    if coverage_binding != artifact.coverage_proof_source:
        raise ValueError("portable coverage proof path identity changed")
    coverage_proof = FormalStageCoverageProofArtifact.from_dict(
        coverage_binding.reopen()
    )
    if (
        coverage_proof.stage != artifact.stage
        or coverage_proof.phase != artifact.phase
        or coverage_proof.coverage_receipt_sha256 != artifact.coverage_receipt_sha256
        or coverage_proof.materialization_receipt_sha256
        != artifact.materialization_receipt_sha256
    ):
        raise ValueError("portable coverage nested proof identity differs")
    if artifact.stage == "TTS-Cal":
        raise ValueError(_TTS_RAW_EXECUTION_BLOCKED)
    registry_binding = CanonicalJsonProofBinding.bind(
        artifact.registry_layer_source.absolute_path
    )
    if registry_binding != artifact.registry_layer_source:
        raise ValueError("portable coverage registry layer identity changed")
    receipt = load_formal_registry_verification_receipt_path(
        registry_binding.absolute_path,
        now_ns=now_ns,
    )
    if (
        receipt.sha256 != artifact.registry_verification_receipt_sha256
        or receipt.signed_protocol_lock.payload.sha256
        != coverage_proof.protocol_lock_sha256
        or receipt.inventory_sha256 != coverage_proof.inventory_sha256
    ):
        raise ValueError("portable coverage registry immutable root differs")
    if artifact.stage == "E3b":
        from lightcone_spec.experiments.formal_downstream_prefix import (
            rebuild_formal_downstream_pilot_precoverage,
        )

        assert artifact.downstream_pilot_precoverage_source is not None
        precoverage_binding = CanonicalJsonProofBinding.bind(
            artifact.downstream_pilot_precoverage_source.absolute_path
        )
        if precoverage_binding != artifact.downstream_pilot_precoverage_source:
            raise ValueError("portable E3b precoverage path identity changed")
        precoverage = rebuild_formal_downstream_pilot_precoverage(
            precoverage_binding.absolute_path,
            now_ns=now_ns,
        )
        downstream = precoverage.materialization
        predecessor = downstream.immediate_predecessor
        lock = receipt.signed_protocol_lock.payload
        tts_rows = receipt.cumulative_tts_calibration_authorities
        signed_tts_rows = receipt.cumulative_signed_tts_calibration_seals
        final_recipes = tuple(
            row.payload.final_recipe
            for row in receipt.cumulative_signed_e2_staged_selections
            if row.payload.round_index == 3 and row.payload.final_recipe is not None
        )
        grid = default_e2_recipe_grid_authority()
        if (
            precoverage.artifact.phase != "e3b_pilot"
            or downstream.registry_verification_receipt != receipt
            or downstream.artifact.registry_layer_source != registry_binding
            or precoverage.signed_materialization.payload != downstream.materialization
            or downstream.materialization.sha256
            != coverage_proof.materialization_receipt_sha256
            or predecessor.artifact.phase != "e4_profiler"
            or len(tts_rows) != 1
            or len(signed_tts_rows) != 1
            or len(final_recipes) != 1
            or tts_rows[0].sha256 != lock.tts_calibration_authority_sha256
            or grid.sha256 != lock.e2_recipe_grid_authority_sha256
        ):
            raise ValueError("portable E3b coverage source DAG differs")
        context = rebuild_formal_stage_coverage_context(
            coverage_binding.absolute_path,
            now_ns=now_ns,
            tts_authority=tts_rows[0],
            signed_tts_seal=signed_tts_rows[0],
            e1_recipe_anchor_authority=(predecessor.e1_recipe_anchor_authority),
            e2_recipe_grid_authority=grid,
            lightcone_recipe=final_recipes[0],
            registry_verification_receipt=receipt,
            stage_source_inputs=downstream.stage_source_inputs,
        )
        if (
            context.coverage.sha256 != artifact.coverage_receipt_sha256
            or context.materialization != downstream.materialization
        ):
            raise ValueError("portable E3b coverage replay changed its result")
        return context
    if artifact.stage in {"E3a", "TTS-Cal"}:
        _require_exact_precoverage_registry_state(
            receipt=receipt,
            coverage_proof=coverage_proof,
            prior=None,
        )
        context = rebuild_formal_stage_coverage_context(
            coverage_binding.absolute_path,
            now_ns=now_ns,
        )
        if (
            context.coverage.sha256 != artifact.coverage_receipt_sha256
            or context.materialization.sha256 != artifact.materialization_receipt_sha256
        ):
            raise ValueError("portable initial coverage replay changed its result")
        return context

    tts_rows = receipt.cumulative_tts_calibration_authorities
    signed_tts_rows = receipt.cumulative_signed_tts_calibration_seals
    if len(tts_rows) != 1 or len(signed_tts_rows) != 1:
        raise ValueError("portable coverage lacks one frozen TTS authority/seal")
    tts = tts_rows[0]
    signed_tts = signed_tts_rows[0]
    lock = receipt.signed_protocol_lock.payload
    grid = default_e2_recipe_grid_authority()
    if (
        tts.sha256 != lock.tts_calibration_authority_sha256
        or grid.sha256 != lock.e2_recipe_grid_authority_sha256
    ):
        raise ValueError("portable coverage recipe roots differ from ProtocolLock")

    phase = _PHASE_BY_STAGE[(artifact.stage, artifact.phase)]
    prior = None
    if artifact.prior_prefix_source is not None:
        prior_binding = CanonicalJsonProofBinding.bind(
            artifact.prior_prefix_source.absolute_path
        )
        if prior_binding != artifact.prior_prefix_source:
            raise ValueError("portable coverage predecessor path identity changed")
        prior = load_and_rebuild_formal_stage_prefix(
            prior_binding.absolute_path,
            now_ns=now_ns,
        )
        if (
            prior.artifact.phase != _PRIOR_BY_PHASE[phase]
            or prior.registry_verification_receipt.sha256
            not in _receipt_chain_sha256s(receipt)
            or prior.artifact_binding
            not in receipt.cumulative_formal_stage_prefix_artifacts
        ):
            raise ValueError("portable coverage predecessor is not exact/current")
        e1_anchor = prior.e1_recipe_anchor_authority
    else:
        assert artifact.e1_recipe_anchor_authority_source is not None
        anchor_binding = CanonicalJsonProofBinding.bind(
            artifact.e1_recipe_anchor_authority_source.absolute_path
        )
        if anchor_binding != artifact.e1_recipe_anchor_authority_source:
            raise ValueError("portable coverage E1 anchor path identity changed")
        anchor_artifact = load_e1_recipe_anchor_authority_artifact(
            anchor_binding.absolute_path
        )
        e1_anchor = anchor_artifact.authority
    _require_exact_precoverage_registry_state(
        receipt=receipt,
        coverage_proof=coverage_proof,
        prior=prior,
    )
    if e1_anchor.sha256 != lock.e1_recipe_anchor_authority_sha256:
        raise ValueError("portable coverage E1 anchor differs from ProtocolLock")

    final_recipes = tuple(
        row.payload.final_recipe
        for row in receipt.cumulative_signed_e2_staged_selections
        if row.payload.round_index == 3 and row.payload.final_recipe is not None
    )
    if phase.startswith("e4_"):
        if len(final_recipes) != 1:
            raise ValueError("portable E4 coverage lacks one frozen LightCone recipe")
        assert prior is not None
        stage_source_inputs = _e4_stage_source_inputs(
            phase,
            receipt=receipt,
            prior=prior,
        )
        lightcone = final_recipes[0]
    else:
        if final_recipes:
            raise ValueError("portable E1/E2 registry includes a future winner")
        stage_source_inputs = None
        lightcone = None
    context = rebuild_formal_stage_coverage_context(
        coverage_binding.absolute_path,
        now_ns=now_ns,
        tts_authority=tts,
        signed_tts_seal=signed_tts,
        e1_recipe_anchor_authority=e1_anchor,
        e2_recipe_grid_authority=grid,
        lightcone_recipe=lightcone,
        registry_verification_receipt=receipt,
        stage_source_inputs=stage_source_inputs,
    )
    if (
        context.coverage.sha256 != artifact.coverage_receipt_sha256
        or context.materialization.sha256 != artifact.materialization_receipt_sha256
    ):
        raise ValueError("portable coverage replay changed its result")
    return context


def bind_formal_portable_stage_coverage_proof_artifact(
    coverage_proof_path: str | Path,
    *,
    registry_layer_path: str | Path,
    prior_prefix_path: str | Path | None = None,
    e1_recipe_anchor_authority_path: str | Path | None = None,
    downstream_pilot_precoverage_path: str | Path | None = None,
    now_ns: int,
) -> FormalPortableStageCoverageProofArtifact:
    """Bind and fully replay one closed coverage graph before publication."""

    coverage_source = CanonicalJsonProofBinding.bind(coverage_proof_path)
    proof = FormalStageCoverageProofArtifact.from_dict(coverage_source.reopen())
    if (proof.stage, proof.phase) not in _PHASE_BY_STAGE:
        raise ValueError("portable coverage phase is not in the closed prefix")
    if proof.stage == "TTS-Cal":
        raise ValueError(_TTS_RAW_EXECUTION_BLOCKED)
    registry_source = CanonicalJsonProofBinding.bind(registry_layer_path)
    prior_source = (
        None
        if prior_prefix_path is None
        else CanonicalJsonProofBinding.bind(prior_prefix_path)
    )
    anchor_source = (
        None
        if e1_recipe_anchor_authority_path is None
        else CanonicalJsonProofBinding.bind(e1_recipe_anchor_authority_path)
    )
    downstream_source = (
        None
        if downstream_pilot_precoverage_path is None
        else CanonicalJsonProofBinding.bind(downstream_pilot_precoverage_path)
    )
    registry_sha256 = load_formal_registry_verification_receipt_path(
        registry_source.absolute_path,
        now_ns=now_ns,
    ).sha256
    artifact = FormalPortableStageCoverageProofArtifact(
        schema_version=3,
        kind="formal_portable_stage_coverage_proof_artifact",
        protocol_sha256=FORMAL_PORTABLE_STAGE_COVERAGE_PROTOCOL_SHA256,
        stage=proof.stage,  # type: ignore[arg-type]
        phase=proof.phase,
        coverage_receipt_sha256=proof.coverage_receipt_sha256,
        materialization_receipt_sha256=proof.materialization_receipt_sha256,
        registry_verification_receipt_sha256=registry_sha256,
        coverage_proof_source=coverage_source,
        registry_layer_source=registry_source,
        prior_prefix_source=prior_source,
        e1_recipe_anchor_authority_source=anchor_source,
        downstream_pilot_precoverage_source=downstream_source,
    )
    _rebuild_portable_stage_coverage(artifact, now_ns=now_ns)
    return artifact


def publish_formal_portable_stage_coverage_proof_artifact(
    artifact: FormalPortableStageCoverageProofArtifact,
    output_path: str | Path,
    *,
    now_ns: int,
) -> CanonicalJsonProofBinding:
    """Replay first, then publish a bounded no-replace portable root."""

    if type(artifact) is not FormalPortableStageCoverageProofArtifact:
        raise TypeError("portable coverage publisher requires an exact artifact")
    _rebuild_portable_stage_coverage(artifact, now_ns=now_ns)
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def revalidate_portable_formal_stage_coverage_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
    relocatable_bundle_manifest_path: str | Path | None = None,
) -> FormalStageCoverageRebuiltContext:
    """Deep-replay one portable root, optionally after a remote A→B pull."""

    if relocatable_bundle_manifest_path is not None:
        from lightcone_spec.runtime.relocatable_evidence import (
            activate_relocatable_evidence_bundle,
        )

        with activate_relocatable_evidence_bundle(
            relocatable_bundle_manifest_path
        ) as bundle:
            remote_path = str(Path(artifact_path))
            if remote_path not in bundle.artifact.entry_remote_absolute_paths:
                raise ValueError("portable coverage proof is not a pulled entry")
            return revalidate_portable_formal_stage_coverage_proof_artifact(
                remote_path,
                now_ns=now_ns,
            )
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalPortableStageCoverageProofArtifact.from_dict(binding.reopen())
    return _rebuild_portable_stage_coverage(artifact, now_ns=now_ns)


__all__ = (
    "FORMAL_PORTABLE_STAGE_COVERAGE_PROTOCOL_SHA256",
    "FormalPortableStageCoverageProofArtifact",
    "bind_formal_portable_stage_coverage_proof_artifact",
    "publish_formal_portable_stage_coverage_proof_artifact",
    "revalidate_portable_formal_stage_coverage_proof_artifact",
)
