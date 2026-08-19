"""Append-only public proof prefixes for sequential formal E1/E2/E4 work.

The final E0 reconstruction artifact is intentionally unsuitable as an input
to an earlier stage: it already contains future decisions.  This module gives
the operator a smaller authority.  Every node binds the historical durable
registry prefix used for that execution, the raw materialization/coverage
proofs, and only its immediate predecessor.  Loading recursively rebuilds the
private verifier tokens; no serialized private seal or caller-authored metric
summary is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.e1_stage_authority import (
    E1CellExecutionEvidence,
    E1StagedParetoEvidenceManifest,
)
from lightcone_spec.experiments.e2_stage_authority import (
    E2CellExecutionEvidence,
    E2StagedRoundEvidenceManifest,
    E2StagedRoundSelectionReceipt,
    SignedE2StagedRoundSelectionReceipt,
    reduce_e2_staged_round_selection_from_proofs,
)
from lightcone_spec.experiments.e4_stage_authority import (
    E4CellExecutionEvidence,
    E4ProfilerCompletionReceipt,
    E4StagedEvidenceManifest,
    E4StageSelectionReceipt,
    SignedE4StageSelectionReceipt,
    reduce_e4_profiler_completion_from_registry,
    reduce_e4_stage_selection_from_proofs,
    require_e4_profiler_completion_authority,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    FormalRegistryVerificationReceipt,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    FormalStageCoverageEvidenceCell,
    FormalStageCoverageEvidenceShard,
    FormalStageCoverageProofArtifact,
    rebuild_formal_stage_coverage_context,
)
from lightcone_spec.experiments.formal_stage_execution import (
    E1RecipeAnchorAuthority,
    E4LocalStageSourceRebuildInputs,
    E4ProfilerStageSourceRebuildInputs,
    E4ScreenStageSourceRebuildInputs,
    FormalServingExecutionRebuildInput,
    VerifiedFormalServingExecutionBinding,
    VerifiedFormalStageMaterializationSource,
    load_e1_recipe_anchor_authority_artifact,
)
from lightcone_spec.experiments.stage_decisions import (
    E1SurvivorSelectionReceipt,
    SignedE1SurvivorSelectionReceipt,
    reduce_e1_survivor_selection_receipt,
)
from lightcone_spec.experiments.stage_materialization import (
    E1Geometry,
    E2CandidateRecipe,
    GpuHourEstimate,
    StageCoverageReceipt,
    StageMaterializationReceipt,
    default_e2_recipe_grid_authority,
    materialize_e2_next_round,
    materialize_e2_round,
    materialize_e4_profiler,
    materialize_e4_strength2_screen,
    materialize_e4_winner_neighborhood,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_STAGE_EXECUTION_REBUILD_SHARD_KIND = (
    "lightcone_formal_stage_execution_rebuild_shard"
)
FORMAL_STAGE_PREFIX_ARTIFACT_KIND = "lightcone_formal_stage_prefix_artifact"

FormalStagePrefixPhase = Literal[
    "e1_selection",
    "e2_round0",
    "e2_round1",
    "e2_round2",
    "e2_round3",
    "e4_screen",
    "e4_local",
    "e4_profiler",
]

FORMAL_STAGE_PREFIX_ORDER: tuple[FormalStagePrefixPhase, ...] = (
    "e1_selection",
    "e2_round0",
    "e2_round1",
    "e2_round2",
    "e2_round3",
    "e4_screen",
    "e4_local",
    "e4_profiler",
)

_PHASE_STAGE = {
    "e1_selection": "E1",
    "e2_round0": "E2",
    "e2_round1": "E2",
    "e2_round2": "E2",
    "e2_round3": "E2",
    "e4_screen": "E4",
    "e4_local": "E4",
    "e4_profiler": "E4",
}
_E4_RULE = {
    "e4_screen": "strength2_8_rows_x_3_loads_x_2_traffic",
    "e4_local": "winner_neighborhood_2pow4_x_3_loads_x_2_traffic",
    "e4_profiler": "three_profiler_only_rows_separate_from_headline",
}


def _strict(label: str, value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _array(label: str, value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _phase_round(phase: FormalStagePrefixPhase) -> int | None:
    return int(phase.removeprefix("e2_round")) if phase.startswith("e2_round") else None


def _materialization_phase(
    materialization: StageMaterializationReceipt,
) -> FormalStagePrefixPhase:
    if materialization.stage == "E1":
        if materialization.expected_cell_count != 68:
            raise ValueError("formal E1 prefix is not the exact 68-cell slice")
        return "e1_selection"
    if materialization.stage == "E2":
        rounds = {dict(cell.dimensions).get("round") for cell in materialization.cells}
        if len(rounds) != 1:
            raise ValueError("formal E2 prefix does not contain one exact round")
        round_index = next(iter(rounds))
        if type(round_index) is not int or round_index not in range(4):
            raise ValueError("formal E2 prefix round is outside [0,4)")
        return FORMAL_STAGE_PREFIX_ORDER[1 + round_index]
    if materialization.stage == "E4":
        for phase, rule in _E4_RULE.items():
            if materialization.materialization_rule == rule:
                return phase  # type: ignore[return-value]
    raise ValueError("formal stage prefix materialization phase is unsupported")


@dataclass(frozen=True)
class FormalStageExecutionRebuildShard:
    """Bounded public execution descriptors for one sequential prefix node."""

    schema_version: Literal[1]
    kind: Literal["lightcone_formal_stage_execution_rebuild_shard"]
    phase: FormalStagePrefixPhase
    materialization_receipt_sha256: str
    stage_source_rebuild_input_sha256: str | None
    descriptors: tuple[FormalServingExecutionRebuildInput, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_STAGE_EXECUTION_REBUILD_SHARD_KIND
        ):
            raise ValueError("formal stage execution shard schema is unsupported")
        if self.phase not in FORMAL_STAGE_PREFIX_ORDER[:-1]:
            raise ValueError("formal profiler prefix cannot carry serving descriptors")
        _sha256(
            "formal stage execution shard materialization",
            self.materialization_receipt_sha256,
        )
        requires_source = self.phase in {"e4_screen", "e4_local"}
        if requires_source:
            _sha256(
                "formal stage execution shard stage source",
                self.stage_source_rebuild_input_sha256,
            )
        elif self.stage_source_rebuild_input_sha256 is not None:
            raise ValueError("formal E1/E2 execution shard carries a stage source")
        cell_ids = tuple(row.subject.materialized_cell_id for row in self.descriptors)
        if (
            type(self.descriptors) is not tuple
            or not self.descriptors
            or any(
                type(row) is not FormalServingExecutionRebuildInput
                for row in self.descriptors
            )
            or cell_ids != tuple(sorted(set(cell_ids)))
            or any(
                row.subject.stage != _PHASE_STAGE[self.phase]
                or row.subject.materialization_receipt_sha256
                != self.materialization_receipt_sha256
                or row.stage_source_binding_sha256
                != (
                    None
                    if not requires_source
                    else row.subject.stage_source_binding_sha256
                )
                for row in self.descriptors
            )
            or len({row.execution_binding_sha256 for row in self.descriptors})
            != len(self.descriptors)
        ):
            raise ValueError("formal stage execution shard descriptors are not exact")
        if requires_source and any(
            row.stage_source_binding_sha256 is None for row in self.descriptors
        ):
            raise ValueError("formal E4 execution shard lacks its stage-source binding")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "phase": self.phase,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "stage_source_rebuild_input_sha256": (
                self.stage_source_rebuild_input_sha256
            ),
            "descriptors": [row.to_dict() for row in self.descriptors],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage execution shard",
            value,
            {*cls.__dataclass_fields__, "shard_sha256"},
        )
        declared = _sha256("formal stage execution shard", row.pop("shard_sha256"))
        row["descriptors"] = tuple(
            FormalServingExecutionRebuildInput.from_dict(item)
            for item in _array("formal stage execution descriptors", row["descriptors"])
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("formal stage execution shard digest differs")
        return shard


def publish_formal_stage_execution_rebuild_shard(
    shard: FormalStageExecutionRebuildShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not FormalStageExecutionRebuildShard:
        raise TypeError("formal stage execution shard publisher requires exact input")
    shard.__post_init__()
    publish_canonical_json_no_replace(output_path, shard.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


@dataclass(frozen=True)
class FormalStagePrefixArtifact:
    """One immutable node in the E1 -> E2 -> E4 public proof chain."""

    schema_version: Literal[2]
    kind: Literal["lightcone_formal_stage_prefix_artifact"]
    phase: FormalStagePrefixPhase
    registry_verification_receipt_source: CanonicalJsonProofBinding
    coverage_proof_source: CanonicalJsonProofBinding
    e1_recipe_anchor_authority_source: CanonicalJsonProofBinding | None
    prior_prefix_source: CanonicalJsonProofBinding | None

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.kind != FORMAL_STAGE_PREFIX_ARTIFACT_KIND:
            raise ValueError("formal stage prefix artifact schema is unsupported")
        if self.phase not in FORMAL_STAGE_PREFIX_ORDER:
            raise ValueError("formal stage prefix phase is unsupported")
        for value in (
            self.registry_verification_receipt_source,
            self.coverage_proof_source,
        ):
            if type(value) is not CanonicalJsonProofBinding:
                raise TypeError("formal stage prefix core sources are not path-bound")
        root = self.phase == "e1_selection"
        if root != (
            type(self.e1_recipe_anchor_authority_source) is CanonicalJsonProofBinding
        ):
            raise ValueError("formal E1 prefix recipe-anchor source coverage differs")
        if root != (self.prior_prefix_source is None):
            raise ValueError("formal stage prefix predecessor coverage differs")
        if not root and type(self.prior_prefix_source) is not CanonicalJsonProofBinding:
            raise TypeError("formal stage prefix predecessor is not path-bound")
        paths = tuple(
            row.absolute_path
            for row in (
                self.registry_verification_receipt_source,
                self.coverage_proof_source,
                *(
                    ()
                    if self.e1_recipe_anchor_authority_source is None
                    else (self.e1_recipe_anchor_authority_source,)
                ),
                *(
                    ()
                    if self.prior_prefix_source is None
                    else (self.prior_prefix_source,)
                ),
            )
        )
        if len(paths) != len(set(paths)):
            raise ValueError("formal stage prefix reuses a source path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        def optional(value: CanonicalJsonProofBinding | None) -> object:
            return None if value is None else value.to_dict()

        value = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "phase": self.phase,
            "registry_verification_receipt_source": (
                self.registry_verification_receipt_source.to_dict()
            ),
            "coverage_proof_source": self.coverage_proof_source.to_dict(),
            "e1_recipe_anchor_authority_source": optional(
                self.e1_recipe_anchor_authority_source
            ),
            "prior_prefix_source": optional(self.prior_prefix_source),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal stage prefix artifact",
            value,
            {*cls.__dataclass_fields__, "artifact_sha256"},
        )
        declared = _sha256("formal stage prefix artifact", row.pop("artifact_sha256"))
        for name in (
            "registry_verification_receipt_source",
            "coverage_proof_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "e1_recipe_anchor_authority_source",
            "prior_prefix_source",
        ):
            row[name] = (
                None
                if row[name] is None
                else CanonicalJsonProofBinding.from_dict(row[name])
            )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal stage prefix artifact digest differs")
        return artifact


def bind_formal_stage_prefix_artifact(
    *,
    phase: FormalStagePrefixPhase,
    registry_verification_receipt_path: str | Path,
    formal_runtime_authority_manifest_path: str | Path | None,
    inventory_path: str | Path | None,
    materialization_path: str | Path | None,
    coverage_path: str | Path | None,
    coverage_proof_path: str | Path,
    stage_source_rebuild_path: str | Path | None,
    execution_rebuild_shard_paths: tuple[str | Path, ...],
    e1_recipe_anchor_authority_path: str | Path | None,
    prior_prefix_path: str | Path | None,
) -> FormalStagePrefixArtifact:
    """Bind direct operator paths into one strict, append-only prefix node."""

    if phase == "e4_profiler":
        require_e4_profiler_completion_authority()

    def optional(value: str | Path | None) -> CanonicalJsonProofBinding | None:
        return None if value is None else CanonicalJsonProofBinding.bind(value)

    coverage_proof_binding = CanonicalJsonProofBinding.bind(coverage_proof_path)
    coverage_proof = FormalStageCoverageProofArtifact.from_dict(
        coverage_proof_binding.reopen()
    )
    if (coverage_proof.stage, coverage_proof.phase) != (
        _PHASE_STAGE[phase],
        {
            "e1_selection": "selection",
            "e2_round0": "round0",
            "e2_round1": "round1",
            "e2_round2": "round2",
            "e2_round3": "round3",
            "e4_screen": "screen",
            "e4_local": "local",
            "e4_profiler": "profiler",
        }[phase],
    ):
        raise ValueError("formal stage prefix coverage proof phase differs")
    for label, supplied, expected in (
        (
            "runtime authority",
            formal_runtime_authority_manifest_path,
            coverage_proof.runtime_authority_source,
        ),
        ("inventory", inventory_path, coverage_proof.inventory_source),
        (
            "materialization",
            materialization_path,
            coverage_proof.materialization_source,
        ),
        (
            "stage source rebuild",
            stage_source_rebuild_path,
            coverage_proof.stage_source_rebuild_input_source,
        ),
    ):
        if (
            supplied is not None
            and CanonicalJsonProofBinding.bind(supplied) != expected
        ):
            raise ValueError(f"formal stage prefix {label} differs from coverage proof")
    if execution_rebuild_shard_paths:
        supplied_execution = tuple(
            CanonicalJsonProofBinding.bind(path)
            for path in execution_rebuild_shard_paths
        )
        if supplied_execution != coverage_proof.execution_rebuild_shard_sources:
            raise ValueError(
                "formal stage prefix execution shards differ from coverage proof"
            )
    if coverage_path is not None:
        from lightcone_spec.experiments.formal_registry import (
            stage_coverage_receipt_from_dict,
        )

        declared_coverage = stage_coverage_receipt_from_dict(
            CanonicalJsonProofBinding.bind(coverage_path).reopen()
        )
        if declared_coverage.sha256 != coverage_proof.coverage_receipt_sha256:
            raise ValueError("formal stage prefix coverage differs from proof")
    artifact = FormalStagePrefixArtifact(
        schema_version=2,
        kind=FORMAL_STAGE_PREFIX_ARTIFACT_KIND,
        phase=phase,
        registry_verification_receipt_source=CanonicalJsonProofBinding.bind(
            registry_verification_receipt_path
        ),
        coverage_proof_source=coverage_proof_binding,
        e1_recipe_anchor_authority_source=optional(e1_recipe_anchor_authority_path),
        prior_prefix_source=optional(prior_prefix_path),
    )
    artifact.__post_init__()
    return artifact


def publish_formal_stage_prefix_artifact(
    artifact: FormalStagePrefixArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalStagePrefixArtifact:
        raise TypeError("formal stage prefix publisher requires an exact artifact")
    artifact.__post_init__()
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


@dataclass(frozen=True)
class RebuiltFormalStagePrefix:
    artifact_binding: CanonicalJsonProofBinding
    artifact: FormalStagePrefixArtifact
    registry_verification_receipt: FormalRegistryVerificationReceipt
    materialization: StageMaterializationReceipt
    coverage: StageCoverageReceipt
    evidence_manifest: (
        E1StagedParetoEvidenceManifest
        | E2StagedRoundEvidenceManifest
        | E4StagedEvidenceManifest
        | None
    )
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    stage_source: VerifiedFormalStageMaterializationSource | None
    source_recipes: tuple[E2CandidateRecipe, ...]
    e1_recipe_anchor_authority: E1RecipeAnchorAuthority
    prior: RebuiltFormalStagePrefix | None


def _reopen_typed_source(binding, *, label: str, decoder):
    before = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if before != binding:
        raise ValueError(f"{label} path identity changed")
    value = decoder(before.reopen())
    if CanonicalJsonProofBinding.bind(binding.absolute_path) != before:
        raise RuntimeError(f"{label} changed while reopened")
    return value


def _e2_source_recipes(
    materialization: StageMaterializationReceipt,
) -> tuple[E2CandidateRecipe, ...]:
    grid = default_e2_recipe_grid_authority()
    recipes: dict[str, E2CandidateRecipe] = {}
    for cell in materialization.cells:
        if cell.method_role != "LightCone-candidate":
            continue
        dimensions = dict(cell.dimensions)
        rank = dimensions.get("rank")
        alpha = dimensions.get("alpha_over_rank")
        geometry = E1Geometry(
            scope=str(dimensions.get("scope")),
            parameterization=str(dimensions.get("parameterization")),  # type: ignore[arg-type]
            rank=None if rank == "none" else int(rank),  # type: ignore[arg-type]
            alpha_over_rank=None if alpha == "none" else float(alpha),  # type: ignore[arg-type]
        )
        recipe = E2CandidateRecipe(
            geometry=geometry,
            optimizer=str(dimensions.get("optimizer")),
            schedule=str(dimensions.get("schedule")),
            learning_rate=float(dimensions.get("learning_rate")),  # type: ignore[arg-type]
            optimizer_recipe_authority_sha256=(grid.optimizer_recipe_authority.sha256),
        )
        if recipe.sha256 != cell.recipe_sha256:
            raise ValueError("formal E2 prefix cell differs from its typed recipe")
        recipes[recipe.sha256] = recipe
    result = tuple(recipes[digest] for digest in sorted(recipes))
    if len(result) != sum(
        cell.method_role == "LightCone-candidate" for cell in materialization.cells
    ):
        raise ValueError("formal E2 prefix source recipe universe is not exact")
    return result


def _one(rows, *, label: str, predicate):
    selected = tuple(row for row in rows if predicate(row))
    if len(selected) != 1:
        raise ValueError(f"formal stage prefix requires one exact {label}")
    return selected[0]


def _registered_current_materialization(
    receipt: FormalRegistryVerificationReceipt,
    materialization: StageMaterializationReceipt,
) -> None:
    rows = tuple(
        row
        for row in receipt.cumulative_signed_materializations
        if row.payload == materialization
    )
    coverage = tuple(
        row
        for row in receipt.cumulative_signed_coverage
        if row.payload.materialization_receipt_sha256 == materialization.sha256
    )
    if len(rows) != 1 or coverage:
        raise ValueError(
            "formal stage prefix registry must contain current materialization "
            "but no future coverage"
        )


def _registered_completed_predecessor(
    receipt: FormalRegistryVerificationReceipt,
    prior: RebuiltFormalStagePrefix,
) -> None:
    materializations = tuple(
        row
        for row in receipt.cumulative_signed_materializations
        if row.payload == prior.materialization
    )
    coverages = tuple(
        row
        for row in receipt.cumulative_signed_coverage
        if row.payload == prior.coverage
    )
    if len(materializations) != 1 or len(coverages) != 1:
        raise ValueError("formal stage prefix registry lacks its completed predecessor")
    phase = prior.artifact.phase
    if phase == "e1_selection":
        rows = tuple(
            row
            for row in receipt.cumulative_signed_e1_survivor_selections
            if row.payload.materialization_receipt_sha256
            == prior.materialization.sha256
            and row.payload.coverage_receipt_sha256 == prior.coverage.sha256
        )
    elif phase.startswith("e2_round"):
        rows = tuple(
            row
            for row in receipt.cumulative_signed_e2_staged_selections
            if row.payload.materialization_receipt_sha256
            == prior.materialization.sha256
            and row.payload.coverage_receipt_sha256 == prior.coverage.sha256
        )
    elif phase in {"e4_screen", "e4_local"}:
        rows = tuple(
            row
            for row in receipt.cumulative_signed_e4_stage_selections
            if row.payload.materialization_receipt_sha256
            == prior.materialization.sha256
            and row.payload.coverage_receipt_sha256 == prior.coverage.sha256
        )
    else:
        rows = tuple(
            row
            for row in receipt.cumulative_formal_stage_prefix_artifacts
            if row == prior.artifact_binding
        )
        if any(row.status != "COMPLETE" for row in prior.coverage.dispositions):
            raise ValueError("formal profiler predecessor is not terminal-complete")
    if len(rows) != 1:
        raise ValueError("formal stage prefix registry lacks predecessor reduction")


def _validate_evidence_bindings(
    manifest: object,
    bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
) -> None:
    evidence = {row.materialized_cell_id: row for row in manifest.cells}  # type: ignore[attr-defined]
    rebuilt = {row.subject.materialized_cell_id: row for row in bindings}
    if len(evidence) != len(manifest.cells) or set(evidence) != set(rebuilt):  # type: ignore[attr-defined]
        raise ValueError("formal stage evidence/binding coverage differs")
    for cell_id, row in evidence.items():
        binding = rebuilt[cell_id]
        if (
            row.execution_binding_sha256 != binding.sha256
            or row.execution_identity != binding.subject.execution_identity
        ):
            raise ValueError("formal stage evidence names a foreign binding")


def _coverage_evidence_cells(
    proof: FormalStageCoverageProofArtifact,
) -> tuple[FormalStageCoverageEvidenceCell, ...]:
    shards = tuple(
        _reopen_typed_source(
            binding,
            label="formal stage coverage evidence shard",
            decoder=FormalStageCoverageEvidenceShard.from_dict,
        )
        for binding in proof.evidence_shard_sources
    )
    if (
        tuple(row.shard_index for row in shards) != tuple(range(len(shards)))
        or any(row.shard_count != len(shards) for row in shards)
        or any(
            (
                row.protocol_lock_sha256,
                row.materialization_receipt_sha256,
                row.inventory_sha256,
                row.stage,
                row.phase,
            )
            != (
                proof.protocol_lock_sha256,
                proof.materialization_receipt_sha256,
                proof.inventory_sha256,
                proof.stage,
                proof.phase,
            )
            for row in shards
        )
    ):
        raise ValueError("formal stage coverage evidence shard graph differs")
    cells = tuple(cell for shard in shards for cell in shard.cells)
    ids = tuple(row.materialized_cell_id for row in cells)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("formal stage coverage evidence union is not canonical")
    return cells


def _proof_derived_reducer_manifest(
    *,
    phase: FormalStagePrefixPhase,
    proof: FormalStageCoverageProofArtifact,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
) -> (
    E1StagedParetoEvidenceManifest
    | E2StagedRoundEvidenceManifest
    | E4StagedEvidenceManifest
    | None
):
    if phase == "e4_profiler":
        return None
    evidence = _coverage_evidence_cells(proof)
    binding_by_cell = {row.subject.materialized_cell_id: row for row in bindings}
    if set(binding_by_cell) != {row.materialized_cell_id for row in evidence}:
        raise ValueError("formal reducer manifest binding coverage differs")

    def paths(row: FormalStageCoverageEvidenceCell) -> tuple[str, str]:
        return (
            row.native_result_proof.absolute_path,
            row.stage_itl_proof.absolute_path,
        )

    if phase == "e1_selection":
        cells = tuple(
            E1CellExecutionEvidence.bind(
                execution_binding=binding_by_cell[row.materialized_cell_id],
                native_result_proof_path=paths(row)[0],
                stage_itl_proof_path=paths(row)[1],
            )
            for row in evidence
        )
        return E1StagedParetoEvidenceManifest(
            schema_version=1,
            protocol_lock_sha256=proof.protocol_lock_sha256,
            materialization_receipt_sha256=materialization.sha256,
            coverage_receipt_sha256=coverage.sha256,
            e3a_selection_receipt_sha256=materialization.source_decision_sha256,
            inventory_sha256=proof.inventory_sha256,
            cells=cells,
        )
    if phase.startswith("e2_round"):
        cells = tuple(
            E2CellExecutionEvidence.bind(
                execution_binding=binding_by_cell[row.materialized_cell_id],
                native_result_proof_path=paths(row)[0],
                stage_itl_proof_path=paths(row)[1],
            )
            for row in evidence
        )
        round_index = _phase_round(phase)
        assert round_index is not None
        return E2StagedRoundEvidenceManifest(
            schema_version=1,
            protocol_lock_sha256=proof.protocol_lock_sha256,
            materialization_receipt_sha256=materialization.sha256,
            coverage_receipt_sha256=coverage.sha256,
            source_selection_sha256=materialization.source_decision_sha256,
            inventory_sha256=proof.inventory_sha256,
            round_index=round_index,
            cells=cells,
        )
    cells = tuple(
        E4CellExecutionEvidence.bind(
            execution_binding=binding_by_cell[row.materialized_cell_id],
            native_result_proof_path=paths(row)[0],
            stage_itl_proof_path=paths(row)[1],
        )
        for row in evidence
    )
    return E4StagedEvidenceManifest(
        schema_version=1,
        phase="screen" if phase == "e4_screen" else "local",
        protocol_lock_sha256=proof.protocol_lock_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        upstream_signed_authority_sha256=(materialization.source_decision_sha256),
        inventory_sha256=proof.inventory_sha256,
        cells=cells,
    )


def _load_prefix_binding(
    binding: CanonicalJsonProofBinding,
    *,
    now_ns: int,
    seen_paths: frozenset[str],
) -> RebuiltFormalStagePrefix:
    if binding.absolute_path in seen_paths:
        raise ValueError("formal stage prefix chain contains a cycle")
    artifact = _reopen_typed_source(
        binding,
        label="formal stage prefix artifact",
        decoder=FormalStagePrefixArtifact.from_dict,
    )
    if artifact.phase == "e4_profiler":
        require_e4_profiler_completion_authority()
    # Import lazily because registry layers rebuild their current prefix proofs.
    # The binding is checked before and after the recursive load so replacing a
    # layer in place cannot change the registry observed by this prefix.
    from lightcone_spec.experiments.formal_registry_layers import (
        load_formal_registry_verification_receipt_path,
    )

    registry_source = artifact.registry_verification_receipt_source
    if CanonicalJsonProofBinding.bind(registry_source.absolute_path) != registry_source:
        raise ValueError("formal stage prefix registry path identity changed")
    receipt = load_formal_registry_verification_receipt_path(
        registry_source.absolute_path,
        now_ns=now_ns,
    )
    if CanonicalJsonProofBinding.bind(registry_source.absolute_path) != registry_source:
        raise RuntimeError("formal stage prefix registry changed while rebuilt")
    lock = receipt.signed_protocol_lock.payload
    coverage_proof = _reopen_typed_source(
        artifact.coverage_proof_source,
        label="formal stage coverage proof artifact",
        decoder=FormalStageCoverageProofArtifact.from_dict,
    )
    expected_coverage_phase = {
        "e1_selection": ("E1", "selection"),
        "e2_round0": ("E2", "round0"),
        "e2_round1": ("E2", "round1"),
        "e2_round2": ("E2", "round2"),
        "e2_round3": ("E2", "round3"),
        "e4_screen": ("E4", "screen"),
        "e4_local": ("E4", "local"),
        "e4_profiler": ("E4", "profiler"),
    }[artifact.phase]
    if (coverage_proof.stage, coverage_proof.phase) != expected_coverage_phase:
        raise ValueError("formal stage prefix coverage proof phase differs")

    prior = (
        None
        if artifact.prior_prefix_source is None
        else _load_prefix_binding(
            artifact.prior_prefix_source,
            now_ns=now_ns,
            seen_paths=seen_paths | {binding.absolute_path},
        )
    )
    index = FORMAL_STAGE_PREFIX_ORDER.index(artifact.phase)
    expected_prior = None if index == 0 else FORMAL_STAGE_PREFIX_ORDER[index - 1]
    if (None if prior is None else prior.artifact.phase) != expected_prior:
        raise ValueError("formal stage prefix predecessor order differs")
    if prior is not None:
        prior_lock = prior.registry_verification_receipt.signed_protocol_lock.payload
        if (
            prior_lock != lock
            or prior.registry_verification_receipt.inventory_sha256
            != receipt.inventory_sha256
        ):
            raise ValueError("formal stage prefix predecessor changes immutable root")
        _registered_completed_predecessor(receipt, prior)

    stage_source_inputs = None
    if artifact.phase == "e4_screen":
        assert prior is not None
        signed = _one(
            receipt.cumulative_signed_e2_staged_selections,
            label="E2 final selection",
            predicate=lambda row: (
                row.payload.round_index == 3
                and row.payload.materialization_receipt_sha256
                == prior.materialization.sha256
            ),
        )
        assert type(prior.evidence_manifest) is E2StagedRoundEvidenceManifest
        stage_source_inputs = E4ScreenStageSourceRebuildInputs(
            registry_verification_receipt=receipt,
            signed_e2_final_selection=signed,
            e2_materialization=prior.materialization,
            e2_coverage=prior.coverage,
            e2_source_recipes=prior.source_recipes,
            e2_evidence_manifest=prior.evidence_manifest,
            e2_execution_bindings=prior.execution_bindings,
        )
    elif artifact.phase == "e4_local":
        assert prior is not None
        signed = _one(
            receipt.cumulative_signed_e4_stage_selections,
            label="E4 screen selection",
            predicate=lambda row: (
                row.payload.phase == "screen"
                and row.payload.materialization_receipt_sha256
                == prior.materialization.sha256
            ),
        )
        assert type(prior.evidence_manifest) is E4StagedEvidenceManifest
        stage_source_inputs = E4LocalStageSourceRebuildInputs(
            registry_verification_receipt=receipt,
            signed_e4_screen_selection=signed,
            screen_materialization=prior.materialization,
            screen_coverage=prior.coverage,
            screen_evidence_manifest=prior.evidence_manifest,
            screen_execution_bindings=prior.execution_bindings,
        )
    elif artifact.phase == "e4_profiler":
        assert prior is not None
        signed = _one(
            receipt.cumulative_signed_e4_stage_selections,
            label="E4 local selection",
            predicate=lambda row: (
                row.payload.phase == "local"
                and row.payload.materialization_receipt_sha256
                == prior.materialization.sha256
            ),
        )
        assert type(prior.evidence_manifest) is E4StagedEvidenceManifest
        stage_source_inputs = E4ProfilerStageSourceRebuildInputs(
            registry_verification_receipt=receipt,
            signed_e4_final_selection=signed,
            local_materialization=prior.materialization,
            local_coverage=prior.coverage,
            local_evidence_manifest=prior.evidence_manifest,
            local_execution_bindings=prior.execution_bindings,
        )

    tts_rows = receipt.cumulative_tts_calibration_authorities
    signed_tts_rows = receipt.cumulative_signed_tts_calibration_seals
    if len(tts_rows) != 1 or len(signed_tts_rows) != 1:
        raise ValueError("formal stage prefix lacks one frozen TTS authority")
    tts = tts_rows[0]
    signed_tts = signed_tts_rows[0]
    grid = default_e2_recipe_grid_authority()
    if (
        tts.sha256 != lock.tts_calibration_authority_sha256
        or grid.sha256 != lock.e2_recipe_grid_authority_sha256
    ):
        raise ValueError("formal stage prefix recipe authorities differ")
    if artifact.e1_recipe_anchor_authority_source is not None:
        anchor_binding = artifact.e1_recipe_anchor_authority_source
        if (
            CanonicalJsonProofBinding.bind(anchor_binding.absolute_path)
            != anchor_binding
        ):
            raise ValueError("formal E1 recipe-anchor artifact path changed")
        anchor_artifact = load_e1_recipe_anchor_authority_artifact(
            anchor_binding.absolute_path
        )
        if (
            CanonicalJsonProofBinding.bind(anchor_binding.absolute_path)
            != anchor_binding
            or anchor_artifact.sha256 != anchor_binding.semantic_sha256
            or anchor_artifact.authority.sha256
            != lock.e1_recipe_anchor_authority_sha256
        ):
            raise ValueError("formal E1 recipe-anchor authority differs")
        e1_anchor = anchor_artifact.authority
    else:
        assert prior is not None
        e1_anchor = prior.e1_recipe_anchor_authority
    final_recipes = tuple(
        row.payload.final_recipe
        for row in receipt.cumulative_signed_e2_staged_selections
        if row.payload.round_index == 3 and row.payload.final_recipe is not None
    )
    lightcone = final_recipes[0] if len(final_recipes) == 1 else None
    context = rebuild_formal_stage_coverage_context(
        artifact.coverage_proof_source.absolute_path,
        now_ns=now_ns,
        tts_authority=tts,
        signed_tts_seal=signed_tts,
        e1_recipe_anchor_authority=e1_anchor,
        e2_recipe_grid_authority=grid,
        lightcone_recipe=lightcone,
        registry_verification_receipt=receipt,
        stage_source_inputs=stage_source_inputs,
    )
    materialization = context.materialization
    coverage = context.coverage
    bindings = context.execution_bindings
    stage_source = context.stage_source
    if (
        coverage_proof.coverage_receipt_sha256 != coverage.sha256
        or _materialization_phase(materialization) != artifact.phase
        or materialization.protocol_lock_sha256 != lock.sha256
    ):
        raise ValueError("formal stage prefix proof-derived lineage differs")
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("formal stage prefix coverage is not all COMPLETE")
    _registered_current_materialization(receipt, materialization)
    evidence = _proof_derived_reducer_manifest(
        phase=artifact.phase,
        proof=coverage_proof,
        materialization=materialization,
        coverage=coverage,
        bindings=bindings,
    )
    if evidence is not None:
        _validate_evidence_bindings(evidence, bindings)
    source_recipes = (
        _e2_source_recipes(materialization)
        if artifact.phase.startswith("e2_round")
        else ()
    )

    return RebuiltFormalStagePrefix(
        artifact_binding=binding,
        artifact=artifact,
        registry_verification_receipt=receipt,
        materialization=materialization,
        coverage=coverage,
        evidence_manifest=evidence,
        execution_bindings=bindings,
        stage_source=stage_source,
        source_recipes=source_recipes,
        e1_recipe_anchor_authority=e1_anchor,
        prior=prior,
    )


def load_and_rebuild_formal_stage_prefix(
    artifact_path: str | Path,
    *,
    now_ns: int,
) -> RebuiltFormalStagePrefix:
    """Deep-open one current-only prefix and recursively rebuild its proofs."""

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal stage prefix verification time is invalid")
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    return _load_prefix_binding(binding, now_ns=now_ns, seen_paths=frozenset())


def materialize_next_formal_stage_from_prefix(
    prefix: RebuiltFormalStagePrefix,
    *,
    registry_verification_receipt: FormalRegistryVerificationReceipt,
    now_ns: int,
) -> StageMaterializationReceipt:
    """Materialize exactly the next node from a completed public prefix."""

    if type(prefix) is not RebuiltFormalStagePrefix:
        raise TypeError("formal next-stage materializer requires a rebuilt prefix")
    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("formal next-stage materializer requires durable registry")
    registry_verification_receipt.revalidate(current_ns=now_ns)
    phase = prefix.artifact.phase
    lock = registry_verification_receipt.signed_protocol_lock.payload
    if lock != prefix.registry_verification_receipt.signed_protocol_lock.payload:
        raise ValueError("formal next-stage registry changes ProtocolLock")
    if phase == "e1_selection":
        signed = _one(
            registry_verification_receipt.cumulative_signed_e1_survivor_selections,
            label="signed E1 survivor selection",
            predicate=lambda row: (
                row.payload.materialization_receipt_sha256
                == prefix.materialization.sha256
            ),
        )
        assert type(prefix.evidence_manifest) is E1StagedParetoEvidenceManifest
        return materialize_e2_round(
            registry_verification_receipt=registry_verification_receipt,
            protocol_lock=lock,
            signed_e1_selection=signed,
            e1_materialization=prefix.materialization,
            e1_coverage=prefix.coverage,
            pareto_evidence_manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
            gpu_hours=GpuHourEstimate.unmeasured(),
        )
    if phase in {"e2_round0", "e2_round1", "e2_round2"}:
        round_index = _phase_round(phase)
        assert round_index is not None
        signed = _one(
            registry_verification_receipt.cumulative_signed_e2_staged_selections,
            label=f"signed E2 round {round_index} selection",
            predicate=lambda row: (
                row.payload.round_index == round_index
                and row.payload.materialization_receipt_sha256
                == prefix.materialization.sha256
            ),
        )
        assert type(prefix.evidence_manifest) is E2StagedRoundEvidenceManifest
        return materialize_e2_next_round(
            registry_verification_receipt=registry_verification_receipt,
            protocol_lock=lock,
            signed_prior_selection=signed,
            prior_materialization=prefix.materialization,
            prior_coverage=prefix.coverage,
            source_recipes=prefix.source_recipes,
            evidence_manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
            gpu_hours=GpuHourEstimate.unmeasured(),
        )
    if phase == "e2_round3":
        signed = _one(
            registry_verification_receipt.cumulative_signed_e2_staged_selections,
            label="signed E2 final selection",
            predicate=lambda row: (
                row.payload.round_index == 3
                and row.payload.materialization_receipt_sha256
                == prefix.materialization.sha256
            ),
        )
        assert type(prefix.evidence_manifest) is E2StagedRoundEvidenceManifest
        return materialize_e4_strength2_screen(
            registry_verification_receipt=registry_verification_receipt,
            signed_e2_final_selection=signed,
            e2_materialization=prefix.materialization,
            e2_coverage=prefix.coverage,
            e2_source_recipes=prefix.source_recipes,
            e2_evidence_manifest=prefix.evidence_manifest,
            e2_execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
        )
    if phase == "e4_screen":
        signed = _one(
            registry_verification_receipt.cumulative_signed_e4_stage_selections,
            label="signed E4 screen selection",
            predicate=lambda row: (
                row.payload.phase == "screen"
                and row.payload.materialization_receipt_sha256
                == prefix.materialization.sha256
            ),
        )
        assert type(prefix.evidence_manifest) is E4StagedEvidenceManifest
        return materialize_e4_winner_neighborhood(
            registry_verification_receipt=registry_verification_receipt,
            signed_e4_screen_selection=signed,
            screen_materialization=prefix.materialization,
            screen_coverage=prefix.coverage,
            screen_evidence_manifest=prefix.evidence_manifest,
            screen_execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
        )
    if phase == "e4_local":
        signed = _one(
            registry_verification_receipt.cumulative_signed_e4_stage_selections,
            label="signed E4 local selection",
            predicate=lambda row: (
                row.payload.phase == "local"
                and row.payload.materialization_receipt_sha256
                == prefix.materialization.sha256
            ),
        )
        assert type(prefix.evidence_manifest) is E4StagedEvidenceManifest
        return materialize_e4_profiler(
            registry_verification_receipt=registry_verification_receipt,
            signed_e4_final_selection=signed,
            local_materialization=prefix.materialization,
            local_coverage=prefix.coverage,
            local_evidence_manifest=prefix.evidence_manifest,
            local_execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
        )
    raise ValueError("formal stage prefix has no registered materialization successor")


def reduce_formal_stage_prefix(
    prefix: RebuiltFormalStagePrefix,
    *,
    now_ns: int,
) -> (
    E1SurvivorSelectionReceipt
    | E2StagedRoundSelectionReceipt
    | E4ProfilerCompletionReceipt
    | E4StageSelectionReceipt
):
    """Run the current E2/E4 headline reducer from proof-derived inputs."""

    if type(prefix) is not RebuiltFormalStagePrefix:
        raise TypeError("formal prefix reducer requires a rebuilt prefix")
    lock = prefix.registry_verification_receipt.signed_protocol_lock.payload
    phase = prefix.artifact.phase
    if phase == "e1_selection":
        assert type(prefix.evidence_manifest) is E1StagedParetoEvidenceManifest
        return reduce_e1_survivor_selection_receipt(
            registry_verification_receipt=prefix.registry_verification_receipt,
            protocol_lock=lock,
            e1_materialization=prefix.materialization,
            e1_coverage=prefix.coverage,
            pareto_evidence_manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
        )
    if phase.startswith("e2_round"):
        assert type(prefix.evidence_manifest) is E2StagedRoundEvidenceManifest
        return reduce_e2_staged_round_selection_from_proofs(
            protocol_lock=lock,
            materialization=prefix.materialization,
            coverage=prefix.coverage,
            source_selection_sha256=prefix.materialization.source_decision_sha256,
            source_recipes=prefix.source_recipes,
            manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
        )
    if phase in {"e4_screen", "e4_local"}:
        assert type(prefix.evidence_manifest) is E4StagedEvidenceManifest
        return reduce_e4_stage_selection_from_proofs(
            protocol_lock=lock,
            materialization=prefix.materialization,
            coverage=prefix.coverage,
            upstream_signed_authority_sha256=(
                prefix.materialization.source_decision_sha256
            ),
            manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            now_ns=now_ns,
        )
    if phase == "e4_profiler":
        require_e4_profiler_completion_authority()
        return reduce_e4_profiler_completion_from_registry(
            registry_verification_receipt=(prefix.registry_verification_receipt),
            materialization=prefix.materialization,
            now_ns=now_ns,
        )
    raise ValueError("formal prefix phase has no E2/E4 headline reducer")


def verify_signed_formal_stage_prefix_result(
    prefix: RebuiltFormalStagePrefix,
    signed: (
        SignedE1SurvivorSelectionReceipt
        | SignedE2StagedRoundSelectionReceipt
        | SignedE4StageSelectionReceipt
    ),
    *,
    now_ns: int,
) -> (
    E1SurvivorSelectionReceipt | E2StagedRoundSelectionReceipt | E4StageSelectionReceipt
):
    """Re-reduce and verify one externally signed current-prefix result."""

    expected = reduce_formal_stage_prefix(prefix, now_ns=now_ns)
    if signed.payload != expected:
        raise ValueError("signed formal prefix result differs from proof reducer")
    policy = prefix.registry_verification_receipt.trusted_release_policy(
        current_ns=now_ns
    )
    if type(signed) is SignedE1SurvivorSelectionReceipt:
        assert type(prefix.evidence_manifest) is E1StagedParetoEvidenceManifest
        artifacts = prefix.registry_verification_receipt.cumulative_e3a_staged_selection_artifacts
        selections = (
            prefix.registry_verification_receipt.cumulative_signed_e3a_staged_selections
        )
        if len(artifacts) != 1 or len(selections) != 1:
            raise ValueError("formal E1 prefix lacks its staged E3a authority")
        return signed.verify(
            protocol_lock=prefix.registry_verification_receipt.signed_protocol_lock.payload,
            e1_materialization=prefix.materialization,
            e1_coverage=prefix.coverage,
            e3a_selection_artifact=artifacts[0],
            signed_e3a_selection=selections[0],
            e3a_policy=policy,
            expected_e3a_policy_sha256=policy.sha256,
            pareto_evidence_manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    if type(signed) is SignedE2StagedRoundSelectionReceipt:
        assert type(prefix.evidence_manifest) is E2StagedRoundEvidenceManifest
        return signed.verify(
            protocol_lock=prefix.registry_verification_receipt.signed_protocol_lock.payload,
            materialization=prefix.materialization,
            coverage=prefix.coverage,
            source_recipes=prefix.source_recipes,
            manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    if type(signed) is SignedE4StageSelectionReceipt:
        assert type(prefix.evidence_manifest) is E4StagedEvidenceManifest
        return signed.verify(
            protocol_lock=prefix.registry_verification_receipt.signed_protocol_lock.payload,
            materialization=prefix.materialization,
            coverage=prefix.coverage,
            manifest=prefix.evidence_manifest,
            execution_bindings=prefix.execution_bindings,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    raise TypeError("signed formal prefix result wrapper is unsupported")


__all__ = [
    "FORMAL_STAGE_EXECUTION_REBUILD_SHARD_KIND",
    "FORMAL_STAGE_PREFIX_ARTIFACT_KIND",
    "FORMAL_STAGE_PREFIX_ORDER",
    "FormalStageExecutionRebuildShard",
    "FormalStagePrefixArtifact",
    "FormalStagePrefixPhase",
    "RebuiltFormalStagePrefix",
    "bind_formal_stage_prefix_artifact",
    "load_and_rebuild_formal_stage_prefix",
    "materialize_next_formal_stage_from_prefix",
    "publish_formal_stage_execution_rebuild_shard",
    "publish_formal_stage_prefix_artifact",
    "reduce_formal_stage_prefix",
    "verify_signed_formal_stage_prefix_result",
]
