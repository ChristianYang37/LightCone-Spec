"""Durable proof replay for the formal 360-row E3a selection.

The E3a receipt is a derived scientific result.  A structurally valid receipt
or a signature over one is not sufficient authority: the six locked outputs
must be reproduced from every terminal/ITL proof used by the proof-derived
coverage reducer.  This module stores only a bounded path graph and result
commitments; every load reopens the coverage graph, reconstructs the exact
360-row evidence manifest, and reruns the registered E3a reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.e3a_stage_authority import (
    E3aCellExecutionEvidence,
    E3aStagedEvidenceManifest,
    E3aStagedSelectionArtifact,
    E3aStagedSelectionReceipt,
    build_e3a_staged_selection_receipt,
    reduce_e3a_staged_selection_from_proofs,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry_layers import (
    load_formal_registry_verification_receipt_path,
    validate_formal_precoverage_registry_state,
)
from lightcone_spec.experiments.formal_stage_coverage import (
    FormalStageCoverageEvidenceShard,
    FormalStageCoverageProofArtifact,
    rebuild_formal_stage_coverage_context,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_E3A_STAGED_SELECTION_PROOF_KIND = "formal_e3a_staged_selection_proof_artifact"
FORMAL_E3A_STAGED_SELECTION_PROOF_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e3a_staged_selection_proof_protocol",
        "input": "proof_derived_exact_360_row_e3a_coverage_graph",
        "evidence": "coverage_evidence_shards_rebound_as_e3a_manifest",
        "reducer": "reduce_e3a_staged_selection_from_proofs",
        "output": "selection_artifact_and_signable_receipt_exact_commitments",
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


@dataclass(frozen=True)
class FormalE3aStagedSelectionProofArtifact:
    """Small path-bound commitment to one exact E3a reducer invocation."""

    schema_version: Literal[1]
    kind: Literal["formal_e3a_staged_selection_proof_artifact"]
    protocol_sha256: str
    coverage_proof_source: CanonicalJsonProofBinding
    registry_layer_source: CanonicalJsonProofBinding
    registry_verification_receipt_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    inventory_sha256: str
    evidence_manifest_sha256: str
    selection_artifact_sha256: str
    selection_receipt_sha256: str
    reduced_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != FORMAL_E3A_STAGED_SELECTION_PROOF_KIND
            or self.protocol_sha256 != FORMAL_E3A_STAGED_SELECTION_PROOF_PROTOCOL_SHA256
        ):
            raise ValueError("formal E3a selection proof identity differs")
        if (
            type(self.coverage_proof_source) is not CanonicalJsonProofBinding
            or type(self.registry_layer_source) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal E3a selection sources must be path-bound")
        if (
            self.coverage_proof_source.absolute_path
            == self.registry_layer_source.absolute_path
        ):
            raise ValueError("formal E3a selection proof aliases its registry")
        for label, digest in (
            ("registry receipt", self.registry_verification_receipt_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("selection artifact", self.selection_artifact_sha256),
            ("selection receipt", self.selection_receipt_sha256),
        ):
            _sha256(f"formal E3a selection proof {label}", digest)
        if type(self.reduced_ns) is not int or self.reduced_ns < 1:
            raise ValueError("formal E3a selection proof time must be positive")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "coverage_proof_source": self.coverage_proof_source.to_dict(),
            "registry_layer_source": self.registry_layer_source.to_dict(),
            "registry_verification_receipt_sha256": (
                self.registry_verification_receipt_sha256
            ),
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "selection_artifact_sha256": self.selection_artifact_sha256,
            "selection_receipt_sha256": self.selection_receipt_sha256,
            "reduced_ns": self.reduced_ns,
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal E3a selection proof fields differ")
        row = dict(value)
        declared = _sha256(
            "formal E3a selection proof artifact", row.pop("artifact_sha256")
        )
        row["coverage_proof_source"] = CanonicalJsonProofBinding.from_dict(
            row["coverage_proof_source"]
        )
        row["registry_layer_source"] = CanonicalJsonProofBinding.from_dict(
            row["registry_layer_source"]
        )
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal E3a selection proof digest differs")
        return artifact


def _coverage_evidence_rows(
    coverage_proof: FormalStageCoverageProofArtifact,
) -> tuple[E3aCellExecutionEvidence, ...]:
    shards = tuple(
        FormalStageCoverageEvidenceShard.from_dict(source.reopen())
        for source in coverage_proof.evidence_shard_sources
    )
    if (
        not shards
        or tuple(row.shard_index for row in shards) != tuple(range(len(shards)))
        or any(row.shard_count != len(shards) for row in shards)
    ):
        raise ValueError("formal E3a coverage evidence shard sequence differs")
    generic_rows = tuple(cell for shard in shards for cell in shard.cells)
    if len(generic_rows) != 360 or tuple(
        row.materialized_cell_id for row in generic_rows
    ) != tuple(sorted({row.materialized_cell_id for row in generic_rows})):
        raise ValueError("formal E3a coverage proof lacks exact 360 evidence rows")
    return tuple(
        E3aCellExecutionEvidence(
            schema_version=1,
            materialized_cell_id=row.materialized_cell_id,
            execution_binding_sha256=row.execution_binding_sha256,
            execution_identity=row.execution_identity,
            native_result_proof_path=row.native_result_proof.absolute_path,
            native_result_proof_raw_sha256=row.native_result_proof.raw_sha256,
            native_result_proof_semantic_sha256=(
                row.native_result_proof.semantic_sha256
            ),
            stage_itl_proof_path=row.stage_itl_proof.absolute_path,
            stage_itl_proof_raw_sha256=row.stage_itl_proof.raw_sha256,
            stage_itl_proof_semantic_sha256=row.stage_itl_proof.semantic_sha256,
        )
        for row in generic_rows
    )


def _reduce_from_coverage_source(
    source: CanonicalJsonProofBinding,
    registry_source: CanonicalJsonProofBinding,
    *,
    reduced_ns: int,
) -> tuple[
    E3aStagedSelectionArtifact,
    E3aStagedSelectionReceipt,
    E3aStagedEvidenceManifest,
]:
    observed = CanonicalJsonProofBinding.bind(source.absolute_path)
    if observed != source:
        raise ValueError("formal E3a coverage proof path identity changed")
    coverage_proof = FormalStageCoverageProofArtifact.from_dict(observed.reopen())
    if (coverage_proof.stage, coverage_proof.phase) != ("E3a", "capacity"):
        raise ValueError("formal E3a selection proof consumes another stage")
    context = rebuild_formal_stage_coverage_context(
        observed.absolute_path,
        now_ns=reduced_ns,
    )
    if (
        context.materialization.expected_cell_count != 360
        or context.coverage.stage != "E3a"
        or len(context.execution_bindings) != 360
    ):
        raise ValueError("formal E3a selection proof context is incomplete")
    registry = load_formal_registry_verification_receipt_path(
        registry_source.absolute_path,
        now_ns=reduced_ns,
    )
    validate_formal_precoverage_registry_state(
        registry,
        stage="E3a",
        phase="capacity",
        materialization=context.materialization,
    )
    if (
        registry.signed_protocol_lock.payload != context.protocol_lock
        or registry.inventory_sha256 != context.inventory.sha256
    ):
        raise ValueError("formal E3a selection registry authority differs")
    evidence = E3aStagedEvidenceManifest(
        schema_version=1,
        protocol_lock_sha256=context.protocol_lock.sha256,
        materialization_receipt_sha256=context.materialization.sha256,
        coverage_receipt_sha256=context.coverage.sha256,
        inventory_sha256=context.inventory.sha256,
        reducer_authority_member_sha256=context.formal_runtime_authority_manifest.member(
            "e3a_selection_reducer"
        ).sha256,
        cells=_coverage_evidence_rows(coverage_proof),
    )
    selection = reduce_e3a_staged_selection_from_proofs(
        protocol_lock=context.protocol_lock,
        formal_runtime_authority_manifest=(context.formal_runtime_authority_manifest),
        materialization=context.materialization,
        coverage=context.coverage,
        manifest=evidence,
        execution_bindings=context.execution_bindings,
        now_ns=reduced_ns,
    )
    receipt = build_e3a_staged_selection_receipt(selection)
    return selection, receipt, evidence


def bind_formal_e3a_staged_selection_proof_artifact(
    *,
    coverage_proof_path: str | Path,
    registry_layer_path: str | Path,
    now_ns: int,
) -> FormalE3aStagedSelectionProofArtifact:
    """Build a result commitment only after rerunning all 360 proof rows."""

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal E3a selection proof time must be positive")
    source = CanonicalJsonProofBinding.bind(coverage_proof_path)
    registry_source = CanonicalJsonProofBinding.bind(registry_layer_path)
    selection, receipt, evidence = _reduce_from_coverage_source(
        source,
        registry_source,
        reduced_ns=now_ns,
    )
    registry = load_formal_registry_verification_receipt_path(
        registry_source.absolute_path,
        now_ns=now_ns,
    )
    artifact = FormalE3aStagedSelectionProofArtifact(
        schema_version=1,
        kind=FORMAL_E3A_STAGED_SELECTION_PROOF_KIND,
        protocol_sha256=FORMAL_E3A_STAGED_SELECTION_PROOF_PROTOCOL_SHA256,
        coverage_proof_source=source,
        registry_layer_source=registry_source,
        registry_verification_receipt_sha256=registry.sha256,
        protocol_lock_sha256=selection.protocol_lock_sha256,
        materialization_receipt_sha256=selection.materialization_receipt_sha256,
        coverage_receipt_sha256=selection.coverage_receipt_sha256,
        inventory_sha256=selection.inventory_sha256,
        evidence_manifest_sha256=evidence.sha256,
        selection_artifact_sha256=selection.sha256,
        selection_receipt_sha256=receipt.sha256,
        reduced_ns=now_ns,
    )
    _revalidate_artifact(artifact)
    return artifact


def _revalidate_artifact(
    artifact: FormalE3aStagedSelectionProofArtifact,
) -> tuple[E3aStagedSelectionArtifact, E3aStagedSelectionReceipt]:
    selection, receipt, evidence = _reduce_from_coverage_source(
        artifact.coverage_proof_source,
        artifact.registry_layer_source,
        reduced_ns=artifact.reduced_ns,
    )
    registry = load_formal_registry_verification_receipt_path(
        artifact.registry_layer_source.absolute_path,
        now_ns=artifact.reduced_ns,
    )
    if (
        registry.sha256 != artifact.registry_verification_receipt_sha256
        or selection.protocol_lock_sha256 != artifact.protocol_lock_sha256
        or selection.materialization_receipt_sha256
        != artifact.materialization_receipt_sha256
        or selection.coverage_receipt_sha256 != artifact.coverage_receipt_sha256
        or selection.inventory_sha256 != artifact.inventory_sha256
        or evidence.sha256 != artifact.evidence_manifest_sha256
        or selection.sha256 != artifact.selection_artifact_sha256
        or receipt.sha256 != artifact.selection_receipt_sha256
    ):
        raise ValueError("formal E3a selection proof reducer output changed")
    return selection, receipt


def publish_formal_e3a_staged_selection_proof_artifact(
    artifact: FormalE3aStagedSelectionProofArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish one immutable proof only after an immediate deep replay."""

    if type(artifact) is not FormalE3aStagedSelectionProofArtifact:
        raise TypeError("formal E3a selection publisher requires an exact artifact")
    _revalidate_artifact(artifact)
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(output_path)


def revalidate_formal_e3a_staged_selection_proof_artifact(
    artifact_path: str | Path,
    *,
    now_ns: int,
    relocatable_bundle_manifest_path: str | Path | None = None,
) -> tuple[E3aStagedSelectionArtifact, E3aStagedSelectionReceipt]:
    """Deep-replay a durable E3a proof at its recorded decision time."""

    if type(now_ns) is not int or now_ns < 1:
        raise ValueError("formal E3a selection verification time must be positive")
    if relocatable_bundle_manifest_path is not None:
        from lightcone_spec.runtime.relocatable_evidence import (
            activate_relocatable_evidence_bundle,
        )

        with activate_relocatable_evidence_bundle(
            relocatable_bundle_manifest_path
        ) as bundle:
            path = str(Path(artifact_path))
            if path not in bundle.artifact.entry_remote_absolute_paths:
                raise ValueError("formal E3a proof is not a pulled-evidence entry")
            return revalidate_formal_e3a_staged_selection_proof_artifact(
                path,
                now_ns=now_ns,
            )
    binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalE3aStagedSelectionProofArtifact.from_dict(binding.reopen())
    if now_ns < artifact.reduced_ns:
        raise ValueError("formal E3a selection proof is from the future")
    return _revalidate_artifact(artifact)


__all__ = (
    "FORMAL_E3A_STAGED_SELECTION_PROOF_KIND",
    "FORMAL_E3A_STAGED_SELECTION_PROOF_PROTOCOL_SHA256",
    "FormalE3aStagedSelectionProofArtifact",
    "bind_formal_e3a_staged_selection_proof_artifact",
    "publish_formal_e3a_staged_selection_proof_artifact",
    "revalidate_formal_e3a_staged_selection_proof_artifact",
)
