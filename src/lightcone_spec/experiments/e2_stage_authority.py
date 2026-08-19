"""Signed-staged E2 successive-halving authority.

The historical E2 reducer is tied to the eager diagnostic registry.  This
module instead derives each round from the exact signed materialization and
durable per-cell terminal/timestamp proofs.  Only LightCone-candidate rows are
ranked; Target-only, Static, frozen TTS, and frozen L0-naive remain fixed
mechanism anchors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

from lightcone_spec.experiments.e1_stage_authority import (
    _integer_counter,
    _paired_confidence_lower,
    _request_identity,
    _validated_cell,
)
from lightcone_spec.experiments.formal_protocol import (
    ProtocolLock,
    content_sha256,
    reject_banned_model_identity,
    verify_signed_payload,
)
from lightcone_spec.experiments.formal_stage_execution import (
    VerifiedFormalServingExecutionBinding,
    require_verified_formal_serving_execution_binding,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    E2_OPTIMIZERS,
    E2_SCHEDULES,
    E2CandidateRecipe,
    MaterializedCell,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.runtime.attestation import (
    AttestationChallenge,
    SignedAttestation,
    TrustedAttesterPolicy,
)
from lightcone_spec.runtime.proof_artifact import CanonicalJsonProofBinding

E2_STAGED_HALVING_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_e2_staged_successive_halving_protocol",
        "rounds": 4,
        "round_zero": "105_recipes_per_e1_survivor_geometry",
        "retention": "max(ceil(n/4),21)",
        "family_floor": "one_per_each_7_optimizer_x_3_schedule_family",
        "final_round": "one_locked_recipe_after_round_3",
        "anchors": ("Target-only", "Static", "TTS", "L0-naive"),
        "ranking": (
            "minimum_paired_95pct_lower_request_rate_ratio",
            "peak_hbm_bytes",
            "within_request_p99_itl_us",
            "exposed_update_us",
            "recipe_sha256",
        ),
        "confirmation_data_visible": False,
    }
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


def _round_index(materialization: StageMaterializationReceipt) -> int:
    values = {dict(cell.dimensions).get("round") for cell in materialization.cells}
    if len(values) != 1:
        raise ValueError("E2 materialization must bind one exact round")
    result = next(iter(values))
    if type(result) is not int or result not in range(4):
        raise ValueError("E2 round index is outside [0,4)")
    return result


@dataclass(frozen=True)
class E2CellExecutionEvidence:
    schema_version: int
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
        if self.schema_version != 1:
            raise ValueError("only E2 cell execution evidence schema 1 is supported")
        _require_sha256("E2 evidence cell", self.materialized_cell_id)
        _require_sha256("E2 evidence execution binding", self.execution_binding_sha256)
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("E2 evidence requires an exact execution identity")
        if self.execution_identity.materialized_cell_id != self.materialized_cell_id:
            raise ValueError("E2 execution identity names another cell")
        for label, path, raw_sha256, semantic_sha256 in (
            (
                "E2 native result proof",
                self.native_result_proof_path,
                self.native_result_proof_raw_sha256,
                self.native_result_proof_semantic_sha256,
            ),
            (
                "E2 stage ITL proof",
                self.stage_itl_proof_path,
                self.stage_itl_proof_raw_sha256,
                self.stage_itl_proof_semantic_sha256,
            ),
        ):
            _require_absolute_path(label, path)
            binding = CanonicalJsonProofBinding.bind(path)
            if (
                binding.raw_sha256 != raw_sha256
                or binding.semantic_sha256 != semantic_sha256
            ):
                raise ValueError(f"{label} content changed after binding")
        if self.native_result_proof_path == self.stage_itl_proof_path:
            raise ValueError("E2 result and timing proofs must be distinct")

    @classmethod
    def bind(
        cls,
        *,
        execution_binding: VerifiedFormalServingExecutionBinding,
        native_result_proof_path: str,
        stage_itl_proof_path: str,
    ) -> E2CellExecutionEvidence:
        verified = require_verified_formal_serving_execution_binding(execution_binding)
        if verified.subject.stage != "E2":
            raise ValueError("E2 evidence cannot consume another stage binding")
        result = CanonicalJsonProofBinding.bind(native_result_proof_path)
        timing = CanonicalJsonProofBinding.bind(stage_itl_proof_path)
        return cls(
            schema_version=1,
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
class E2StagedRoundEvidenceManifest:
    schema_version: int
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    source_selection_sha256: str
    inventory_sha256: str
    round_index: int
    cells: tuple[E2CellExecutionEvidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 staged evidence schema 1 is supported")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("source selection", self.source_selection_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"E2 evidence {label}", value)
        if self.round_index not in range(4):
            raise ValueError("E2 evidence round is outside [0,4)")
        ids = tuple(row.materialized_cell_id for row in self.cells)
        if (
            type(self.cells) is not tuple
            or not self.cells
            or any(type(row) is not E2CellExecutionEvidence for row in self.cells)
            or ids != tuple(sorted(set(ids)))
        ):
            raise ValueError("E2 evidence cells are not exact sorted unique rows")
        runs = tuple(
            (
                row.execution_identity.run_id,
                row.execution_identity.run_nonce_sha256,
                row.execution_identity.attempt_id,
            )
            for row in self.cells
        )
        if len(runs) != len(set(runs)):
            raise ValueError("E2 evidence reuses a run identity")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2StagedCandidateEvaluation:
    recipe: E2CandidateRecipe
    cell_id: str
    confidence_lower_request_rate_ratio: float
    peak_hbm_bytes: int
    p99_itl_us: int
    exposed_update_us: int
    launched_updates: int
    published_updates: int

    def __post_init__(self) -> None:
        if type(self.recipe) is not E2CandidateRecipe:
            raise TypeError("E2 evaluation requires an exact recipe")
        _require_sha256("E2 evaluation cell", self.cell_id)
        if (
            type(self.confidence_lower_request_rate_ratio) is not float
            or not math.isfinite(self.confidence_lower_request_rate_ratio)
            or self.confidence_lower_request_rate_ratio <= 0
        ):
            raise ValueError("E2 request-rate confidence lower bound is invalid")
        for label, value in (
            ("peak HBM", self.peak_hbm_bytes),
            ("p99 ITL", self.p99_itl_us),
            ("exposed update", self.exposed_update_us),
            ("launched updates", self.launched_updates),
            ("published updates", self.published_updates),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"E2 {label} must be non-negative")
        if self.published_updates > self.launched_updates:
            raise ValueError("E2 published updates exceed launched updates")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class E2StagedRoundSelectionReceipt:
    schema_version: int
    protocol_lock_sha256: str
    registry_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    source_selection_sha256: str
    evidence_manifest_sha256: str
    inventory_sha256: str
    round_index: int
    source_candidate_count: int
    evaluations: tuple[E2StagedCandidateEvaluation, ...]
    survivor_recipes: tuple[E2CandidateRecipe, ...]
    final_recipe: E2CandidateRecipe | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only E2 staged round selection schema 1 is supported")
        for label, value in (
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
            ("source selection", self.source_selection_sha256),
            ("evidence manifest", self.evidence_manifest_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"E2 selection {label}", value)
        if self.round_index not in range(4):
            raise ValueError("E2 selection round is outside [0,4)")
        if (
            type(self.source_candidate_count) is not int
            or self.source_candidate_count < 1
        ):
            raise ValueError("E2 source candidate count must be positive")
        evaluation_ids = tuple(row.recipe.sha256 for row in self.evaluations)
        survivor_ids = tuple(row.sha256 for row in self.survivor_recipes)
        if evaluation_ids != tuple(sorted(set(evaluation_ids))):
            raise ValueError("E2 evaluations are not canonical")
        if survivor_ids != tuple(sorted(set(survivor_ids))):
            raise ValueError("E2 survivors are not canonical")
        if not set(survivor_ids) <= set(evaluation_ids):
            raise ValueError("E2 survivor is foreign to evaluated recipes")
        if len(self.evaluations) > self.source_candidate_count:
            raise ValueError("E2 evaluated candidates exceed the source universe")
        if self.round_index < 3:
            expected_count = max(math.ceil(self.source_candidate_count / 4), 21)
            expected_families = {
                (optimizer, schedule)
                for optimizer in E2_OPTIMIZERS
                for schedule in E2_SCHEDULES
            }
            if (
                len(self.survivor_recipes) != expected_count
                or {(row.optimizer, row.schedule) for row in self.survivor_recipes}
                != expected_families
                or self.final_recipe is not None
            ):
                raise ValueError("E2 survivors violate quarter/family-floor protocol")
        elif (
            len(self.survivor_recipes) != 1
            or self.final_recipe != self.survivor_recipes[0]
        ):
            raise ValueError("E2 round three must lock exactly one final recipe")
        reject_banned_model_identity(self)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True)
class SignedE2StagedRoundSelectionReceipt:
    payload: E2StagedRoundSelectionReceipt
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def verify(
        self,
        *,
        protocol_lock: ProtocolLock,
        materialization: StageMaterializationReceipt,
        coverage: StageCoverageReceipt,
        source_recipes: tuple[E2CandidateRecipe, ...],
        manifest: E2StagedRoundEvidenceManifest,
        execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
        policy: TrustedAttesterPolicy,
        expected_policy_sha256: str,
        now_ns: int,
    ) -> E2StagedRoundSelectionReceipt:
        if type(self.payload) is not E2StagedRoundSelectionReceipt:
            raise TypeError("signed E2 staged selection payload has the wrong type")
        expected = reduce_e2_staged_round_selection_from_proofs(
            protocol_lock=protocol_lock,
            materialization=materialization,
            coverage=coverage,
            source_selection_sha256=self.payload.source_selection_sha256,
            source_recipes=source_recipes,
            manifest=manifest,
            execution_bindings=execution_bindings,
            now_ns=now_ns,
        )
        if self.payload != expected:
            raise ValueError("signed E2 staged selection differs from proof reducer")
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


def _ranking(row: E2StagedCandidateEvaluation) -> tuple[object, ...]:
    return (
        -row.confidence_lower_request_rate_ratio,
        row.peak_hbm_bytes,
        row.p99_itl_us,
        row.exposed_update_us,
        row.recipe.sha256,
    )


def _select_survivor_recipes(
    *,
    source_recipes: tuple[E2CandidateRecipe, ...],
    evaluations: tuple[E2StagedCandidateEvaluation, ...],
    round_index: int,
) -> tuple[tuple[E2CandidateRecipe, ...], E2CandidateRecipe | None]:
    """Apply the registered quarter retention and exact 21-family floor."""

    if type(source_recipes) is not tuple or any(
        type(row) is not E2CandidateRecipe for row in source_recipes
    ):
        raise TypeError("E2 survivor selection requires exact typed source recipes")
    source_ids = tuple(row.sha256 for row in source_recipes)
    if source_ids != tuple(sorted(set(source_ids))):
        raise ValueError("E2 survivor source recipes are not canonical")
    if type(evaluations) is not tuple or any(
        type(row) is not E2StagedCandidateEvaluation for row in evaluations
    ):
        raise TypeError("E2 survivor selection requires exact typed evaluations")
    evaluation_ids = tuple(row.recipe.sha256 for row in evaluations)
    if evaluation_ids != tuple(sorted(set(evaluation_ids))) or not set(
        evaluation_ids
    ) <= set(source_ids):
        raise ValueError("E2 survivor evaluations are duplicate or foreign")
    if type(round_index) is not int or round_index not in range(4):
        raise ValueError("E2 survivor round is outside [0,4)")
    ranked = tuple(sorted(evaluations, key=_ranking))
    if not ranked:
        raise ValueError("E2 survivor selection has no safe evaluated candidate")
    if round_index == 3:
        final_recipe = ranked[0].recipe
        return (final_recipe,), final_recipe
    families: dict[tuple[str, str], list[E2StagedCandidateEvaluation]] = {
        (optimizer, schedule): []
        for optimizer in E2_OPTIMIZERS
        for schedule in E2_SCHEDULES
    }
    for row in evaluations:
        families[(row.recipe.optimizer, row.recipe.schedule)].append(row)
    if any(not rows for rows in families.values()):
        raise ValueError(
            "E2 safe evidence does not preserve every optimizer/schedule family"
        )
    survivor_count = max(math.ceil(len(source_recipes) / 4), 21)
    if len(evaluations) < survivor_count:
        raise ValueError("E2 safe evidence cannot fill the registered survivor count")
    selected = {min(rows, key=_ranking).recipe.sha256 for rows in families.values()}
    for row in ranked:
        if len(selected) >= survivor_count:
            break
        selected.add(row.recipe.sha256)
    recipes = {row.sha256: row for row in source_recipes}
    return (
        tuple(
            sorted((recipes[digest] for digest in selected), key=lambda row: row.sha256)
        ),
        None,
    )


def reduce_e2_staged_round_selection_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    coverage: StageCoverageReceipt,
    source_selection_sha256: str,
    source_recipes: tuple[E2CandidateRecipe, ...],
    manifest: E2StagedRoundEvidenceManifest,
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> E2StagedRoundSelectionReceipt:
    """Deep-open one complete E2 round and deterministically seal survivors."""

    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("E2 staged reduction requires an exact ProtocolLock")
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("E2 staged reduction requires exact materialization")
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("E2 staged reduction requires exact coverage")
    if type(manifest) is not E2StagedRoundEvidenceManifest:
        raise TypeError("E2 staged reduction requires exact evidence manifest")
    _require_sha256("E2 source selection", source_selection_sha256)
    round_index = _round_index(materialization)
    if (
        materialization.stage != "E2"
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.source_decision_sha256 != source_selection_sha256
        or manifest.protocol_lock_sha256 != protocol_lock.sha256
        or manifest.materialization_receipt_sha256 != materialization.sha256
        or manifest.coverage_receipt_sha256 != coverage.sha256
        or manifest.source_selection_sha256 != source_selection_sha256
        or manifest.round_index != round_index
    ):
        raise ValueError("E2 staged evidence differs from signed lineage")
    coverage.validate_against(materialization)
    if any(row.status != "COMPLETE" for row in coverage.dispositions):
        raise ValueError("E2 staged reduction requires all-COMPLETE coverage")
    if type(source_recipes) is not tuple or any(
        type(row) is not E2CandidateRecipe for row in source_recipes
    ):
        raise TypeError("E2 source recipes must be exact sorted unique typed rows")
    expected_source_ids = tuple(row.sha256 for row in source_recipes)
    if expected_source_ids != tuple(sorted(set(expected_source_ids))):
        raise ValueError("E2 source recipes must be exact sorted unique typed rows")
    roles: dict[str, list[MaterializedCell]] = {}
    for cell in materialization.cells:
        roles.setdefault(cell.method_role, []).append(cell)
    candidates = roles.get("LightCone-candidate", [])
    if (
        {
            role: len(roles.get(role, []))
            for role in ("Target-only", "Static", "TTS", "L0-naive")
        }
        != {"Target-only": 1, "Static": 1, "TTS": 1, "L0-naive": 1}
        or len(candidates) != len(source_recipes)
        or tuple(
            sorted(cell.recipe_sha256 for cell in candidates if cell.recipe_sha256)
        )
        != expected_source_ids
    ):
        raise ValueError("E2 materialization differs from typed source recipe universe")
    evidence_by_cell = {row.materialized_cell_id: row for row in manifest.cells}
    bindings_by_cell = {}
    for row in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(row)
        if verified.subject.materialized_cell_id in bindings_by_cell:
            raise ValueError("E2 reuses an execution binding")
        bindings_by_cell[verified.subject.materialized_cell_id] = verified
    expected_cells = {cell.cell_id for cell in materialization.cells}
    if (
        set(evidence_by_cell) != expected_cells
        or set(bindings_by_cell) != expected_cells
    ):
        raise ValueError("E2 proof/binding coverage is not exact")
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
            expected_stage="E2",
        )
        for cell in materialization.cells
    }
    target = validated[roles["Target-only"][0].cell_id]
    static = validated[roles["Static"][0].cell_id]
    tts = validated[roles["TTS"][0].cell_id]
    l0 = validated[roles["L0-naive"][0].cell_id]
    target_identity = _request_identity(target.metrics)
    for label, row in (
        ("Target-only", target),
        ("Static", static),
        ("TTS", tts),
        ("L0-naive", l0),
    ):
        if row.safety_reasons or _request_identity(row.metrics) != target_identity:
            raise ValueError(f"E2 {label} anchor is unsafe or trajectory-mismatched")
    if tts.published_updates < 1 or l0.published_updates < 1:
        raise ValueError("E2 TTS/L0 anchors lack a published update")
    evaluations = []
    recipes = {row.sha256: row for row in source_recipes}
    for cell in candidates:
        assert cell.recipe_sha256 is not None
        row = validated[cell.cell_id]
        if row.safety_reasons or _request_identity(row.metrics) != target_identity:
            continue
        launched = _integer_counter(row.result.performance_counters, "updates_launched")
        if launched < 1 or row.published_updates < 1:
            continue
        recipe = recipes[cell.recipe_sha256]
        evaluation = E2StagedCandidateEvaluation(
            recipe=recipe,
            cell_id=cell.cell_id,
            confidence_lower_request_rate_ratio=min(
                _paired_confidence_lower(row.metrics, static.metrics),
                _paired_confidence_lower(row.metrics, tts.metrics),
            ),
            peak_hbm_bytes=row.peak_hbm_bytes,
            p99_itl_us=max(
                math.ceil(metric.p99_itl_ns / 1_000) for metric in row.metrics
            ),
            exposed_update_us=row.exposed_update_us,
            launched_updates=launched,
            published_updates=row.published_updates,
        )
        evaluations.append(evaluation)
    canonical_evaluations = tuple(
        sorted(evaluations, key=lambda row: row.recipe.sha256)
    )
    survivors, final_recipe = _select_survivor_recipes(
        source_recipes=source_recipes,
        evaluations=canonical_evaluations,
        round_index=round_index,
    )
    receipt = E2StagedRoundSelectionReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        source_selection_sha256=source_selection_sha256,
        evidence_manifest_sha256=manifest.sha256,
        inventory_sha256=manifest.inventory_sha256,
        round_index=round_index,
        source_candidate_count=len(source_recipes),
        evaluations=canonical_evaluations,
        survivor_recipes=survivors,
        final_recipe=final_recipe,
    )
    receipt.__post_init__()
    return receipt


__all__ = [
    "E2_STAGED_HALVING_PROTOCOL_SHA256",
    "E2CellExecutionEvidence",
    "E2StagedCandidateEvaluation",
    "E2StagedRoundEvidenceManifest",
    "E2StagedRoundSelectionReceipt",
    "SignedE2StagedRoundSelectionReceipt",
    "reduce_e2_staged_round_selection_from_proofs",
]
