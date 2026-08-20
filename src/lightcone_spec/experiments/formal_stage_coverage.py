"""Proof-derived, durable coverage authority for formal experiment stages.

Coverage is deliberately reconstructed from path-bound terminal evidence.  A
caller never supplies a cell status or terminal digest to the reducer.  The
durable top-level artifact points at the exact materialization, runtime
authority, inventory, evidence shards, execution-rebuild shards, optional
stage-source descriptor, candidate-replay proofs, and reducer-derived coverage
shards needed to repeat that reconstruction from a clean checkout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from lightcone_spec.experiments.formal_failure_actuator import (
    FormalFailureActuationProofArtifact,
    validate_formal_failure_actuation_proof_artifact,
)
from lightcone_spec.experiments.formal_failure_execution import (
    FormalFailureExecutionRebuildInput,
    VerifiedFormalFailureExecutionBinding,
    rebuild_formal_failure_execution_binding,
    require_verified_formal_failure_execution_binding,
)
from lightcone_spec.experiments.formal_materialization_shards import (
    revalidate_formal_materialization_shard_index,
)
from lightcone_spec.experiments.formal_protocol import (
    E6_MODELS,
    TTS_L0_CANDIDATE_STATE_COVERAGE_STAGES,
    FormalRuntimeAuthorityManifest,
    ProtocolLock,
    TtsL0CandidateStateCoverage,
    content_sha256,
)
from lightcone_spec.experiments.formal_stage_execution import (
    E6FinalStageSourceRebuildInputs,
    E6PilotStageSourceRebuildInputs,
    FormalServingExecutionRebuildInput,
    FormalStageSourceRebuildInput,
    FormalStageSourceRebuildInputs,
    VerifiedFormalServingExecutionBinding,
    VerifiedFormalStageMaterializationSource,
    rebuild_formal_serving_execution_binding,
    rebuild_formal_stage_materialization_source,
    require_verified_formal_serving_execution_binding,
    require_verified_formal_stage_materialization_source,
)
from lightcone_spec.experiments.gpu_pool import GpuInventory
from lightcone_spec.experiments.industrial_analysis import (
    BoundArtifact,
    RawTtsCalibrationEvidenceManifest,
    raw_tts_calibration_manifest_from_dict,
    validate_raw_evidence_manifest_sidecars,
)
from lightcone_spec.experiments.industrial_analysis import (
    _bound_json as _industrial_bound_json,
)
from lightcone_spec.experiments.itl_authority import (
    StageItlExecutionIdentity,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_ALL_NA_MATERIALIZATION_RULE,
    E6_TASKS,
    TTS_CAL_MATERIALIZATION_RULE,
    MaterializedCell,
    SignedStageCoverageReceipt,
    StageCellDisposition,
    StageCoverageReceipt,
    StageMaterializationReceipt,
)
from lightcone_spec.experiments.statistics import TTFT_LIMIT_MS
from lightcone_spec.experiments.tts_calibration_authority import (
    TtsCalibrationAuthority,
)
from lightcone_spec.orchestration.formal_failure_physical import (
    validate_formal_e5_failure_lifecycle_proof_artifact,
)
from lightcone_spec.orchestration.formal_serving_lift import (
    validate_formal_serving_itl_proof,
)
from lightcone_spec.orchestration.formal_terminal_result import (
    validate_formal_terminal_result_proof_artifact,
)
from lightcone_spec.orchestration.native_terminal import (
    REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256,
    prepare_native_terminal_external_control,
    validate_candidate_state_replay_proof_artifact,
)
from lightcone_spec.runtime.attestation import AttestationChallenge, SignedAttestation
from lightcone_spec.runtime.control_attestation import (
    ControlArtifactAttestation,
    verify_release_control_artifact_attestation,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 3,
        "kind": "lightcone_formal_stage_proof_derived_coverage_protocol",
        "statuses": "derived_only_from_deep_revalidated_terminal_proofs",
        "nonserving_closed_unions": (
            "tts_calibration_raw_288",
            "e5_failure_actuation_and_integrated_lifecycle_264",
            "e6_signed_nextn_model_compatibility_2_plus_serving",
        ),
        "profiler": "blocked_until_dedicated_nsys_ncu_reducer",
        "materialization": "every_and_only_materialized_cell",
        "serving": "sealed_execution_rebuild_plus_result_plus_native_itl",
        "tts_calibration": (
            "raw_288_terminal_external_controls_qualification_locks_and_"
            "current_request_reset_evidence"
        ),
        "candidate_state": "path_bound_native_replay_proof_exact_pair_coverage",
        "failure_policy": "E5_final_requires_separate_failure_proof_adapter",
        "durability": (
            "path_bound_sharded_rebuild_and_derived_receipt_plus_"
            "caller_free_portable_prefix_replay"
        ),
    }
)
FORMAL_STAGE_COVERAGE_RUNNER_SHA256 = content_sha256(
    {
        "schema_version": 2,
        "kind": "lightcone_formal_runtime_runner_semantic_identity",
        "member_id": "stage_coverage_reducer",
        "entrypoint_sources": (
            "src/lightcone_spec/experiments/formal_downstream_prefix.py",
            "src/lightcone_spec/experiments/formal_stage_coverage.py",
            "src/lightcone_spec/experiments/formal_stage_coverage_portable.py",
            "src/lightcone_spec/experiments/formal_materialization_shards.py",
            "src/lightcone_spec/runtime/scientific_source_validation.py",
        ),
    }
)
FORMAL_STAGE_COVERAGE_TEST_SET_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_runtime_test_set_semantic_identity",
        "member_id": "stage_coverage_reducer",
        "pytest_nodes": (
            "tests/test_formal_stage_coverage.py",
            "tests/test_formal_stage_coverage_portable.py",
            "tests/test_offline_scientific_signing.py",
            "tests/test_tts_calibration_authority.py",
        ),
    }
)
FORMAL_E0_ALL_NA_COVERAGE_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_e0_all_na_coverage_protocol",
        "compatibility": "exact_signed_108_all_na",
        "materialization": "exact_zero_cell_all_na_rule",
        "completion": "signed_zero_valid_zero_cell_completion",
        "fdr": "signed_756_excluded_na_zero_decisions",
        "coverage": "empty_dispositions_only_for_this_closed_branch",
    }
)

FormalCoveragePhase = Literal[
    "capacity",
    "calibration",
    "selection",
    "round0",
    "round1",
    "round2",
    "round3",
    "screen",
    "local",
    "profiler",
    "excluded_pilot",
    "verification",
    "final",
    "final_and_one_shot_failure",
    "excluded_pilot_and_model_preflight",
    "onlinespec_tuning",
]

_STAGE_PHASES: frozenset[tuple[str, str]] = frozenset(
    {
        ("E3a", "capacity"),
        ("TTS-Cal", "calibration"),
        ("E1", "selection"),
        ("E2", "round0"),
        ("E2", "round1"),
        ("E2", "round2"),
        ("E2", "round3"),
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
    }
)

_MATERIALIZATION_RULE_PHASES = {
    E0_ALL_NA_MATERIALIZATION_RULE: ("E0", "final"),
    "exact_360_row_capacity_width_and_drift_grid": ("E3a", "capacity"),
    TTS_CAL_MATERIALIZATION_RULE: ("TTS-Cal", "calibration"),
    "four_fixed_anchors_plus_32_geometries_x_2_optimizers": ("E1", "selection"),
    "strength2_8_rows_x_3_loads_x_2_traffic": ("E4", "screen"),
    "winner_neighborhood_2pow4_x_3_loads_x_2_traffic": ("E4", "local"),
    "three_profiler_only_rows_separate_from_headline": ("E4", "profiler"),
    "e3b_exact_480_rows_x_4_excluded_pilot_blocks": ("E3b", "excluded_pilot"),
    ("five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only"): (
        "E3b",
        "final",
    ),
    "58_configurations_x_2_verification_modes": ("E1a", "verification"),
    "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks": (
        "E5",
        "excluded_pilot",
    ),
    "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics": (
        "E5",
        "final_and_one_shot_failure",
    ),
    "e6_exact_two_model_preflights_plus_60_rows_x_4_excluded_pilot_blocks": (
        "E6",
        "excluded_pilot_and_model_preflight",
    ),
    "60_final_rows_per_block_reusing_global_model_preflights": ("E6", "final"),
    "e0_full_registered_onlinespec_grid_per_valid_combination_tuning_only": (
        "E0",
        "onlinespec_tuning",
    ),
    "e0_exact_16_rows_per_valid_combination_x_4_excluded_pilot_blocks": (
        "E0",
        "excluded_pilot",
    ),
    "valid_compatibilities_x_8_roles_x_2_loads_x_final_only_powered_prefix": (
        "E0",
        "final",
    ),
}


def _require_sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return value


def _require_phase(stage: object, phase: object) -> tuple[str, str]:
    if (
        type(stage) is not str
        or type(phase) is not str
        or (stage, phase) not in (_STAGE_PHASES)
    ):
        raise ValueError("formal coverage stage/phase is not in the closed union")
    return stage, phase


def _phase_for_materialization(
    materialization: StageMaterializationReceipt,
) -> tuple[str, str]:
    if type(materialization) is not StageMaterializationReceipt:
        raise TypeError("formal coverage requires an exact materialization")
    if materialization.stage == "E2":
        expected_rule = (
            "e2_round_0_105_per_geometry_plus_four_anchors"
            if {dict(cell.dimensions).get("round") for cell in materialization.cells}
            == {0}
            else "e2_quarter_retention_floor_21_plus_four_anchors"
        )
        rounds = {dict(cell.dimensions).get("round") for cell in materialization.cells}
        if (
            materialization.materialization_rule != expected_rule
            or len(rounds) != 1
            or next(iter(rounds)) not in range(4)
        ):
            raise ValueError("E2 coverage materialization round/rule is not exact")
        return "E2", f"round{next(iter(rounds))}"
    try:
        observed = _MATERIALIZATION_RULE_PHASES[materialization.materialization_rule]
    except KeyError as error:
        raise ValueError(
            "formal coverage materialization rule is unsupported"
        ) from error
    if observed[0] != materialization.stage:
        raise ValueError("formal coverage materialization stage/rule differs")
    return observed


def _validate_materialization_cardinality(
    materialization: StageMaterializationReceipt,
    *,
    phase: str,
    e0_valid_compatibility_decision_ids: tuple[str, ...] | None = None,
) -> None:
    count = materialization.expected_cell_count
    exact = {
        ("E3a", "capacity"): 360,
        ("TTS-Cal", "calibration"): 288,
        ("E1", "selection"): 68,
        ("E4", "screen"): 48,
        ("E4", "local"): 96,
        ("E4", "profiler"): 3,
        ("E3b", "excluded_pilot"): 1_920,
        ("E1a", "verification"): 116,
        ("E5", "excluded_pilot"): 1_800,
        ("E6", "excluded_pilot_and_model_preflight"): 242,
    }
    expected = exact.get((materialization.stage, phase))
    if expected is not None and count != expected:
        raise ValueError("formal coverage materialization cardinality differs")
    if materialization.stage == "E2":
        round_index = int(phase.removeprefix("round"))
        if round_index == 0:
            if count < 109 or (count - 4) % 105:
                raise ValueError("E2 round-0 coverage cardinality differs")
        elif count < 25:
            raise ValueError("E2 retained-round coverage is below its floor")
    elif materialization.stage == "E3b" and phase == "final":
        if count % 480 or not 12 <= count // 480 <= 20:
            raise ValueError("E3b final coverage cardinality differs")
    elif materialization.stage == "E5" and phase == "final_and_one_shot_failure":
        if (count - 264) % 450 or not 12 <= (count - 264) // 450 <= 20:
            raise ValueError("E5 final coverage cardinality differs")
    elif materialization.stage == "E6" and phase == "final":
        if count % 60 or not 12 <= count // 60 <= 20:
            raise ValueError("E6 final coverage cardinality differs")
    elif materialization.stage == "E0" and count == 0:
        # This is only a shape acknowledgement.  The separate E0 V=0 adapter
        # must still deep-verify all 108 signed compatibility decisions and
        # the all-EXCLUDED_NA FDR/completion projection before coverage exists.
        raise ValueError("E0 V=0 requires the dedicated all-N/A coverage adapter")
    elif materialization.stage == "E0":
        ids = e0_valid_compatibility_decision_ids
        if type(ids) is not tuple or not ids or ids != tuple(sorted(set(ids))):
            raise ValueError(
                "positive E0 coverage requires deep-rebuilt signed compatibility"
            )
        for decision_id in ids:
            _require_sha256("E0 valid compatibility decision", decision_id)
        by_decision: dict[str, int] = {}
        for cell in materialization.cells:
            decision_id = dict(cell.dimensions).get("compatibility_decision_id")
            if type(decision_id) is not str:
                raise ValueError("E0 coverage cell lacks a compatibility decision")
            by_decision[decision_id] = by_decision.get(decision_id, 0) + 1
        if set(by_decision) != set(ids):
            raise ValueError("E0 coverage does not match every signed VALID decision")
        valid_count = len(ids)
        if phase == "onlinespec_tuning":
            if count != 239 * valid_count or set(by_decision.values()) != {239}:
                raise ValueError("E0 tuning coverage cardinality differs")
        elif phase == "excluded_pilot":
            if count != 64 * valid_count or set(by_decision.values()) != {64}:
                raise ValueError("E0 excluded-pilot coverage cardinality differs")
        elif phase == "final":
            per_valid = set(by_decision.values())
            if len(per_valid) != 1:
                raise ValueError("E0 final coverage uses unequal powered prefixes")
            rows = next(iter(per_valid))
            if rows % 16 or not 12 <= rows // 16 <= 20:
                raise ValueError("E0 final coverage cardinality differs")
        else:  # pragma: no cover - phase union is closed above
            raise AssertionError("unsupported E0 coverage phase")


def _runtime_method(cell: MaterializedCell) -> str:
    try:
        return {
            "Target-only": "target_only",
            "Static": "static",
            "TTS": "tts",
            "L0-naive": "l0",
            "LightCone-candidate": "l0",
            "LightCone": "l0",
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
        }[cell.method_role]
    except KeyError as error:
        raise ValueError(
            "formal coverage materialization role is unsupported"
        ) from error


def _e0_valid_compatibility_decision_ids(
    source_inputs: FormalStageSourceRebuildInputs | None,
) -> tuple[str, ...]:
    """Project V only from the deep-replayed, signed 108-row source union."""

    from lightcone_spec.experiments.formal_stage_execution import (
        E0FinalStageSourceRebuildInputs,
        E0PilotStageSourceRebuildInputs,
        E0TuningStageSourceRebuildInputs,
    )

    if type(source_inputs) in {
        E0TuningStageSourceRebuildInputs,
        E0PilotStageSourceRebuildInputs,
    }:
        compatibility = source_inputs.signed_compatibility_receipt.payload  # type: ignore[union-attr]
    elif type(source_inputs) is E0FinalStageSourceRebuildInputs:
        compatibility = source_inputs.authority_bundle.signed_compatibility.payload
    else:
        raise TypeError("E0 coverage lacks its signed compatibility source")
    if len(compatibility.decisions) != 108:
        raise ValueError("E0 coverage compatibility universe is incomplete")
    return tuple(
        sorted(
            row.decision_id
            for row in compatibility.decisions
            if row.disposition == "VALID"
        )
    )


def _validate_runtime_authority(
    protocol_lock: ProtocolLock,
    manifest: FormalRuntimeAuthorityManifest,
) -> None:
    if type(protocol_lock) is not ProtocolLock:
        raise TypeError("formal coverage requires an exact ProtocolLock")
    if type(manifest) is not FormalRuntimeAuthorityManifest:
        raise TypeError("formal coverage requires an exact runtime authority")
    member = manifest.member("stage_coverage_reducer")
    if (
        manifest.sha256 != protocol_lock.formal_runtime_authority_manifest_sha256
        or member.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
        or member.runner_sha256 != FORMAL_STAGE_COVERAGE_RUNNER_SHA256
        or member.test_set_sha256 != FORMAL_STAGE_COVERAGE_TEST_SET_SHA256
    ):
        raise ValueError("formal coverage runtime authority differs from ProtocolLock")


@dataclass(frozen=True)
class FormalStageCoverageEvidenceCell:
    """No-status path binding for one serving terminal and ITL proof pair."""

    schema_version: Literal[1]
    materialized_cell_id: str
    execution_binding_sha256: str
    execution_identity: StageItlExecutionIdentity
    native_result_proof: CanonicalJsonProofBinding
    stage_itl_proof: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only formal coverage evidence-cell schema 1 is supported")
        _require_sha256("formal coverage cell", self.materialized_cell_id)
        _require_sha256(
            "formal coverage execution binding", self.execution_binding_sha256
        )
        if type(self.execution_identity) is not StageItlExecutionIdentity:
            raise TypeError("formal coverage evidence requires an execution identity")
        self.execution_identity.__post_init__()
        if self.execution_identity.materialized_cell_id != self.materialized_cell_id:
            raise ValueError("formal coverage execution identity names another cell")
        if (
            type(self.native_result_proof) is not CanonicalJsonProofBinding
            or type(self.stage_itl_proof) is not CanonicalJsonProofBinding
        ):
            raise TypeError("formal coverage evidence proofs must be path-bound")
        if self.native_result_proof.absolute_path == self.stage_itl_proof.absolute_path:
            raise ValueError("formal coverage terminal and ITL proofs must differ")
        if (
            CanonicalJsonProofBinding.bind(self.native_result_proof.absolute_path)
            != self.native_result_proof
            or CanonicalJsonProofBinding.bind(self.stage_itl_proof.absolute_path)
            != self.stage_itl_proof
        ):
            raise ValueError("formal coverage proof path changed after binding")

    @classmethod
    def bind(
        cls,
        *,
        execution_binding: VerifiedFormalServingExecutionBinding,
        native_result_proof_path: str,
        stage_itl_proof_path: str,
    ) -> Self:
        verified = require_verified_formal_serving_execution_binding(execution_binding)
        return cls(
            schema_version=1,
            materialized_cell_id=verified.subject.materialized_cell_id,
            execution_binding_sha256=verified.sha256,
            execution_identity=verified.subject.execution_identity,
            native_result_proof=CanonicalJsonProofBinding.bind(
                native_result_proof_path
            ),
            stage_itl_proof=CanonicalJsonProofBinding.bind(stage_itl_proof_path),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "materialized_cell_id": self.materialized_cell_id,
            "execution_binding_sha256": self.execution_binding_sha256,
            "execution_identity": self.execution_identity.to_dict(),
            "native_result_proof": self.native_result_proof.to_dict(),
            "stage_itl_proof": self.stage_itl_proof.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal coverage evidence-cell fields differ")
        row = dict(value)
        row["execution_identity"] = StageItlExecutionIdentity.from_dict(
            row["execution_identity"]
        )
        row["native_result_proof"] = CanonicalJsonProofBinding.from_dict(
            row["native_result_proof"]
        )
        row["stage_itl_proof"] = CanonicalJsonProofBinding.from_dict(
            row["stage_itl_proof"]
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalStageCoverageEvidenceShard:
    """Bounded shard of the no-status serving evidence manifest."""

    schema_version: Literal[1]
    kind: Literal["formal_stage_coverage_evidence_shard"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    stage: str
    phase: str
    shard_index: int
    shard_count: int
    cells: tuple[FormalStageCoverageEvidenceCell, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_stage_coverage_evidence_shard"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
        ):
            raise ValueError("formal coverage evidence-shard identity differs")
        _require_phase(self.stage, self.phase)
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"formal coverage {label}", digest)
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
            or type(self.cells) is not tuple
            or not self.cells
            or any(
                type(row) is not FormalStageCoverageEvidenceCell for row in self.cells
            )
            or tuple(row.materialized_cell_id for row in self.cells)
            != tuple(sorted({row.materialized_cell_id for row in self.cells}))
        ):
            raise ValueError("formal coverage evidence shard is not exact/canonical")
        paths = tuple(
            path
            for row in self.cells
            for path in (
                row.native_result_proof.absolute_path,
                row.stage_itl_proof.absolute_path,
            )
        )
        bindings = tuple(row.execution_binding_sha256 for row in self.cells)
        runs = tuple(
            (
                row.execution_identity.run_id,
                row.execution_identity.run_nonce_sha256,
                row.execution_identity.attempt_id,
            )
            for row in self.cells
        )
        if (
            len(paths) != len(set(paths))
            or len(bindings) != len(set(bindings))
            or len(runs) != len(set(runs))
        ):
            raise ValueError("formal coverage evidence shard reuses proof/run identity")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "stage": self.stage,
            "phase": self.phase,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "cells": [row.to_dict() for row in self.cells],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal coverage evidence-shard fields differ")
        row = dict(value)
        declared = _require_sha256(
            "formal coverage evidence shard", row.pop("shard_sha256")
        )
        raw_cells = row["cells"]
        if type(raw_cells) is not list:
            raise TypeError("formal coverage evidence-shard cells must be an array")
        row["cells"] = tuple(
            FormalStageCoverageEvidenceCell.from_dict(item) for item in raw_cells
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("formal coverage evidence-shard digest differs")
        return shard


@dataclass(frozen=True)
class FormalStageCoverageExecutionRebuildShard:
    """Durable descriptors that rebuild private serving-execution tokens."""

    schema_version: Literal[1]
    kind: Literal["formal_stage_coverage_execution_rebuild_shard"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    stage: str
    phase: str
    shard_index: int
    shard_count: int
    rebuild_inputs: tuple[FormalServingExecutionRebuildInput, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_stage_coverage_execution_rebuild_shard"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
        ):
            raise ValueError("formal coverage execution-shard identity differs")
        _require_phase(self.stage, self.phase)
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"formal coverage execution {label}", digest)
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
            or type(self.rebuild_inputs) is not tuple
            or not self.rebuild_inputs
            or any(
                type(row) is not FormalServingExecutionRebuildInput
                for row in self.rebuild_inputs
            )
        ):
            raise ValueError("formal coverage execution rebuild shard is invalid")
        ids = tuple(row.subject.materialized_cell_id for row in self.rebuild_inputs)
        if ids != tuple(sorted(set(ids))):
            raise ValueError(
                "formal coverage execution rebuild cells are not canonical"
            )
        if any(
            row.subject.stage != self.stage
            or row.subject.protocol_lock_sha256 != self.protocol_lock_sha256
            or row.subject.materialization_receipt_sha256
            != self.materialization_receipt_sha256
            or row.subject.inventory_sha256 != self.inventory_sha256
            for row in self.rebuild_inputs
        ):
            raise ValueError("formal coverage execution rebuild input is foreign")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "stage": self.stage,
            "phase": self.phase,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "rebuild_inputs": [row.to_dict() for row in self.rebuild_inputs],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal coverage execution-shard fields differ")
        row = dict(value)
        declared = _require_sha256(
            "formal coverage execution shard", row.pop("shard_sha256")
        )
        raw_inputs = row["rebuild_inputs"]
        if type(raw_inputs) is not list:
            raise TypeError("formal coverage rebuild inputs must be an array")
        row["rebuild_inputs"] = tuple(
            FormalServingExecutionRebuildInput.from_dict(item) for item in raw_inputs
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("formal coverage execution-shard digest differs")
        return shard


@dataclass(frozen=True)
class FormalE5FailureCoverageEvidenceCell:
    """Actuation and integrated-process lifecycle proofs for one E5 fault row."""

    schema_version: Literal[1]
    materialized_cell_id: str
    failure_execution_binding_sha256: str
    assignment_sha256: str
    serving_execution_plan_sha256: str
    plan: CanonicalJsonProofBinding
    actuation_proof: CanonicalJsonProofBinding
    lifecycle_proof: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("E5 failure coverage evidence schema differs")
        for label, digest in (
            ("cell", self.materialized_cell_id),
            ("failure binding", self.failure_execution_binding_sha256),
            ("assignment", self.assignment_sha256),
            ("serving plan", self.serving_execution_plan_sha256),
        ):
            _require_sha256(f"E5 failure coverage {label}", digest)
        bindings = (self.plan, self.actuation_proof, self.lifecycle_proof)
        if any(type(row) is not CanonicalJsonProofBinding for row in bindings):
            raise TypeError("E5 failure coverage proofs must be path-bound")
        if len({row.absolute_path for row in bindings}) != 3 or any(
            CanonicalJsonProofBinding.bind(row.absolute_path) != row for row in bindings
        ):
            raise ValueError("E5 failure coverage proof paths are reused or changed")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "materialized_cell_id": self.materialized_cell_id,
            "failure_execution_binding_sha256": (self.failure_execution_binding_sha256),
            "assignment_sha256": self.assignment_sha256,
            "serving_execution_plan_sha256": self.serving_execution_plan_sha256,
            "plan": self.plan.to_dict(),
            "actuation_proof": self.actuation_proof.to_dict(),
            "lifecycle_proof": self.lifecycle_proof.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("E5 failure coverage evidence fields differ")
        row = dict(value)
        for name in ("plan", "actuation_proof", "lifecycle_proof"):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FormalE5FailureCoverageEvidenceShard:
    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_coverage_evidence_shard"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    stage: Literal["E5"]
    phase: Literal["final_and_one_shot_failure"]
    shard_index: int
    shard_count: int
    cells: tuple[FormalE5FailureCoverageEvidenceCell, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_e5_failure_coverage_evidence_shard"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
            or (self.stage, self.phase) != ("E5", "final_and_one_shot_failure")
        ):
            raise ValueError("E5 failure coverage evidence shard identity differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"E5 failure coverage {label}", digest)
        ids = tuple(row.materialized_cell_id for row in self.cells)
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
            or not self.cells
            or any(
                type(row) is not FormalE5FailureCoverageEvidenceCell
                for row in self.cells
            )
            or ids != tuple(sorted(set(ids)))
        ):
            raise ValueError("E5 failure coverage evidence shard is not canonical")
        paths = tuple(
            binding.absolute_path
            for row in self.cells
            for binding in (row.plan, row.actuation_proof, row.lifecycle_proof)
        )
        if len(paths) != len(set(paths)):
            raise ValueError("E5 failure coverage evidence shard reuses a proof path")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "stage": self.stage,
            "phase": self.phase,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "cells": [row.to_dict() for row in self.cells],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("E5 failure coverage evidence shard fields differ")
        row = dict(value)
        declared = _require_sha256(
            "E5 failure coverage evidence shard", row.pop("shard_sha256")
        )
        raw_cells = row["cells"]
        if type(raw_cells) is not list:
            raise TypeError("E5 failure coverage evidence cells must be an array")
        row["cells"] = tuple(
            FormalE5FailureCoverageEvidenceCell.from_dict(item) for item in raw_cells
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("E5 failure coverage evidence shard digest differs")
        return shard


@dataclass(frozen=True)
class FormalE5FailureCoverageExecutionRebuildRow:
    schema_version: Literal[1]
    serving_execution: FormalServingExecutionRebuildInput
    failure_execution: FormalFailureExecutionRebuildInput

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("E5 failure coverage rebuild row schema differs")
        if (
            type(self.serving_execution) is not FormalServingExecutionRebuildInput
            or type(self.failure_execution) is not FormalFailureExecutionRebuildInput
        ):
            raise TypeError("E5 failure coverage rebuild row types differ")
        if (
            self.failure_execution.serving_execution_rebuild_input_sha256
            != self.serving_execution.sha256
            or self.failure_execution.subject.materialized_cell_id
            != self.serving_execution.subject.materialized_cell_id
        ):
            raise ValueError("E5 failure/serving rebuild descriptors differ")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "serving_execution": self.serving_execution.to_dict(),
            "failure_execution": self.failure_execution.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("E5 failure coverage rebuild row fields differ")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            serving_execution=FormalServingExecutionRebuildInput.from_dict(
                value["serving_execution"]
            ),
            failure_execution=FormalFailureExecutionRebuildInput.from_dict(
                value["failure_execution"]
            ),
        )


@dataclass(frozen=True)
class FormalE5FailureCoverageExecutionRebuildShard:
    schema_version: Literal[1]
    kind: Literal["formal_e5_failure_coverage_execution_rebuild_shard"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    stage: Literal["E5"]
    phase: Literal["final_and_one_shot_failure"]
    shard_index: int
    shard_count: int
    rows: tuple[FormalE5FailureCoverageExecutionRebuildRow, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_e5_failure_coverage_execution_rebuild_shard"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
            or (self.stage, self.phase) != ("E5", "final_and_one_shot_failure")
        ):
            raise ValueError("E5 failure coverage rebuild shard identity differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"E5 failure coverage rebuild {label}", digest)
        ids = tuple(
            row.failure_execution.subject.materialized_cell_id for row in self.rows
        )
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
            or not self.rows
            or any(
                type(row) is not FormalE5FailureCoverageExecutionRebuildRow
                for row in self.rows
            )
            or ids != tuple(sorted(set(ids)))
            or any(
                row.serving_execution.subject.protocol_lock_sha256
                != self.protocol_lock_sha256
                or row.serving_execution.subject.materialization_receipt_sha256
                != self.materialization_receipt_sha256
                or row.serving_execution.subject.inventory_sha256
                != self.inventory_sha256
                for row in self.rows
            )
        ):
            raise ValueError("E5 failure coverage rebuild rows are not exact")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "stage": self.stage,
            "phase": self.phase,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "rows": [row.to_dict() for row in self.rows],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("E5 failure coverage rebuild shard fields differ")
        row = dict(value)
        declared = _require_sha256(
            "E5 failure coverage rebuild shard", row.pop("shard_sha256")
        )
        raw_rows = row["rows"]
        if type(raw_rows) is not list:
            raise TypeError("E5 failure coverage rebuild rows must be an array")
        row["rows"] = tuple(
            FormalE5FailureCoverageExecutionRebuildRow.from_dict(item)
            for item in raw_rows
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("E5 failure coverage rebuild shard digest differs")
        return shard


def publish_formal_stage_coverage_evidence_shard(
    shard: FormalStageCoverageEvidenceShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not FormalStageCoverageEvidenceShard:
        raise TypeError("formal coverage evidence publisher requires an exact shard")
    publish_canonical_json_no_replace(output_path, shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def publish_formal_stage_coverage_execution_rebuild_shard(
    shard: FormalStageCoverageExecutionRebuildShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not FormalStageCoverageExecutionRebuildShard:
        raise TypeError("formal coverage rebuild publisher requires an exact shard")
    publish_canonical_json_no_replace(output_path, shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def publish_formal_e5_failure_coverage_evidence_shard(
    shard: FormalE5FailureCoverageEvidenceShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not FormalE5FailureCoverageEvidenceShard:
        raise TypeError("E5 failure coverage publisher requires an exact shard")
    publish_canonical_json_no_replace(output_path, shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def publish_formal_e5_failure_coverage_execution_rebuild_shard(
    shard: FormalE5FailureCoverageExecutionRebuildShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not FormalE5FailureCoverageExecutionRebuildShard:
        raise TypeError("E5 failure coverage rebuild publisher requires exact shard")
    publish_canonical_json_no_replace(output_path, shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def _validate_candidate_coverages(
    candidate_coverages: tuple[TtsL0CandidateStateCoverage, ...],
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    candidate_replay_proof_artifact_paths: tuple[str, ...],
    terminal_by_cell: dict[str, str],
    now_ns: int,
) -> None:
    if type(candidate_coverages) is not tuple or any(
        type(row) is not TtsL0CandidateStateCoverage for row in candidate_coverages
    ):
        raise TypeError("formal coverage candidate-state rows are not exact")
    if materialization.stage in TTS_L0_CANDIDATE_STATE_COVERAGE_STAGES:
        if not candidate_coverages:
            raise ValueError("formal coverage lacks required TTS/L0 candidate proofs")
    elif candidate_coverages:
        raise ValueError("formal coverage attaches candidate proofs to another stage")
    expected_pointer_sha256s = {
        digest
        for candidate in candidate_coverages
        for digest in (
            candidate.tts_native_replay_pointer_sha256,
            candidate.l0_naive_native_replay_pointer_sha256,
        )
    }
    if len(candidate_replay_proof_artifact_paths) != len(
        set(candidate_replay_proof_artifact_paths)
    ):
        raise ValueError("formal coverage candidate replay paths are duplicated")
    pointers = tuple(
        validate_candidate_state_replay_proof_artifact(
            path,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        for path in candidate_replay_proof_artifact_paths
    )
    by_sha = {row.semantic_commitment_sha256: row for row in pointers}
    if len(by_sha) != len(pointers) or set(by_sha) != expected_pointer_sha256s:
        raise ValueError("formal coverage candidate replay proof set is not exact")
    for candidate in candidate_coverages:
        candidate.validate_identity(
            stage=materialization.stage,
            protocol_lock_sha256=protocol_lock.sha256,
            materialization_receipt_sha256=materialization.sha256,
        )
        candidate.validate_native_replay_pointers(pointers)
        tts_terminals = {
            row.tts_terminal_receipt_sha256 for row in candidate.terminal_pairs
        }
        l0_terminals = {
            row.l0_naive_terminal_receipt_sha256 for row in candidate.terminal_pairs
        }
        if tts_terminals != {
            terminal_by_cell.get(candidate.tts_cell_id)
        } or l0_terminals != {terminal_by_cell.get(candidate.l0_naive_cell_id)}:
            raise ValueError("candidate replay terminal differs from serving proof")


def _derive_serving_terminals(
    *,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    stage: str,
    materialized_cells: tuple[MaterializedCell, ...],
    evidence_cells: tuple[FormalStageCoverageEvidenceCell, ...],
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    now_ns: int,
) -> dict[str, str]:
    ids = tuple(row.materialized_cell_id for row in evidence_cells)
    expected_ids = tuple(sorted(row.cell_id for row in materialized_cells))
    if (
        type(evidence_cells) is not tuple
        or any(
            type(row) is not FormalStageCoverageEvidenceCell for row in evidence_cells
        )
        or ids != tuple(sorted(set(ids)))
        or ids != expected_ids
    ):
        raise ValueError(
            "formal coverage evidence lacks every and only materialized cell"
        )
    if type(execution_bindings) is not tuple or len(execution_bindings) != len(
        evidence_cells
    ):
        raise ValueError("formal coverage execution binding count differs")
    bindings_by_sha: dict[str, VerifiedFormalServingExecutionBinding] = {}
    for binding in execution_bindings:
        verified = require_verified_formal_serving_execution_binding(binding)
        if verified.sha256 in bindings_by_sha:
            raise ValueError("formal coverage reuses an execution binding")
        bindings_by_sha[verified.sha256] = verified
    evidence_by_id = {row.materialized_cell_id: row for row in evidence_cells}
    terminal_by_cell: dict[str, str] = {}
    result_paths: set[str] = set()
    timing_paths: set[str] = set()
    for cell in sorted(materialized_cells, key=lambda row: row.cell_id):
        evidence = evidence_by_id[cell.cell_id]
        binding = bindings_by_sha.get(evidence.execution_binding_sha256)
        if binding is None:
            raise ValueError("formal coverage lacks its sealed execution binding")
        subject = binding.subject
        identity = evidence.execution_identity
        if (
            subject.stage != stage
            or subject.protocol_lock_sha256 != protocol_lock.sha256
            or subject.materialization_receipt_sha256 != materialization.sha256
            or subject.materialized_cell_id != cell.cell_id
            or subject.inventory_sha256 != inventory.sha256
            or subject.execution_identity != identity
            or identity.registry_sha256 != protocol_lock.registry_sha256
            or identity.inventory_sha256 != inventory.sha256
            or identity.method != _runtime_method(cell)
        ):
            raise ValueError("formal coverage execution identity is foreign")
        result_path = evidence.native_result_proof.absolute_path
        timing_path = evidence.stage_itl_proof.absolute_path
        if result_path in result_paths or timing_path in timing_paths:
            raise ValueError("formal coverage reuses a result or timing proof")
        result_paths.add(result_path)
        timing_paths.add(timing_path)
        result = validate_formal_terminal_result_proof_artifact(
            result_path,
            expected_inventory_sha256=inventory.sha256,
            expected_registry_sha256=protocol_lock.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            expected_execution_plan_sha256=identity.execution_plan_sha256,
            expected_rank_config_sha256=identity.rank_config_sha256,
            expected_run_id=identity.run_id,
            expected_run_nonce_sha256=identity.run_nonce_sha256,
            expected_attempt_id=identity.attempt_id,
            expected_method=identity.method,
            expected_stage=stage,
            expected_topology=subject.topology_mode,
            now_ns=now_ns,
        )
        timing = validate_formal_serving_itl_proof(
            timing_path,
            expected_registry_sha256=identity.registry_sha256,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        if (
            timing.execution_identity != identity
            or timing.native_result_proof_path != result_path
            or timing.native_result_proof_raw_sha256
            != evidence.native_result_proof.raw_sha256
            or timing.native_result_proof_semantic_sha256
            != evidence.native_result_proof.semantic_sha256
        ):
            raise ValueError("formal coverage terminal/ITL lineage differs")
        terminal_by_cell[cell.cell_id] = result.terminal_sha256
    if len(bindings_by_sha) != len(terminal_by_cell):
        raise ValueError("formal coverage includes a foreign execution binding")
    return terminal_by_cell


def reduce_e0_all_na_stage_coverage_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    registry_verification_receipt: object,
    signed_e6_confirmation: object,
    signed_compatibility: object,
    signed_final_completion: object,
    signed_formal_fdr: object,
    now_ns: int,
) -> StageCoverageReceipt:
    """Derive E0 V=0 coverage from signed 108-row N/A and formal FDR proofs."""

    from lightcone_spec.experiments.breadth_fdr_authority import (
        SignedE0FormalBreadthFdrReceipt,
        reduce_formal_e0_breadth_fdr_from_projection,
    )
    from lightcone_spec.experiments.e0_authority_artifact import (
        E0FinalAnalysisProjection,
        E0FinalCompletionReceipt,
        SignedE0FinalCompletionReceipt,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        SignedE6ConfirmationReceipt,
    )
    from lightcone_spec.experiments.formal_protocol import verify_signed_payload
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )
    from lightcone_spec.experiments.registry import build_industrial_registry
    from lightcone_spec.experiments.stage_materialization import (
        E0_ALL_NA_MATERIALIZATION_PROTOCOL_SHA256,
        SignedE0CompatibilityReceipt,
    )

    _validate_runtime_authority(protocol_lock, formal_runtime_authority_manifest)
    if type(inventory) is not GpuInventory:
        raise TypeError("E0 all-N/A coverage requires an exact inventory")
    if (
        type(registry_verification_receipt) is not FormalRegistryVerificationReceipt
        or type(signed_e6_confirmation) is not SignedE6ConfirmationReceipt
        or type(signed_compatibility) is not SignedE0CompatibilityReceipt
        or type(signed_final_completion) is not SignedE0FinalCompletionReceipt
        or type(signed_formal_fdr) is not SignedE0FormalBreadthFdrReceipt
    ):
        raise TypeError("E0 all-N/A coverage signed source union is not exact")
    if (
        materialization.stage != "E0"
        or materialization.materialization_rule != E0_ALL_NA_MATERIALIZATION_RULE
        or materialization.expected_cell_count != 0
        or materialization.cells
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
    ):
        raise ValueError("E0 all-N/A coverage materialization is not exact")
    manifest = registry_verification_receipt.revalidate(current_ns=now_ns)
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or registry_verification_receipt.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("E0 all-N/A coverage registry lineage differs")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    for signed in (
        signed_e6_confirmation,
        signed_compatibility,
        signed_final_completion,
        signed_formal_fdr,
    ):
        verify_signed_payload(
            signed.payload,
            payload_sha256=signed.payload_sha256,
            challenge=signed.challenge,
            attestation=signed.attestation,
            policy=policy,
            expected_policy_sha256=policy.sha256,
            now_ns=now_ns,
        )
    e6 = signed_e6_confirmation.payload
    compatibility = signed_compatibility.payload
    if (
        e6.status != "CONFIRMED"
        or e6.protocol_lock_sha256 != protocol_lock.sha256
        or e6.registry_sha256 != protocol_lock.registry_sha256
        or e6.materialization_receipt_sha256
        not in {row.materialization_receipt_sha256 for row in manifest.materializations}
        or e6.coverage_receipt_sha256
        not in {row.coverage_receipt_sha256 for row in manifest.coverage}
        or compatibility.protocol_lock_sha256 != protocol_lock.sha256
        or compatibility.upstream_e6_receipt_sha256 != e6.materialization_receipt_sha256
        or len(compatibility.decisions) != 108
        or compatibility.valid_count != 0
        or any(row.disposition != "N/A" for row in compatibility.decisions)
        or signed_compatibility.sha256
        not in {
            row.sha256
            for row in registry_verification_receipt.cumulative_signed_e0_compatibilities
        }
    ):
        raise ValueError("E0 all-N/A signed 108-row authority differs")
    expected_source = content_sha256(
        {
            "protocol_sha256": E0_ALL_NA_MATERIALIZATION_PROTOCOL_SHA256,
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_e0_compatibility_sha256": signed_compatibility.sha256,
        }
    )
    if (
        materialization.source_decision_sha256 != expected_source
        or materialization.upstream_receipt_sha256s
        != (e6.materialization_receipt_sha256,)
    ):
        raise ValueError("E0 all-N/A materialization source decision differs")
    coverage = StageCoverageReceipt(
        schema_version=2,
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=(),
        tts_l0_candidate_state_coverages=(),
    )
    coverage.validate_against(materialization)
    lineage = content_sha256(
        {
            "protocol_sha256": FORMAL_E0_ALL_NA_COVERAGE_PROTOCOL_SHA256,
            "registry_verification_receipt_sha256": (
                registry_verification_receipt.sha256
            ),
            "signed_e6_confirmation_sha256": signed_e6_confirmation.sha256,
            "signed_compatibility_sha256": signed_compatibility.sha256,
            "materialization_receipt_sha256": materialization.sha256,
        }
    )
    completion = E0FinalCompletionReceipt(
        schema_version=1,
        protocol_lock_sha256=protocol_lock.sha256,
        registry_sha256=protocol_lock.registry_sha256,
        prior_registry_verification_receipt_sha256=(
            registry_verification_receipt.sha256
        ),
        current_registry_verification_receipt_sha256=(
            registry_verification_receipt.sha256
        ),
        materialization_receipt_sha256=materialization.sha256,
        coverage_receipt_sha256=coverage.sha256,
        stage_source_binding_sha256=lineage,
        evidence_manifest_sha256=content_sha256(
            {"all_na_lineage_sha256": lineage, "terminal_count": 0}
        ),
        inventory_sha256=inventory.sha256,
        rebuild_artifact_sha256=content_sha256(
            {"all_na_lineage_sha256": lineage, "rebuild_kind": "zero_cell"}
        ),
        selected_final_prefix=(),
        valid_compatibility_count=0,
        cells=(),
        protocol_sha256=signed_final_completion.payload.protocol_sha256,
    )
    if signed_final_completion.payload != completion:
        raise ValueError("E0 all-N/A signed completion differs from proof reducer")
    registry = build_industrial_registry()
    if registry.sha256 != protocol_lock.registry_sha256:
        raise ValueError("E0 all-N/A registered hypothesis universe differs")
    projection = E0FinalAnalysisProjection(
        schema_version=1,
        completion_receipt=completion,
        compatibility_decisions=compatibility.decisions,
        cells=(),
    )
    fdr = reduce_formal_e0_breadth_fdr_from_projection(registry, projection)
    if (
        signed_formal_fdr.payload != fdr
        or len(fdr.hypotheses) != 756
        or any(row.status != "EXCLUDED_NA" for row in fdr.hypotheses)
        or fdr.decisions
    ):
        raise ValueError("E0 all-N/A formal FDR projection differs")
    return coverage


def reduce_formal_serving_stage_coverage_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    stage: str,
    phase: str,
    evidence_cells: tuple[FormalStageCoverageEvidenceCell, ...],
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    candidate_coverages: tuple[TtsL0CandidateStateCoverage, ...] = (),
    candidate_replay_proof_artifact_paths: tuple[str, ...] = (),
    e0_valid_compatibility_decision_ids: tuple[str, ...] | None = None,
    now_ns: int,
) -> StageCoverageReceipt:
    """Derive COMPLETE dispositions from exact serving result/ITL proofs."""

    _validate_runtime_authority(protocol_lock, formal_runtime_authority_manifest)
    if type(inventory) is not GpuInventory:
        raise TypeError("formal coverage requires an exact GPU inventory")
    expected_stage, expected_phase = _phase_for_materialization(materialization)
    if (stage, phase) != (expected_stage, expected_phase):
        raise ValueError("formal coverage stage/phase differs from materialization")
    _require_phase(stage, phase)
    _validate_materialization_cardinality(
        materialization,
        phase=phase,
        e0_valid_compatibility_decision_ids=(e0_valid_compatibility_decision_ids),
    )
    if stage == "TTS-Cal":
        raise ValueError("TTS-Cal requires its raw non-serving coverage reducer")
    if stage == "E4" and phase == "profiler":
        raise ValueError(
            "E4 profiler coverage requires its dedicated profiler-proof reducer"
        )
    if stage == "E5" and phase == "final_and_one_shot_failure":
        raise ValueError("E5 final coverage requires its closed failure-proof reducer")
    if stage == "E6":
        raise ValueError(
            "E6 coverage requires its closed model-compatibility and serving reducer"
        )
    terminal_by_cell = _derive_serving_terminals(
        protocol_lock=protocol_lock,
        materialization=materialization,
        inventory=inventory,
        stage=stage,
        materialized_cells=materialization.cells,
        evidence_cells=evidence_cells,
        execution_bindings=execution_bindings,
        now_ns=now_ns,
    )
    _validate_candidate_coverages(
        candidate_coverages,
        protocol_lock=protocol_lock,
        materialization=materialization,
        inventory=inventory,
        candidate_replay_proof_artifact_paths=(candidate_replay_proof_artifact_paths),
        terminal_by_cell=terminal_by_cell,
        now_ns=now_ns,
    )
    receipt = StageCoverageReceipt(
        schema_version=2,
        stage=stage,
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage=stage,
                cell_id=cell_id,
                status="COMPLETE",
                reason_code="proof_derived_terminal_complete",
                terminal_receipt_sha256=terminal_by_cell[cell_id],
            )
            for cell_id in sorted(terminal_by_cell)
        ),
        tts_l0_candidate_state_coverages=candidate_coverages,
    )
    receipt.validate_against(materialization)
    return receipt


def reduce_e6_stage_coverage_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    phase: str,
    signed_model_compatibility: object,
    compatibility_sources: tuple[object, ...],
    registry_verification_receipt: object,
    stage_source: VerifiedFormalStageMaterializationSource,
    serving_evidence_cells: tuple[FormalStageCoverageEvidenceCell, ...],
    serving_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    candidate_coverages: tuple[TtsL0CandidateStateCoverage, ...] = (),
    candidate_replay_proof_artifact_paths: tuple[str, ...] = (),
    now_ns: int,
) -> StageCoverageReceipt:
    """Join two signed NEXTN preflights with only the true serving rows."""

    from lightcone_spec.experiments.e6_stage_authority import (
        E6NextnModelAuthorityInput,
        SignedE6ModelCompatibilityReceipt,
    )
    from lightcone_spec.experiments.formal_registry import (
        FormalRegistryVerificationReceipt,
    )

    _validate_runtime_authority(protocol_lock, formal_runtime_authority_manifest)
    if type(inventory) is not GpuInventory:
        raise TypeError("E6 coverage requires an exact GPU inventory")
    if _phase_for_materialization(materialization) != ("E6", phase) or phase not in {
        "excluded_pilot_and_model_preflight",
        "final",
    }:
        raise ValueError("E6 coverage materialization phase differs")
    _validate_materialization_cardinality(materialization, phase=phase)
    if type(signed_model_compatibility) is not SignedE6ModelCompatibilityReceipt:
        raise TypeError("E6 coverage requires signed model compatibility")
    if type(compatibility_sources) is not tuple or any(
        type(row) is not E6NextnModelAuthorityInput for row in compatibility_sources
    ):
        raise TypeError("E6 coverage compatibility sources are not exact")
    if type(registry_verification_receipt) is not FormalRegistryVerificationReceipt:
        raise TypeError("E6 coverage requires a durable formal registry")
    registry_verification_receipt.revalidate(current_ns=now_ns)
    verified_stage_source = require_verified_formal_stage_materialization_source(
        stage_source,
        materialization=materialization,
    )
    expected_source_authorities = {
        signed_model_compatibility.sha256,
        *(row.sha256 for row in compatibility_sources),
    }
    if (
        registry_verification_receipt.signed_protocol_lock.payload != protocol_lock
        or registry_verification_receipt.inventory_sha256 != inventory.sha256
        or verified_stage_source.phase != phase
        or not expected_source_authorities.issubset(
            verified_stage_source.typed_source_authority_sha256s
        )
    ):
        raise ValueError("E6 coverage lacks its registry-rooted compatibility source")
    policy = registry_verification_receipt.trusted_release_policy(current_ns=now_ns)
    compatibility = signed_model_compatibility.verify(
        protocol_lock=protocol_lock,
        sources=compatibility_sources,  # type: ignore[arg-type]
        expected_inventory_sha256=inventory.sha256,
        policy=policy,
        expected_policy_sha256=policy.sha256,
        now_ns=now_ns,
    )
    compatibility_by_model = {row.model: row for row in compatibility.models}
    if tuple(compatibility_by_model) != E6_MODELS:
        raise ValueError("E6 coverage compatibility panel differs")
    preflight_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "immutable_metadata_interface_and_fit_preflight"
    )
    serving_cells = tuple(
        cell for cell in materialization.cells if cell.task in E6_TASKS
    )
    if (
        tuple(cell.model for cell in preflight_cells) != E6_MODELS
        or len(preflight_cells) != 2
        or len(preflight_cells) + len(serving_cells) != len(materialization.cells)
    ):
        raise ValueError("E6 coverage lacks its exact preflight/serving partition")
    terminal_by_cell = _derive_serving_terminals(
        protocol_lock=protocol_lock,
        materialization=materialization,
        inventory=inventory,
        stage="E6",
        materialized_cells=serving_cells,
        evidence_cells=serving_evidence_cells,
        execution_bindings=serving_execution_bindings,
        now_ns=now_ns,
    )
    for cell in preflight_cells:
        compatibility_row = compatibility_by_model[cell.model]
        dimensions = dict(cell.dimensions)
        if (
            cell.method_role != "Target-only"
            or cell.backend != "NEXTN"
            or cell.publication_policy != "none"
            or cell.recipe_sha256 is not None
            or dimensions.get("topology") != "tp2_dp1"
            or dimensions.get("e6_model_compatibility_row_sha256")
            != compatibility_row.sha256
            or dimensions.get("e6_verified_authority_sha256")
            != compatibility_row.verified_authority_sha256
            or dimensions.get("inventory_sha256") != inventory.sha256
            or dimensions.get("target_member_id") != compatibility_row.target_member_id
            or dimensions.get("drafter_member_id")
            != compatibility_row.drafter_member_id
            or dimensions.get("interface_sha256") != compatibility_row.interface_sha256
            or dimensions.get("signed_e6_model_compatibility_sha256")
            != signed_model_compatibility.sha256
        ):
            raise ValueError("E6 preflight differs from signed NEXTN compatibility")
        terminal_by_cell[cell.cell_id] = compatibility_row.verified_authority_sha256
    if len(terminal_by_cell) != len(materialization.cells):
        raise ValueError("E6 closed coverage union is incomplete")
    _validate_candidate_coverages(
        candidate_coverages,
        protocol_lock=protocol_lock,
        materialization=materialization,
        inventory=inventory,
        candidate_replay_proof_artifact_paths=(candidate_replay_proof_artifact_paths),
        terminal_by_cell=terminal_by_cell,
        now_ns=now_ns,
    )
    preflight_ids = {cell.cell_id for cell in preflight_cells}
    receipt = StageCoverageReceipt(
        schema_version=2,
        stage="E6",
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="E6",
                cell_id=cell_id,
                status="COMPLETE",
                reason_code=(
                    "proof_derived_e6_model_compatibility_complete"
                    if cell_id in preflight_ids
                    else "proof_derived_terminal_complete"
                ),
                terminal_receipt_sha256=terminal_by_cell[cell_id],
            )
            for cell_id in sorted(terminal_by_cell)
        ),
        tts_l0_candidate_state_coverages=candidate_coverages,
    )
    receipt.validate_against(materialization)
    return receipt


def reduce_e5_final_stage_coverage_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    headline_evidence_cells: tuple[FormalStageCoverageEvidenceCell, ...],
    headline_execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...],
    failure_evidence_cells: tuple[FormalE5FailureCoverageEvidenceCell, ...],
    failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...],
    candidate_coverages: tuple[TtsL0CandidateStateCoverage, ...],
    candidate_replay_proof_artifact_paths: tuple[str, ...],
    now_ns: int,
) -> StageCoverageReceipt:
    """Derive the E5 450N headline + exact 264 one-shot failure coverage."""

    _validate_runtime_authority(protocol_lock, formal_runtime_authority_manifest)
    if type(inventory) is not GpuInventory:
        raise TypeError("E5 final coverage requires an exact GPU inventory")
    if _phase_for_materialization(materialization) != (
        "E5",
        "final_and_one_shot_failure",
    ):
        raise ValueError("E5 final coverage materialization identity differs")
    _validate_materialization_cardinality(
        materialization, phase="final_and_one_shot_failure"
    )
    headline_cells = tuple(
        row
        for row in materialization.cells
        if row.task == "production_slo_power_prefix"
    )
    failure_cells = tuple(
        row
        for row in materialization.cells
        if row.task == "deterministic_failure_injection"
    )
    if len(failure_cells) != 264 or len(headline_cells) + len(failure_cells) != len(
        materialization.cells
    ):
        raise ValueError("E5 final coverage lacks the exact closed cell union")
    terminal_by_cell = _derive_serving_terminals(
        protocol_lock=protocol_lock,
        materialization=materialization,
        inventory=inventory,
        stage="E5",
        materialized_cells=headline_cells,
        evidence_cells=headline_evidence_cells,
        execution_bindings=headline_execution_bindings,
        now_ns=now_ns,
    )
    failure_ids = tuple(row.materialized_cell_id for row in failure_evidence_cells)
    expected_failure_ids = tuple(sorted(row.cell_id for row in failure_cells))
    if (
        type(failure_evidence_cells) is not tuple
        or any(
            type(row) is not FormalE5FailureCoverageEvidenceCell
            for row in failure_evidence_cells
        )
        or failure_ids != tuple(sorted(set(failure_ids)))
        or failure_ids != expected_failure_ids
    ):
        raise ValueError("E5 failure evidence lacks every and only 264 cells")
    bindings_by_cell: dict[str, VerifiedFormalFailureExecutionBinding] = {}
    if type(failure_execution_bindings) is not tuple:
        raise TypeError("E5 failure coverage bindings must be an exact tuple")
    for binding in failure_execution_bindings:
        verified = require_verified_formal_failure_execution_binding(binding)
        cell_id = verified.subject.materialized_cell_id
        if cell_id in bindings_by_cell:
            raise ValueError("E5 failure coverage reuses an execution binding")
        bindings_by_cell[cell_id] = verified
    if set(bindings_by_cell) != set(expected_failure_ids):
        raise ValueError("E5 failure execution binding coverage is not exact")
    evidence_by_cell = {row.materialized_cell_id: row for row in failure_evidence_cells}
    proof_paths: set[str] = set()
    for cell in sorted(failure_cells, key=lambda row: row.cell_id):
        evidence = evidence_by_cell[cell.cell_id]
        binding = bindings_by_cell[cell.cell_id]
        subject = binding.subject
        dimensions = dict(cell.dimensions)
        if (
            subject.protocol_lock_sha256 != protocol_lock.sha256
            or subject.materialization_receipt_sha256 != materialization.sha256
            or subject.inventory_sha256 != inventory.sha256
            or subject.registry_sha256 != protocol_lock.registry_sha256
            or evidence.failure_execution_binding_sha256 != binding.sha256
            or evidence.assignment_sha256 != subject.assignment_sha256
            or evidence.serving_execution_plan_sha256
            != subject.serving_execution_plan_sha256
            or subject.scenario != dimensions.get("failure")
            or subject.topology != dimensions.get("topology")
            or subject.cohort_count != dimensions.get("cohort_count")
        ):
            raise ValueError("E5 failure evidence differs from materialized assignment")
        paths = {
            evidence.plan.absolute_path,
            evidence.actuation_proof.absolute_path,
            evidence.lifecycle_proof.absolute_path,
        }
        if len(paths) != 3 or proof_paths.intersection(paths):
            raise ValueError("E5 failure coverage reuses plan/proof path")
        proof_paths.update(paths)
        actuation = validate_formal_failure_actuation_proof_artifact(
            evidence.actuation_proof.absolute_path,
            binding=binding,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        lifecycle = validate_formal_e5_failure_lifecycle_proof_artifact(
            evidence.lifecycle_proof.absolute_path,
            plan_path=evidence.plan.absolute_path,
            execution_binding=binding.serving_execution,
            failure_execution_binding=binding,
            expected_root_manifest_sha256=(
                protocol_lock.offline_release_trust_root_sha256
            ),
            now_ns=now_ns,
        )
        actuation_artifact = FormalFailureActuationProofArtifact.from_dict(
            evidence.actuation_proof.reopen()
        )
        if (
            actuation.assignment_sha256 != subject.assignment_sha256
            or actuation.cell_id != cell.cell_id
            or actuation.inventory_sha256 != inventory.sha256
            or actuation.registry_sha256 != protocol_lock.registry_sha256
            or actuation.scenario != subject.scenario
            or actuation.correctness_only is not True
            or lifecycle.formal_failure_execution_binding_sha256 != binding.sha256
            or lifecycle.raw_failure_terminal != actuation_artifact.raw_terminal
            or lifecycle.run_nonce_sha256 != subject.run_nonce_sha256
            or lifecycle.inventory_sha256 != inventory.sha256
            or lifecycle.registry_sha256 != protocol_lock.registry_sha256
            or lifecycle.topology_mode != subject.topology
            or lifecycle.gpu_uuids != binding.serving_execution.subject.gpu_uuids
        ):
            raise ValueError("E5 actuation/lifecycle proof join differs")
        terminal_by_cell[cell.cell_id] = actuation.sha256
    if len(terminal_by_cell) != len(materialization.cells):
        raise ValueError("E5 final coverage closed proof union is incomplete")
    _validate_candidate_coverages(
        candidate_coverages,
        protocol_lock=protocol_lock,
        materialization=materialization,
        inventory=inventory,
        candidate_replay_proof_artifact_paths=(candidate_replay_proof_artifact_paths),
        terminal_by_cell=terminal_by_cell,
        now_ns=now_ns,
    )
    receipt = StageCoverageReceipt(
        schema_version=2,
        stage="E5",
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="E5",
                cell_id=cell_id,
                status="COMPLETE",
                reason_code=(
                    "proof_derived_failure_actuation_and_lifecycle_complete"
                    if cell_id in evidence_by_cell
                    else "proof_derived_terminal_complete"
                ),
                terminal_receipt_sha256=terminal_by_cell[cell_id],
            )
            for cell_id in sorted(terminal_by_cell)
        ),
        tts_l0_candidate_state_coverages=candidate_coverages,
    )
    receipt.validate_against(materialization)
    return receipt


def _load_bound_json(reference: BoundArtifact, *, label: str) -> dict[str, object]:
    if type(reference) is not BoundArtifact:
        raise TypeError(f"{label} requires an exact BoundArtifact")
    return _industrial_bound_json(reference.path, reference.sha256, label=label)


def _validate_tts_qualification_lock(
    reference: BoundArtifact,
    *,
    block: int,
    registry_sha256: str,
    protocol_lock: ProtocolLock,
    materialization: StageMaterializationReceipt,
    authority: TtsCalibrationAuthority,
    manifest: RawTtsCalibrationEvidenceManifest,
) -> None:
    value = _load_bound_json(reference, label="TTS qualification lock")
    required = {
        "schema_version",
        "kind",
        "registry_sha256",
        "protocol_lock_sha256",
        "materialization_receipt_sha256",
        "authority_sha256",
        "tuning_window_sha256",
        "runtime_sha256",
        "split_sha256",
        "block",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
        "rows",
    }
    if set(value) != required or any(
        value.get(name) != expected
        for name, expected in (
            ("schema_version", 1),
            ("kind", "tts_calibration_request_qualification_lock"),
            ("registry_sha256", registry_sha256),
            ("protocol_lock_sha256", protocol_lock.sha256),
            ("materialization_receipt_sha256", materialization.sha256),
            ("authority_sha256", authority.sha256),
            ("tuning_window_sha256", manifest.tuning_window.sha256),
            ("block", block),
        )
    ):
        raise ValueError("TTS qualification lock differs from exact stage lineage")
    for name in (
        "runtime_sha256",
        "split_sha256",
        "corpus_sha256",
        "arrival_trace_sha256",
        "request_ids_sha256",
        "sampling_profile_sha256",
        "model_lock_sha256",
    ):
        _require_sha256(f"TTS qualification {name}", value[name])
    rows = value["rows"]
    if type(rows) is not list or not rows:
        raise ValueError("TTS qualification lock requires request rows")
    request_ids: list[str] = []
    for row in rows:
        if type(row) is not dict or set(row) != {
            "request_id",
            "prompt_bucket",
            "eligible",
        }:
            raise ValueError("TTS qualification row schema differs")
        if (
            type(row["request_id"]) is not str
            or not row["request_id"]
            or row["prompt_bucket"] not in TTFT_LIMIT_MS
            or type(row["eligible"]) is not bool
        ):
            raise ValueError("TTS qualification row is incomplete")
        request_ids.append(row["request_id"])
    if (
        len(request_ids) != len(set(request_ids))
        or content_sha256(request_ids) != value["request_ids_sha256"]
    ):
        raise ValueError("TTS qualification request coverage differs")


def _require_tts_calibration_request_reset_evidence(evidence: object) -> None:
    """Require the deep-validated current request-reset contract for TTS-Cal.

    Zero round/update rows are valid for short or aborted requests.  Coverage
    is therefore joined to submitted server lifecycle rows through reset
    receipts instead of manufacturing adaptation rows.
    """

    binding = getattr(evidence, "binding", None)
    reset_receipt = getattr(evidence, "reset_receipt", None)
    scored_resets = getattr(evidence, "request_source_point_resets", None)
    warmup_resets = getattr(reset_receipt, "warmup_request_source_point_resets", None)
    if (
        getattr(evidence, "terminal_schema_version", None) != 2
        or getattr(binding, "method", None) != "tts"
        or getattr(binding, "reset_scope", None) != "request"
        or getattr(binding, "request_admission_policy", None)
        != "serialized_native_scheduler_v1"
        or getattr(
            reset_receipt,
            "request_source_point_reset_protocol_sha256",
            None,
        )
        != REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
    ):
        raise ValueError("TTS coverage requires current request-reset identity")
    for phase, resets, requests in (
        (
            "warmup",
            warmup_resets,
            getattr(reset_receipt, "warmup_requests", None),
        ),
        ("scored", scored_resets, getattr(evidence, "requests", None)),
    ):
        if (
            resets is None
            or getattr(resets, "reset_scope", None) != "request"
            or getattr(resets, "request_admission_policy", None)
            != "serialized_native_scheduler_v1"
            or getattr(resets, "protocol_sha256", None)
            != REQUEST_SOURCE_POINT_RESET_PROTOCOL_SHA256
            or type(requests) is not tuple
            or type(getattr(resets, "receipts", None)) is not tuple
        ):
            raise ValueError(f"TTS coverage {phase} reset envelope differs")
        submitted_ids = tuple(
            request.request_id for request in requests if request.submitted_to_server
        )
        receipt_ids = tuple(receipt.request_id for receipt in resets.receipts)
        # Receipt order is the native acquire/terminal order (and therefore
        # carries epoch semantics); it need not equal caller submission order.
        # The deep terminal validator already proves both sequences unique.
        if len(receipt_ids) != len(submitted_ids) or set(receipt_ids) != set(
            submitted_ids
        ):
            raise ValueError(f"TTS coverage {phase} reset coverage differs")


def reduce_tts_calibration_stage_coverage_from_proofs(
    *,
    protocol_lock: ProtocolLock,
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    authority: TtsCalibrationAuthority,
    manifest: RawTtsCalibrationEvidenceManifest,
    now_ns: int,
) -> StageCoverageReceipt:
    """Reconstruct TTS-Cal coverage without consuming control challenges again."""

    _validate_runtime_authority(protocol_lock, formal_runtime_authority_manifest)
    if type(inventory) is not GpuInventory:
        raise TypeError("TTS coverage requires an exact GPU inventory")
    if type(authority) is not TtsCalibrationAuthority:
        raise TypeError("TTS coverage requires an exact calibration authority")
    if type(manifest) is not RawTtsCalibrationEvidenceManifest:
        raise TypeError("TTS coverage requires an exact raw manifest")
    if (
        _phase_for_materialization(materialization) != ("TTS-Cal", "calibration")
        or materialization.protocol_lock_sha256 != protocol_lock.sha256
        or materialization.source_decision_sha256 != authority.sha256
        or protocol_lock.tts_calibration_authority_sha256 != authority.sha256
        or manifest.tuning_window.sha256 != authority.tuning_window_sha256
        or materialization.expected_cell_count != 288
    ):
        raise ValueError("TTS coverage identity differs from calibration authority")
    _validate_materialization_cardinality(materialization, phase="calibration")
    validate_raw_evidence_manifest_sidecars(manifest)
    from lightcone_spec.experiments.registry import build_industrial_registry

    registry = build_industrial_registry()
    if registry.sha256 != protocol_lock.registry_sha256:
        raise ValueError("TTS coverage registry differs from ProtocolLock")
    registry_cells = {row.cell_id: row for row in registry.cells_for("TTS-Cal")}
    materialized_by_registry: dict[str, MaterializedCell] = {}
    for cell in materialization.cells:
        registry_cell_id = dict(cell.dimensions).get("registry_cell_id")
        _require_sha256("TTS materialized registry cell", registry_cell_id)
        if registry_cell_id in materialized_by_registry:
            raise ValueError("TTS materialization repeats a registry cell")
        materialized_by_registry[str(registry_cell_id)] = cell
    manifest_by_registry = {row.cell_id: row for row in manifest.cells}
    if (
        len(registry_cells) != 288
        or set(materialized_by_registry) != set(registry_cells)
        or set(manifest_by_registry) != set(registry_cells)
        or len(manifest.cells) != 288
    ):
        raise ValueError("TTS coverage cell universe is missing or foreign")
    terminal_by_cell: dict[str, str] = {}
    challenges: set[str] = set()
    deployment_challenges: set[str] = set()
    run_identities: set[tuple[str, str]] = set()
    control_paths: set[Path] = set()
    terminal_paths: set[Path] = set()
    for pilot in manifest.pilots:
        _validate_tts_qualification_lock(
            pilot.qualification_lock,
            block=pilot.block,
            registry_sha256=registry.sha256,
            protocol_lock=protocol_lock,
            materialization=materialization,
            authority=authority,
            manifest=manifest,
        )
        pilot_ids = {row.cell_id for row in pilot.cells}
        expected_pilot_ids = {
            row.cell_id
            for row in registry_cells.values()
            if row.identity.block == pilot.block
        }
        if pilot_ids != expected_pilot_ids or len(pilot.cells) != 72:
            raise ValueError("TTS coverage pilot lacks its exact 72 cells")
        for cell, control_reference in zip(
            pilot.cells,
            pilot.terminal_control_attestations,
            strict=True,
        ):
            if len(cell.terminal_receipts) != 1:
                raise ValueError("TTS coverage requires one TP1 terminal per cell")
            terminal_reference = cell.terminal_receipts[0]
            if (
                terminal_reference.path in terminal_paths
                or control_reference.path in control_paths
            ):
                raise ValueError("TTS coverage reuses terminal/control path")
            terminal_paths.add(terminal_reference.path)
            control_paths.add(control_reference.path)
            terminal_value = _load_bound_json(
                terminal_reference, label="TTS native terminal"
            )
            control_value = _load_bound_json(
                control_reference, label="TTS terminal external control"
            )
            control = ControlArtifactAttestation.from_dict(control_value)
            prepared = prepare_native_terminal_external_control(
                terminal_value,
                control_attestation=control,
                expected_inventory_sha256=inventory.sha256,
                expected_registry_sha256=registry.sha256,
            )
            _require_tts_calibration_request_reset_evidence(prepared.evidence)
            verified_control = verify_release_control_artifact_attestation(
                control,
                expected_inventory_sha256=inventory.sha256,
                now_ns=now_ns,
                consumed_challenge_sha256s=(),
            )
            if (
                control.deployment_policy_authorization.root_manifest_sha256
                != protocol_lock.offline_release_trust_root_sha256
                or prepared.binding.canonical_raw_sha256 != terminal_reference.sha256
                or verified_control.artifact_sha256 != prepared.binding.sha256
                or prepared.evidence.binding.method != "tts"
            ):
                raise ValueError("TTS terminal control/proof lineage differs")
            run_identity = (
                prepared.evidence.binding.run_id,
                prepared.evidence.binding.run_nonce_sha256,
            )
            if (
                verified_control.challenge_sha256 in challenges
                or verified_control.deployment_policy_challenge_sha256
                in deployment_challenges
                or run_identity in run_identities
            ):
                raise ValueError("TTS coverage reuses challenge or native run")
            challenges.add(verified_control.challenge_sha256)
            deployment_challenges.add(
                verified_control.deployment_policy_challenge_sha256
            )
            run_identities.add(run_identity)
            materialized = materialized_by_registry[cell.cell_id]
            registry_cell = registry_cells[cell.cell_id]
            dimensions = dict(materialized.dimensions)
            learning_rate = registry_cell.identity.learning_rate
            stride = int(
                registry_cell.identity.variant.removeprefix("tts_calibration:stride=")
            )
            if (
                type(learning_rate) is not float
                or dimensions.get("block") != pilot.block
                or dimensions.get("learning_rate") != learning_rate
                or dimensions.get("stride") != stride
                or materialized.recipe_sha256
                != authority.candidate_id(learning_rate=learning_rate, stride=stride)
            ):
                raise ValueError("TTS materialized candidate differs from registry")
            terminal_by_cell[materialized.cell_id] = terminal_reference.sha256
    if len(terminal_by_cell) != 288:
        raise ValueError("TTS coverage terminal proof set is incomplete")
    receipt = StageCoverageReceipt(
        schema_version=2,
        stage="TTS-Cal",
        protocol_lock_sha256=protocol_lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        dispositions=tuple(
            StageCellDisposition(
                stage="TTS-Cal",
                cell_id=cell_id,
                status="COMPLETE",
                reason_code="proof_derived_external_control_terminal_complete",
                terminal_receipt_sha256=terminal_by_cell[cell_id],
            )
            for cell_id in sorted(terminal_by_cell)
        ),
    )
    receipt.validate_against(materialization)
    return receipt


@dataclass(frozen=True)
class FormalStageDerivedCoverageShard:
    """Bounded durable projection of one reducer-derived coverage receipt."""

    schema_version: Literal[1]
    kind: Literal["formal_stage_derived_coverage_shard"]
    protocol_sha256: str
    stage: str
    phase: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    coverage_receipt_sha256: str
    shard_index: int
    shard_count: int
    dispositions: tuple[StageCellDisposition, ...]
    candidate_coverages: tuple[TtsL0CandidateStateCoverage, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_stage_derived_coverage_shard"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
        ):
            raise ValueError("derived coverage shard identity differs")
        _require_phase(self.stage, self.phase)
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("coverage", self.coverage_receipt_sha256),
        ):
            _require_sha256(f"derived coverage {label}", digest)
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count < 1
            or self.shard_index not in range(self.shard_count)
            or type(self.dispositions) is not tuple
            or any(type(row) is not StageCellDisposition for row in self.dispositions)
            or tuple(row.cell_id for row in self.dispositions)
            != tuple(sorted({row.cell_id for row in self.dispositions}))
            or type(self.candidate_coverages) is not tuple
            or any(
                type(row) is not TtsL0CandidateStateCoverage
                for row in self.candidate_coverages
            )
            or tuple(row.pair_id for row in self.candidate_coverages)
            != tuple(sorted({row.pair_id for row in self.candidate_coverages}))
        ):
            raise ValueError("derived coverage shard rows are not canonical")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        from lightcone_spec.experiments.formal_registry import (
            tts_l0_candidate_state_coverage_to_dict,
        )

        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "phase": self.phase,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "dispositions": [asdict(row) for row in self.dispositions],
            "candidate_coverages": [
                tts_l0_candidate_state_coverage_to_dict(row)
                for row in self.candidate_coverages
            ],
        }
        if include_sha256:
            value["shard_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        from lightcone_spec.experiments.formal_registry import (
            tts_l0_candidate_state_coverage_from_dict,
        )

        fields = {*cls.__dataclass_fields__, "shard_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("derived coverage shard fields differ")
        row = dict(value)
        declared = _require_sha256("derived coverage shard", row.pop("shard_sha256"))
        raw_dispositions = row["dispositions"]
        raw_candidates = row["candidate_coverages"]
        if type(raw_dispositions) is not list or type(raw_candidates) is not list:
            raise TypeError("derived coverage shard rows must be arrays")
        row["dispositions"] = tuple(
            StageCellDisposition(**item)
            if type(item) is dict
            and set(item) == set(StageCellDisposition.__dataclass_fields__)
            else (_ for _ in ()).throw(
                ValueError("derived coverage disposition fields differ")
            )
            for item in raw_dispositions
        )
        row["candidate_coverages"] = tuple(
            tts_l0_candidate_state_coverage_from_dict(item) for item in raw_candidates
        )
        shard = cls(**row)  # type: ignore[arg-type]
        if shard.sha256 != declared:
            raise ValueError("derived coverage shard digest differs")
        return shard


def derived_coverage_shards(
    coverage: StageCoverageReceipt,
    *,
    phase: str,
    maximum_dispositions_per_shard: int = 512,
) -> tuple[FormalStageDerivedCoverageShard, ...]:
    if type(coverage) is not StageCoverageReceipt:
        raise TypeError("derived coverage sharding requires an exact receipt")
    _require_phase(coverage.stage, phase)
    if (
        type(maximum_dispositions_per_shard) is not int
        or maximum_dispositions_per_shard < 1
    ):
        raise ValueError("derived coverage shard size must be positive")
    chunks = tuple(
        coverage.dispositions[index : index + maximum_dispositions_per_shard]
        for index in range(
            0, len(coverage.dispositions), maximum_dispositions_per_shard
        )
    ) or ((),)
    pair_chunks: list[list[TtsL0CandidateStateCoverage]] = [[] for _chunk in chunks]
    pair_chunks[0].extend(coverage.tts_l0_candidate_state_coverages)
    return tuple(
        FormalStageDerivedCoverageShard(
            schema_version=1,
            kind="formal_stage_derived_coverage_shard",
            protocol_sha256=FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
            stage=coverage.stage,
            phase=phase,
            protocol_lock_sha256=coverage.protocol_lock_sha256,
            materialization_receipt_sha256=(coverage.materialization_receipt_sha256),
            coverage_receipt_sha256=coverage.sha256,
            shard_index=index,
            shard_count=len(chunks),
            dispositions=chunk,
            candidate_coverages=tuple(
                sorted(pair_chunks[index], key=lambda row: row.pair_id)
            ),
        )
        for index, chunk in enumerate(chunks)
    )


def publish_formal_stage_derived_coverage_shard(
    shard: FormalStageDerivedCoverageShard,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(shard) is not FormalStageDerivedCoverageShard:
        raise TypeError("derived coverage publisher requires an exact shard")
    publish_canonical_json_no_replace(output_path, shard.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


@dataclass(frozen=True)
class FormalE0AllNaCoverageAuthorityArtifact:
    """Closed, path-bound source union for E0's legitimate zero-row branch."""

    schema_version: Literal[1]
    kind: Literal["formal_e0_all_na_coverage_authority_artifact"]
    protocol_sha256: str
    protocol_lock_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    registry_verification_receipt_source: CanonicalJsonProofBinding
    signed_e6_confirmation_source: CanonicalJsonProofBinding
    signed_compatibility_source: CanonicalJsonProofBinding
    signed_final_completion_source: CanonicalJsonProofBinding
    signed_formal_fdr_source: CanonicalJsonProofBinding

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_e0_all_na_coverage_authority_artifact"
            or self.protocol_sha256 != FORMAL_E0_ALL_NA_COVERAGE_PROTOCOL_SHA256
        ):
            raise ValueError("E0 all-N/A coverage authority identity differs")
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
        ):
            _require_sha256(f"E0 all-N/A coverage {label}", digest)
        sources = (
            self.registry_verification_receipt_source,
            self.signed_e6_confirmation_source,
            self.signed_compatibility_source,
            self.signed_final_completion_source,
            self.signed_formal_fdr_source,
        )
        if any(type(row) is not CanonicalJsonProofBinding for row in sources):
            raise TypeError("E0 all-N/A coverage sources must be path-bound")
        paths = tuple(row.absolute_path for row in sources)
        if len(paths) != len(set(paths)):
            raise ValueError("E0 all-N/A coverage source paths alias")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "materialization_receipt_sha256": (self.materialization_receipt_sha256),
            "inventory_sha256": self.inventory_sha256,
            "registry_verification_receipt_source": (
                self.registry_verification_receipt_source.to_dict()
            ),
            "signed_e6_confirmation_source": (
                self.signed_e6_confirmation_source.to_dict()
            ),
            "signed_compatibility_source": (self.signed_compatibility_source.to_dict()),
            "signed_final_completion_source": (
                self.signed_final_completion_source.to_dict()
            ),
            "signed_formal_fdr_source": self.signed_formal_fdr_source.to_dict(),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("E0 all-N/A coverage authority fields differ")
        row = dict(value)
        declared = _require_sha256(
            "E0 all-N/A coverage authority", row.pop("artifact_sha256")
        )
        for name in (
            "registry_verification_receipt_source",
            "signed_e6_confirmation_source",
            "signed_compatibility_source",
            "signed_final_completion_source",
            "signed_formal_fdr_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("E0 all-N/A coverage authority digest differs")
        return artifact


def publish_formal_e0_all_na_coverage_authority_artifact(
    artifact: FormalE0AllNaCoverageAuthorityArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalE0AllNaCoverageAuthorityArtifact:
        raise TypeError("E0 all-N/A coverage publisher requires exact authority")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


@dataclass(frozen=True)
class FormalStageCoverageProofArtifact:
    """Small path graph that makes one coverage receipt independently replayable."""

    schema_version: Literal[1]
    kind: Literal["formal_stage_coverage_proof_artifact"]
    protocol_sha256: str
    stage: str
    phase: str
    protocol_lock_sha256: str
    formal_runtime_authority_manifest_sha256: str
    materialization_receipt_sha256: str
    inventory_sha256: str
    coverage_receipt_sha256: str
    protocol_lock_source: CanonicalJsonProofBinding
    runtime_authority_source: CanonicalJsonProofBinding
    materialization_source: CanonicalJsonProofBinding
    inventory_source: CanonicalJsonProofBinding
    tts_authority_source: CanonicalJsonProofBinding | None
    raw_tts_evidence_source: CanonicalJsonProofBinding | None
    stage_source_rebuild_input_source: CanonicalJsonProofBinding | None
    evidence_shard_sources: tuple[CanonicalJsonProofBinding, ...]
    execution_rebuild_shard_sources: tuple[CanonicalJsonProofBinding, ...]
    candidate_replay_proof_sources: tuple[CanonicalJsonProofBinding, ...]
    derived_coverage_shard_sources: tuple[CanonicalJsonProofBinding, ...]
    e0_all_na_authority_source: CanonicalJsonProofBinding | None = None

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_stage_coverage_proof_artifact"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
        ):
            raise ValueError("formal stage coverage artifact identity differs")
        _require_phase(self.stage, self.phase)
        for label, digest in (
            ("protocol lock", self.protocol_lock_sha256),
            ("runtime authority", self.formal_runtime_authority_manifest_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("inventory", self.inventory_sha256),
            ("coverage", self.coverage_receipt_sha256),
        ):
            _require_sha256(f"formal stage coverage artifact {label}", digest)
        required_bindings = (
            self.protocol_lock_source,
            self.runtime_authority_source,
            self.materialization_source,
            self.inventory_source,
        )
        if any(type(row) is not CanonicalJsonProofBinding for row in required_bindings):
            raise TypeError("formal coverage authority sources must be path-bound")
        collections = (
            self.evidence_shard_sources,
            self.execution_rebuild_shard_sources,
            self.candidate_replay_proof_sources,
            self.derived_coverage_shard_sources,
        )
        if any(
            type(rows) is not tuple
            or any(type(row) is not CanonicalJsonProofBinding for row in rows)
            for rows in collections
        ):
            raise TypeError("formal coverage artifact source sets must be path-bound")
        if not self.derived_coverage_shard_sources:
            raise ValueError("formal coverage artifact lacks derived coverage shards")
        e0_all_na = self.e0_all_na_authority_source is not None
        if e0_all_na:
            if (
                (self.stage, self.phase) != ("E0", "final")
                or type(self.e0_all_na_authority_source)
                is not CanonicalJsonProofBinding
                or self.tts_authority_source is not None
                or self.raw_tts_evidence_source is not None
                or self.stage_source_rebuild_input_source is not None
                or self.evidence_shard_sources
                or self.execution_rebuild_shard_sources
                or self.candidate_replay_proof_sources
            ):
                raise ValueError("E0 all-N/A coverage source union is not exact")
        elif self.stage == "TTS-Cal":
            if (
                type(self.tts_authority_source) is not CanonicalJsonProofBinding
                or type(self.raw_tts_evidence_source) is not CanonicalJsonProofBinding
                or self.stage_source_rebuild_input_source is not None
                or self.evidence_shard_sources
                or self.execution_rebuild_shard_sources
                or self.candidate_replay_proof_sources
            ):
                raise ValueError("TTS coverage artifact source union is not exact")
        elif (
            self.tts_authority_source is not None
            or self.raw_tts_evidence_source is not None
            or not self.evidence_shard_sources
            or not self.execution_rebuild_shard_sources
        ):
            raise ValueError("serving coverage artifact source union is not exact")
        all_bindings = tuple(
            row
            for row in (
                *required_bindings,
                self.tts_authority_source,
                self.raw_tts_evidence_source,
                self.stage_source_rebuild_input_source,
                *self.evidence_shard_sources,
                *self.execution_rebuild_shard_sources,
                *self.candidate_replay_proof_sources,
                *self.derived_coverage_shard_sources,
                self.e0_all_na_authority_source,
            )
            if row is not None
        )
        paths = tuple(row.absolute_path for row in all_bindings)
        if len(paths) != len(set(paths)):
            raise ValueError("formal coverage artifact reuses a source path")

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
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "formal_runtime_authority_manifest_sha256": (
                self.formal_runtime_authority_manifest_sha256
            ),
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "inventory_sha256": self.inventory_sha256,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "protocol_lock_source": self.protocol_lock_source.to_dict(),
            "runtime_authority_source": self.runtime_authority_source.to_dict(),
            "materialization_source": self.materialization_source.to_dict(),
            "inventory_source": self.inventory_source.to_dict(),
            "tts_authority_source": optional(self.tts_authority_source),
            "raw_tts_evidence_source": optional(self.raw_tts_evidence_source),
            "stage_source_rebuild_input_source": optional(
                self.stage_source_rebuild_input_source
            ),
            "evidence_shard_sources": [
                row.to_dict() for row in self.evidence_shard_sources
            ],
            "execution_rebuild_shard_sources": [
                row.to_dict() for row in self.execution_rebuild_shard_sources
            ],
            "candidate_replay_proof_sources": [
                row.to_dict() for row in self.candidate_replay_proof_sources
            ],
            "derived_coverage_shard_sources": [
                row.to_dict() for row in self.derived_coverage_shard_sources
            ],
            "e0_all_na_authority_source": optional(self.e0_all_na_authority_source),
        }
        if include_sha256:
            value["artifact_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "artifact_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("formal coverage proof artifact fields differ")
        row = dict(value)
        declared = _require_sha256(
            "formal coverage proof artifact", row.pop("artifact_sha256")
        )
        for name in (
            "protocol_lock_source",
            "runtime_authority_source",
            "materialization_source",
            "inventory_source",
        ):
            row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "tts_authority_source",
            "raw_tts_evidence_source",
            "stage_source_rebuild_input_source",
            "e0_all_na_authority_source",
        ):
            if row[name] is not None:
                row[name] = CanonicalJsonProofBinding.from_dict(row[name])
        for name in (
            "evidence_shard_sources",
            "execution_rebuild_shard_sources",
            "candidate_replay_proof_sources",
            "derived_coverage_shard_sources",
        ):
            raw = row[name]
            if type(raw) is not list:
                raise TypeError(f"formal coverage {name} must be an array")
            row[name] = tuple(CanonicalJsonProofBinding.from_dict(item) for item in raw)
        artifact = cls(**row)  # type: ignore[arg-type]
        if artifact.sha256 != declared:
            raise ValueError("formal coverage proof artifact digest differs")
        return artifact


def publish_formal_stage_coverage_proof_artifact(
    artifact: FormalStageCoverageProofArtifact,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    if type(artifact) is not FormalStageCoverageProofArtifact:
        raise TypeError("formal coverage publisher requires an exact artifact")
    publish_canonical_json_no_replace(output_path, artifact.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


@dataclass(frozen=True)
class FormalSignedStageCoverageProofWrapper:
    """Compact signature header for one path-bound coverage proof graph."""

    schema_version: Literal[1]
    kind: Literal["formal_signed_stage_coverage_proof_wrapper"]
    protocol_sha256: str
    coverage_receipt_sha256: str
    signed_coverage_receipt_sha256: str
    coverage_proof_source: CanonicalJsonProofBinding
    payload_sha256: str
    challenge: AttestationChallenge
    attestation: SignedAttestation

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "formal_signed_stage_coverage_proof_wrapper"
            or self.protocol_sha256 != FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256
        ):
            raise ValueError("signed coverage proof wrapper identity differs")
        for label, digest in (
            ("coverage", self.coverage_receipt_sha256),
            ("signed coverage", self.signed_coverage_receipt_sha256),
            ("payload", self.payload_sha256),
        ):
            _require_sha256(f"signed coverage proof wrapper {label}", digest)
        if type(self.coverage_proof_source) is not CanonicalJsonProofBinding:
            raise TypeError("signed coverage proof source must be path-bound")
        if (
            type(self.challenge) is not AttestationChallenge
            or type(self.attestation) is not SignedAttestation
        ):
            raise TypeError("signed coverage proof signature fields are not exact")
        self.challenge.validate()
        self.attestation.validate()
        if (
            self.payload_sha256 != self.coverage_receipt_sha256
            or self.challenge.subject_sha256 != self.coverage_receipt_sha256
            or self.attestation.payload_sha256 != self.payload_sha256
            or self.attestation.challenge_sha256 != self.challenge.sha256
        ):
            raise ValueError("signed coverage proof signature lineage differs")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict(include_sha256=False))

    def to_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "protocol_sha256": self.protocol_sha256,
            "coverage_receipt_sha256": self.coverage_receipt_sha256,
            "signed_coverage_receipt_sha256": (self.signed_coverage_receipt_sha256),
            "coverage_proof_source": self.coverage_proof_source.to_dict(),
            "payload_sha256": self.payload_sha256,
            "challenge": asdict(self.challenge),
            "attestation": asdict(self.attestation),
        }
        if include_sha256:
            value["wrapper_sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        fields = {*cls.__dataclass_fields__, "wrapper_sha256"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("signed coverage proof wrapper fields differ")
        row = dict(value)
        declared = _require_sha256(
            "signed coverage proof wrapper", row.pop("wrapper_sha256")
        )
        challenge = row["challenge"]
        attestation = row["attestation"]
        if (
            type(challenge) is not dict
            or set(challenge) != set(AttestationChallenge.__dataclass_fields__)
            or type(attestation) is not dict
            or set(attestation) != set(SignedAttestation.__dataclass_fields__)
        ):
            raise ValueError("signed coverage proof signature fields differ")
        row["challenge"] = AttestationChallenge(**challenge)
        row["attestation"] = SignedAttestation(**attestation)
        row["coverage_proof_source"] = CanonicalJsonProofBinding.from_dict(
            row["coverage_proof_source"]
        )
        wrapper = cls(**row)  # type: ignore[arg-type]
        if wrapper.sha256 != declared:
            raise ValueError("signed coverage proof wrapper digest differs")
        return wrapper


def publish_formal_signed_stage_coverage_proof_wrapper(
    signed_coverage: SignedStageCoverageReceipt,
    *,
    coverage_proof_source: CanonicalJsonProofBinding,
    output_path: str | Path,
) -> CanonicalJsonProofBinding:
    """Publish a bounded signature header without embedding coverage rows."""

    if type(signed_coverage) is not SignedStageCoverageReceipt:
        raise TypeError("signed coverage proof wrapper requires exact coverage")
    if type(coverage_proof_source) is not CanonicalJsonProofBinding:
        raise TypeError("signed coverage proof wrapper requires a path binding")
    artifact = FormalStageCoverageProofArtifact.from_dict(
        coverage_proof_source.reopen()
    )
    if artifact.coverage_receipt_sha256 != signed_coverage.payload.sha256:
        raise ValueError("coverage proof source names another signed payload")
    wrapper = FormalSignedStageCoverageProofWrapper(
        schema_version=1,
        kind="formal_signed_stage_coverage_proof_wrapper",
        protocol_sha256=FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256,
        coverage_receipt_sha256=signed_coverage.payload.sha256,
        signed_coverage_receipt_sha256=signed_coverage.sha256,
        coverage_proof_source=coverage_proof_source,
        payload_sha256=signed_coverage.payload_sha256,
        challenge=signed_coverage.challenge,
        attestation=signed_coverage.attestation,
    )
    publish_canonical_json_no_replace(output_path, wrapper.to_dict())
    return CanonicalJsonProofBinding.bind(str(output_path))


def rebuild_formal_signed_stage_coverage_proof_wrapper(
    wrapper_path: str,
    **coverage_revalidation_kwargs: object,
) -> SignedStageCoverageReceipt:
    """Deep-replay a compact wrapper into the original signed receipt."""

    binding = CanonicalJsonProofBinding.bind(wrapper_path)
    wrapper = FormalSignedStageCoverageProofWrapper.from_dict(binding.reopen())
    coverage = revalidate_formal_stage_coverage_proof_artifact(
        wrapper.coverage_proof_source.absolute_path,
        **coverage_revalidation_kwargs,
    )
    signed = SignedStageCoverageReceipt(
        payload=coverage,
        payload_sha256=wrapper.payload_sha256,
        challenge=wrapper.challenge,
        attestation=wrapper.attestation,
    )
    if (
        coverage.sha256 != wrapper.coverage_receipt_sha256
        or signed.sha256 != wrapper.signed_coverage_receipt_sha256
    ):
        raise ValueError("signed coverage proof wrapper reconstructs another receipt")
    return signed


def _open_binding(binding: CanonicalJsonProofBinding) -> dict:
    observed = CanonicalJsonProofBinding.bind(binding.absolute_path)
    if observed != binding:
        raise ValueError("formal coverage artifact source path identity changed")
    return observed.reopen()


def rebuild_formal_stage_bound_materialization(
    binding: CanonicalJsonProofBinding,
    *,
    expected_receipt_sha256: str,
) -> StageMaterializationReceipt:
    """Deep-open a direct or sharded materialization binding."""

    value = _open_binding(binding)
    if value.get("kind") == "formal_materialization_shard_index":
        return revalidate_formal_materialization_shard_index(
            binding.absolute_path,
            expected_materialization_receipt_sha256=expected_receipt_sha256,
        )
    from lightcone_spec.experiments.formal_registry import (
        stage_materialization_receipt_from_dict,
    )

    receipt = stage_materialization_receipt_from_dict(value)
    if receipt.sha256 != expected_receipt_sha256:
        raise ValueError("formal coverage materialization source identity differs")
    return receipt


def reconstruct_formal_stage_derived_coverage(
    artifact: FormalStageCoverageProofArtifact,
) -> StageCoverageReceipt:
    """Reconstruct reducer-owned coverage from its durable ordered shards."""

    shards = tuple(
        FormalStageDerivedCoverageShard.from_dict(_open_binding(binding))
        for binding in artifact.derived_coverage_shard_sources
    )
    if (
        tuple(row.shard_index for row in shards) != tuple(range(len(shards)))
        or any(row.shard_count != len(shards) for row in shards)
        or any(
            (
                row.stage,
                row.phase,
                row.protocol_lock_sha256,
                row.materialization_receipt_sha256,
                row.coverage_receipt_sha256,
            )
            != (
                artifact.stage,
                artifact.phase,
                artifact.protocol_lock_sha256,
                artifact.materialization_receipt_sha256,
                artifact.coverage_receipt_sha256,
            )
            for row in shards
        )
    ):
        raise ValueError("derived coverage shard graph differs from artifact")
    dispositions = tuple(row for shard in shards for row in shard.dispositions)
    candidates = tuple(row for shard in shards for row in shard.candidate_coverages)
    receipt = StageCoverageReceipt(
        schema_version=2,
        stage=artifact.stage,
        protocol_lock_sha256=artifact.protocol_lock_sha256,
        materialization_receipt_sha256=artifact.materialization_receipt_sha256,
        dispositions=dispositions,
        tts_l0_candidate_state_coverages=candidates,
    )
    if receipt.sha256 != artifact.coverage_receipt_sha256:
        raise ValueError("derived coverage shards reconstruct another receipt")
    return receipt


def _reduce_e0_all_na_from_authority_source(
    source: CanonicalJsonProofBinding,
    *,
    protocol_lock: ProtocolLock,
    runtime_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    now_ns: int,
) -> StageCoverageReceipt:
    from lightcone_spec.experiments.breadth_fdr_authority import (
        signed_formal_e0_breadth_fdr_from_dict,
    )
    from lightcone_spec.experiments.e0_authority_artifact import (
        signed_e0_final_completion_from_dict,
        signed_e6_confirmation_from_dict,
    )
    from lightcone_spec.experiments.formal_registry import (
        formal_registry_verification_receipt_from_dict,
        signed_e0_compatibility_from_dict,
    )

    authority = FormalE0AllNaCoverageAuthorityArtifact.from_dict(_open_binding(source))
    if (
        authority.protocol_lock_sha256 != protocol_lock.sha256
        or authority.materialization_receipt_sha256 != materialization.sha256
        or authority.inventory_sha256 != inventory.sha256
    ):
        raise ValueError("E0 all-N/A durable authority lineage differs")
    return reduce_e0_all_na_stage_coverage_from_proofs(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        registry_verification_receipt=formal_registry_verification_receipt_from_dict(
            _open_binding(authority.registry_verification_receipt_source)
        ),
        signed_e6_confirmation=signed_e6_confirmation_from_dict(
            _open_binding(authority.signed_e6_confirmation_source)
        ),
        signed_compatibility=signed_e0_compatibility_from_dict(
            _open_binding(authority.signed_compatibility_source)
        ),
        signed_final_completion=signed_e0_final_completion_from_dict(
            _open_binding(authority.signed_final_completion_source)
        ),
        signed_formal_fdr=signed_formal_e0_breadth_fdr_from_dict(
            _open_binding(authority.signed_formal_fdr_source)
        ),
        now_ns=now_ns,
    )


@dataclass(frozen=True)
class FormalStageCoverageRebuiltContext:
    protocol_lock: ProtocolLock
    formal_runtime_authority_manifest: FormalRuntimeAuthorityManifest
    materialization: StageMaterializationReceipt
    inventory: GpuInventory
    coverage: StageCoverageReceipt
    stage_source: VerifiedFormalStageMaterializationSource | None
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...]
    failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...] = ()
    tts_calibration_authority: TtsCalibrationAuthority | None = None
    raw_tts_evidence_manifest: RawTtsCalibrationEvidenceManifest | None = None

    def __post_init__(self) -> None:
        if (
            type(self.protocol_lock) is not ProtocolLock
            or type(self.formal_runtime_authority_manifest)
            is not FormalRuntimeAuthorityManifest
            or type(self.materialization) is not StageMaterializationReceipt
            or type(self.inventory) is not GpuInventory
            or type(self.coverage) is not StageCoverageReceipt
            or type(self.execution_bindings) is not tuple
            or type(self.failure_execution_bindings) is not tuple
            or (
                self.tts_calibration_authority is not None
                and type(self.tts_calibration_authority) is not TtsCalibrationAuthority
            )
            or (
                self.raw_tts_evidence_manifest is not None
                and type(self.raw_tts_evidence_manifest)
                is not RawTtsCalibrationEvidenceManifest
            )
        ):
            raise TypeError("formal coverage rebuilt context types differ")
        if (
            self.materialization.protocol_lock_sha256 != self.protocol_lock.sha256
            or self.coverage.protocol_lock_sha256 != self.protocol_lock.sha256
            or self.coverage.materialization_receipt_sha256
            != self.materialization.sha256
            or self.formal_runtime_authority_manifest.sha256
            != self.protocol_lock.formal_runtime_authority_manifest_sha256
        ):
            raise ValueError("formal coverage rebuilt context lineage differs")
        if self.materialization.stage == "TTS-Cal":
            if (
                self.tts_calibration_authority is None
                or self.raw_tts_evidence_manifest is None
            ):
                raise ValueError("TTS coverage context lacks durable reducer inputs")
        elif self.raw_tts_evidence_manifest is not None:
            raise ValueError("non-TTS coverage context carries raw TTS evidence")


def _require_complete_shard_sequence(rows: tuple[object, ...], *, label: str) -> None:
    if (
        not rows
        or tuple(getattr(row, "shard_index", None) for row in rows)
        != tuple(range(len(rows)))
        or any(getattr(row, "shard_count", None) != len(rows) for row in rows)
    ):
        raise ValueError(f"{label} shard graph is incomplete")


def rebuild_formal_stage_coverage_context(
    artifact_path: str,
    *,
    now_ns: int,
    tts_authority: TtsCalibrationAuthority | None = None,
    signed_tts_seal: object | None = None,
    e1_recipe_anchor_authority: object | None = None,
    e2_recipe_grid_authority: object | None = None,
    lightcone_recipe: object | None = None,
    registry_verification_receipt: object | None = None,
    stage_source_inputs: FormalStageSourceRebuildInputs | None = None,
    relocatable_bundle_manifest_path: str | None = None,
) -> FormalStageCoverageRebuiltContext:
    """Deep-reopen sources and return the exact reusable verified context."""

    if relocatable_bundle_manifest_path is not None:
        from lightcone_spec.runtime.relocatable_evidence import (
            activate_relocatable_evidence_bundle,
        )

        with activate_relocatable_evidence_bundle(
            relocatable_bundle_manifest_path
        ) as bundle:
            if artifact_path not in bundle.artifact.entry_remote_absolute_paths:
                raise ValueError(
                    "formal coverage artifact is not a pulled-evidence entry"
                )
            return rebuild_formal_stage_coverage_context(
                artifact_path,
                now_ns=now_ns,
                tts_authority=tts_authority,
                signed_tts_seal=signed_tts_seal,
                e1_recipe_anchor_authority=e1_recipe_anchor_authority,
                e2_recipe_grid_authority=e2_recipe_grid_authority,
                lightcone_recipe=lightcone_recipe,
                registry_verification_receipt=registry_verification_receipt,
                stage_source_inputs=stage_source_inputs,
                relocatable_bundle_manifest_path=None,
            )

    artifact_binding = CanonicalJsonProofBinding.bind(artifact_path)
    artifact = FormalStageCoverageProofArtifact.from_dict(artifact_binding.reopen())
    from lightcone_spec.experiments.formal_registry import (
        formal_runtime_authority_manifest_from_dict,
        protocol_lock_from_dict,
        tts_calibration_authority_from_dict,
    )

    protocol_lock = protocol_lock_from_dict(
        _open_binding(artifact.protocol_lock_source)
    )
    runtime_manifest = formal_runtime_authority_manifest_from_dict(
        _open_binding(artifact.runtime_authority_source)
    )
    materialization = rebuild_formal_stage_bound_materialization(
        artifact.materialization_source,
        expected_receipt_sha256=artifact.materialization_receipt_sha256,
    )
    inventory = GpuInventory.from_dict(_open_binding(artifact.inventory_source))
    if (
        protocol_lock.sha256 != artifact.protocol_lock_sha256
        or runtime_manifest.sha256 != artifact.formal_runtime_authority_manifest_sha256
        or materialization.sha256 != artifact.materialization_receipt_sha256
        or inventory.sha256 != artifact.inventory_sha256
        or _phase_for_materialization(materialization)
        != (artifact.stage, artifact.phase)
    ):
        raise ValueError("formal coverage durable authority identity differs")
    expected = reconstruct_formal_stage_derived_coverage(artifact)
    stage_source: VerifiedFormalStageMaterializationSource | None = None
    execution_bindings: tuple[VerifiedFormalServingExecutionBinding, ...] = ()
    failure_execution_bindings: tuple[VerifiedFormalFailureExecutionBinding, ...] = ()
    durable_tts_authority: TtsCalibrationAuthority | None = None
    raw_tts_manifest: RawTtsCalibrationEvidenceManifest | None = None
    if artifact.e0_all_na_authority_source is not None:
        if stage_source_inputs is not None:
            raise ValueError("E0 all-N/A coverage accepts no serving stage source")
        rebuilt = _reduce_e0_all_na_from_authority_source(
            artifact.e0_all_na_authority_source,
            protocol_lock=protocol_lock,
            runtime_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=now_ns,
        )
    elif artifact.stage == "TTS-Cal":
        assert artifact.tts_authority_source is not None
        assert artifact.raw_tts_evidence_source is not None
        durable_authority = tts_calibration_authority_from_dict(
            _open_binding(artifact.tts_authority_source)
        )
        if tts_authority is not None and tts_authority != durable_authority:
            raise ValueError("caller TTS authority differs from durable source")
        raw_manifest = raw_tts_calibration_manifest_from_dict(
            _open_binding(artifact.raw_tts_evidence_source)
        )
        durable_tts_authority = durable_authority
        raw_tts_manifest = raw_manifest
        rebuilt = reduce_tts_calibration_stage_coverage_from_proofs(
            protocol_lock=protocol_lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            authority=durable_authority,
            manifest=raw_manifest,
            now_ns=now_ns,
        )
    else:
        evidence_rows = tuple(
            (binding, _open_binding(binding))
            for binding in artifact.evidence_shard_sources
        )
        rebuild_rows = tuple(
            (binding, _open_binding(binding))
            for binding in artifact.execution_rebuild_shard_sources
        )
        serving_evidence_shards = tuple(
            FormalStageCoverageEvidenceShard.from_dict(value)
            for _binding, value in evidence_rows
            if value.get("kind") == "formal_stage_coverage_evidence_shard"
        )
        failure_evidence_shards = tuple(
            FormalE5FailureCoverageEvidenceShard.from_dict(value)
            for _binding, value in evidence_rows
            if value.get("kind") == "formal_e5_failure_coverage_evidence_shard"
        )
        serving_rebuild_shards = tuple(
            FormalStageCoverageExecutionRebuildShard.from_dict(value)
            for _binding, value in rebuild_rows
            if value.get("kind") == "formal_stage_coverage_execution_rebuild_shard"
        )
        failure_rebuild_shards = tuple(
            FormalE5FailureCoverageExecutionRebuildShard.from_dict(value)
            for _binding, value in rebuild_rows
            if value.get("kind") == "formal_e5_failure_coverage_execution_rebuild_shard"
        )
        if len(serving_evidence_shards) + len(failure_evidence_shards) != len(
            evidence_rows
        ) or len(serving_rebuild_shards) + len(failure_rebuild_shards) != len(
            rebuild_rows
        ):
            raise ValueError("formal coverage shard kind is unsupported")
        _require_complete_shard_sequence(
            serving_evidence_shards, label="formal coverage serving evidence"
        )
        _require_complete_shard_sequence(
            serving_rebuild_shards, label="formal coverage serving rebuild"
        )
        evidence_cells = tuple(
            row for shard in serving_evidence_shards for row in shard.cells
        )
        rebuild_inputs = tuple(
            row for shard in serving_rebuild_shards for row in shard.rebuild_inputs
        )
        if tuple(row.materialized_cell_id for row in evidence_cells) != tuple(
            row.subject.materialized_cell_id for row in rebuild_inputs
        ):
            raise ValueError("formal coverage evidence/rebuild shard cells differ")
        if artifact.stage_source_rebuild_input_source is not None:
            if stage_source_inputs is None:
                raise TypeError(
                    "formal coverage revalidation lacks stage source inputs"
                )
            descriptor = FormalStageSourceRebuildInput.from_dict(
                _open_binding(artifact.stage_source_rebuild_input_source)
            )
            stage_source = rebuild_formal_stage_materialization_source(
                descriptor,
                materialization=materialization,
                source_inputs=stage_source_inputs,
                now_ns=now_ns,
            )
        elif stage_source_inputs is not None:
            raise ValueError("formal coverage has unexpected stage source inputs")
        execution_bindings = tuple(
            rebuild_formal_serving_execution_binding(
                descriptor,
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=runtime_manifest,
                materialization=materialization,
                inventory=inventory,
                tts_authority=tts_authority,
                signed_tts_seal=signed_tts_seal,  # type: ignore[arg-type]
                e1_recipe_anchor_authority=e1_recipe_anchor_authority,  # type: ignore[arg-type]
                e2_recipe_grid_authority=e2_recipe_grid_authority,  # type: ignore[arg-type]
                lightcone_recipe=lightcone_recipe,  # type: ignore[arg-type]
                stage_source=stage_source,
                now_ns=now_ns,
                registry_verification_receipt=registry_verification_receipt,
            )
            for descriptor in rebuild_inputs
        )
        for binding in artifact.candidate_replay_proof_sources:
            _open_binding(binding)
        candidate_paths = tuple(
            row.absolute_path for row in artifact.candidate_replay_proof_sources
        )
        if artifact.stage == "E5" and artifact.phase == "final_and_one_shot_failure":
            _require_complete_shard_sequence(
                failure_evidence_shards,
                label="formal coverage E5 failure evidence",
            )
            _require_complete_shard_sequence(
                failure_rebuild_shards,
                label="formal coverage E5 failure rebuild",
            )
            failure_evidence_cells = tuple(
                row for shard in failure_evidence_shards for row in shard.cells
            )
            failure_rebuild_rows = tuple(
                row for shard in failure_rebuild_shards for row in shard.rows
            )
            if tuple(
                row.materialized_cell_id for row in failure_evidence_cells
            ) != tuple(
                row.failure_execution.subject.materialized_cell_id
                for row in failure_rebuild_rows
            ):
                raise ValueError("E5 failure evidence/rebuild cells differ")
            failure_serving_bindings = tuple(
                rebuild_formal_serving_execution_binding(
                    row.serving_execution,
                    protocol_lock=protocol_lock,
                    formal_runtime_authority_manifest=runtime_manifest,
                    materialization=materialization,
                    inventory=inventory,
                    tts_authority=tts_authority,
                    signed_tts_seal=signed_tts_seal,  # type: ignore[arg-type]
                    e1_recipe_anchor_authority=e1_recipe_anchor_authority,  # type: ignore[arg-type]
                    e2_recipe_grid_authority=e2_recipe_grid_authority,  # type: ignore[arg-type]
                    lightcone_recipe=lightcone_recipe,  # type: ignore[arg-type]
                    stage_source=stage_source,
                    now_ns=now_ns,
                    registry_verification_receipt=registry_verification_receipt,
                )
                for row in failure_rebuild_rows
            )
            failure_execution_bindings = tuple(
                rebuild_formal_failure_execution_binding(
                    row.failure_execution,
                    serving_execution_rebuild_input=row.serving_execution,
                    serving_execution=serving,
                    protocol_lock=protocol_lock,
                    formal_runtime_authority_manifest=runtime_manifest,
                    materialization=materialization,
                )
                for row, serving in zip(
                    failure_rebuild_rows,
                    failure_serving_bindings,
                    strict=True,
                )
            )
            rebuilt = reduce_e5_final_stage_coverage_from_proofs(
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=runtime_manifest,
                materialization=materialization,
                inventory=inventory,
                headline_evidence_cells=evidence_cells,
                headline_execution_bindings=execution_bindings,
                failure_evidence_cells=failure_evidence_cells,
                failure_execution_bindings=failure_execution_bindings,
                candidate_coverages=expected.tts_l0_candidate_state_coverages,
                candidate_replay_proof_artifact_paths=candidate_paths,
                now_ns=now_ns,
            )
            execution_bindings = (*execution_bindings, *failure_serving_bindings)
        elif artifact.stage == "E6":
            if failure_evidence_shards or failure_rebuild_shards:
                raise ValueError("E6 coverage carries foreign failure proof shards")
            if stage_source is None:
                raise ValueError("E6 coverage lacks its sealed stage source")
            if artifact.phase == "excluded_pilot_and_model_preflight":
                if type(stage_source_inputs) is not E6PilotStageSourceRebuildInputs:
                    raise TypeError("E6 pilot coverage source DAG is not exact")
            elif artifact.phase == "final":
                if type(stage_source_inputs) is not E6FinalStageSourceRebuildInputs:
                    raise TypeError("E6 final coverage source DAG is not exact")
            else:  # pragma: no cover - phase union is closed above
                raise AssertionError("unsupported E6 coverage phase")
            rebuilt = reduce_e6_stage_coverage_from_proofs(
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=runtime_manifest,
                materialization=materialization,
                inventory=inventory,
                phase=artifact.phase,
                signed_model_compatibility=(
                    stage_source_inputs.signed_model_compatibility
                ),
                compatibility_sources=stage_source_inputs.compatibility_sources,
                registry_verification_receipt=(
                    stage_source_inputs.registry_verification_receipt
                ),
                stage_source=stage_source,
                serving_evidence_cells=evidence_cells,
                serving_execution_bindings=execution_bindings,
                candidate_coverages=expected.tts_l0_candidate_state_coverages,
                candidate_replay_proof_artifact_paths=candidate_paths,
                now_ns=now_ns,
            )
        else:
            if failure_evidence_shards or failure_rebuild_shards:
                raise ValueError("non-E5-final coverage carries failure proof shards")
            rebuilt = reduce_formal_serving_stage_coverage_from_proofs(
                protocol_lock=protocol_lock,
                formal_runtime_authority_manifest=runtime_manifest,
                materialization=materialization,
                inventory=inventory,
                stage=artifact.stage,
                phase=artifact.phase,
                evidence_cells=evidence_cells,
                execution_bindings=execution_bindings,
                candidate_coverages=expected.tts_l0_candidate_state_coverages,
                candidate_replay_proof_artifact_paths=candidate_paths,
                e0_valid_compatibility_decision_ids=(
                    _e0_valid_compatibility_decision_ids(stage_source_inputs)
                    if artifact.stage == "E0"
                    else None
                ),
                now_ns=now_ns,
            )
    if rebuilt != expected or rebuilt.sha256 != artifact.coverage_receipt_sha256:
        raise ValueError("formal coverage proof replay changed derived coverage")
    return FormalStageCoverageRebuiltContext(
        protocol_lock=protocol_lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        coverage=rebuilt,
        stage_source=stage_source,
        execution_bindings=execution_bindings,
        failure_execution_bindings=failure_execution_bindings,
        tts_calibration_authority=(
            durable_tts_authority
            if durable_tts_authority is not None
            else tts_authority
        ),
        raw_tts_evidence_manifest=raw_tts_manifest,
    )


def revalidate_formal_stage_coverage_proof_artifact(
    artifact_path: str,
    **kwargs,
) -> StageCoverageReceipt:
    """Compatibility wrapper returning only the exact derived coverage."""

    return rebuild_formal_stage_coverage_context(artifact_path, **kwargs).coverage


__all__ = (
    "FORMAL_E0_ALL_NA_COVERAGE_PROTOCOL_SHA256",
    "FORMAL_STAGE_COVERAGE_PROTOCOL_SHA256",
    "FORMAL_STAGE_COVERAGE_RUNNER_SHA256",
    "FORMAL_STAGE_COVERAGE_TEST_SET_SHA256",
    "FormalE0AllNaCoverageAuthorityArtifact",
    "FormalE5FailureCoverageEvidenceCell",
    "FormalE5FailureCoverageEvidenceShard",
    "FormalE5FailureCoverageExecutionRebuildRow",
    "FormalE5FailureCoverageExecutionRebuildShard",
    "FormalSignedStageCoverageProofWrapper",
    "FormalStageCoverageEvidenceCell",
    "FormalStageCoverageEvidenceShard",
    "FormalStageCoverageExecutionRebuildShard",
    "FormalStageCoverageProofArtifact",
    "FormalStageCoverageRebuiltContext",
    "FormalStageDerivedCoverageShard",
    "derived_coverage_shards",
    "publish_formal_e0_all_na_coverage_authority_artifact",
    "publish_formal_e5_failure_coverage_evidence_shard",
    "publish_formal_e5_failure_coverage_execution_rebuild_shard",
    "publish_formal_signed_stage_coverage_proof_wrapper",
    "publish_formal_stage_coverage_evidence_shard",
    "publish_formal_stage_coverage_execution_rebuild_shard",
    "publish_formal_stage_coverage_proof_artifact",
    "publish_formal_stage_derived_coverage_shard",
    "rebuild_formal_signed_stage_coverage_proof_wrapper",
    "rebuild_formal_stage_bound_materialization",
    "rebuild_formal_stage_coverage_context",
    "reconstruct_formal_stage_derived_coverage",
    "reduce_e0_all_na_stage_coverage_from_proofs",
    "reduce_e5_final_stage_coverage_from_proofs",
    "reduce_e6_stage_coverage_from_proofs",
    "reduce_formal_serving_stage_coverage_from_proofs",
    "reduce_tts_calibration_stage_coverage_from_proofs",
    "revalidate_formal_stage_coverage_proof_artifact",
)
