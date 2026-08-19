"""Current-result downstream reducers for ``formal_single_operator_v1``.

The older formal pipeline reduces signed proof wrappers.  The trusted
single-operator pipeline instead already deep-reopens every run manifest and
passes a :class:`FormalSingleOperatorValidatedActual` to the stage reducer.
This module keeps that boundary: it never manufactures a signed wrapper and it
never treats a directory name as evidence.

Only scientific values observed in the current predecessor may materialize the
next node.  In particular, E3b final block count is fixed from its four
excluded pilots and E1a is fixed from the complete 116-row verification grid.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from lightcone_spec.experiments.formal_slo_metrics import (
    FormalSloGoodputObservation,
    FormalSloRequestEvidence,
    linear_p99_ns,
    reduce_formal_slo_goodput,
    require_paired_completed_output_exactness,
    require_paired_primary_goodputs,
)
from lightcone_spec.experiments.stage_materialization import (
    E0_LOADS,
    E1A_FIXED_VERIFICATION_BUDGET,
    E1A_NATIVE_VERIFICATION_BUDGET,
    E5_BACKENDS,
    E5_TOPOLOGIES,
    E6_MODELS,
    E6_TASKS,
    FORMAL_METHOD_ROLES,
    E0CompatibilityDecision,
    E0CompatibilityReceipt,
    E5SelectedP99Anchor,
    GpuHourEstimate,
    MaterializedCell,
    StageMaterializationReceipt,
    _cell,
    _e5_headline_cells,
    _e6_cells_from_verified_sources,
    _materialize_e1a_diagnostic,
    _materialize_e3b_diagnostic,
    _receipt,
    default_e5_failure_diagnostic_authority,
)
from lightcone_spec.experiments.statistics import (
    PILOT_BLOCK_COUNT,
    PRIMARY_CONTRASTS,
    PairedBcaContrast,
    PilotBlock,
    UnresolvedPairedContrast,
    UnresolvedPowerSizing,
    benjamini_hochberg,
    guard_p99_claim,
    hierarchical_block_request_bootstrap,
    holm_primary_contrasts,
    resolve_paired_bca_contrast,
    resolve_preregistered_power_sizing,
    time_block_bootstrap,
)

if TYPE_CHECKING:
    from lightcone_spec.experiments.formal_protocol import ProtocolLock
    from lightcone_spec.experiments.formal_single_operator_stages import (
        FormalSingleOperatorDecisionDraft,
        FormalSingleOperatorValidatedActual,
        RebuiltFormalSingleOperatorStageCompletion,
    )


_E3B_SCIENTIFIC_AXES = ("context", "load", "regime", "width_panel")
_E5_P99_BOOTSTRAP_REPETITIONS = 10_000
_E5_P99_REDUCER_METHOD = "registered_time_block_bootstrap_native_itl_linear_p99"
_CORE_CONTRAST_ROLES = {
    "lightcone_vs_tts": ("LightCone", "TTS"),
    "lightcone_vs_static": ("LightCone", "Static"),
    "l0_naive_vs_tts": ("L0-naive", "TTS"),
    "lightcone_vs_l0_naive": ("LightCone", "L0-naive"),
    "lightcone_vs_target_only": ("LightCone", "Target-only"),
}


def _stages() -> object:
    # Kept lazy so ``formal_single_operator_stages`` can register these
    # adapters without creating an import cycle.
    from lightcone_spec.experiments import formal_single_operator_stages

    return formal_single_operator_stages


def _completion_for_node(
    completion: RebuiltFormalSingleOperatorStageCompletion,
    node: str,
) -> RebuiltFormalSingleOperatorStageCompletion | None:
    current: RebuiltFormalSingleOperatorStageCompletion | None = completion
    while current is not None:
        if current.artifact.node == node:
            return current
        current = current.predecessor
    return None


def _sha(label: str, value: object) -> str:
    stages = _stages()
    return stages._require_sha256(label, value)  # type: ignore[attr-defined,no-any-return]


def _text(label: str, value: object) -> str:
    stages = _stages()
    return stages._require_text(label, value)  # type: ignore[attr-defined,no-any-return]


def _digest(value: object) -> str:
    stages = _stages()
    return stages._content_sha256(value)  # type: ignore[attr-defined,no-any-return]


def _serving_observations(
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> dict[str, dict[str, object]]:
    stages = _stages()
    cells = {cell.cell_id: cell for cell in materialization.cells}
    actual = {row.cell_id: row for row in actual_results}
    if set(cells) != set(actual) or len(actual) != len(actual_results):
        raise ValueError("single-operator downstream actual coverage differs")
    return {
        cell_id: stages._serving_observation(actual[cell_id], cell)  # type: ignore[attr-defined]
        for cell_id, cell in cells.items()
    }


def _request_evidence(
    observation: dict[str, object],
) -> tuple[FormalSloRequestEvidence, ...]:
    stages = _stages()
    rows = stages._single_operator_request_rows(observation)  # type: ignore[attr-defined]
    evidence = []
    for row in rows:
        status = str(row["terminal_status"])
        evidence.append(
            FormalSloRequestEvidence(
                request_id=str(row["request_id"]),
                input_token_ids=tuple(row["input_token_ids"]),
                output_token_ids=tuple(row["output_token_ids"]),
                request_started_ns=int(row["request_started_ns"]),
                request_terminal_ns=int(row["request_terminal_ns"]),
                token_observed_ns=tuple(row["token_observed_ns"]),
                eligible=True,
                completed=status == "completed",
                # Client outcome is the scientific denominator.  A registered
                # timeout/cancel/reject is not an infrastructure error; only a
                # request left unfinished by transport/abort-proof failure is.
                error=status == "unfinished",
            )
        )
    return tuple(evidence)


def _slo(
    observation: dict[str, object],
) -> FormalSloGoodputObservation:
    return reduce_formal_slo_goodput(
        _request_evidence(observation),
        source_request_pool_sha256=str(observation["source_request_pool_sha256"]),
    )


def _adaptive_safe(
    observation: dict[str, object],
    *,
    require_published_update: bool,
) -> bool:
    stages = _stages()
    return not stages._adaptive_safety_reasons(  # type: ignore[attr-defined]
        observation,
        require_published_update=require_published_update,
    )


def _paired_role_goodputs(
    rows: dict[str, tuple[MaterializedCell, dict[str, object]]],
    *,
    excluded_roles: frozenset[str] = frozenset(),
) -> dict[str, FormalSloGoodputObservation]:
    if set(rows) != set(FORMAL_METHOD_ROLES):
        raise ValueError("single-operator family method coverage differs")
    if not excluded_roles <= set(rows):
        raise ValueError("single-operator exclusions name an unknown method role")
    request_evidence = {
        role: _request_evidence(value[1]) for role, value in rows.items()
    }
    pool_sha256s = {
        role: str(value[1]["source_request_pool_sha256"])
        for role, value in rows.items()
    }
    exactness_evidence = {
        role: evidence
        for role, evidence in request_evidence.items()
        if role not in excluded_roles
    }
    if len(exactness_evidence) >= 2:
        require_paired_completed_output_exactness(
            exactness_evidence,
            source_request_pool_sha256s={
                role: pool_sha256s[role] for role in exactness_evidence
            },
        )
    observations = {role: _slo(value[1]) for role, value in rows.items()}
    if len({row.source_request_pool_sha256 for row in observations.values()}) != 1:
        raise ValueError("single-operator family request pools are unpaired")
    require_paired_primary_goodputs(
        {role: observations[role] for role in ("Static", "TTS", "LightCone")}
    )
    return observations


def _e3b_grouped(
    materialization: StageMaterializationReceipt,
    observations: dict[str, dict[str, object]],
) -> dict[
    tuple[tuple[str, str | int], ...],
    dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
]:
    grouped: dict[
        tuple[tuple[str, str | int], ...],
        dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
    ] = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if type(block) is not int:
            raise ValueError("E3b cell lacks an integer block")
        family_values = []
        for axis in _E3B_SCIENTIFIC_AXES:
            value = dimensions.get(axis)
            if type(value) not in {str, int}:
                raise ValueError("E3b cell lacks a scientific axis")
            family_values.append((axis, value))
        family = tuple(family_values)
        by_role = grouped.setdefault(family, {}).setdefault(block, {})
        if cell.method_role in by_role or cell.method_role not in FORMAL_METHOD_ROLES:
            raise ValueError("E3b family repeats or changes a method role")
        observation = observations[cell.cell_id]
        by_role[cell.method_role] = (cell, observation)
    return grouped


def _frozen_recipes(
    completion: RebuiltFormalSingleOperatorStageCompletion,
) -> tuple[str, str]:
    tts = _completion_for_node(completion, "tts_cal")
    e2 = _completion_for_node(completion, "e2_r3")
    if tts is None or e2 is None:
        raise ValueError("downstream chain lacks frozen TTS or LightCone selection")
    frozen_tts = _sha(
        "single-operator frozen TTS recipe",
        tts.decision.payload.get("candidate_id"),
    )
    final_recipe = e2.decision.payload.get("final_recipe")
    if type(final_recipe) is not dict:
        raise ValueError("downstream chain lacks the sealed E2 recipe")
    lightcone = _sha(
        "single-operator sealed LightCone recipe",
        final_recipe.get("recipe_sha256"),
    )
    return frozen_tts, lightcone


def _pilot_power_resolution(
    pilots: tuple[PilotBlock, ...],
    exclusions: dict[str, dict[str, object]],
) -> object:
    measured = resolve_preregistered_power_sizing(pilots)
    if {"Static", "TTS", "LightCone"} & set(exclusions):
        return UnresolvedPowerSizing(
            status="POWER_UNRESOLVED",
            reason_code="UNSAFE_OR_INACTIVE_PILOT",
            pilot_block_ids=tuple(row.block_id for row in pilots),
            selected_final_blocks=None,
            minimum_final_blocks=12,
            maximum_final_blocks=20,
            target_power=0.80,
            family_alpha=0.05,
            minimum_relative_effect=0.03,
        )
    return measured


def reduce_single_operator_e4_profiler(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Seal three descriptive profiler terminals without inventing CIs."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e4_local"
        or materialization.stage != "E4"
        or materialization.materialization_rule
        != "three_profiler_only_rows_separate_from_headline"
        or len(materialization.cells) != 3
    ):
        raise ValueError("E4 profiler reducer received another node")
    rows = []
    variants = set()
    for cell, actual in zip(materialization.cells, actual_results, strict=True):
        if cell.task != "mechanism_profile_only" or actual.status != "COMPLETE":
            raise ValueError("E4 profiler requires complete diagnostic-only rows")
        variant = dict(cell.dimensions).get("profiler")
        if type(variant) is not str or variant in variants:
            raise ValueError("E4 profiler variants are not exact")
        variants.add(variant)
        terminal = actual.reducer_payload.get("profiler_terminal")
        if type(terminal) is not dict:
            # The validator may expose the terminal fields directly; both
            # representations are canonical JSON, not caller-authored values.
            terminal = actual.reducer_payload
        raw_sha = terminal.get("raw_profile_sha256")
        raw_size = terminal.get("raw_profile_size_bytes")
        if type(raw_sha) is not str or type(raw_size) is not int or raw_size <= 0:
            raise ValueError("E4 profiler terminal lacks its raw report identity")
        rows.append(
            {
                "cell_id": cell.cell_id,
                "variant": variant,
                "raw_profile_sha256": _sha("E4 raw profile", raw_sha),
                "raw_profile_size_bytes": raw_size,
                "terminal_identity_sha256": actual.result_identity_sha256,
                "headline_eligible": False,
            }
        )
    expected = {"nvtx", "nsight_systems", "nsight_compute"}
    if variants != expected:
        raise ValueError("E4 profiler coverage differs from NVTX/nsys/ncu")
    local = predecessor.decision.payload
    frozen_tts, lightcone = _frozen_recipes(predecessor)
    completion = {
        "schema_version": 1,
        "kind": "formal_single_operator_e4_profiler_completion",
        "model": _text("E4 profiler model", local.get("model")),
        "frozen_tts_recipe_sha256": frozen_tts,
        "lightcone_recipe_sha256": lightcone,
        "local_selection_sha256": _sha(
            "E4 local selection", local.get("selection_sha256")
        ),
        "profiler_rows": sorted(rows, key=lambda row: str(row["variant"])),
    }
    completion_sha = _digest(completion)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e4_profiler_actual_3_reduced",
        next_materialization_source_decision_sha256=completion_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**completion, "completion_sha256": completion_sha},
    )


def materialize_single_operator_e3b_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    """Materialize only four excluded E3b blocks after profiler completion."""

    if predecessor is None or predecessor.artifact.node != "e4_profiler":
        raise ValueError("E3b pilots require completed E4 profiler")
    decision = predecessor.decision
    payload = decision.payload
    source = _sha("E3b profiler completion", payload.get("completion_sha256"))
    if (
        decision.next_materialization_source_decision_sha256 != source
        or decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E3b profiler predecessor lineage differs")
    frozen_tts = _sha("E3b frozen TTS recipe", payload.get("frozen_tts_recipe_sha256"))
    lightcone = _sha("E3b LightCone recipe", payload.get("lightcone_recipe_sha256"))
    fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=predecessor.materialization.sha256,
        source_decision_sha256=source,
        model=_text("E3b model", payload.get("model")),
        frozen_tts_recipe_sha256=frozen_tts,
        lightcone_recipe_sha256=lightcone,
        final_blocks=12,
        gpu_hours=GpuHourEstimate.unmeasured(),
        lineage_dimensions={
            "e4_profiler_completion_sha256": source,
            "frozen_tts_recipe_sha256": frozen_tts,
            "lightcone_recipe_sha256": lightcone,
        },
    )
    cells = tuple(cell for cell in fixture.cells if dict(cell.dimensions)["block"] < 4)
    if len(cells) != 480 * PILOT_BLOCK_COUNT:
        raise AssertionError("E3b excluded pilots must contain exactly 1,920 rows")
    return _receipt(
        stage="E3b",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule="e3b_exact_480_rows_x_4_excluded_pilot_blocks",
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def reduce_single_operator_e3b_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Power every E3b family from exactly four excluded blocks."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e4_profiler"
        or materialization.stage != "E3b"
        or materialization.materialization_rule
        != "e3b_exact_480_rows_x_4_excluded_pilot_blocks"
        or len(materialization.cells) != 1_920
    ):
        raise ValueError("E3b pilot reducer received another node")
    observations = _serving_observations(materialization, actual_results)
    grouped = _e3b_grouped(materialization, observations)
    if len(grouped) != 96:
        raise ValueError("E3b pilot lacks the exact 96 scientific families")
    plans = []
    underpowered = []
    power_unresolved = []
    selected = []
    for family in sorted(grouped, key=_digest):
        blocks = grouped[family]
        if set(blocks) != set(range(PILOT_BLOCK_COUNT)):
            raise ValueError("E3b family lacks exactly four excluded pilots")
        exclusions = _family_role_exclusions(blocks)
        pilot_rows = []
        observation_ids = []
        for block in range(PILOT_BLOCK_COUNT):
            by_role = _paired_role_goodputs(
                blocks[block], excluded_roles=frozenset(exclusions)
            )
            paired = dict(
                require_paired_primary_goodputs(
                    {role: by_role[role] for role in ("Static", "TTS", "LightCone")}
                )
            )
            pilot_rows.append(
                PilotBlock(
                    block_id=f"E3b:{_digest(family)}:excluded_pilot:{block}",
                    static_goodput=float(paired["Static"]),
                    tts_goodput=float(paired["TTS"]),
                    lightcone_goodput=float(paired["LightCone"]),
                )
            )
            observation_ids.extend(
                (block, role, by_role[role].sha256)
                for role in ("Static", "TTS", "LightCone")
            )
        plan = _pilot_power_resolution(tuple(pilot_rows), exclusions)
        family_sha = _digest({"stage": "E3b", "dimensions": family})
        row = {
            "family_sha256": family_sha,
            "dimensions": [list(value) for value in family],
            "pilot_observation_sha256s": [
                list(value) for value in sorted(observation_ids)
            ],
            "power_sizing": asdict(plan),
            "scientific_exclusions": exclusions,
        }
        row["commitment_sha256"] = _digest(row)
        plans.append(row)
        if plan.status == "POWER_UNRESOLVED":
            power_unresolved.append(family_sha)
        elif plan.underpowered or plan.selected_final_blocks is None:
            underpowered.append(family_sha)
        else:
            selected.append(plan.selected_final_blocks)
    profiler = predecessor.decision.payload
    base = {
        "schema_version": 1,
        "kind": "formal_single_operator_e3b_power_prefix",
        "model": _text("E3b model", profiler.get("model")),
        "frozen_tts_recipe_sha256": _sha(
            "E3b frozen TTS recipe", profiler.get("frozen_tts_recipe_sha256")
        ),
        "lightcone_recipe_sha256": _sha(
            "E3b LightCone recipe", profiler.get("lightcone_recipe_sha256")
        ),
        "pilot_materialization_sha256": materialization.sha256,
        "family_commitments": plans,
        "underpowered_family_sha256s": sorted(underpowered),
        "power_unresolved_family_sha256s": sorted(power_unresolved),
    }
    if underpowered or power_unresolved:
        payload = {
            **base,
            "status": "POWER_UNRESOLVED" if power_unresolved else "UNDERPOWERED",
            "selected_final_blocks": None,
            "reason_codes": [
                *(
                    ["one_or_more_families_power_unresolved"]
                    if power_unresolved
                    else []
                ),
                *(["one_or_more_families_underpowered"] if underpowered else []),
            ],
        }
        payload_sha = _digest(payload)
        return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
            decision_kind=(
                "e3b_pilot_power_unresolved"
                if power_unresolved
                else "e3b_pilot_underpowered"
            ),
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**payload, "power_prefix_sha256": payload_sha},
        )
    final_blocks = max(selected)
    payload = {
        **base,
        "status": "READY",
        "selected_final_blocks": final_blocks,
        "selected_final_prefix": list(
            range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + final_blocks)
        ),
    }
    payload_sha = _digest(payload)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e3b_pilot_actual_1920_powered",
        next_materialization_source_decision_sha256=payload_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**payload, "power_prefix_sha256": payload_sha},
    )


def materialize_single_operator_e3b_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    """Materialize the exact powered E3b final prefix; pilots never re-enter."""

    if predecessor is None or predecessor.artifact.node != "e3b_pilot":
        raise ValueError("E3b final requires its excluded pilots")
    payload = predecessor.decision.payload
    status = payload.get("status")
    if status in {"UNDERPOWERED", "POWER_UNRESOLVED"}:
        stages = _stages()
        raise stages.FormalSingleOperatorStageBlocked(  # type: ignore[attr-defined]
            f"E3b pilot families cannot advance: {status}"
        )
    if status != "READY":
        raise ValueError("E3b pilot power status is malformed")
    count = payload.get("selected_final_blocks")
    if type(count) is not int or not 12 <= count <= 20:
        raise ValueError("E3b selected final block count differs")
    source = _sha("E3b power prefix", payload.get("power_prefix_sha256"))
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E3b power predecessor lineage differs")
    frozen_tts = _sha("E3b frozen TTS recipe", payload.get("frozen_tts_recipe_sha256"))
    lightcone = _sha("E3b LightCone recipe", payload.get("lightcone_recipe_sha256"))
    fixture = _materialize_e3b_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=predecessor.materialization.sha256,
        source_decision_sha256=source,
        model=_text("E3b model", payload.get("model")),
        frozen_tts_recipe_sha256=frozen_tts,
        lightcone_recipe_sha256=lightcone,
        final_blocks=PILOT_BLOCK_COUNT + count,
        gpu_hours=GpuHourEstimate.unmeasured(),
        lineage_dimensions={
            "pilot_materialization_sha256": predecessor.materialization.sha256,
            "power_prefix_sha256": source,
        },
    )
    expected_prefix = set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
    cells = tuple(
        cell
        for cell in fixture.cells
        if dict(cell.dimensions).get("block") in expected_prefix
    )
    if len(cells) != 480 * count:
        raise AssertionError("E3b final prefix must contain exactly 480N rows")
    return _receipt(
        stage="E3b",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule=(
            "five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _paired_request_bootstrap_rows(
    numerator: dict[str, object],
    denominator: dict[str, object],
) -> np.ndarray:
    left = {row.request_id: row for row in _request_evidence(numerator)}
    right = {row.request_id: row for row in _request_evidence(denominator)}
    if set(left) != set(right) or not left:
        raise ValueError("hierarchical bootstrap request IDs differ")
    rows = []
    for request_id in sorted(left):
        lhs = left[request_id]
        rhs = right[request_id]
        if lhs.trajectory_sha256 != rhs.trajectory_sha256:
            raise ValueError("hierarchical bootstrap token trajectories differ")
        rows.append(
            (
                len(lhs.output_token_ids) if lhs.qualifies else 0,
                lhs.request_started_ns,
                lhs.request_terminal_ns,
                len(rhs.output_token_ids) if rhs.qualifies else 0,
                rhs.request_started_ns,
                rhs.request_terminal_ns,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def _hierarchical_log_goodput(rows: np.ndarray) -> float:
    numerator_tokens = float(rows[:, 0].sum())
    denominator_tokens = float(rows[:, 3].sum())
    numerator_window = float(rows[:, 2].max() - rows[:, 1].min())
    denominator_window = float(rows[:, 5].max() - rows[:, 4].min())
    if (
        min(numerator_tokens, denominator_tokens, numerator_window, denominator_window)
        <= 0
    ):
        raise ValueError("hierarchical goodput replicate is non-positive")
    return math.log(
        (numerator_tokens / numerator_window)
        / (denominator_tokens / denominator_window)
    )


def _contrast_payload(contrast: object) -> dict[str, object]:
    payload = asdict(contrast)  # type: ignore[arg-type]
    if type(contrast) is PairedBcaContrast:
        payload.update({"status": "RESOLVED", "reason_codes": []})
    return payload


def _family_role_exclusions(
    blocks: dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
) -> dict[str, dict[str, object]]:
    """Return deterministic scientific exclusions without hiding malformed rows."""

    reasons: dict[str, set[str]] = {}
    evidence: dict[str, set[str]] = {}
    for block in sorted(blocks):
        for role, (cell, observation) in blocks[block].items():
            require_update = role not in {"Target-only", "Static"}
            observed = _stages()._adaptive_safety_reasons(  # type: ignore[attr-defined]
                observation,
                require_published_update=require_update,
            )
            if observed:
                reasons.setdefault(role, set()).update(observed)
                evidence.setdefault(role, set()).add(cell.cell_id)
    return {
        role: {
            "reason_codes": sorted(values),
            "evidence_cell_ids": sorted(evidence[role]),
        }
        for role, values in sorted(reasons.items())
    }


def _excluded_contrast_payload(
    name: str,
    roles: tuple[str, str],
    exclusions: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    affected = tuple(role for role in roles if role in exclusions)
    if not affected:
        return None
    reason_codes = sorted(
        {
            f"{role}:{reason}"
            for role in affected
            for reason in exclusions[role]["reason_codes"]  # type: ignore[union-attr]
        }
    )
    evidence_ids = sorted(
        {
            cell_id
            for role in affected
            for cell_id in exclusions[role]["evidence_cell_ids"]  # type: ignore[union-attr]
        }
    )
    return {
        "name": name,
        "status": "EXCLUDED_UNSAFE_OR_INACTIVE",
        "reason_codes": reason_codes,
        "excluded_roles": list(affected),
        "evidence_cell_ids": evidence_ids,
        "independent_unit": "paired_block",
    }


def _resolve_family_contrasts(
    *,
    paired: dict[str, dict[str, tuple[float, float]]],
    contrast_roles: dict[str, tuple[str, str]],
    exclusions: dict[str, dict[str, object]],
    target_slo_pass: bool,
) -> dict[str, object]:
    """Reduce every independently valid contrast and seal invalid ones visibly."""

    resolved: dict[str, PairedBcaContrast | UnresolvedPairedContrast] = {}
    payloads: dict[str, dict[str, object]] = {}
    for name, roles in contrast_roles.items():
        excluded = _excluded_contrast_payload(name, roles, exclusions)
        if excluded is not None:
            payloads[name] = excluded
            continue
        resolution = resolve_paired_bca_contrast(name, paired[name])
        resolved[name] = resolution
        payloads[name] = _contrast_payload(resolution)

    primary_objects = {
        name: resolved[name]
        for name in PRIMARY_CONTRASTS
        if type(resolved.get(name)) is PairedBcaContrast
    }
    if set(primary_objects) == set(PRIMARY_CONTRASTS):
        holm = holm_primary_contrasts(primary_objects)  # type: ignore[arg-type]
        holm_payload = [asdict(row) for row in holm]
        primary_confirmed = all(
            row.rejected and primary_objects[row.name].ci_lower_relative_gain > 0  # type: ignore[union-attr]
            for row in holm
        )
        holm_status = "RESOLVED"
    else:
        holm_payload = []
        primary_confirmed = False
        holm_status = "UNRESOLVED"

    target = resolved.get("lightcone_vs_target_only")
    target_effect_pass = (
        type(target) is PairedBcaContrast and target.ci_lower_relative_gain > 0
    )
    target_reasons = []
    if not target_slo_pass:
        target_reasons.append("target_only_slo_failed")
    target_payload = payloads.get("lightcone_vs_target_only")
    if target_payload is None:
        raise ValueError("deployment contrast is absent")
    if target_payload["status"] != "RESOLVED":
        target_reasons.extend(target_payload.get("reason_codes", []))  # type: ignore[arg-type]
    elif not target_effect_pass:
        target_reasons.append("lightcone_did_not_outperform_target_only")
    target_gate = {
        "passed": bool(target_slo_pass and target_effect_pass),
        "target_only_slo_passed": target_slo_pass,
        "contrast": target_payload,
        "reason_codes": sorted(set(target_reasons)),
    }
    family_reasons = {
        reason
        for payload in payloads.values()
        if payload["status"] != "RESOLVED"
        for reason in payload.get("reason_codes", [])  # type: ignore[union-attr]
    }
    family_reasons.update(target_gate["reason_codes"])  # type: ignore[arg-type]
    if holm_status != "RESOLVED":
        family_reasons.add("primary_holm_family_unresolved")
    elif not primary_confirmed:
        family_reasons.add("primary_holm_gate_failed")
    all_registered_contrasts_resolved = all(
        payload["status"] == "RESOLVED" for payload in payloads.values()
    )
    if not all_registered_contrasts_resolved:
        family_reasons.add("registered_contrast_family_incomplete")
    return {
        "contrast_payloads": payloads,
        "holm_decisions": holm_payload,
        "holm_status": holm_status,
        "primary_confirmed": primary_confirmed,
        "target_only_gate": target_gate,
        "deployment_confirmed": bool(
            primary_confirmed
            and target_gate["passed"]
            and all_registered_contrasts_resolved
        ),
        "all_registered_contrasts_resolved": all_registered_contrasts_resolved,
        "reason_codes": sorted(family_reasons),
    }


def _hierarchical_interval_payload(
    name: str,
    blocks: dict[str, np.ndarray],
) -> dict[str, object]:
    """Resolve one hierarchical interval without emitting a degenerate CI."""

    block_effects = []
    saw_zero_goodput = False
    for block_id in sorted(blocks):
        rows = blocks[block_id]
        if (
            type(rows) is not np.ndarray
            or rows.ndim != 2
            or rows.shape[1] != 6
            or rows.size == 0
            or not np.isfinite(rows).all()
        ):
            raise ValueError("hierarchical bootstrap rows are malformed")
        numerator_tokens = float(rows[:, 0].sum())
        denominator_tokens = float(rows[:, 3].sum())
        numerator_window = float(rows[:, 2].max() - rows[:, 1].min())
        denominator_window = float(rows[:, 5].max() - rows[:, 4].min())
        if np.any(rows[:, 0] < 0) or np.any(rows[:, 3] < 0):
            raise ValueError("hierarchical bootstrap tokens are negative")
        if numerator_window <= 0 or denominator_window <= 0:
            raise ValueError("hierarchical bootstrap window is not positive")
        if numerator_tokens == 0 or denominator_tokens == 0:
            saw_zero_goodput = True
            continue
        block_effects.append(
            math.log(
                (numerator_tokens / numerator_window)
                / (denominator_tokens / denominator_window)
            )
        )
    if saw_zero_goodput:
        return {
            "name": name,
            "status": "UNRESOLVED_ZERO_GOODPUT",
            "reason_codes": ["UNRESOLVED_ZERO_GOODPUT"],
            "independent_units": ["block", "request"],
        }
    if float(np.std(np.asarray(block_effects), ddof=1)) <= np.finfo(np.float64).tiny:
        return {
            "name": name,
            "status": "UNRESOLVED_ZERO_VARIANCE",
            "reason_codes": ["UNRESOLVED_ZERO_VARIANCE"],
            "independent_units": ["block", "request"],
        }
    try:
        interval = hierarchical_block_request_bootstrap(
            blocks,
            _hierarchical_log_goodput,
        )
    except ValueError as error:
        if "replicate is non-positive" not in str(error):
            raise
        return {
            "name": name,
            "status": "UNRESOLVED_ZERO_GOODPUT",
            "reason_codes": ["UNRESOLVED_ZERO_GOODPUT"],
            "independent_units": ["block", "request"],
        }
    return {
        "name": name,
        "status": "RESOLVED",
        "reason_codes": [],
        "mean_log_ratio": interval.estimate[0],
        "mean_relative_gain": math.expm1(interval.estimate[0]),
        "ci_lower_relative_gain": math.expm1(interval.ci_lower[0]),
        "ci_upper_relative_gain": math.expm1(interval.ci_upper[0]),
        "confidence": interval.confidence,
        "repetitions": interval.repetitions,
        "independent_units": list(interval.independent_units),
    }


def reduce_single_operator_e3b_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Reduce powered E3b with paired/Holm and hierarchical request CIs."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e3b_pilot"
        or materialization.stage != "E3b"
        or materialization.materialization_rule
        != "five_roles_x_8_contexts_x_3_regimes_x_2_loads_x_2_widths_final_only"
    ):
        raise ValueError("E3b final reducer received another node")
    count, remainder = divmod(len(materialization.cells), 480)
    if remainder or not 12 <= count <= 20:
        raise ValueError("E3b final materialization is not exactly 480N")
    observations = _serving_observations(materialization, actual_results)
    grouped = _e3b_grouped(materialization, observations)
    if len(grouped) != 96:
        raise ValueError("E3b final lacks 96 scientific families")
    expected_blocks = set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
    family_results = []
    for family in sorted(grouped, key=_digest):
        blocks = grouped[family]
        if set(blocks) != expected_blocks:
            raise ValueError("E3b family lacks every powered final block")
        exclusions = _family_role_exclusions(blocks)
        paired: dict[str, dict[str, tuple[float, float]]] = {
            "lightcone_vs_tts": {},
            "lightcone_vs_static": {},
            "l0_naive_vs_tts": {},
            "lightcone_vs_l0_naive": {},
            "lightcone_vs_target_only": {},
        }
        hierarchical: dict[str, dict[str, np.ndarray]] = {name: {} for name in paired}
        request_count = 0
        target_slo_pass = True
        for block in sorted(blocks):
            role_rows = blocks[block]
            slo = _paired_role_goodputs(role_rows, excluded_roles=frozenset(exclusions))
            block_id = f"E3b:final:{block - PILOT_BLOCK_COUNT:02d}"
            goodput = {
                role: float(row.goodput_tokens_per_second) for role, row in slo.items()
            }
            target_slo_pass &= slo["Target-only"].status == "PASS"
            for name, (left, right) in _CORE_CONTRAST_ROLES.items():
                paired[name][block_id] = (goodput[left], goodput[right])
                if left not in exclusions and right not in exclusions:
                    hierarchical[name][block_id] = _paired_request_bootstrap_rows(
                        role_rows[left][1], role_rows[right][1]
                    )
            request_count += slo["LightCone"].eligible_requests
        reduction = _resolve_family_contrasts(
            paired=paired,
            contrast_roles=_CORE_CONTRAST_ROLES,
            exclusions=exclusions,
            target_slo_pass=target_slo_pass,
        )
        hierarchical_rows = []
        contrast_payloads = reduction["contrast_payloads"]
        assert isinstance(contrast_payloads, dict)
        for name in paired:
            contrast = contrast_payloads[name]
            if contrast["status"] == "EXCLUDED_UNSAFE_OR_INACTIVE":
                hierarchical_rows.append(
                    {
                        "name": name,
                        "status": "EXCLUDED_UNSAFE_OR_INACTIVE",
                        "reason_codes": contrast["reason_codes"],
                        "evidence_cell_ids": contrast["evidence_cell_ids"],
                        "independent_units": ["block", "request"],
                    }
                )
            else:
                hierarchical_rows.append(
                    _hierarchical_interval_payload(name, hierarchical[name])
                )
        primary_hierarchical_resolved = all(
            row["status"] == "RESOLVED"
            for row in hierarchical_rows
            if row["name"] in PRIMARY_CONTRASTS
        )
        reason_codes = set(reduction["reason_codes"])
        if not primary_hierarchical_resolved:
            reason_codes.add("primary_hierarchical_interval_unresolved")
        family_row = {
            "family_sha256": _digest({"stage": "E3b", "dimensions": family}),
            "dimensions": [list(value) for value in family],
            "block_count": count,
            "request_count": request_count,
            "paired": True,
            "primary_contrasts": [
                contrast_payloads[name] for name in PRIMARY_CONTRASTS
            ],
            "holm_decisions": reduction["holm_decisions"],
            "holm_status": reduction["holm_status"],
            "all_registered_contrasts_resolved": reduction[
                "all_registered_contrasts_resolved"
            ],
            "mechanism_contrasts": [
                contrast_payloads[name]
                for name in ("l0_naive_vs_tts", "lightcone_vs_l0_naive")
            ],
            "target_only_gate": reduction["target_only_gate"],
            "hierarchical_intervals": hierarchical_rows,
            "scientific_exclusions": exclusions,
            "reason_codes": sorted(reason_codes),
            "reducer": "paired_block_bca_plus_block_request_hierarchical_bootstrap",
            "status": (
                "CONFIRMED"
                if reduction["deployment_confirmed"] and primary_hierarchical_resolved
                else "NOT_CONFIRMED"
            ),
        }
        family_row["result_sha256"] = _digest(family_row)
        family_results.append(family_row)
    pilot = predecessor.decision.payload
    result = {
        "schema_version": 1,
        "kind": "formal_single_operator_e3b_confirmation",
        "status": (
            "CONFIRMED"
            if all(row["status"] == "CONFIRMED" for row in family_results)
            else "NOT_CONFIRMED"
        ),
        "model": _text("E3b model", pilot.get("model")),
        "frozen_tts_recipe_sha256": _sha(
            "E3b frozen TTS recipe", pilot.get("frozen_tts_recipe_sha256")
        ),
        "lightcone_recipe_sha256": _sha(
            "E3b LightCone recipe", pilot.get("lightcone_recipe_sha256")
        ),
        "block_count": count,
        "materialization_sha256": materialization.sha256,
        "family_results": family_results,
    }
    result_sha = _digest(result)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e3b_final_actual_reduced",
        next_materialization_source_decision_sha256=result_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**result, "confirmation_sha256": result_sha},
    )


def materialize_single_operator_e1a(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    if predecessor is None or predecessor.artifact.node != "e3b_final":
        raise ValueError("E1a requires exact completed E3b final")
    payload = predecessor.decision.payload
    source = _sha("E3b confirmation", payload.get("confirmation_sha256"))
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E1a predecessor lineage differs")
    return _materialize_e1a_diagnostic(
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256=predecessor.materialization.sha256,
        source_decision_sha256=source,
        model=_text("E1a model", payload.get("model")),
        frozen_tts_recipe_sha256=_sha(
            "E1a frozen TTS recipe", payload.get("frozen_tts_recipe_sha256")
        ),
        lightcone_recipe_sha256=_sha(
            "E1a source LightCone recipe", payload.get("lightcone_recipe_sha256")
        ),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def reduce_single_operator_e1a(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Freeze one DSpark recipe from all 56 configurations and two modes."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e3b_final"
        or materialization.stage != "E1a"
        or materialization.materialization_rule
        != "58_configurations_x_2_verification_modes"
        or len(materialization.cells) != 116
    ):
        raise ValueError("E1a reducer requires the exact 116-row grid")
    observations = _serving_observations(materialization, actual_results)
    by_mode: dict[str, list[MaterializedCell]] = {}
    static: dict[str, MaterializedCell] = {}
    candidates: dict[tuple[tuple[str, str | int], ...], list[MaterializedCell]] = {}
    source_recipes = {
        cell.recipe_sha256
        for cell in materialization.cells
        if cell.method_role == "LightCone-candidate"
    }
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        mode = dimensions.get("verification_mode")
        if mode not in {"fixed_verification_budget", "native_scheduler"}:
            raise ValueError("E1a verification mode differs")
        if dimensions.get("fixed_verification_budget") != (
            E1A_FIXED_VERIFICATION_BUDGET
            if mode == "fixed_verification_budget"
            else E1A_NATIVE_VERIFICATION_BUDGET
        ):
            raise ValueError("E1a fixed verification budget differs")
        by_mode.setdefault(str(mode), []).append(cell)
        if cell.method_role == "Static":
            if mode in static:
                raise ValueError("E1a repeats a Static anchor")
            static[str(mode)] = cell
        elif cell.method_role == "LightCone-candidate":
            configuration = (
                ("parameterization", dimensions["parameterization"]),
                ("rank", dimensions["rank"]),
                ("scope", dimensions["scope"]),
            )
            candidates.setdefault(configuration, []).append(cell)
    if (
        set(by_mode) != {"fixed_verification_budget", "native_scheduler"}
        or set(static) != set(by_mode)
        or any(len(rows) != 58 for rows in by_mode.values())
        or len(candidates) != 56
        or any(len(rows) != 2 for rows in candidates.values())
        or len(source_recipes) != 1
        or None in source_recipes
    ):
        raise ValueError("E1a configuration/mode coverage differs")
    for rows in by_mode.values():
        identities = {
            stages._single_operator_request_identity(observations[cell.cell_id])  # type: ignore[attr-defined]
            for cell in rows
        }
        if len(identities) != 1:
            raise ValueError("E1a verification requests are unpaired")
    static_reasons = {
        mode: tuple(
            stages._adaptive_safety_reasons(  # type: ignore[attr-defined]
                observations[cell.cell_id],
                require_published_update=False,
            )
        )
        for mode, cell in static.items()
    }
    evaluations = []
    for configuration, rows in candidates.items():
        reasons = {
            f"{mode}:{reason}"
            for mode, values in static_reasons.items()
            for reason in values
        }
        reasons.update(
            f"{dict(cell.dimensions)['verification_mode']}:{reason}"
            for cell in rows
            for reason in stages._adaptive_safety_reasons(  # type: ignore[attr-defined]
                observations[cell.cell_id],
                require_published_update=True,
            )
        )
        lower = []
        peaks = []
        p99 = []
        exposed = []
        for cell in rows:
            mode = str(dict(cell.dimensions)["verification_mode"])
            if reasons:
                continue
            candidate_metrics = stages._request_metrics(observations[cell.cell_id])  # type: ignore[attr-defined]
            static_metrics = stages._request_metrics(  # type: ignore[attr-defined]
                observations[static[mode].cell_id]
            )
            lower.append(
                stages._paired_confidence_lower(candidate_metrics, static_metrics)
            )  # type: ignore[attr-defined]
            peaks.append(
                int(stages._counter(observations[cell.cell_id], "peak_hbm_bytes"))
            )  # type: ignore[attr-defined]
            p99.append(
                max(
                    math.ceil(Fraction(metric["p99_itl_ns"]) / 1_000)
                    for metric in candidate_metrics
                )
            )
            exposed.append(
                math.ceil(
                    stages._finite_counter(
                        observations[cell.cell_id], "exposed_update_ms"
                    )
                    * 1_000
                )  # type: ignore[attr-defined]
            )
        row = {
            "configuration": [list(value) for value in configuration],
            "cell_ids": sorted(cell.cell_id for cell in rows),
            "eligible": not reasons,
            "reason_codes": sorted(reasons),
            "evidence_ids": sorted(
                {
                    *(cell.cell_id for cell in rows),
                    *(
                        actual.result_identity_sha256
                        for cell in rows
                        for actual in actual_results
                        if actual.cell_id == cell.cell_id
                    ),
                    *(cell.cell_id for cell in static.values()),
                    *(
                        actual.result_identity_sha256
                        for cell in static.values()
                        for actual in actual_results
                        if actual.cell_id == cell.cell_id
                    ),
                }
            ),
            "minimum_confidence_lower_request_rate_ratio": (
                min(lower) if not reasons else None
            ),
            "peak_hbm_bytes": max(peaks) if not reasons else None,
            "p99_itl_us": max(p99) if not reasons else None,
            "exposed_update_us": max(exposed) if not reasons else None,
        }
        row["evaluation_sha256"] = _digest(row)
        evaluations.append(row)
    eligible = [row for row in evaluations if row["eligible"] is True]
    source_recipe = next(iter(source_recipes))
    assert source_recipe is not None
    frozen_tts = {
        dict(cell.dimensions).get("frozen_tts_recipe_sha256")
        for cell in materialization.cells
    }
    models = {cell.model for cell in materialization.cells}
    if len(frozen_tts) != 1 or None in frozen_tts or len(models) != 1:
        raise ValueError("E1a model/TTS identity differs")
    if not eligible:
        negative = {
            "schema_version": 1,
            "kind": "formal_single_operator_e1a_verification",
            "status": "NO_SAFE_CONFIGURATION",
            "model": next(iter(models)),
            "frozen_tts_recipe_sha256": next(iter(frozen_tts)),
            "source_lightcone_recipe_sha256": source_recipe,
            "selected_configuration": None,
            "selected_dspark_recipe_sha256": None,
            "evaluations": sorted(
                evaluations, key=lambda row: str(row["evaluation_sha256"])
            ),
            "reason_codes": ["no_safe_e1a_configuration"],
        }
        result_sha = _digest(negative)
        return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
            decision_kind="e1a_no_safe_configuration",
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**negative, "verification_sha256": result_sha},
        )
    winner = min(
        eligible,
        key=lambda row: (
            -float(row["minimum_confidence_lower_request_rate_ratio"]),
            int(row["peak_hbm_bytes"]),
            int(row["p99_itl_us"]),
            int(row["exposed_update_us"]),
            str(row["evaluation_sha256"]),
        ),
    )
    selected_recipe = _digest(
        {
            "schema_version": 1,
            "kind": "lightcone_e1a_selected_dspark_recipe",
            "source_lightcone_recipe_sha256": source_recipe,
            "configuration": winner["configuration"],
            "rule": (
                "max_minimum_confidence_lower_ratio_then_min_hbm_p99_exposed_digest"
            ),
        }
    )
    result = {
        "schema_version": 1,
        "kind": "formal_single_operator_e1a_verification",
        "status": "READY",
        "model": next(iter(models)),
        "frozen_tts_recipe_sha256": next(iter(frozen_tts)),
        "source_lightcone_recipe_sha256": source_recipe,
        "selected_configuration": winner["configuration"],
        "selected_dspark_recipe_sha256": selected_recipe,
        "evaluations": sorted(
            evaluations, key=lambda row: str(row["evaluation_sha256"])
        ),
    }
    result_sha = _digest(result)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e1a_actual_116_reduced",
        next_materialization_source_decision_sha256=result_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**result, "verification_sha256": result_sha},
    )


def materialize_single_operator_e5_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    """Materialize exactly 1,800 E5 headline pilots and no diagnostics."""

    if predecessor is None or predecessor.artifact.node != "e1a":
        raise ValueError("E5 pilots require completed E1a verification")
    payload = predecessor.decision.payload
    status = payload.get("status")
    if status == "NO_SAFE_CONFIGURATION":
        stages = _stages()
        raise stages.FormalSingleOperatorStageBlocked(  # type: ignore[attr-defined]
            f"E5 cannot advance from E1a: {status}"
        )
    if status != "READY":
        raise ValueError("E1a selection status is malformed")
    source = _sha("E1a verification", payload.get("verification_sha256"))
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E5 pilot predecessor lineage differs")
    frozen_tts = _sha("E5 frozen TTS recipe", payload.get("frozen_tts_recipe_sha256"))
    dflash = _sha(
        "E5 DFlash LightCone recipe",
        payload.get("source_lightcone_recipe_sha256"),
    )
    dspark = _sha(
        "E5 DSpark LightCone recipe",
        payload.get("selected_dspark_recipe_sha256"),
    )
    cells = _e5_headline_cells(
        model=_text("E5 model", payload.get("model")),
        frozen_tts_recipe_sha256=frozen_tts,
        dflash_lightcone_recipe_sha256=dflash,
        dspark_lightcone_recipe_sha256=dspark,
        blocks=PILOT_BLOCK_COUNT,
        anchors=(),
        anchor_receipt_sha256=None,
        lineage_dimensions={
            "upstream_e1a_verification_sha256": source,
            "frozen_tts_recipe_sha256": frozen_tts,
            "dflash_lightcone_recipe_sha256": dflash,
            "dspark_lightcone_recipe_sha256": dspark,
        },
    )
    if len(cells) != 450 * PILOT_BLOCK_COUNT:
        raise AssertionError("E5 excluded pilots must contain exactly 1,800 rows")
    return _receipt(
        stage="E5",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule="e5_exact_450_headline_rows_x_4_excluded_pilot_blocks",
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e5_family(
    dimensions: dict[str, object],
) -> tuple[tuple[str, str | int | float], ...]:
    family = dimensions.get("family")
    if family == "closed_loop":
        variant = ("concurrency", dimensions.get("concurrency"))
    elif family == "open_loop":
        variant = ("load_factor", dimensions.get("load_factor"))
    elif family == "trace_or_soak":
        variant = ("arrival", dimensions.get("arrival"))
    elif family == "topology_cohort":
        variant = (
            ("cohort_count", dimensions.get("cohort_count")),
            ("cohort_distribution", dimensions.get("cohort_distribution")),
        )
    else:
        raise ValueError("E5 headline family is outside the registered matrix")
    variants = (variant,) if type(variant[0]) is str else variant
    rows = (
        ("backend_authority", dimensions.get("backend_authority")),
        ("family", family),
        ("family_id", dimensions.get("family_id")),
        ("topology", dimensions.get("topology")),
        *variants,
    )
    if any(
        type(name) is not str or type(value) not in {str, int, float}
        for name, value in rows
    ):
        raise ValueError("E5 headline family axes are incomplete")
    return tuple(sorted(rows))  # type: ignore[return-value]


def _e5_grouped(
    materialization: StageMaterializationReceipt,
    observations: dict[str, dict[str, object]],
) -> dict[
    tuple[tuple[str, str | int | float], ...],
    dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
]:
    grouped: dict[
        tuple[tuple[str, str | int | float], ...],
        dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
    ] = {}
    for cell in materialization.cells:
        if cell.task != "production_slo_power_prefix":
            continue
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        if type(block) is not int:
            raise ValueError("E5 headline cell lacks an integer block")
        family = _e5_family(dimensions)
        by_role = grouped.setdefault(family, {}).setdefault(block, {})
        if cell.method_role in by_role or cell.method_role not in FORMAL_METHOD_ROLES:
            raise ValueError("E5 headline repeats or changes a method role")
        observation = observations[cell.cell_id]
        by_role[cell.method_role] = (cell, observation)
    return grouped


def _maximum_request_p99_ns(observation: dict[str, object]) -> int:
    events = []
    for row in _request_evidence(observation):
        if not row.completed:
            continue
        events.extend(
            right - left
            for left, right in zip(
                row.token_observed_ns,
                row.token_observed_ns[1:],
                strict=False,
            )
        )
    value = linear_p99_ns(events)
    if value is None:
        raise ValueError("E5 p99 anchor lacks native ITL samples")
    return math.ceil(value)


def _linear_native_p99_ms(rows: np.ndarray) -> float:
    """Apply the registered exact linear p99 to bootstrap-resampled ITLs."""

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("E5 p99 bootstrap rows must be finite and non-empty")
    integral = values.astype(np.int64)
    if np.any(values != integral) or np.any(integral < 0):
        raise ValueError("E5 p99 bootstrap rows must be non-negative integer ns")
    value = linear_p99_ns(tuple(int(row) for row in integral))
    if value is None:
        raise ValueError("E5 p99 bootstrap has no native ITL samples")
    return float(value / 1_000_000)


def _e5_p99_anchor_row(
    anchor: E5SelectedP99Anchor,
    block_observations: dict[
        int,
        tuple[MaterializedCell, dict[str, object]],
    ],
    *,
    expected_blocks: set[int],
    selection_receipt_sha256: str,
) -> dict[str, object]:
    """Reduce one locked p99 anchor without pooling away a failed block gate."""

    if type(anchor) is not E5SelectedP99Anchor:
        raise TypeError("E5 p99 reduction requires an exact selected anchor")
    selection_receipt_sha256 = _sha(
        "E5 p99 selection receipt",
        selection_receipt_sha256,
    )
    if (
        type(expected_blocks) is not set
        or len(expected_blocks) < 2
        or any(type(block) is not int or block < 0 for block in expected_blocks)
        or set(block_observations) != expected_blocks
    ):
        raise ValueError("E5 p99 anchor lacks every expected independent block")

    total_offered = 0
    total_completed = 0
    aggregate_status_counts: dict[str, int] = {}
    bootstrap_rows: dict[str, np.ndarray] = {}
    block_rows = []
    failed_minimum_blocks = []
    missing_itl_blocks = []
    safety_reasons: set[str] = set()
    safety_evidence_cell_ids: set[str] = set()
    for block in sorted(expected_blocks):
        cell, observation = block_observations[block]
        dimensions = dict(cell.dimensions)
        if (
            cell.stage != "E5"
            or cell.method_role != "LightCone"
            or cell.task != "production_slo_power_prefix"
            or cell.backend != anchor.backend
            or dimensions.get("block") != block
            or dimensions.get("backend_authority") != anchor.backend
            or dimensions.get("topology") != anchor.topology
            or dimensions.get("family_id") != anchor.family_id
            or dimensions.get("p99_anchor_id") != anchor.anchor_id
            or dimensions.get("p99_minimum_completions") != anchor.minimum_completions
            or dimensions.get("p99_selection_receipt_sha256")
            != selection_receipt_sha256
        ):
            raise ValueError("E5 p99 anchor cell binding differs")

        requests = _request_evidence(observation)
        if len({row.request_id for row in requests}) != len(requests):
            raise ValueError("E5 p99 anchor repeats a request ID within a block")
        stages = _stages()
        raw_requests = stages._single_operator_request_rows(observation)  # type: ignore[attr-defined]
        if len(raw_requests) != len(requests):
            raise ValueError("E5 p99 offered-request denominator differs")
        observed_safety_reasons = stages._adaptive_safety_reasons(  # type: ignore[attr-defined]
            observation,
            require_published_update=True,
        )
        if observed_safety_reasons:
            safety_reasons.update(
                f"LightCone:{reason}" for reason in observed_safety_reasons
            )
            safety_evidence_cell_ids.add(cell.cell_id)
        status_counts: dict[str, int] = {}
        for raw in raw_requests:
            status = _text("E5 p99 terminal status", raw.get("terminal_status"))
            status_counts[status] = status_counts.get(status, 0) + 1
            aggregate_status_counts[status] = aggregate_status_counts.get(status, 0) + 1
        completed = sum(int(row.completed) for row in requests)
        if status_counts.get("completed", 0) != completed:
            raise ValueError("E5 p99 completed-request denominator differs")
        offered = len(requests)
        intervals = tuple(
            right - left
            for row in requests
            if row.completed
            for left, right in zip(
                row.token_observed_ns,
                row.token_observed_ns[1:],
                strict=False,
            )
        )
        if completed < anchor.minimum_completions:
            failed_minimum_blocks.append(block)
        if not intervals:
            missing_itl_blocks.append(block)
        else:
            bootstrap_rows[f"E5:final:{block:02d}"] = np.asarray(
                intervals,
                dtype=np.float64,
            )
        total_offered += offered
        total_completed += completed
        block_rows.append(
            {
                "block": block,
                "offered_request_count": offered,
                "completed_request_count": completed,
                "incomplete_request_count": offered - completed,
                "terminal_status_counts": {
                    name: status_counts[name] for name in sorted(status_counts)
                },
                "minimum_completed_request_count": anchor.minimum_completions,
                "minimum_completion_gate": (
                    "PASS" if completed >= anchor.minimum_completions else "FAIL"
                ),
                "native_itl_sample_count": len(intervals),
            }
        )

    base: dict[str, object] = {
        "anchor_id": anchor.anchor_id,
        "backend": anchor.backend,
        "topology": anchor.topology,
        "family_id": anchor.family_id,
        "block_count": len(expected_blocks),
        "independent_block_count": len(expected_blocks),
        "request_count": total_offered,
        "offered_request_count": total_offered,
        "completed_request_count": total_completed,
        "incomplete_request_count": total_offered - total_completed,
        "terminal_status_counts": {
            name: aggregate_status_counts[name]
            for name in sorted(aggregate_status_counts)
        },
        "minimum_completions_per_block": anchor.minimum_completions,
        "required_completed_requests_across_blocks": (
            anchor.minimum_completions * len(expected_blocks)
        ),
        "paired": False,
        "block_evidence": block_rows,
        "registered_reducer_method": _E5_P99_REDUCER_METHOD,
    }
    if safety_reasons:
        return {
            "anchor_id": anchor.anchor_id,
            "completed_requests": total_completed,
            "observed_p99_ms": None,
            "minimum_completions": anchor.minimum_completions,
            **base,
            "status": "EXCLUDED_UNSAFE_OR_INACTIVE",
            "reason_codes": sorted(safety_reasons),
            "excluded_roles": ["LightCone"],
            "evidence_cell_ids": sorted(safety_evidence_cell_ids),
            "failed_minimum_blocks": failed_minimum_blocks,
            "missing_itl_blocks": missing_itl_blocks,
        }
    if failed_minimum_blocks or missing_itl_blocks:
        claim = guard_p99_claim(
            anchor.anchor_id,
            completed_requests=total_completed,
            observed_p99_ms=None,
            minimum_completions=anchor.minimum_completions,
            preregistered_anchor_locked=False,
        )
        return {
            **asdict(claim),
            **base,
            "status": "UNRESOLVED",
            "reason_codes": [
                *(
                    ["per_block_minimum_completions_not_met"]
                    if failed_minimum_blocks
                    else []
                ),
                *(["native_itl_samples_missing"] if missing_itl_blocks else []),
            ],
            "failed_minimum_blocks": failed_minimum_blocks,
            "missing_itl_blocks": missing_itl_blocks,
        }

    interval = time_block_bootstrap(
        bootstrap_rows,
        _linear_native_p99_ms,
        repetitions=_E5_P99_BOOTSTRAP_REPETITIONS,
        seed=0,
    )
    if (
        interval.confidence != 0.95
        or interval.independent_units != ("time_block",)
        or len(interval.estimate) != 1
        or len(interval.ci_lower) != 1
        or len(interval.ci_upper) != 1
    ):
        raise ValueError("E5 p99 registered bootstrap contract differs")
    point_estimate = interval.estimate[0]
    claim = guard_p99_claim(
        anchor.anchor_id,
        completed_requests=total_completed,
        observed_p99_ms=point_estimate,
        minimum_completions=anchor.minimum_completions,
        preregistered_anchor_locked=True,
    )
    return {
        **asdict(claim),
        **base,
        "point_estimate": point_estimate,
        "ci_low": interval.ci_lower[0],
        "ci_high": interval.ci_upper[0],
        "confidence": interval.confidence,
        "bootstrap_repetitions": interval.repetitions,
        "independent_units": list(interval.independent_units),
        "reducer_method": _E5_P99_REDUCER_METHOD,
        "metric_name": "native_p99_itl_ms",
        "reason_codes": [],
        "failed_minimum_blocks": [],
        "missing_itl_blocks": [],
    }


def reduce_single_operator_e5_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Power E5 families and lock six backend/topology p99 anchors."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e1a"
        or materialization.stage != "E5"
        or materialization.materialization_rule
        != "e5_exact_450_headline_rows_x_4_excluded_pilot_blocks"
        or len(materialization.cells) != 1_800
    ):
        raise ValueError("E5 pilot reducer received another node")
    observations = _serving_observations(materialization, actual_results)
    grouped = _e5_grouped(materialization, observations)
    if len(grouped) != 90:
        raise ValueError("E5 pilot lacks exactly 90 backend/family strata")
    commitments = []
    underpowered = []
    power_unresolved = []
    selected = []
    anchor_subjects: dict[
        tuple[str, str],
        list[
            tuple[
                str,
                dict[
                    int,
                    dict[str, tuple[MaterializedCell, dict[str, object]]],
                ],
            ]
        ],
    ] = {}
    for family in sorted(grouped, key=_digest):
        blocks = grouped[family]
        if set(blocks) != set(range(PILOT_BLOCK_COUNT)):
            raise ValueError("E5 family lacks exactly four excluded pilots")
        exclusions = _family_role_exclusions(blocks)
        pilot_rows = []
        observations_sha = []
        for block in range(PILOT_BLOCK_COUNT):
            slo = _paired_role_goodputs(
                blocks[block], excluded_roles=frozenset(exclusions)
            )
            paired = dict(
                require_paired_primary_goodputs(
                    {role: slo[role] for role in ("Static", "TTS", "LightCone")}
                )
            )
            pilot_rows.append(
                PilotBlock(
                    block_id=f"E5:{_digest(family)}:excluded_pilot:{block}",
                    static_goodput=float(paired["Static"]),
                    tts_goodput=float(paired["TTS"]),
                    lightcone_goodput=float(paired["LightCone"]),
                )
            )
            observations_sha.extend(
                (block, role, slo[role].sha256)
                for role in ("Static", "TTS", "LightCone")
            )
        plan = _pilot_power_resolution(tuple(pilot_rows), exclusions)
        family_sha = _digest({"stage": "E5", "dimensions": family})
        row = {
            "family_sha256": family_sha,
            "dimensions": [list(value) for value in family],
            "pilot_observation_sha256s": [
                list(value) for value in sorted(observations_sha)
            ],
            "power_sizing": asdict(plan),
            "scientific_exclusions": exclusions,
        }
        row["commitment_sha256"] = _digest(row)
        commitments.append(row)
        if plan.status == "POWER_UNRESOLVED":
            power_unresolved.append(family_sha)
        elif plan.underpowered or plan.selected_final_blocks is None:
            underpowered.append(family_sha)
        else:
            selected.append(plan.selected_final_blocks)
        axes = dict(family)
        backend = str(axes["backend_authority"])
        topology = str(axes["topology"])
        family_id = str(axes["family_id"])
        anchor_subjects.setdefault((backend, topology), []).append((family_id, blocks))
    anchors = []
    if not underpowered and not power_unresolved:
        for backend in E5_BACKENDS:
            for topology in E5_TOPOLOGIES:
                subjects = anchor_subjects.get((backend, topology), [])
                if not subjects:
                    raise ValueError("E5 p99 anchor panel lacks a backend/topology")
                candidates = [
                    (
                        max(
                            _maximum_request_p99_ns(blocks[block]["LightCone"][1])
                            for block in range(PILOT_BLOCK_COUNT)
                        ),
                        family_id,
                    )
                    for family_id, blocks in subjects
                ]
                _p99, family_id = max(candidates, key=lambda row: (row[0], row[1]))
                anchors.append(
                    E5SelectedP99Anchor(
                        backend=backend,
                        topology=topology,
                        family_id=family_id,
                        minimum_completions=10_000,
                    )
                )
    e1a = predecessor.decision.payload
    base = {
        "schema_version": 1,
        "kind": "formal_single_operator_e5_power_and_anchor_prefix",
        "model": _text("E5 model", e1a.get("model")),
        "frozen_tts_recipe_sha256": _sha(
            "E5 frozen TTS recipe", e1a.get("frozen_tts_recipe_sha256")
        ),
        "dflash_lightcone_recipe_sha256": _sha(
            "E5 DFlash recipe", e1a.get("source_lightcone_recipe_sha256")
        ),
        "dspark_lightcone_recipe_sha256": _sha(
            "E5 DSpark recipe", e1a.get("selected_dspark_recipe_sha256")
        ),
        "pilot_materialization_sha256": materialization.sha256,
        "family_commitments": commitments,
        "p99_anchors": [
            {
                "backend": row.backend,
                "topology": row.topology,
                "family_id": row.family_id,
                "minimum_completions": row.minimum_completions,
                "anchor_id": row.anchor_id,
            }
            for row in sorted(anchors, key=lambda row: row.anchor_id)
        ],
        "underpowered_family_sha256s": sorted(underpowered),
        "power_unresolved_family_sha256s": sorted(power_unresolved),
    }
    if underpowered or power_unresolved:
        payload = {
            **base,
            "status": "POWER_UNRESOLVED" if power_unresolved else "UNDERPOWERED",
            "selected_final_blocks": None,
            "reason_codes": [
                *(
                    ["one_or_more_families_power_unresolved"]
                    if power_unresolved
                    else []
                ),
                *(["one_or_more_families_underpowered"] if underpowered else []),
            ],
        }
        result_sha = _digest(payload)
        return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
            decision_kind=(
                "e5_pilot_power_unresolved"
                if power_unresolved
                else "e5_pilot_underpowered"
            ),
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**payload, "power_prefix_sha256": result_sha},
        )
    final_blocks = max(selected)
    payload = {
        **base,
        "status": "READY",
        "selected_final_blocks": final_blocks,
        "selected_final_prefix": list(
            range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + final_blocks)
        ),
    }
    result_sha = _digest(payload)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e5_pilot_actual_1800_powered",
        next_materialization_source_decision_sha256=result_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**payload, "power_prefix_sha256": result_sha},
    )


def _anchors_from_payload(value: object) -> tuple[E5SelectedP99Anchor, ...]:
    if type(value) is not list:
        raise TypeError("E5 p99 anchors must be a JSON array")
    anchors = []
    for item in value:
        if type(item) is not dict:
            raise TypeError("E5 p99 anchor must be a JSON object")
        row = dict(item)
        declared = row.pop("anchor_id", None)
        anchor = E5SelectedP99Anchor(**row)  # type: ignore[arg-type]
        if anchor.anchor_id != declared:
            raise ValueError("E5 p99 anchor identity differs")
        anchors.append(anchor)
    result = tuple(sorted(anchors, key=lambda row: row.anchor_id))
    if len(result) != 6 or len({row.anchor_id for row in result}) != 6:
        raise ValueError("E5 requires exactly six distinct p99 anchors")
    return result


def materialize_single_operator_e5_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
) -> StageMaterializationReceipt:
    """Materialize 450N final headlines plus 264 one-shot failures."""

    stages = _stages()
    if predecessor is None or predecessor.artifact.node != "e5_pilot":
        raise ValueError("E5 final requires its excluded pilots")
    payload = predecessor.decision.payload
    status = payload.get("status")
    if status in {"UNDERPOWERED", "POWER_UNRESOLVED"}:
        raise stages.FormalSingleOperatorStageBlocked(  # type: ignore[attr-defined]
            f"E5 pilot families cannot advance: {status}"
        )
    if status != "READY":
        raise ValueError("E5 pilot power status is malformed")
    count = payload.get("selected_final_blocks")
    if type(count) is not int or not 12 <= count <= 20:
        raise ValueError("E5 selected final block count differs")
    source = _sha("E5 power prefix", payload.get("power_prefix_sha256"))
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E5 power predecessor lineage differs")
    anchors = _anchors_from_payload(payload.get("p99_anchors"))
    model = _text("E5 model", payload.get("model"))
    frozen_tts = _sha("E5 frozen TTS recipe", payload.get("frozen_tts_recipe_sha256"))
    dflash = _sha("E5 DFlash recipe", payload.get("dflash_lightcone_recipe_sha256"))
    dspark = _sha("E5 DSpark recipe", payload.get("dspark_lightcone_recipe_sha256"))
    headline_fixture = _e5_headline_cells(
        model=model,
        frozen_tts_recipe_sha256=frozen_tts,
        dflash_lightcone_recipe_sha256=dflash,
        dspark_lightcone_recipe_sha256=dspark,
        blocks=PILOT_BLOCK_COUNT + count,
        anchors=anchors,
        anchor_receipt_sha256=source,
        lineage_dimensions={
            "pilot_materialization_sha256": predecessor.materialization.sha256,
            "power_prefix_sha256": source,
        },
    )
    expected_blocks = set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
    headline = tuple(
        cell
        for cell in headline_fixture
        if dict(cell.dimensions).get("block") in expected_blocks
    )
    authority = default_e5_failure_diagnostic_authority()
    failures = tuple(
        _cell(
            stage="E5",
            method_role="LightCone",
            model=model,
            backend=member.backend,
            task="deterministic_failure_injection",
            publication_policy="diagnostic_only",
            recipe_sha256=dflash if member.backend == "DFLASH" else dspark,
            dimensions={
                "diagnostic_only": "true",
                "failure": member.failure,
                "failure_authority_sha256": authority.sha256,
                "failure_member_id": member.member_id,
                "topology": member.topology,
                "cohort_count": member.cohort_count,
                "power_prefix_sha256": source,
            },
        )
        for member in authority.members
    )
    cells = headline + failures
    if len(headline) != 450 * count or len(failures) != 264:
        raise AssertionError("E5 final differs from 450N + 264")
    return _receipt(
        stage="E5",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule=(
            "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _failure_payload(
    actual: FormalSingleOperatorValidatedActual,
    cell: MaterializedCell,
) -> dict[str, object]:
    payload = actual.reducer_payload.get("failure_terminal")
    if type(payload) is not dict:
        payload = actual.reducer_payload
    status = payload.get("diagnostic_status", payload.get("status"))
    if status not in {"PASS", "FAIL"}:
        raise ValueError("E5 failure terminal lacks PASS/FAIL disposition")
    recovered = payload.get("recovered")
    if recovered is None:
        recovered = status == "PASS"
    if type(recovered) is not bool or recovered != (status == "PASS"):
        raise ValueError("E5 failure recovery flag and disposition differ")
    dimensions = dict(cell.dimensions)
    for field in ("failure", "topology", "cohort_count"):
        observed = payload.get(field)
        if observed is not None and observed != dimensions[field]:
            raise ValueError("E5 failure terminal names another diagnostic")
    return {
        "cell_id": cell.cell_id,
        "failure": dimensions["failure"],
        "backend": cell.backend,
        "topology": dimensions["topology"],
        "cohort_count": dimensions["cohort_count"],
        "status": status,
        "recovered": recovered,
        "terminal_identity_sha256": actual.result_identity_sha256,
        "evidence_sha256": _digest(payload),
    }


def reduce_single_operator_e5_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Reduce production headlines, six p99 anchors, and 264 diagnostics."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e5_pilot"
        or materialization.stage != "E5"
        or materialization.materialization_rule
        != "450_final_headline_rows_per_block_plus_264_one_shot_failure_diagnostics"
    ):
        raise ValueError("E5 final reducer received another node")
    headline_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "production_slo_power_prefix"
    )
    failure_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task == "deterministic_failure_injection"
    )
    count, remainder = divmod(len(headline_cells), 450)
    if remainder or not 12 <= count <= 20 or len(failure_cells) != 264:
        raise ValueError("E5 final is not exactly 450N + 264")
    actual_by_cell = {row.cell_id: row for row in actual_results}
    cells_by_id = {cell.cell_id: cell for cell in materialization.cells}
    if set(actual_by_cell) != set(cells_by_id):
        raise ValueError("E5 final actual coverage differs")
    headline_receipt = _receipt(
        stage=materialization.stage,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        upstream_receipt_sha256s=materialization.upstream_receipt_sha256s,
        source_decision_sha256=materialization.source_decision_sha256,
        materialization_rule=materialization.materialization_rule,
        cells=headline_cells,
        gpu_hours=materialization.gpu_hours,
    )
    observations = _serving_observations(
        headline_receipt,
        tuple(actual_by_cell[cell.cell_id] for cell in headline_cells),
    )
    grouped = _e5_grouped(headline_receipt, observations)
    if len(grouped) != 90:
        raise ValueError("E5 final lacks 90 backend/family strata")
    expected_blocks = set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
    family_results = []
    anchor_observations: dict[
        str,
        dict[int, tuple[MaterializedCell, dict[str, object]]],
    ] = {}
    for family in sorted(grouped, key=_digest):
        blocks = grouped[family]
        if set(blocks) != expected_blocks:
            raise ValueError("E5 family lacks every powered final block")
        exclusions = _family_role_exclusions(blocks)
        paired: dict[str, dict[str, tuple[float, float]]] = {
            "lightcone_vs_tts": {},
            "lightcone_vs_static": {},
            "l0_naive_vs_tts": {},
            "lightcone_vs_l0_naive": {},
            "lightcone_vs_target_only": {},
        }
        request_count = 0
        target_slo_pass = True
        for block in sorted(blocks):
            role_rows = blocks[block]
            slo = _paired_role_goodputs(role_rows, excluded_roles=frozenset(exclusions))
            block_id = f"E5:final:{block - PILOT_BLOCK_COUNT:02d}"
            goodput = {
                role: float(row.goodput_tokens_per_second) for role, row in slo.items()
            }
            target_slo_pass &= slo["Target-only"].status == "PASS"
            for name, (left, right) in _CORE_CONTRAST_ROLES.items():
                paired[name][block_id] = (goodput[left], goodput[right])
            request_count += slo["LightCone"].eligible_requests
            lc_cell, lc_observation = role_rows["LightCone"]
            anchor_id = dict(lc_cell.dimensions).get("p99_anchor_id")
            if type(anchor_id) is str:
                by_block = anchor_observations.setdefault(anchor_id, {})
                if block in by_block:
                    raise ValueError("E5 p99 anchor repeats an independent block")
                by_block[block] = (lc_cell, lc_observation)
        reduction = _resolve_family_contrasts(
            paired=paired,
            contrast_roles=_CORE_CONTRAST_ROLES,
            exclusions=exclusions,
            target_slo_pass=target_slo_pass,
        )
        contrast_payloads = reduction["contrast_payloads"]
        assert isinstance(contrast_payloads, dict)
        row = {
            "family_sha256": _digest({"stage": "E5", "dimensions": family}),
            "dimensions": [list(value) for value in family],
            "block_count": count,
            "request_count": request_count,
            "paired": True,
            "primary_contrasts": [
                contrast_payloads[name] for name in PRIMARY_CONTRASTS
            ],
            "holm_decisions": reduction["holm_decisions"],
            "holm_status": reduction["holm_status"],
            "all_registered_contrasts_resolved": reduction[
                "all_registered_contrasts_resolved"
            ],
            "mechanism_contrasts": [
                contrast_payloads[name]
                for name in ("l0_naive_vs_tts", "lightcone_vs_l0_naive")
            ],
            "target_only_gate": reduction["target_only_gate"],
            "scientific_exclusions": exclusions,
            "reason_codes": reduction["reason_codes"],
            "reducer": "paired_time_block_bca",
            "status": (
                "CONFIRMED" if reduction["deployment_confirmed"] else "NOT_CONFIRMED"
            ),
        }
        row["result_sha256"] = _digest(row)
        family_results.append(row)
    anchor_rows = []
    expected_anchors = _anchors_from_payload(
        predecessor.decision.payload.get("p99_anchors")
    )
    if set(anchor_observations) != {anchor.anchor_id for anchor in expected_anchors}:
        raise ValueError("E5 p99 anchor actual coverage differs")
    for anchor in expected_anchors:
        anchor_rows.append(
            _e5_p99_anchor_row(
                anchor,
                anchor_observations[anchor.anchor_id],
                expected_blocks=expected_blocks,
                selection_receipt_sha256=materialization.source_decision_sha256,
            )
        )
    failures = tuple(
        _failure_payload(actual_by_cell[cell.cell_id], cell) for cell in failure_cells
    )
    failure_complete = len(failures) == 264
    failure_pass = all(row["status"] == "PASS" for row in failures)
    pilot = predecessor.decision.payload
    result = {
        "schema_version": 1,
        "kind": "formal_single_operator_e5_confirmation",
        "status": (
            "CONFIRMED"
            if all(row["status"] == "CONFIRMED" for row in family_results)
            and all(row["status"] == "CLAIMABLE" for row in anchor_rows)
            and failure_complete
            and failure_pass
            else "NOT_CONFIRMED"
        ),
        "model": _text("E5 model", pilot.get("model")),
        "frozen_tts_recipe_sha256": _sha(
            "E5 frozen TTS recipe", pilot.get("frozen_tts_recipe_sha256")
        ),
        "dflash_lightcone_recipe_sha256": _sha(
            "E5 DFlash recipe", pilot.get("dflash_lightcone_recipe_sha256")
        ),
        "dspark_lightcone_recipe_sha256": _sha(
            "E5 DSpark recipe", pilot.get("dspark_lightcone_recipe_sha256")
        ),
        "block_count": count,
        "materialization_sha256": materialization.sha256,
        "family_results": family_results,
        "p99_anchor_claims": anchor_rows,
        "failure_results": list(failures),
    }
    result_sha = _digest(result)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e5_final_actual_reduced",
        next_materialization_source_decision_sha256=result_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**result, "confirmation_sha256": result_sha},
    )


def _e6_compatibility_from_auxiliary(
    protocol_lock: ProtocolLock,
    value: object,
) -> tuple[object, str]:
    """Deep-reopen the two actual NEXTN interface/fit source DAGs."""

    if (
        type(value) is dict
        and value.get("schema_version") == 2
        and value.get("kind") == "formal_single_operator_e6_interface_fit_bundle"
        and value.get("trust_mode") == "trusted_single_operator_empirical_no_signature"
    ):
        from lightcone_spec.experiments.formal_single_operator_e6_interface import (
            revalidate_formal_single_operator_e6_interface_fit_bundle_value,
        )

        bundle = revalidate_formal_single_operator_e6_interface_fit_bundle_value(
            value,
            protocol_lock=protocol_lock,
        )
        return bundle.compatibility, _digest(value)

    from lightcone_spec.experiments.e0_authority_artifact import (
        e6_nextn_model_authority_input_from_dict,
    )
    from lightcone_spec.experiments.e6_stage_authority import (
        reduce_e6_model_compatibility_from_proofs,
    )

    expected_fields = {
        "schema_version",
        "kind",
        "protocol_lock_sha256",
        "expected_inventory_sha256",
        "verified_ns",
        "sources",
        "compatibility_sha256",
    }
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError("E6 interface/fit auxiliary fields differ")
    row = dict(value)
    if (
        row["schema_version"] != 1
        or row["kind"] != "formal_single_operator_e6_interface_fit_bundle"
        or row["protocol_lock_sha256"] != protocol_lock.sha256
        or type(row["verified_ns"]) is not int
        or row["verified_ns"] < 1
        or type(row["sources"]) is not list
    ):
        raise ValueError("E6 interface/fit auxiliary identity differs")
    sources = tuple(
        e6_nextn_model_authority_input_from_dict(item) for item in row["sources"]
    )
    receipt = reduce_e6_model_compatibility_from_proofs(
        protocol_lock=protocol_lock,
        sources=sources,
        expected_inventory_sha256=_sha(
            "E6 auxiliary inventory", row["expected_inventory_sha256"]
        ),
        now_ns=row["verified_ns"],
    )
    if receipt.sha256 != _sha(
        "E6 auxiliary compatibility", row["compatibility_sha256"]
    ):
        raise ValueError("E6 interface/fit auxiliary result changed")
    return receipt, _digest(value)


def materialize_single_operator_e6_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    e6_interface_fit_auxiliary: object,
) -> StageMaterializationReceipt:
    """Materialize two globally deduplicated preflights plus 240 pilots."""

    if predecessor is None or predecessor.artifact.node != "e5_final":
        raise ValueError("E6 pilots require completed E5 final")
    payload = predecessor.decision.payload
    source = _sha("E5 confirmation", payload.get("confirmation_sha256"))
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E6 pilot predecessor lineage differs")
    compatibility, auxiliary_sha = _e6_compatibility_from_auxiliary(
        protocol_lock,
        e6_interface_fit_auxiliary,
    )
    cells = _e6_cells_from_verified_sources(
        signed_e5_confirmation_sha256=source,
        signed_model_compatibility_sha256=auxiliary_sha,
        model_compatibility=compatibility,
        frozen_tts_recipe_sha256=_sha(
            "E6 frozen TTS recipe", payload.get("frozen_tts_recipe_sha256")
        ),
        lightcone_recipe_sha256=_sha(
            "E6 LightCone recipe", payload.get("dflash_lightcone_recipe_sha256")
        ),
        block_indices=tuple(range(PILOT_BLOCK_COUNT)),
    )
    return _receipt(
        stage="E6",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule=(
            "two_model_preflights_plus_60_excluded_pilot_rows_per_block"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e6_grouped(
    materialization: StageMaterializationReceipt,
    observations: dict[str, dict[str, object]],
) -> dict[
    tuple[tuple[str, str | int], ...],
    dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
]:
    grouped: dict[
        tuple[tuple[str, str | int], ...],
        dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
    ] = {}
    for cell in materialization.cells:
        if cell.task == "immutable_metadata_interface_and_fit_preflight":
            continue
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        context = dimensions.get("context")
        if (
            type(block) is not int
            or type(context) is not int
            or cell.task not in E6_TASKS
        ):
            raise ValueError("E6 serving scientific axes differ")
        family = (
            ("context", context),
            ("model", cell.model),
            ("task", cell.task),
        )
        by_role = grouped.setdefault(family, {}).setdefault(block, {})
        if cell.method_role in by_role or cell.method_role not in FORMAL_METHOD_ROLES:
            raise ValueError("E6 serving repeats or changes a method role")
        observation = observations[cell.cell_id]
        by_role[cell.method_role] = (cell, observation)
    return grouped


def _e6_preflight_results(
    materialization: StageMaterializationReceipt,
    actual_by_cell: dict[str, FormalSingleOperatorValidatedActual],
) -> tuple[dict[str, object], ...]:
    rows = []
    for cell in materialization.cells:
        if cell.task != "immutable_metadata_interface_and_fit_preflight":
            continue
        actual = actual_by_cell[cell.cell_id]
        payload = actual.reducer_payload.get("e6_interface_preflight")
        if type(payload) is not dict:
            payload = actual.reducer_payload
        dimensions = dict(cell.dimensions)
        expected = dimensions.get("e6_verified_authority_sha256")
        observed = payload.get(
            "verified_authority_sha256", actual.result_identity_sha256
        )
        if observed != expected:
            raise ValueError("E6 preflight result differs from compatibility authority")
        rows.append(
            {
                "cell_id": cell.cell_id,
                "model": cell.model,
                "verified_authority_sha256": _sha("E6 verified authority", expected),
                "terminal_identity_sha256": actual.result_identity_sha256,
            }
        )
    result = tuple(sorted(rows, key=lambda row: str(row["model"])))
    if len(result) != 2 or len({row["model"] for row in result}) != 2:
        raise ValueError("E6 preflight coverage must contain both models once")
    return result


def reduce_single_operator_e6_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Power twelve model/task/context transfer families from four pilots."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e5_final"
        or materialization.stage != "E6"
        or materialization.materialization_rule
        != "two_model_preflights_plus_60_excluded_pilot_rows_per_block"
        or len(materialization.cells) != 242
    ):
        raise ValueError("E6 pilot reducer received another node")
    actual_by_cell = {row.cell_id: row for row in actual_results}
    cells_by_id = {cell.cell_id: cell for cell in materialization.cells}
    if set(actual_by_cell) != set(cells_by_id):
        raise ValueError("E6 pilot actual coverage differs")
    preflights = _e6_preflight_results(materialization, actual_by_cell)
    serving_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task != "immutable_metadata_interface_and_fit_preflight"
    )
    serving_receipt = _receipt(
        stage="E6",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        upstream_receipt_sha256s=materialization.upstream_receipt_sha256s,
        source_decision_sha256=materialization.source_decision_sha256,
        materialization_rule=materialization.materialization_rule,
        cells=serving_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    observations = _serving_observations(
        serving_receipt,
        tuple(actual_by_cell[cell.cell_id] for cell in serving_cells),
    )
    grouped = _e6_grouped(serving_receipt, observations)
    if len(grouped) != 12:
        raise ValueError("E6 pilots lack twelve model/task/context families")
    commitments = []
    underpowered = []
    power_unresolved = []
    selected = []
    for family in sorted(grouped, key=_digest):
        blocks = grouped[family]
        if set(blocks) != set(range(PILOT_BLOCK_COUNT)):
            raise ValueError("E6 family lacks exactly four excluded pilots")
        exclusions = _family_role_exclusions(blocks)
        pilots = []
        for block in range(PILOT_BLOCK_COUNT):
            slo = _paired_role_goodputs(
                blocks[block], excluded_roles=frozenset(exclusions)
            )
            paired = dict(
                require_paired_primary_goodputs(
                    {role: slo[role] for role in ("Static", "TTS", "LightCone")}
                )
            )
            pilots.append(
                PilotBlock(
                    block_id=f"E6:{_digest(family)}:excluded_pilot:{block}",
                    static_goodput=float(paired["Static"]),
                    tts_goodput=float(paired["TTS"]),
                    lightcone_goodput=float(paired["LightCone"]),
                )
            )
        plan = _pilot_power_resolution(tuple(pilots), exclusions)
        family_sha = _digest({"stage": "E6", "dimensions": family})
        row = {
            "family_sha256": family_sha,
            "dimensions": [list(value) for value in family],
            "power_sizing": asdict(plan),
            "scientific_exclusions": exclusions,
        }
        row["commitment_sha256"] = _digest(row)
        commitments.append(row)
        if plan.status == "POWER_UNRESOLVED":
            power_unresolved.append(family_sha)
        elif plan.underpowered or plan.selected_final_blocks is None:
            underpowered.append(family_sha)
        else:
            selected.append(plan.selected_final_blocks)
    e5 = predecessor.decision.payload
    compatibility_ids = {
        dict(cell.dimensions).get("signed_e6_model_compatibility_sha256")
        for cell in materialization.cells
    }
    if len(compatibility_ids) != 1 or None in compatibility_ids:
        raise ValueError("E6 materialization changes compatibility identity")
    base = {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_power_prefix",
        "upstream_e5_confirmation_sha256": _sha(
            "E5 confirmation", e5.get("confirmation_sha256")
        ),
        "frozen_tts_recipe_sha256": _sha(
            "E6 frozen TTS recipe", e5.get("frozen_tts_recipe_sha256")
        ),
        "lightcone_recipe_sha256": _sha(
            "E6 LightCone recipe", e5.get("dflash_lightcone_recipe_sha256")
        ),
        "compatibility_bundle_sha256": next(iter(compatibility_ids)),
        "preflight_results": list(preflights),
        "pilot_materialization_sha256": materialization.sha256,
        "family_commitments": commitments,
        "underpowered_family_sha256s": sorted(underpowered),
        "power_unresolved_family_sha256s": sorted(power_unresolved),
    }
    if underpowered or power_unresolved:
        payload = {
            **base,
            "status": "POWER_UNRESOLVED" if power_unresolved else "UNDERPOWERED",
            "selected_final_blocks": None,
            "reason_codes": [
                *(
                    ["one_or_more_families_power_unresolved"]
                    if power_unresolved
                    else []
                ),
                *(["one_or_more_families_underpowered"] if underpowered else []),
            ],
        }
        result_sha = _digest(payload)
        return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
            decision_kind=(
                "e6_pilot_power_unresolved"
                if power_unresolved
                else "e6_pilot_underpowered"
            ),
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**payload, "power_prefix_sha256": result_sha},
        )
    final_blocks = max(selected)
    payload = {
        **base,
        "status": "READY",
        "selected_final_blocks": final_blocks,
        "selected_final_prefix": list(
            range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + final_blocks)
        ),
    }
    result_sha = _digest(payload)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e6_pilot_actual_242_powered",
        next_materialization_source_decision_sha256=result_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**payload, "power_prefix_sha256": result_sha},
    )


def materialize_single_operator_e6_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    e6_interface_fit_auxiliary: object,
) -> StageMaterializationReceipt:
    """Materialize 60N final rows and reuse the two sealed pilot preflights."""

    stages = _stages()
    if predecessor is None or predecessor.artifact.node != "e6_pilot":
        raise ValueError("E6 final requires its excluded pilots")
    payload = predecessor.decision.payload
    status = payload.get("status")
    if status in {"UNDERPOWERED", "POWER_UNRESOLVED"}:
        raise stages.FormalSingleOperatorStageBlocked(  # type: ignore[attr-defined]
            f"E6 pilot families cannot advance: {status}"
        )
    if status != "READY":
        raise ValueError("E6 pilot power status is malformed")
    count = payload.get("selected_final_blocks")
    if type(count) is not int or not 12 <= count <= 20:
        raise ValueError("E6 selected final block count differs")
    source = _sha("E6 power prefix", payload.get("power_prefix_sha256"))
    compatibility, auxiliary_sha = _e6_compatibility_from_auxiliary(
        protocol_lock,
        e6_interface_fit_auxiliary,
    )
    if auxiliary_sha != payload.get("compatibility_bundle_sha256"):
        raise ValueError("E6 final switched its interface/fit auxiliary")
    cells = _e6_cells_from_verified_sources(
        signed_e5_confirmation_sha256=_sha(
            "E5 confirmation", payload.get("upstream_e5_confirmation_sha256")
        ),
        signed_model_compatibility_sha256=auxiliary_sha,
        model_compatibility=compatibility,
        frozen_tts_recipe_sha256=_sha(
            "E6 frozen TTS recipe", payload.get("frozen_tts_recipe_sha256")
        ),
        lightcone_recipe_sha256=_sha(
            "E6 LightCone recipe", payload.get("lightcone_recipe_sha256")
        ),
        block_indices=tuple(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count)),
        # Deliberately omit optional power lineage from the individual cells.
        # The receipt binds it globally and this keeps the two preflight cell
        # IDs byte-identical for physical evidence deduplication.
    )
    return _receipt(
        stage="E6",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule=(
            "60_final_rows_per_block_reusing_global_model_preflights"
        ),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def reduce_single_operator_e6_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Reduce two-model NEXTN transfer with paired primary/target gates."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e6_pilot"
        or materialization.stage != "E6"
        or materialization.materialization_rule
        != "60_final_rows_per_block_reusing_global_model_preflights"
    ):
        raise ValueError("E6 final reducer received another node")
    serving_cells = tuple(
        cell
        for cell in materialization.cells
        if cell.task != "immutable_metadata_interface_and_fit_preflight"
    )
    count, remainder = divmod(len(serving_cells), 60)
    if remainder or not 12 <= count <= 20 or len(materialization.cells) != 60 * count:
        raise ValueError("E6 final is not exactly 60N")
    actual_by_cell = {row.cell_id: row for row in actual_results}
    if set(actual_by_cell) != {cell.cell_id for cell in materialization.cells}:
        raise ValueError("E6 final actual coverage differs")
    pilot_preflights = predecessor.decision.payload.get("preflight_results")
    if type(pilot_preflights) is not list or len(pilot_preflights) != 2:
        raise ValueError("E6 final lacks the two globally sealed pilot preflights")
    preflights = tuple(pilot_preflights)
    if any(type(row) is not dict for row in preflights):
        raise ValueError("E6 final pilot preflight provenance differs")
    if {row.get("model") for row in preflights} != set(E6_MODELS):
        raise ValueError("E6 final pilot preflight provenance differs")
    for row in preflights:
        _sha("E6 preflight cell", row.get("cell_id"))
        _sha(
            "E6 verified preflight authority",
            row.get("verified_authority_sha256"),
        )
        _sha("E6 preflight terminal", row.get("terminal_identity_sha256"))
    serving_receipt = _receipt(
        stage="E6",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        upstream_receipt_sha256s=materialization.upstream_receipt_sha256s,
        source_decision_sha256=materialization.source_decision_sha256,
        materialization_rule=materialization.materialization_rule,
        cells=serving_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    observations = _serving_observations(
        serving_receipt,
        tuple(actual_by_cell[cell.cell_id] for cell in serving_cells),
    )
    grouped = _e6_grouped(serving_receipt, observations)
    if len(grouped) != 12:
        raise ValueError("E6 final lacks twelve model/task/context families")
    expected_blocks = set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
    results = []
    for family in sorted(grouped, key=_digest):
        blocks = grouped[family]
        if set(blocks) != expected_blocks:
            raise ValueError("E6 family lacks every powered final block")
        exclusions = _family_role_exclusions(blocks)
        paired: dict[str, dict[str, tuple[float, float]]] = {
            "lightcone_vs_tts": {},
            "lightcone_vs_static": {},
            "l0_naive_vs_tts": {},
            "lightcone_vs_l0_naive": {},
            "lightcone_vs_target_only": {},
        }
        request_count = 0
        target_slo_pass = True
        for block in sorted(blocks):
            slo = _paired_role_goodputs(
                blocks[block], excluded_roles=frozenset(exclusions)
            )
            goodput = {
                role: float(row.goodput_tokens_per_second) for role, row in slo.items()
            }
            target_slo_pass &= slo["Target-only"].status == "PASS"
            block_id = f"E6:final:{block - PILOT_BLOCK_COUNT:02d}"
            for name, (left, right) in _CORE_CONTRAST_ROLES.items():
                paired[name][block_id] = (goodput[left], goodput[right])
            request_count += slo["LightCone"].eligible_requests
        reduction = _resolve_family_contrasts(
            paired=paired,
            contrast_roles=_CORE_CONTRAST_ROLES,
            exclusions=exclusions,
            target_slo_pass=target_slo_pass,
        )
        contrast_payloads = reduction["contrast_payloads"]
        assert isinstance(contrast_payloads, dict)
        row = {
            "family_sha256": _digest({"stage": "E6", "dimensions": family}),
            "dimensions": [list(value) for value in family],
            "block_count": count,
            "request_count": request_count,
            "paired": True,
            "primary_contrasts": [
                contrast_payloads[name] for name in PRIMARY_CONTRASTS
            ],
            "holm_decisions": reduction["holm_decisions"],
            "holm_status": reduction["holm_status"],
            "all_registered_contrasts_resolved": reduction[
                "all_registered_contrasts_resolved"
            ],
            "mechanism_contrasts": [
                contrast_payloads[name]
                for name in ("l0_naive_vs_tts", "lightcone_vs_l0_naive")
            ],
            "target_only_gate": reduction["target_only_gate"],
            "scientific_exclusions": exclusions,
            "reason_codes": reduction["reason_codes"],
            "reducer": "paired_block_bca",
            "status": (
                "CONFIRMED" if reduction["deployment_confirmed"] else "NOT_CONFIRMED"
            ),
        }
        row["result_sha256"] = _digest(row)
        results.append(row)
    pilot = predecessor.decision.payload
    result = {
        "schema_version": 1,
        "kind": "formal_single_operator_e6_confirmation",
        "status": (
            "CONFIRMED"
            if all(row["status"] == "CONFIRMED" for row in results)
            else "NOT_CONFIRMED"
        ),
        "frozen_tts_recipe_sha256": _sha(
            "E6 frozen TTS recipe", pilot.get("frozen_tts_recipe_sha256")
        ),
        "lightcone_recipe_sha256": _sha(
            "E6 LightCone recipe", pilot.get("lightcone_recipe_sha256")
        ),
        "compatibility_bundle_sha256": _sha(
            "E6 compatibility", pilot.get("compatibility_bundle_sha256")
        ),
        "block_count": count,
        "materialization_sha256": materialization.sha256,
        "preflight_results": list(preflights),
        "family_results": results,
    }
    result_sha = _digest(result)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e6_final_actual_reduced",
        next_materialization_source_decision_sha256=result_sha,
        next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
        payload={**result, "confirmation_sha256": result_sha},
    )


_E0_ONLINE_ROLE_BY_METHOD = {
    "onlinespec_ogd": "OnlineSPEC-OGD",
    "onlinespec_opt": "OnlineSPEC-OPT",
    "onlinespec_ens": "OnlineSPEC-ENS",
}
_E0_METHOD_ROLES = FORMAL_METHOD_ROLES + tuple(_E0_ONLINE_ROLE_BY_METHOD.values())
_E0_CORE_BREADTH_CONTRASTS = (
    "lightcone_vs_tts",
    "lightcone_vs_static",
    "l0_naive_vs_tts",
    "lightcone_vs_l0_naive",
)
_E0_ONLINE_BREADTH_CONTRASTS = (
    "onlinespec_ogd_vs_static",
    "onlinespec_opt_vs_static",
    "onlinespec_ens_vs_static",
)
_E0_CONTRAST_ROLES = {
    **_CORE_CONTRAST_ROLES,
    "onlinespec_ogd_vs_static": ("OnlineSPEC-OGD", "Static"),
    "onlinespec_opt_vs_static": ("OnlineSPEC-OPT", "Static"),
    "onlinespec_ens_vs_static": ("OnlineSPEC-ENS", "Static"),
}


def _e0_compatibility_from_auxiliary(
    predecessor: RebuiltFormalSingleOperatorStageCompletion,
    protocol_lock: ProtocolLock,
    value: object,
) -> tuple[E0CompatibilityReceipt, object | None, str, str]:
    """Deep-open the trusted compatibility probe and OnlineSPEC authority."""

    from lightcone_spec.experiments.formal_registry import (
        e0_compatibility_receipt_from_dict,
        e0_onlinespec_source_authority_from_dict,
    )

    expected_fields = {
        "schema_version",
        "kind",
        "protocol_lock_sha256",
        "upstream_e6_materialization_sha256",
        "upstream_e6_confirmation_sha256",
        "compatibility",
        "compatibility_sha256",
        "compatibility_evidence_manifest_sha256",
        "onlinespec_source_authority",
        "onlinespec_source_authority_sha256",
        "started_ns",
        "finished_ns",
        "bundle_sha256",
    }
    if type(value) is dict and value.get("schema_version") in {2, 3}:
        expected_fields.add("compatibility_evidence_manifest")
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError("E0 compatibility auxiliary fields differ")
    row = dict(value)
    started_ns = row["started_ns"]
    finished_ns = row["finished_ns"]
    confirmation_sha256 = _sha(
        "E0 upstream E6 confirmation",
        predecessor.decision.payload.get("confirmation_sha256"),
    )
    if (
        row["schema_version"] not in {1, 2, 3}
        or row["kind"] != "formal_single_operator_e0_compatibility_bundle"
        or row["protocol_lock_sha256"] != protocol_lock.sha256
        or row["upstream_e6_materialization_sha256"]
        != predecessor.materialization.sha256
        or row["upstream_e6_confirmation_sha256"] != confirmation_sha256
        or type(started_ns) is not int
        or type(finished_ns) is not int
        or started_ns < 1
        or finished_ns <= started_ns
    ):
        raise ValueError("E0 compatibility auxiliary identity differs")
    if (
        predecessor.decision.next_materialization_source_decision_sha256
        != confirmation_sha256
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E0 compatibility predecessor lineage differs")
    compatibility = e0_compatibility_receipt_from_dict(row["compatibility"])
    if (
        compatibility.protocol_lock_sha256 != protocol_lock.sha256
        or compatibility.upstream_e6_receipt_sha256
        != predecessor.materialization.sha256
        or compatibility.sha256
        != _sha("E0 compatibility receipt", row["compatibility_sha256"])
    ):
        raise ValueError("E0 compatibility receipt lineage differs")
    evidence_sha256 = _sha(
        "E0 compatibility evidence manifest",
        row["compatibility_evidence_manifest_sha256"],
    )
    if row["schema_version"] in {2, 3}:
        from lightcone_spec.experiments.formal_single_operator_e0_compatibility import (
            revalidate_trusted_e0_compatibility_bundle_value,
        )

        trusted = revalidate_trusted_e0_compatibility_bundle_value(row)
        if (
            trusted.compatibility != compatibility
            or trusted.evidence_manifest.sha256 != evidence_sha256
        ):
            raise ValueError("trusted E0 compatibility replay differs")
    authority_value = row["onlinespec_source_authority"]
    authority_sha256 = row["onlinespec_source_authority_sha256"]
    authority: object | None
    if compatibility.valid_count == 0:
        if authority_value is not None or authority_sha256 is not None:
            raise ValueError("all-N/A E0 compatibility cannot claim OnlineSPEC source")
        authority = None
    else:
        if authority_value is None:
            raise ValueError("VALID E0 compatibility lacks OnlineSPEC source")
        authority = e0_onlinespec_source_authority_from_dict(authority_value)
        if authority.sha256 != _sha("E0 OnlineSPEC source authority", authority_sha256):
            raise ValueError("E0 OnlineSPEC source authority changed")
    expected_bundle_sha256 = _sha("E0 compatibility bundle", row["bundle_sha256"])
    row_without_sha = dict(row)
    row_without_sha.pop("bundle_sha256")
    if _digest(row_without_sha) != expected_bundle_sha256:
        raise ValueError("E0 compatibility bundle digest differs")
    return compatibility, authority, expected_bundle_sha256, evidence_sha256


def _e0_compatibility_dimensions(
    decision: E0CompatibilityDecision,
    *,
    compatibility_sha256: str,
    evidence_sha256: str,
    bundle_sha256: str,
) -> dict[str, str]:
    return {
        "compatibility_decision_id": decision.decision_id,
        "deployment_task": decision.task,
        "disposition": decision.disposition,
        "reason_code": decision.reason_code,
        "interface_sha256": decision.interface_sha256,
        "task_native_workload_sha256": decision.task_native_workload_sha256,
        "compatibility_receipt_sha256": compatibility_sha256,
        "compatibility_evidence_manifest_sha256": evidence_sha256,
        "e0_compatibility_bundle_sha256": bundle_sha256,
    }


def materialize_single_operator_e0_tuning(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    e0_compatibility_auxiliary: object,
) -> StageMaterializationReceipt:
    """Materialize 108 decisions and the independent ``239V`` tuning grid."""

    from lightcone_spec.experiments.onlinespec import onlinespec_candidates

    if predecessor is None or predecessor.artifact.node != "e6_final":
        raise ValueError("E0 tuning requires completed E6 final")
    compatibility, authority, bundle_sha256, evidence_sha256 = (
        _e0_compatibility_from_auxiliary(
            predecessor,
            protocol_lock,
            e0_compatibility_auxiliary,
        )
    )
    frozen_tts = _sha(
        "E0 frozen TTS recipe",
        predecessor.decision.payload.get("frozen_tts_recipe_sha256"),
    )
    authority_sha256 = None if authority is None else authority.sha256
    cells: list[MaterializedCell] = []
    for decision in compatibility.decisions:
        dimensions = _e0_compatibility_dimensions(
            decision,
            compatibility_sha256=compatibility.sha256,
            evidence_sha256=evidence_sha256,
            bundle_sha256=bundle_sha256,
        )
        cells.append(
            _cell(
                stage="E0",
                method_role="Compatibility",
                model=decision.model,
                backend=decision.backend,
                task="compatibility_decision",
                publication_policy="decision_only",
                recipe_sha256=None,
                dimensions=dimensions,
            )
        )
        if decision.disposition == "N/A":
            continue
        assert authority_sha256 is not None
        tuning_dimensions = {
            **dimensions,
            "e0_onlinespec_source_authority_sha256": authority_sha256,
            "tuning_window": "task_native_disjoint",
        }
        for role in ("Static", "TTS", "L0-naive"):
            cells.append(
                _cell(
                    stage="E0",
                    method_role=role,
                    model=decision.model,
                    backend=decision.backend,
                    task="independent_onlinespec_tuning",
                    publication_policy=(
                        "fixed_barrier"
                        if role == "TTS"
                        else "first_ready"
                        if role == "L0-naive"
                        else "none"
                    ),
                    recipe_sha256=(frozen_tts if role in {"TTS", "L0-naive"} else None),
                    dimensions=tuning_dimensions,
                )
            )
        for candidate in sorted(
            onlinespec_candidates(), key=lambda row: row.candidate_id
        ):
            role = _E0_ONLINE_ROLE_BY_METHOD[candidate.method]
            cells.append(
                _cell(
                    stage="E0",
                    method_role=role,
                    model=decision.model,
                    backend=decision.backend,
                    task="independent_onlinespec_tuning",
                    publication_policy="tuning_only",
                    recipe_sha256=candidate.candidate_id,
                    dimensions={
                        **tuning_dimensions,
                        "candidate_id": candidate.candidate_id,
                        "onlinespec_method": candidate.method,
                    },
                )
            )
    expected = 108 + 239 * compatibility.valid_count
    if len(cells) != expected:
        raise AssertionError("E0 tuning differs from 108 + 239V")
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=bundle_sha256,
        materialization_rule="108_compatibility_decisions_plus_239_rows_per_valid",
        cells=tuple(cells),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e0_validate_compatibility_actuals(
    materialization: StageMaterializationReceipt,
    actual_by_cell: dict[str, FormalSingleOperatorValidatedActual],
) -> tuple[dict[str, object], ...]:
    rows = []
    for cell in materialization.cells:
        if cell.task != "compatibility_decision":
            continue
        actual = actual_by_cell[cell.cell_id]
        payload = actual.reducer_payload.get("e0_compatibility_decision")
        if payload is None:
            payload = actual.reducer_payload
        expected_fields = {
            "schema_version",
            "compatibility_decision_id",
            "disposition",
            "reason_code",
            "interface_sha256",
            "task_native_workload_sha256",
            "compatibility_evidence_manifest_sha256",
        }
        if type(payload) is not dict or set(payload) != expected_fields:
            raise ValueError("E0 compatibility actual fields differ")
        dimensions = dict(cell.dimensions)
        if payload["schema_version"] != 1 or any(
            payload[key] != dimensions[key]
            for key in expected_fields - {"schema_version"}
        ):
            raise ValueError("E0 compatibility actual differs from typed receipt")
        rows.append(
            {
                **payload,
                "cell_id": cell.cell_id,
                "result_identity_sha256": actual.result_identity_sha256,
            }
        )
    result = tuple(sorted(rows, key=lambda row: str(row["compatibility_decision_id"])))
    if len(result) != 108:
        raise ValueError("E0 compatibility actual coverage is not exactly 108")
    return result


def _e0_receipt_from_payload(payload: dict[str, object]) -> E0CompatibilityReceipt:
    from lightcone_spec.experiments.formal_registry import (
        e0_compatibility_receipt_from_dict,
    )

    return e0_compatibility_receipt_from_dict(payload.get("compatibility"))


def reduce_single_operator_e0_tuning(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Freeze one safe/SLO winner independently for OGD, OPT, and ENS."""

    from lightcone_spec.experiments.formal_registry import (
        e0_compatibility_receipt_to_dict,
    )

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e6_final"
        or materialization.stage != "E0"
        or materialization.materialization_rule
        != "108_compatibility_decisions_plus_239_rows_per_valid"
    ):
        raise ValueError("E0 tuning reducer received another node")
    actual_by_cell = {row.cell_id: row for row in actual_results}
    if len(actual_by_cell) != len(actual_results) or set(actual_by_cell) != {
        cell.cell_id for cell in materialization.cells
    }:
        raise ValueError("E0 tuning actual coverage differs")
    compatibility_actuals = _e0_validate_compatibility_actuals(
        materialization,
        actual_by_cell,
    )
    compatibility_cells = [
        cell for cell in materialization.cells if cell.task == "compatibility_decision"
    ]
    serving_cells = tuple(
        cell for cell in materialization.cells if cell.task != "compatibility_decision"
    )
    compatibility_sha256s = {
        dict(cell.dimensions)["compatibility_receipt_sha256"]
        for cell in compatibility_cells
    }
    bundle_sha256s = {
        dict(cell.dimensions)["e0_compatibility_bundle_sha256"]
        for cell in compatibility_cells
    }
    evidence_sha256s = {
        dict(cell.dimensions)["compatibility_evidence_manifest_sha256"]
        for cell in compatibility_cells
    }
    authority_sha256s = {
        dict(cell.dimensions)["e0_onlinespec_source_authority_sha256"]
        for cell in serving_cells
    }
    if (
        len(compatibility_sha256s) != 1
        or len(bundle_sha256s) != 1
        or len(evidence_sha256s) != 1
        or len(authority_sha256s) != (1 if serving_cells else 0)
    ):
        raise ValueError("E0 compatibility lineage changes within tuning")
    decisions = tuple(
        E0CompatibilityDecision(
            model=cell.model,
            backend=cell.backend,
            task=_text("E0 deployment task", dict(cell.dimensions)["deployment_task"]),
            disposition=dict(cell.dimensions)["disposition"],  # type: ignore[arg-type]
            reason_code=_text("E0 reason", dict(cell.dimensions)["reason_code"]),
            interface_sha256=_sha(
                "E0 interface", dict(cell.dimensions)["interface_sha256"]
            ),
            task_native_workload_sha256=_sha(
                "E0 workload",
                dict(cell.dimensions)["task_native_workload_sha256"],
            ),
        )
        for cell in compatibility_cells
    )
    compatibility = E0CompatibilityReceipt(
        schema_version=1,
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        upstream_e6_receipt_sha256=predecessor.materialization.sha256,
        decisions=tuple(sorted(decisions, key=lambda row: row.decision_id)),
    )
    if compatibility.sha256 != next(iter(compatibility_sha256s)):
        raise ValueError("E0 compatibility receipt changed during materialization")
    frozen_tts = _sha(
        "E0 frozen TTS recipe",
        predecessor.decision.payload.get("frozen_tts_recipe_sha256"),
    )
    lightcone = _sha(
        "E0 LightCone recipe",
        predecessor.decision.payload.get("lightcone_recipe_sha256"),
    )
    serving_receipt = _receipt(
        stage="E0",
        protocol_lock_sha256=materialization.protocol_lock_sha256,
        upstream_receipt_sha256s=materialization.upstream_receipt_sha256s,
        source_decision_sha256=materialization.source_decision_sha256,
        materialization_rule=materialization.materialization_rule,
        cells=serving_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    observations = (
        {}
        if not serving_cells
        else _serving_observations(
            serving_receipt,
            tuple(actual_by_cell[cell.cell_id] for cell in serving_cells),
        )
    )
    winners = []
    missing = []
    anchor_evaluations = []
    candidate_evaluations = []
    for decision in compatibility.decisions:
        if decision.disposition == "N/A":
            continue
        rows = [
            cell
            for cell in serving_cells
            if dict(cell.dimensions)["compatibility_decision_id"]
            == decision.decision_id
        ]
        anchors = [
            cell for cell in rows if cell.method_role in {"Static", "TTS", "L0-naive"}
        ]
        candidates = [
            cell for cell in rows if cell.method_role.startswith("OnlineSPEC-")
        ]
        if len(anchors) != 3 or len(candidates) != 236 or len(rows) != 239:
            raise ValueError("E0 tuning decision does not contain 239 rows")
        anchor_slo = {
            cell.method_role: _slo(observations[cell.cell_id]) for cell in anchors
        }
        if set(anchor_slo) != {"Static", "TTS", "L0-naive"}:
            raise ValueError("E0 tuning anchor role coverage differs")
        for cell in anchors:
            slo = anchor_slo[cell.method_role]
            reasons = set(
                stages._adaptive_safety_reasons(  # type: ignore[attr-defined]
                    observations[cell.cell_id],
                    require_published_update=cell.method_role in {"TTS", "L0-naive"},
                )
            )
            if slo.status != "PASS":
                reasons.add("slo_failed")
            anchor_evaluations.append(
                {
                    "compatibility_decision_id": decision.decision_id,
                    "method_role": cell.method_role,
                    "cell_id": cell.cell_id,
                    "eligible": not reasons,
                    "reason_codes": sorted(reasons),
                    "evidence_ids": sorted(
                        {
                            cell.cell_id,
                            actual_by_cell[cell.cell_id].result_identity_sha256,
                            slo.sha256,
                        }
                    ),
                }
            )
        pools = {
            _slo(observations[cell.cell_id]).source_request_pool_sha256 for cell in rows
        }
        if len(pools) != 1:
            raise ValueError("E0 tuning candidates use unpaired request pools")
        for role in _E0_METHOD_ROLES[-3:]:
            eligible = []
            for cell in candidates:
                if cell.method_role != role:
                    continue
                observation = observations[cell.cell_id]
                slo = _slo(observation)
                reasons = set(
                    stages._adaptive_safety_reasons(  # type: ignore[attr-defined]
                        observation,
                        require_published_update=True,
                    )
                )
                if slo.status != "PASS":
                    reasons.add("slo_failed")
                candidate_evaluations.append(
                    {
                        "compatibility_decision_id": decision.decision_id,
                        "method_role": role,
                        "candidate_id": _sha("E0 candidate recipe", cell.recipe_sha256),
                        "cell_id": cell.cell_id,
                        "eligible": not reasons,
                        "reason_codes": sorted(reasons),
                        "evidence_ids": sorted(
                            {
                                cell.cell_id,
                                actual_by_cell[cell.cell_id].result_identity_sha256,
                                slo.sha256,
                            }
                        ),
                        "goodput_tokens_per_second": float(
                            slo.goodput_tokens_per_second
                        ),
                    }
                )
                if not reasons:
                    eligible.append((cell, slo))
            if not eligible:
                missing.append((decision.decision_id, role))
                continue
            winner, slo = min(
                eligible,
                key=lambda value: (
                    -float(value[1].goodput_tokens_per_second),
                    value[0].recipe_sha256,
                    value[0].cell_id,
                ),
            )
            winners.append(
                {
                    "compatibility_decision_id": decision.decision_id,
                    "method_role": role,
                    "candidate_id": _sha("E0 winner candidate", winner.recipe_sha256),
                    "cell_id": winner.cell_id,
                    "goodput_tokens_per_second": float(slo.goodput_tokens_per_second),
                    "eligible_requests": slo.eligible_requests,
                    "selection_rule": "safe_then_slo_then_max_goodput_then_sha256",
                }
            )
    na_decisions = [
        {
            "compatibility_decision_id": row.decision_id,
            "model": row.model,
            "backend": row.backend,
            "task": row.task,
            "reason_code": row.reason_code,
        }
        for row in compatibility.decisions
        if row.disposition == "N/A"
    ]
    base = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_tuning_selection",
        "status": "NO_SAFE_WINNER"
        if missing
        else "ALL_NA"
        if compatibility.valid_count == 0
        else "READY",
        "compatibility": e0_compatibility_receipt_to_dict(compatibility),
        "compatibility_bundle_sha256": next(iter(bundle_sha256s)),
        "compatibility_evidence_manifest_sha256": next(iter(evidence_sha256s)),
        "onlinespec_source_authority_sha256": (
            None if not authority_sha256s else next(iter(authority_sha256s))
        ),
        "compatibility_actuals": list(compatibility_actuals),
        "valid_count": compatibility.valid_count,
        "na_decisions": na_decisions,
        "frozen_tts_recipe_sha256": frozen_tts,
        "lightcone_recipe_sha256": lightcone,
        "selected_onlinespec_recipes": sorted(
            winners,
            key=lambda row: (
                str(row["compatibility_decision_id"]),
                str(row["method_role"]),
            ),
        ),
        "anchor_evaluations": sorted(
            anchor_evaluations,
            key=lambda row: (
                str(row["compatibility_decision_id"]),
                str(row["method_role"]),
            ),
        ),
        "candidate_evaluations": sorted(
            candidate_evaluations,
            key=lambda row: (
                str(row["compatibility_decision_id"]),
                str(row["method_role"]),
                str(row["candidate_id"]),
            ),
        ),
        "missing_winners": [list(row) for row in sorted(missing)],
        "reason_codes": (
            ["one_or_more_onlinespec_roles_lack_safe_slo_candidate"] if missing else []
        ),
        "materialization_sha256": materialization.sha256,
    }
    result_sha = _digest(base)
    can_advance = not missing
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind=(
            "e0_tuning_actual_reduced" if can_advance else "e0_tuning_no_safe_winner"
        ),
        next_materialization_source_decision_sha256=(
            result_sha if can_advance else None
        ),
        next_materialization_upstream_receipt_sha256s=(
            (materialization.sha256,) if can_advance else ()
        ),
        payload={**base, "selection_sha256": result_sha},
    )


def _e0_selected_recipes(payload: dict[str, object]) -> dict[tuple[str, str], str]:
    rows = payload.get("selected_onlinespec_recipes")
    if type(rows) is not list:
        raise ValueError("E0 tuning selection rows differ")
    selected: dict[tuple[str, str], str] = {}
    expected_fields = {
        "compatibility_decision_id",
        "method_role",
        "candidate_id",
        "cell_id",
        "goodput_tokens_per_second",
        "eligible_requests",
        "selection_rule",
    }
    for value in rows:
        if type(value) is not dict or set(value) != expected_fields:
            raise ValueError("E0 tuning selection row fields differ")
        decision_id = _sha("E0 selected decision", value["compatibility_decision_id"])
        role = _text("E0 selected role", value["method_role"])
        if role not in _E0_METHOD_ROLES[-3:]:
            raise ValueError("E0 tuning selected an unknown OnlineSPEC role")
        key = (decision_id, role)
        if key in selected:
            raise ValueError("E0 tuning repeats an OnlineSPEC winner")
        selected[key] = _sha("E0 selected recipe", value["candidate_id"])
    return selected


def _e0_serving_cells(
    *,
    compatibility: E0CompatibilityReceipt,
    compatibility_bundle_sha256: str,
    compatibility_evidence_sha256: str,
    onlinespec_source_authority_sha256: str | None,
    selected_recipes: dict[tuple[str, str], str],
    frozen_tts_recipe_sha256: str,
    lightcone_recipe_sha256: str,
    block_indices: tuple[int, ...],
    final_lineage: dict[str, str] | None = None,
) -> tuple[MaterializedCell, ...]:
    is_pilot = block_indices == tuple(range(PILOT_BLOCK_COUNT))
    is_final = 12 <= len(block_indices) <= 20 and block_indices == tuple(
        range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + len(block_indices))
    )
    if not is_pilot and not is_final:
        raise ValueError("E0 block indices are not an exact pilot/final prefix")
    if (final_lineage is None) != is_pilot:
        raise ValueError("E0 final lineage presence differs from phase")
    valid = tuple(row for row in compatibility.decisions if row.disposition == "VALID")
    if valid:
        authority_sha256 = _sha(
            "E0 OnlineSPEC source authority",
            onlinespec_source_authority_sha256,
        )
    elif onlinespec_source_authority_sha256 is not None:
        raise ValueError("all-N/A E0 serving cannot claim OnlineSPEC source")
    else:
        authority_sha256 = None
    expected_recipe_keys = {
        (row.decision_id, role) for row in valid for role in _E0_METHOD_ROLES[-3:]
    }
    if set(selected_recipes) != expected_recipe_keys:
        raise ValueError("E0 OnlineSPEC winners do not cover every VALID decision")
    cells = []
    if valid:
        assert authority_sha256 is not None
    for block in block_indices:
        for decision in valid:
            for load in E0_LOADS:
                pair_id = _digest(
                    {
                        "stage": "E0",
                        "block": block,
                        "compatibility_decision_id": decision.decision_id,
                        "load": load,
                    }
                )
                for role in _E0_METHOD_ROLES:
                    recipe = (
                        frozen_tts_recipe_sha256
                        if role in {"TTS", "L0-naive"}
                        else lightcone_recipe_sha256
                        if role == "LightCone"
                        else selected_recipes[(decision.decision_id, role)]
                        if role.startswith("OnlineSPEC-")
                        else None
                    )
                    dimensions: dict[str, str | int | float] = {
                        "block": block,
                        "block_phase": "excluded_pilot" if is_pilot else "final",
                        "compatibility_decision_id": decision.decision_id,
                        "compatibility_receipt_sha256": compatibility.sha256,
                        "compatibility_evidence_manifest_sha256": (
                            compatibility_evidence_sha256
                        ),
                        "e0_compatibility_bundle_sha256": (compatibility_bundle_sha256),
                        "e0_onlinespec_source_authority_sha256": (authority_sha256),
                        "interface_sha256": decision.interface_sha256,
                        "load": load,
                        "task_native_workload_sha256": (
                            decision.task_native_workload_sha256
                        ),
                    }
                    if role in {"TTS", "L0-naive"}:
                        dimensions["tts_l0_pair_id"] = pair_id
                    if final_lineage is not None:
                        dimensions.update(final_lineage)
                    cells.append(
                        _cell(
                            stage="E0",
                            method_role=role,
                            model=decision.model,
                            backend=decision.backend,
                            task=decision.task,
                            publication_policy=(
                                "fixed_barrier"
                                if role == "TTS"
                                else "first_ready"
                                if role in {"L0-naive", "LightCone"}
                                else "independent_online"
                                if role.startswith("OnlineSPEC-")
                                else "none"
                            ),
                            recipe_sha256=recipe,
                            dimensions=dimensions,
                        )
                    )
    if len(cells) != 16 * len(valid) * len(block_indices):
        raise AssertionError("E0 serving matrix differs from exact 16VB")
    return tuple(cells)


def _e0_require_auxiliary_identity(
    predecessor: RebuiltFormalSingleOperatorStageCompletion,
    protocol_lock: ProtocolLock,
    auxiliary: object,
) -> tuple[E0CompatibilityReceipt, str, str]:
    e6 = _completion_for_node(predecessor, "e6_final")
    if e6 is None:
        raise ValueError("E0 chain lacks its E6 completion")
    compatibility, _authority, bundle_sha256, evidence_sha256 = (
        _e0_compatibility_from_auxiliary(e6, protocol_lock, auxiliary)
    )
    if predecessor.decision.payload.get("compatibility_bundle_sha256") != bundle_sha256:
        raise ValueError("E0 chain switched compatibility bundle")
    return compatibility, bundle_sha256, evidence_sha256


def materialize_single_operator_e0_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    e0_compatibility_auxiliary: object,
) -> StageMaterializationReceipt:
    """Materialize exactly four excluded ``16V`` serving blocks."""

    if predecessor is None or predecessor.artifact.node != "e0_tuning":
        raise ValueError("E0 pilots require completed independent tuning")
    payload = predecessor.decision.payload
    source = _sha("E0 tuning selection", payload.get("selection_sha256"))
    status = payload.get("status")
    if status == "NO_SAFE_WINNER":
        stages = _stages()
        raise stages.FormalSingleOperatorStageBlocked(  # type: ignore[attr-defined]
            f"E0 tuning cannot advance: {status}"
        )
    if status not in {"READY", "ALL_NA"}:
        raise ValueError("E0 tuning selection status is malformed")
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E0 pilot predecessor lineage differs")
    compatibility, bundle_sha256, evidence_sha256 = _e0_require_auxiliary_identity(
        predecessor,
        protocol_lock,
        e0_compatibility_auxiliary,
    )
    payload_compatibility = _e0_receipt_from_payload(payload)
    if payload_compatibility.sha256 != compatibility.sha256:
        raise ValueError("E0 pilot compatibility receipt changed")
    cells = _e0_serving_cells(
        compatibility=compatibility,
        compatibility_bundle_sha256=bundle_sha256,
        compatibility_evidence_sha256=evidence_sha256,
        onlinespec_source_authority_sha256=payload.get(
            "onlinespec_source_authority_sha256"
        ),  # type: ignore[arg-type]
        selected_recipes=_e0_selected_recipes(payload),
        frozen_tts_recipe_sha256=_sha(
            "E0 frozen TTS recipe", payload.get("frozen_tts_recipe_sha256")
        ),
        lightcone_recipe_sha256=_sha(
            "E0 LightCone recipe", payload.get("lightcone_recipe_sha256")
        ),
        block_indices=tuple(range(PILOT_BLOCK_COUNT)),
    )
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule="valid_x_8_roles_x_2_loads_x_4_excluded_pilots",
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e0_grouped(
    materialization: StageMaterializationReceipt,
    observations: dict[str, dict[str, object]],
) -> dict[
    tuple[str, str], dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]]
]:
    grouped: dict[
        tuple[str, str],
        dict[int, dict[str, tuple[MaterializedCell, dict[str, object]]]],
    ] = {}
    for cell in materialization.cells:
        dimensions = dict(cell.dimensions)
        block = dimensions.get("block")
        load = dimensions.get("load")
        decision_id = dimensions.get("compatibility_decision_id")
        if (
            type(block) is not int
            or load not in E0_LOADS
            or type(decision_id) is not str
        ):
            raise ValueError("E0 serving scientific axes differ")
        family = (decision_id, load)
        by_role = grouped.setdefault(family, {}).setdefault(block, {})
        if cell.method_role not in _E0_METHOD_ROLES or cell.method_role in by_role:
            raise ValueError("E0 serving repeats or changes a method role")
        observation = observations[cell.cell_id]
        by_role[cell.method_role] = (cell, observation)
    return grouped


def _e0_paired_role_goodputs(
    rows: dict[str, tuple[MaterializedCell, dict[str, object]]],
    *,
    excluded_roles: frozenset[str] = frozenset(),
) -> dict[str, FormalSloGoodputObservation]:
    if set(rows) != set(_E0_METHOD_ROLES):
        raise ValueError("E0 serving family method coverage differs")
    if not excluded_roles <= set(rows):
        raise ValueError("E0 exclusions name an unknown method role")
    request_evidence = {
        role: _request_evidence(value[1]) for role, value in rows.items()
    }
    pool_sha256s = {
        role: str(value[1]["source_request_pool_sha256"])
        for role, value in rows.items()
    }
    exactness_evidence = {
        role: evidence
        for role, evidence in request_evidence.items()
        if role not in excluded_roles
    }
    if len(exactness_evidence) >= 2:
        require_paired_completed_output_exactness(
            exactness_evidence,
            source_request_pool_sha256s={
                role: pool_sha256s[role] for role in exactness_evidence
            },
        )
    observations = {role: _slo(value[1]) for role, value in rows.items()}
    if len({value.source_request_pool_sha256 for value in observations.values()}) != 1:
        raise ValueError("E0 serving family request pools are unpaired")
    require_paired_primary_goodputs(
        {role: observations[role] for role in ("Static", "TTS", "LightCone")}
    )
    return observations


def reduce_single_operator_e0_pilot(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Power each decision/load family, then seal the maximum required N."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e0_tuning"
        or materialization.stage != "E0"
        or materialization.materialization_rule
        != "valid_x_8_roles_x_2_loads_x_4_excluded_pilots"
    ):
        raise ValueError("E0 pilot reducer received another node")
    compatibility = _e0_receipt_from_payload(predecessor.decision.payload)
    expected = 16 * compatibility.valid_count * PILOT_BLOCK_COUNT
    if len(materialization.cells) != expected or len(actual_results) != expected:
        raise ValueError("E0 pilot is not exactly 16V x 4")
    base = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_power_prefix",
        "compatibility": predecessor.decision.payload["compatibility"],
        "compatibility_bundle_sha256": _sha(
            "E0 compatibility bundle",
            predecessor.decision.payload.get("compatibility_bundle_sha256"),
        ),
        "compatibility_evidence_manifest_sha256": _sha(
            "E0 compatibility evidence",
            predecessor.decision.payload.get("compatibility_evidence_manifest_sha256"),
        ),
        "onlinespec_source_authority_sha256": (
            predecessor.decision.payload.get("onlinespec_source_authority_sha256")
        ),
        "na_decisions": predecessor.decision.payload.get("na_decisions"),
        "frozen_tts_recipe_sha256": _sha(
            "E0 frozen TTS recipe",
            predecessor.decision.payload.get("frozen_tts_recipe_sha256"),
        ),
        "lightcone_recipe_sha256": _sha(
            "E0 LightCone recipe",
            predecessor.decision.payload.get("lightcone_recipe_sha256"),
        ),
        "selected_onlinespec_recipes": predecessor.decision.payload.get(
            "selected_onlinespec_recipes"
        ),
        "pilot_materialization_sha256": materialization.sha256,
    }
    if compatibility.valid_count == 0:
        if actual_results or materialization.cells:
            raise ValueError("all-N/A E0 pilot must contain zero serving rows")
        payload = {
            **base,
            "status": "ALL_NA",
            "selected_final_blocks": 0,
            "selected_final_prefix": [],
            "family_commitments": [],
            "underpowered_family_sha256s": [],
            "power_unresolved_family_sha256s": [],
        }
        result_sha = _digest(payload)
        return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
            decision_kind="e0_pilot_all_na",
            next_materialization_source_decision_sha256=result_sha,
            next_materialization_upstream_receipt_sha256s=(materialization.sha256,),
            payload={**payload, "power_prefix_sha256": result_sha},
        )
    observations = _serving_observations(materialization, actual_results)
    grouped = _e0_grouped(materialization, observations)
    if len(grouped) != 2 * compatibility.valid_count:
        raise ValueError("E0 pilot lacks every decision/load family")
    commitments = []
    selected = []
    underpowered = []
    power_unresolved = []
    for family in sorted(grouped):
        blocks = grouped[family]
        if set(blocks) != set(range(PILOT_BLOCK_COUNT)):
            raise ValueError("E0 family lacks exactly four excluded pilots")
        exclusions = _family_role_exclusions(blocks)
        pilots = []
        for block in range(PILOT_BLOCK_COUNT):
            paired = _e0_paired_role_goodputs(
                blocks[block], excluded_roles=frozenset(exclusions)
            )
            pilots.append(
                PilotBlock(
                    block_id=f"E0:{family[0]}:{family[1]}:excluded_pilot:{block}",
                    static_goodput=float(paired["Static"].goodput_tokens_per_second),
                    tts_goodput=float(paired["TTS"].goodput_tokens_per_second),
                    lightcone_goodput=float(
                        paired["LightCone"].goodput_tokens_per_second
                    ),
                )
            )
        plan = _pilot_power_resolution(tuple(pilots), exclusions)
        family_sha256 = _digest(
            {"stage": "E0", "decision_id": family[0], "load": family[1]}
        )
        commitment = {
            "family_sha256": family_sha256,
            "compatibility_decision_id": family[0],
            "load": family[1],
            "power_sizing": asdict(plan),
            "scientific_exclusions": exclusions,
        }
        commitment["commitment_sha256"] = _digest(commitment)
        commitments.append(commitment)
        if plan.status == "POWER_UNRESOLVED":
            power_unresolved.append(family_sha256)
        elif plan.underpowered or plan.selected_final_blocks is None:
            underpowered.append(family_sha256)
        else:
            selected.append(plan.selected_final_blocks)
    payload = {
        **base,
        "status": (
            "POWER_UNRESOLVED"
            if power_unresolved
            else "UNDERPOWERED"
            if underpowered
            else "READY"
        ),
        "selected_final_blocks": (
            None if underpowered or power_unresolved else max(selected)
        ),
        "selected_final_prefix": (
            []
            if underpowered or power_unresolved
            else list(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + max(selected)))
        ),
        "family_commitments": commitments,
        "underpowered_family_sha256s": sorted(underpowered),
        "power_unresolved_family_sha256s": sorted(power_unresolved),
        "reason_codes": [
            *(["one_or_more_families_power_unresolved"] if power_unresolved else []),
            *(["one_or_more_families_underpowered"] if underpowered else []),
        ],
    }
    result_sha = _digest(payload)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind=(
            "e0_pilot_power_unresolved"
            if power_unresolved
            else "e0_pilot_underpowered"
            if underpowered
            else "e0_pilot_actual_powered"
        ),
        next_materialization_source_decision_sha256=(
            None if underpowered or power_unresolved else result_sha
        ),
        next_materialization_upstream_receipt_sha256s=(
            () if underpowered or power_unresolved else (materialization.sha256,)
        ),
        payload={**payload, "power_prefix_sha256": result_sha},
    )


def materialize_single_operator_e0_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    protocol_lock: ProtocolLock,
    e0_compatibility_auxiliary: object,
) -> StageMaterializationReceipt:
    """Materialize the exact powered ``16VN`` final prefix, or zero for V=0."""

    if predecessor is None or predecessor.artifact.node != "e0_pilot":
        raise ValueError("E0 final requires completed excluded pilots")
    payload = predecessor.decision.payload
    source = _sha("E0 power prefix", payload.get("power_prefix_sha256"))
    status = payload.get("status")
    if status in {"UNDERPOWERED", "POWER_UNRESOLVED"}:
        stages = _stages()
        raise stages.FormalSingleOperatorStageBlocked(  # type: ignore[attr-defined]
            f"E0 excluded pilots cannot advance: {status}"
        )
    if status not in {"READY", "ALL_NA"}:
        raise ValueError("E0 final power status differs")
    if (
        predecessor.decision.next_materialization_source_decision_sha256 != source
        or predecessor.decision.next_materialization_upstream_receipt_sha256s
        != (predecessor.materialization.sha256,)
    ):
        raise ValueError("E0 final predecessor lineage differs")
    compatibility, bundle_sha256, evidence_sha256 = _e0_require_auxiliary_identity(
        predecessor,
        protocol_lock,
        e0_compatibility_auxiliary,
    )
    count = payload.get("selected_final_blocks")
    if compatibility.valid_count == 0:
        if count != 0:
            raise ValueError("all-N/A E0 final must select zero blocks")
        block_indices: tuple[int, ...] = ()
        cells: tuple[MaterializedCell, ...] = ()
    else:
        if type(count) is not int or not 12 <= count <= 20:
            raise ValueError("E0 final block count is outside [12, 20]")
        block_indices = tuple(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
        cells = _e0_serving_cells(
            compatibility=compatibility,
            compatibility_bundle_sha256=bundle_sha256,
            compatibility_evidence_sha256=evidence_sha256,
            onlinespec_source_authority_sha256=payload.get(
                "onlinespec_source_authority_sha256"
            ),  # type: ignore[arg-type]
            selected_recipes=_e0_selected_recipes(payload),
            frozen_tts_recipe_sha256=_sha(
                "E0 frozen TTS recipe", payload.get("frozen_tts_recipe_sha256")
            ),
            lightcone_recipe_sha256=_sha(
                "E0 LightCone recipe", payload.get("lightcone_recipe_sha256")
            ),
            block_indices=block_indices,
            final_lineage={
                "pilot_materialization_receipt_sha256": (
                    predecessor.materialization.sha256
                ),
                "power_prefix_sha256": source,
            },
        )
    return _receipt(
        stage="E0",
        protocol_lock_sha256=protocol_lock.sha256,
        upstream_receipt_sha256s=(predecessor.materialization.sha256,),
        source_decision_sha256=source,
        materialization_rule="valid_x_8_roles_x_2_loads_x_powered_final_blocks",
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _e0_breadth_fdr(
    common_slo_results: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Apply BH separately to the two registered common-SLO breadth families."""

    family_specs = (
        ("e0_common_slo_core", _E0_CORE_BREADTH_CONTRASTS),
        ("e0_common_slo_onlinespec", _E0_ONLINE_BREADTH_CONTRASTS),
    )
    adjusted = []
    for family_id, contrast_names in family_specs:
        p_values: dict[str, float] = {}
        unresolved = []
        for row in common_slo_results:
            decision_id = _sha("E0 breadth decision", row["compatibility_decision_id"])
            contrasts = row.get("contrasts")
            if type(contrasts) is not dict:
                raise ValueError("E0 breadth family lacks contrast payloads")
            for contrast_name in contrast_names:
                contrast = contrasts.get(contrast_name)
                if type(contrast) is not dict:
                    raise ValueError("E0 breadth contrast is absent")
                hypothesis_id = f"{decision_id}:{contrast_name}"
                status = contrast.get("status")
                if status in {
                    "EXCLUDED_UNSAFE_OR_INACTIVE",
                    "UNRESOLVED_ZERO_GOODPUT",
                    "UNRESOLVED_ZERO_VARIANCE",
                }:
                    reasons = contrast.get("reason_codes")
                    if (
                        type(reasons) is not list
                        or not reasons
                        or reasons != sorted(set(reasons))
                        or any(
                            type(reason) is not str or not reason for reason in reasons
                        )
                    ):
                        raise ValueError("E0 unresolved breadth reasons differ")
                    if status.startswith("UNRESOLVED_ZERO_") and reasons != [status]:
                        raise ValueError("E0 unresolved breadth reason/status differ")
                    if status == "EXCLUDED_UNSAFE_OR_INACTIVE" and (
                        any(":" not in reason for reason in reasons)
                        or type(contrast.get("excluded_roles")) is not list
                        or not contrast["excluded_roles"]
                        or type(contrast.get("evidence_cell_ids")) is not list
                        or not contrast["evidence_cell_ids"]
                    ):
                        raise ValueError("E0 excluded breadth evidence differs")
                    unresolved.append(
                        {
                            "hypothesis_id": hypothesis_id,
                            "status": status,
                            "reason_codes": reasons,
                        }
                    )
                elif status in {None, "RESOLVED"}:
                    # Schema-1 test/replay payloads omitted RESOLVED.  Current
                    # reducers always publish it explicitly.
                    raw_p_value = contrast.get("raw_p_value")
                    if (
                        type(raw_p_value) not in {int, float}
                        or isinstance(raw_p_value, bool)
                        or not math.isfinite(float(raw_p_value))
                        or not 0.0 <= float(raw_p_value) <= 1.0
                    ):
                        raise ValueError("E0 resolved breadth p-value differs")
                    p_values[hypothesis_id] = float(raw_p_value)
                else:
                    raise ValueError("E0 breadth contrast status differs")
        # Scientific exclusions remove only the affected hypotheses.  They do
        # not erase the valid p-values from other model/backend/task rows in
        # the same preregistered breadth family.
        decisions = () if not p_values else benjamini_hochberg(p_values)
        status = (
            "PARTIALLY_RESOLVED"
            if unresolved and p_values
            else "UNRESOLVED"
            if unresolved
            else "RESOLVED"
        )
        adjusted.append(
            {
                "family_id": family_id,
                "load": "common_slo_load",
                "procedure": "benjamini-hochberg",
                "false_discovery_rate": 0.05,
                "status": status,
                "tested_hypothesis_count": len(p_values),
                "decisions": [asdict(row) for row in decisions],
                "unresolved_hypotheses": unresolved,
            }
        )
    return tuple(adjusted)


def _e0_na_hypothesis_exclusions(
    compatibility: E0CompatibilityReceipt,
) -> tuple[dict[str, object], ...]:
    rows = []
    names = _E0_CORE_BREADTH_CONTRASTS + _E0_ONLINE_BREADTH_CONTRASTS
    for decision in compatibility.decisions:
        if decision.disposition != "N/A":
            continue
        for name in names:
            rows.append(
                {
                    "compatibility_decision_id": decision.decision_id,
                    "contrast": name,
                    "status": "EXCLUDED_NA",
                    "reason_code": decision.reason_code,
                    "included_in_multiplicity_family": False,
                }
            )
    return tuple(rows)


def reduce_single_operator_e0_final(
    predecessor: RebuiltFormalSingleOperatorStageCompletion | None,
    materialization: StageMaterializationReceipt,
    actual_results: tuple[FormalSingleOperatorValidatedActual, ...],
) -> FormalSingleOperatorDecisionDraft:
    """Reduce paired E0 breadth evidence without inventing CIs for N/A rows."""

    stages = _stages()
    if (
        predecessor is None
        or predecessor.artifact.node != "e0_pilot"
        or materialization.stage != "E0"
        or materialization.materialization_rule
        != "valid_x_8_roles_x_2_loads_x_powered_final_blocks"
    ):
        raise ValueError("E0 final reducer received another node")
    compatibility = _e0_receipt_from_payload(predecessor.decision.payload)
    count = predecessor.decision.payload.get("selected_final_blocks")
    if compatibility.valid_count == 0:
        if count != 0 or materialization.cells or actual_results:
            raise ValueError("all-N/A E0 final must contain no serving evidence")
        exclusions = _e0_na_hypothesis_exclusions(compatibility)
        result = {
            "schema_version": 1,
            "kind": "formal_single_operator_e0_confirmation",
            "status": "ALL_NA",
            "valid_count": 0,
            "block_count": 0,
            "materialization_sha256": materialization.sha256,
            "compatibility": predecessor.decision.payload["compatibility"],
            "na_decisions": predecessor.decision.payload.get("na_decisions"),
            "family_results": [],
            "breadth_fdr_families": list(_e0_breadth_fdr([])),
            "na_hypothesis_exclusions": list(exclusions),
        }
        result_sha = _digest(result)
        return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
            decision_kind="e0_final_all_na",
            next_materialization_source_decision_sha256=None,
            next_materialization_upstream_receipt_sha256s=(),
            payload={**result, "confirmation_sha256": result_sha},
        )
    if type(count) is not int or not 12 <= count <= 20:
        raise ValueError("E0 final block count differs from powered prefix")
    expected = 16 * compatibility.valid_count * count
    if len(materialization.cells) != expected or len(actual_results) != expected:
        raise ValueError("E0 final is not exactly 16VN")
    observations = _serving_observations(materialization, actual_results)
    grouped = _e0_grouped(materialization, observations)
    if len(grouped) != 2 * compatibility.valid_count:
        raise ValueError("E0 final lacks every decision/load family")
    expected_blocks = set(range(PILOT_BLOCK_COUNT, PILOT_BLOCK_COUNT + count))
    results = []
    common_slo = []
    for family in sorted(grouped):
        blocks = grouped[family]
        if set(blocks) != expected_blocks:
            raise ValueError("E0 final family lacks every powered block")
        exclusions = _family_role_exclusions(blocks)
        pairs: dict[str, dict[str, tuple[float, float]]] = {
            name: {} for name in _E0_CONTRAST_ROLES
        }
        request_count = 0
        target_slo_pass = True
        for block in sorted(blocks):
            slo = _e0_paired_role_goodputs(
                blocks[block], excluded_roles=frozenset(exclusions)
            )
            target_slo_pass &= slo["Target-only"].status == "PASS"
            for name, (left, right) in _E0_CONTRAST_ROLES.items():
                pairs[name][f"E0:final:{block - PILOT_BLOCK_COUNT:02d}"] = (
                    float(slo[left].goodput_tokens_per_second),
                    float(slo[right].goodput_tokens_per_second),
                )
            request_count += slo["LightCone"].eligible_requests
        reduction = _resolve_family_contrasts(
            paired=pairs,
            contrast_roles=_E0_CONTRAST_ROLES,
            exclusions=exclusions,
            target_slo_pass=target_slo_pass,
        )
        contrast_payloads = reduction["contrast_payloads"]
        assert isinstance(contrast_payloads, dict)
        result = {
            "family_sha256": _digest(
                {
                    "stage": "E0",
                    "compatibility_decision_id": family[0],
                    "load": family[1],
                }
            ),
            "compatibility_decision_id": family[0],
            "load": family[1],
            "block_count": count,
            "request_count": request_count,
            "paired": True,
            "reducer": "paired_block_bca",
            "contrasts": contrast_payloads,
            "holm_primary": reduction["holm_decisions"],
            "holm_status": reduction["holm_status"],
            "all_registered_contrasts_resolved": reduction[
                "all_registered_contrasts_resolved"
            ],
            "target_only_gate": reduction["target_only_gate"],
            "scientific_exclusions": exclusions,
            "reason_codes": reduction["reason_codes"],
            "status": (
                "CONFIRMED" if reduction["deployment_confirmed"] else "NOT_CONFIRMED"
            ),
        }
        result["result_sha256"] = _digest(result)
        results.append(result)
        if family[1] == "common_slo_load":
            common_slo.append(result)
    exclusions = _e0_na_hypothesis_exclusions(compatibility)
    result = {
        "schema_version": 1,
        "kind": "formal_single_operator_e0_confirmation",
        "status": (
            "CONFIRMED"
            if all(row["status"] == "CONFIRMED" for row in results)
            else "NOT_CONFIRMED"
        ),
        "valid_count": compatibility.valid_count,
        "block_count": count,
        "materialization_sha256": materialization.sha256,
        "compatibility": predecessor.decision.payload["compatibility"],
        "na_decisions": predecessor.decision.payload.get("na_decisions"),
        "family_results": results,
        "breadth_fdr_families": list(_e0_breadth_fdr(common_slo)),
        "na_hypothesis_exclusions": list(exclusions),
    }
    result_sha = _digest(result)
    return stages.FormalSingleOperatorDecisionDraft(  # type: ignore[attr-defined,no-any-return]
        decision_kind="e0_final_actual_reduced",
        next_materialization_source_decision_sha256=None,
        next_materialization_upstream_receipt_sha256s=(),
        payload={**result, "confirmation_sha256": result_sha},
    )


__all__ = [
    "materialize_single_operator_e0_final",
    "materialize_single_operator_e0_pilot",
    "materialize_single_operator_e0_tuning",
    "materialize_single_operator_e1a",
    "materialize_single_operator_e3b_final",
    "materialize_single_operator_e3b_pilot",
    "materialize_single_operator_e5_final",
    "materialize_single_operator_e5_pilot",
    "materialize_single_operator_e6_final",
    "materialize_single_operator_e6_pilot",
    "reduce_single_operator_e0_final",
    "reduce_single_operator_e0_pilot",
    "reduce_single_operator_e0_tuning",
    "reduce_single_operator_e1a",
    "reduce_single_operator_e3b_final",
    "reduce_single_operator_e3b_pilot",
    "reduce_single_operator_e4_profiler",
    "reduce_single_operator_e5_final",
    "reduce_single_operator_e5_pilot",
    "reduce_single_operator_e6_final",
    "reduce_single_operator_e6_pilot",
]
