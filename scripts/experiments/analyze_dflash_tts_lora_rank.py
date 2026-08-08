#!/usr/bin/env python3
"""Analyze the attested Stage-2 DFlash LoRA rank sweep.

This module is intentionally a derivation layer.  Stage-2 bundle verification
is delegated to ``build_dflash_tts_lora_rank_candidates`` and every run is
validated/reconstructed by ``analyze_dflash_tts_calibration``.  The only new
logic is the preregistered cross-rank comparison:

* ``tuned_envelope`` selects the best safe local LR at each rank before the
  bounded rank comparison; and
* ``fixed_center_control`` varies rank while keeping the Stage-1 optimizer,
  LR, and weight decay fixed.

Both views require non-negative paired delta-A on every locked prompt.  The
published artifact is immutable.  Its locator-free measurement core is hashed
separately from upstream hashes whose source analyses still contain local
paths, so an archive can distinguish portable evidence from locator-bound
provenance without weakening either check.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import analyze_dflash_tts_calibration as calibration_analysis  # noqa: E402
import build_dflash_tts_lora_rank_candidates as rank_builder  # noqa: E402
import run_dflash_tts_calibration_sweep as calibration  # noqa: E402


SCHEMA_VERSION = 1
KIND = "dflash_tts_lora_rank_stage2_analysis"
VIEWS = ("tuned_envelope", "fixed_center_control")
SELECTION_RULE = {
    "scope": "separate_cross_rank_selection_within_each_lora_mode",
    "views": {
        "tuned_envelope": (
            "select the best safe local LR independently at each rank, then "
            "compare the resulting bounded-rank envelope"
        ),
        "fixed_center_control": (
            "compare ranks at the exact Stage-1 winner optimizer, LR, and "
            "weight decay; only rank changes"
        ),
    },
    "eligibility": (
        "exact Stage-2 candidate contract and provenance; fully attested "
        "paired runs; exact Static output; finite A, loss, gradient, HBM, and "
        "target-calls/output metrics"
    ),
    "safety_gate": "paired_delta_A_greater_than_or_equal_to_zero_on_every_prompt",
    "ordering": calibration_analysis.SELECTION_RULE["ordering"],
    "A_semantics": calibration_analysis.SELECTION_RULE["A_semantics"],
    "rank_grid": list(rank_builder.RANKS),
    "boundary_rule": (
        "if the bounded winner is rank 4 or 64, require one-octave extension "
        "toward rank 2 or 128 before any optimum claim"
    ),
    "claim_scope": "bounded_rank_comparison_not_global_rank_optimum",
}


def _expect(observed: Any, expected: Any, label: str) -> None:
    if calibration.frozen._canonical_json(observed) != calibration.frozen._canonical_json(
        expected
    ):
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def _required_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _render(payload: dict[str, Any]) -> str:
    # Match the exclusive JSON writer exactly (json.dump's default
    # ensure_ascii=True), so --check is a byte attestation rather than a second
    # serialization convention.
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _verify_stage2_bundle(
    candidate_spec_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], calibration.CandidateSweep]:
    """Rebuild and byte-verify the Stage-2 spec/provenance pair."""

    verified = rank_builder.verify_published_bundle(candidate_spec_path)
    return verified.candidate_spec, verified.provenance, verified.sweep


def _portable_candidate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop machine-local locators while retaining all measurements/hashes."""

    output: list[dict[str, Any]] = []
    for source in rows:
        row = copy.deepcopy(source)
        for sample in _required_list(
            row.get("sample_results"), f"{row.get('candidate_id')}.sample_results"
        ):
            if not isinstance(sample, dict):
                raise ValueError("candidate sample result must be an object")
            sample.pop("run_root", None)
        output.append(row)
    return output


def _load_stage2_measurement_sources(
    *,
    sweep: calibration.CandidateSweep,
    output_root: Path,
    base_analysis: dict[str, Any],
) -> tuple[
    dict[tuple[int, str], calibration_analysis.BoundRun],
    dict[tuple[int, str], dict[str, Any]],
]:
    """Reuse the canonical loaders to expose attested Stage-2 detail."""

    root = output_root.expanduser().resolve()
    artifact_identity_lock = calibration_analysis._load_artifact_identity_lock(root)
    source_records = {
        (record.get("sample_index"), record.get("candidate_id")): record
        for record in _required_list(
            base_analysis.get("source_runs"), "Stage-2 source_runs"
        )
        if isinstance(record, dict)
    }
    bound: dict[tuple[int, str], calibration_analysis.BoundRun] = {}
    ordered: list[calibration_analysis.BoundRun] = []
    for sample in sweep.samples:
        sample_index = int(sample["sample_index"])
        for candidate_index, candidate in enumerate(sweep.candidates):
            run = calibration_analysis._load_bound_run(
                root,
                artifact_identity_lock=artifact_identity_lock,
                sweep=sweep,
                candidate=candidate,
                candidate_index=candidate_index,
                sample=sample,
            )
            key = (sample_index, candidate.candidate_id)
            expected = source_records.get(key)
            observed = {
                "sample_index": sample_index,
                "candidate_id": candidate.candidate_id,
                **run.evidence_hashes,
            }
            _expect(observed, expected, f"Stage-2 detailed source {key}")
            bound[key] = run
            ordered.append(run)

    metric_rows = calibration_analysis.aggregation.build_long_table(
        [run.artifact_dir for run in ordered],
        bucket_size=calibration_analysis.BUCKET_SIZE,
    )
    by_root = {
        str(Path(row["run_root"]).resolve()): row for row in metric_rows
    }
    if len(by_root) != len(ordered):
        raise ValueError("Stage-2 detailed metric reconstruction lost a run")
    metrics = {
        key: by_root[str(run.artifact_dir.resolve())]
        for key, run in bound.items()
    }
    return bound, metrics


def _loss_context_record(
    run: calibration_analysis.BoundRun,
) -> dict[str, Any]:
    points = []
    for round_row in run.rounds:
        update = round_row.get("update")
        if not isinstance(update, dict) or update.get("applied") is not True:
            continue
        loss, _loss_source = calibration_analysis.aggregation._round_scalar(
            round_row, "loss"
        )
        grad_norm, _grad_source = calibration_analysis.aggregation._round_scalar(
            round_row, "grad_norm"
        )
        if not _finite_number(loss) or not _finite_number(grad_norm):
            raise ValueError(
                f"{run.candidate_id} sample {run.sample_index} has a non-finite "
                "applied loss-context point"
            )
        points.append(
            {
                "round_index": round_row.get("round_index"),
                "optimizer_step": update.get("optimizer_step"),
                "prefix_len_before": calibration_analysis.aggregation._round_prefix(
                    round_row
                ),
                "loss": float(loss),
                "grad_norm": float(grad_norm),
            }
        )
    return {
        "raw_rounds_relative_path": (
            f"sample-{run.sample_index:04d}/{run.candidate_id}/artifact/rounds.jsonl"
        ),
        "raw_rounds_sha256": run.evidence_hashes["rounds_sha256"],
        "applied_update_count": len(points),
        "points": points,
        "points_sha256": calibration.frozen._sha256_json(points),
    }


def _memory_detail(
    metric: dict[str, Any], static_metric: dict[str, Any]
) -> dict[str, Any]:
    allocated = metric.get("whole_process_peak_hbm_bytes")
    reserved = metric.get("whole_process_peak_hbm_reserved_bytes")
    static_allocated = static_metric.get("whole_process_peak_hbm_bytes")
    static_reserved = static_metric.get("whole_process_peak_hbm_reserved_bytes")
    for label, value in (
        ("allocated", allocated),
        ("reserved", reserved),
        ("static allocated", static_allocated),
        ("static reserved", static_reserved),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Stage-2 {label} whole-process HBM is invalid")
    phase_bytes = {
        key: value
        for key, value in metric.items()
        if key.endswith("_bytes")
        and (
            key.startswith("hbm_")
            or key.startswith("whole_process_peak_running_peak_")
        )
    }
    optimizer_ledger = {
        key: metric.get(key)
        for key in (
            *calibration_analysis.aggregation.MEMORY_KEYS,
            "optimizer_resident_bytes",
            "optimizer_update_peak_bytes",
            "optimizer_bytes_per_trainable_parameter",
            "declared_incremental_optimizer_resident_bytes",
            "optimizer_memory_evidence",
            "optimizer_ledger_excludes",
        )
    }
    if metric.get("mode") != "static":
        if optimizer_ledger["optimizer_memory_evidence"] != (
            "exact_declared_optimizer_tensor_ledger"
        ):
            raise ValueError(
                "Stage-2 adaptive run lacks an exact optimizer memory ledger"
            )
        for key in calibration_analysis.aggregation.MEMORY_KEYS:
            value = optimizer_ledger[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Stage-2 optimizer memory ledger {key} is incomplete"
                )
    return {
        "whole_process_peak_allocated_bytes": allocated,
        "whole_process_peak_reserved_bytes": reserved,
        "static_whole_process_peak_allocated_bytes": static_allocated,
        "static_whole_process_peak_reserved_bytes": static_reserved,
        "whole_process_peak_allocated_over_static_bytes": (
            int(allocated) - int(static_allocated)
        ),
        "whole_process_peak_reserved_over_static_bytes": (
            int(reserved) - int(static_reserved)
        ),
        "phase_bytes": phase_bytes,
        "optimizer_ledger": optimizer_ledger,
    }


def _enrich_candidate_measurements(
    *,
    rows: list[dict[str, Any]],
    sweep: calibration.CandidateSweep,
    output_root: Path,
    base_analysis: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    bound, metrics = _load_stage2_measurement_sources(
        sweep=sweep,
        output_root=output_root,
        base_analysis=base_analysis,
    )
    static_candidates = [
        candidate for candidate in sweep.candidates if candidate.mode == "static"
    ]
    if len(static_candidates) != 1:
        raise ValueError("Stage-2 requires exactly one Static candidate")
    static_id = static_candidates[0].candidate_id
    expected_seed = _required_dict(
        provenance.get("derivation"), "Stage-2 derivation"
    ).get("selection_hot_path", {}).get("adapter_seed")
    if isinstance(expected_seed, bool) or not isinstance(expected_seed, int):
        raise ValueError("Stage-2 provenance adapter_seed is invalid")

    row_map = {row["candidate_id"]: row for row in rows}
    for candidate in sweep.candidates:
        row = row_map[candidate.candidate_id]
        reserved_values = []
        reserved_deltas = []
        seeds: set[int | None] = set()
        ledgers = []
        for sample_result in row["sample_results"]:
            sample_index = int(sample_result["sample_index"])
            key = (sample_index, candidate.candidate_id)
            run = bound[key]
            metric = metrics[key]
            static_metric = metrics[(sample_index, static_id)]
            optimization = _required_dict(
                run.identity.get("optimization"),
                f"Stage-2 {candidate.candidate_id} optimization",
            )
            seed = optimization.get("adapter_seed")
            expected_candidate_seed = None if candidate.mode == "static" else expected_seed
            _expect(
                seed,
                expected_candidate_seed,
                f"Stage-2 {candidate.candidate_id} adapter_seed",
            )
            seeds.add(seed)
            memory = _memory_detail(metric, static_metric)
            _expect(
                memory["whole_process_peak_allocated_bytes"],
                sample_result["whole_process_peak_hbm_bytes"],
                f"Stage-2 {candidate.candidate_id} allocated HBM reconstruction",
            )
            _expect(
                metric.get("trainable_parameter_count"),
                sample_result["trainable_parameter_count"],
                f"Stage-2 {candidate.candidate_id} trainable parameters",
            )
            _expect(
                sample_result.get("adapter_seed"),
                seed,
                f"Stage-2 {candidate.candidate_id} sample adapter_seed",
            )
            sample_result["adapter_seed"] = seed
            sample_result["memory"] = memory
            sample_result["loss_context"] = _loss_context_record(run)
            reserved_values.append(memory["whole_process_peak_reserved_bytes"])
            reserved_deltas.append(
                memory["whole_process_peak_reserved_over_static_bytes"]
            )
            ledgers.append(memory["optimizer_ledger"])
        if len(seeds) != 1:
            raise ValueError(
                f"Stage-2 {candidate.candidate_id} adapter_seed differs by prompt"
            )
        if len(
            {calibration.frozen._canonical_json(ledger) for ledger in ledgers}
        ) != 1:
            raise ValueError(
                f"Stage-2 {candidate.candidate_id} optimizer memory ledger "
                "differs by prompt"
            )
        candidate_seed = next(iter(seeds))
        _expect(
            row.get("adapter_seed"),
            candidate_seed,
            f"Stage-2 {candidate.candidate_id} row adapter_seed",
        )
        _expect(
            row["aggregate"].get("adapter_seed"),
            candidate_seed,
            f"Stage-2 {candidate.candidate_id} aggregate adapter_seed",
        )
        row["adapter_seed"] = candidate_seed
        max_reserved = max(reserved_values)
        max_reserved_delta = max(reserved_deltas)
        _expect(
            row["aggregate"].get("max_whole_process_peak_hbm_reserved_bytes"),
            max_reserved,
            f"Stage-2 {candidate.candidate_id} reserved HBM aggregate",
        )
        _expect(
            row["aggregate"].get(
                "max_whole_process_peak_hbm_reserved_over_static_bytes"
            ),
            max_reserved_delta,
            f"Stage-2 {candidate.candidate_id} reserved HBM delta aggregate",
        )
        row["aggregate"]["optimizer_memory_ledger"] = ledgers[0]


def _scope_map(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    derivation = _required_dict(
        provenance.get("derivation"), "Stage-2 provenance.derivation"
    )
    _expect(
        derivation.get("rank_values"),
        list(rank_builder.RANKS),
        "Stage-2 rank grid",
    )
    hot_path = _required_dict(
        derivation.get("selection_hot_path"), "Stage-2 selection hot path"
    )
    adapter_seed = hot_path.get("adapter_seed")
    if isinstance(adapter_seed, bool) or not isinstance(adapter_seed, int):
        raise ValueError("Stage-2 selection hot-path adapter_seed is invalid")
    active_modes = _required_list(
        derivation.get("active_modes"), "Stage-2 active modes"
    )
    if (
        not active_modes
        or active_modes
        != [mode for mode in rank_builder.MODES if mode in active_modes]
        or any(mode not in rank_builder.MODES for mode in active_modes)
    ):
        raise ValueError("Stage-2 active mode order is invalid")
    scopes: dict[str, dict[str, Any]] = {}
    for raw in _required_list(derivation.get("scopes"), "Stage-2 scopes"):
        scope = _required_dict(raw, "Stage-2 scope")
        mode = scope.get("mode")
        if mode not in rank_builder.MODES or mode in scopes:
            raise ValueError(f"Stage-2 scope mode is invalid or duplicated: {mode!r}")
        source_winner = _required_dict(
            scope.get("source_winner"), f"Stage-2 {mode} source winner"
        )
        fixed_center = _required_dict(
            scope.get("fixed_center_control"),
            f"Stage-2 {mode} fixed-center control",
        )
        _expect(
            source_winner.get("adapter_seed"),
            adapter_seed,
            f"Stage-2 {mode} source-winner adapter_seed",
        )
        _expect(
            fixed_center.get("adapter_seed"),
            adapter_seed,
            f"Stage-2 {mode} fixed-center adapter_seed",
        )
        scopes[str(mode)] = scope
    _expect(list(scopes), active_modes, "Stage-2 scope order")
    omissions = _required_list(
        derivation.get("omissions"), "Stage-2 mode omissions"
    )
    omitted_modes = []
    for raw in omissions:
        omission = _required_dict(raw, "Stage-2 mode omission")
        mode = omission.get("mode")
        if (
            mode not in rank_builder.MODES
            or mode in active_modes
            or mode in omitted_modes
            or omission.get("status") != "omitted"
        ):
            raise ValueError("Stage-2 mode omission is invalid")
        omitted_modes.append(mode)
    _expect(
        omitted_modes,
        [mode for mode in rank_builder.MODES if mode not in active_modes],
        "Stage-2 omitted mode coverage",
    )
    return scopes


def _validate_candidate_contract(
    *,
    expected_spec: dict[str, Any],
    sweep: calibration.CandidateSweep,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require exact mode/rank/optimizer/LR/WD agreement with provenance."""

    scopes = _scope_map(provenance)
    _expect(
        len(sweep.candidates),
        1 + 15 * len(scopes),
        "Stage-2 candidate count",
    )
    _expect(len(sweep.samples), 2, "Stage-2 sample count")
    _expect(
        [candidate.candidate_id for candidate in sweep.candidates],
        [candidate["candidate_id"] for candidate in expected_spec["candidates"]],
        "Stage-2 candidate order",
    )
    row_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in row_map:
            raise ValueError(f"Stage-2 candidate row is invalid or duplicated: {candidate_id!r}")
        row_map[candidate_id] = row
    _expect(
        list(row_map),
        [candidate.candidate_id for candidate in sweep.candidates],
        "Stage-2 candidate-row order",
    )

    observed_modes = {
        candidate.mode for candidate in sweep.candidates if candidate.mode != "static"
    }
    _expect(
        sorted(observed_modes),
        sorted(scopes),
        "Stage-2 candidate scope coverage",
    )
    for candidate in sweep.candidates:
        row = row_map[candidate.candidate_id]
        expected_row = next(
            item
            for item in expected_spec["candidates"]
            if item["candidate_id"] == candidate.candidate_id
        )
        for key, observed in (
            ("mode", row.get("mode")),
            ("optimizer", row.get("optimizer")),
            ("learning_rate", row.get("learning_rate")),
            ("weight_decay", row.get("weight_decay")),
            ("rank", row.get("rank")),
            ("draft_cache_policy", row.get("draft_cache_policy")),
            ("diagnostic_kind", row.get("diagnostic_kind")),
        ):
            _expect(
                observed,
                expected_row[key],
                f"Stage-2 {candidate.candidate_id} {key}",
            )
        _expect(
            row.get("candidate_spec_selection_eligible"),
            True,
            f"Stage-2 {candidate.candidate_id} selection eligibility",
        )
        _expect(
            candidate.parameter_audit_stride,
            0,
            f"Stage-2 {candidate.candidate_id} audit stride",
        )
        if candidate.mode == "static":
            continue
        scope = scopes[candidate.mode]
        source_winner = _required_dict(
            scope.get("source_winner"), f"Stage-2 {candidate.mode} source winner"
        )
        _expect(
            row.get("adapter_seed"),
            source_winner.get("adapter_seed"),
            f"Stage-2 {candidate.candidate_id} adapter_seed",
        )
        window = _required_dict(
            scope.get("learning_rate_window"),
            f"Stage-2 {candidate.mode} LR window",
        )
        expected_prefix = f"{candidate.mode}-r{candidate.config.rank}-lr-"
        if not candidate.candidate_id.startswith(expected_prefix):
            raise ValueError(
                f"Stage-2 candidate ID does not bind mode/rank: {candidate.candidate_id}"
            )
        lr_slice = candidate.candidate_id.removeprefix(expected_prefix)
        if lr_slice not in {name for name, _numerator, _denominator in rank_builder.LR_SLICES}:
            raise ValueError(f"Stage-2 candidate has invalid LR slice: {candidate.candidate_id}")
        for key, observed, expected in (
            ("optimizer", candidate.config.optimizer, source_winner["optimizer"]),
            ("weight_decay", candidate.config.weight_decay, source_winner["weight_decay"]),
            ("learning_rate", candidate.config.learning_rate, window[lr_slice]),
        ):
            _expect(observed, expected, f"Stage-2 {candidate.candidate_id} {key}")
        if candidate.config.rank not in rank_builder.RANKS:
            raise ValueError(f"Stage-2 candidate rank is outside the locked grid: {candidate.candidate_id}")
    return row_map


def _prompt_safety(
    row: dict[str, Any], sample_indices: Sequence[int]
) -> list[dict[str, Any]]:
    results = _required_list(
        row.get("sample_results"), f"{row.get('candidate_id')}.sample_results"
    )
    by_sample: dict[int, dict[str, Any]] = {}
    for result in results:
        result = _required_dict(result, "Stage-2 sample result")
        sample_index = result.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError("Stage-2 sample result has invalid sample_index")
        if sample_index in by_sample:
            raise ValueError("Stage-2 sample result is duplicated")
        by_sample[sample_index] = result
    _expect(sorted(by_sample), sorted(sample_indices), "Stage-2 prompt coverage")
    output = []
    for sample_index in sorted(sample_indices):
        delta = by_sample[sample_index].get("paired_delta_A")
        safe = _finite_number(delta) and float(delta) >= 0.0
        output.append(
            {
                "sample_index": sample_index,
                "paired_delta_A": delta,
                "safe_nonnegative": safe,
            }
        )
    return output


def _is_safe(row: dict[str, Any], sample_indices: Sequence[int]) -> bool:
    aggregate = _required_dict(
        row.get("aggregate"), f"{row.get('candidate_id')}.aggregate"
    )
    prompts = _prompt_safety(row, sample_indices)
    computed = (
        aggregate.get("evidence_eligible") is True
        and aggregate.get("safe_for_selection") is True
        and aggregate.get("all_outputs_exact_static") is True
        and aggregate.get("all_losses_and_gradients_finite") is True
        and aggregate.get("ineligibility_reasons") == []
        and all(item["safe_nonnegative"] is True for item in prompts)
    )
    # Fail closed if the upstream aggregate and the explicit prompt gate ever
    # disagree.  Selection must never depend on only an aggregate mean.
    _expect(
        aggregate.get("safe_for_selection") is True,
        computed,
        f"{row.get('candidate_id')} paired prompt safety",
    )
    return computed


def _compact_point(
    row: dict[str, Any], sample_indices: Sequence[int], *, lr_slice: str
) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "mode": row["mode"],
        "rank": row["rank"],
        "lr_slice": lr_slice,
        "optimizer": row["optimizer"],
        "learning_rate": row["learning_rate"],
        "weight_decay": row["weight_decay"],
        "adapter_seed": row["adapter_seed"],
        "prompt_safety": _prompt_safety(row, sample_indices),
        "aggregate": row["aggregate"],
    }


def _rank_boundary(rank: int | None) -> dict[str, Any]:
    lower = min(rank_builder.RANKS)
    upper = max(rank_builder.RANKS)
    at_lower = rank == lower
    at_upper = rank == upper
    extension = [lower // 2] if at_lower else ([upper * 2] if at_upper else [])
    return {
        "tested_rank_bounds": {"minimum": lower, "maximum": upper},
        "selected_rank": rank,
        "at_lower_boundary": at_lower,
        "at_upper_boundary": at_upper,
        "requires_rank_grid_extension_before_optimum_claim": bool(extension),
        "suggested_extension_ranks": extension,
        "global_optimum_claim": False,
    }


def _cross_rank_decision(
    *,
    mode: str,
    view: str,
    points: list[tuple[dict[str, Any], str]],
    sample_indices: Sequence[int],
) -> dict[str, Any]:
    safe = sorted(
        (row for row, _slice in points if _is_safe(row, sample_indices)),
        key=calibration_analysis._selection_order,
    )
    slice_by_id = {row["candidate_id"]: lr_slice for row, lr_slice in points}
    winner = safe[0] if safe else None
    return {
        "mode": mode,
        "view": view,
        "status": "bounded_rank_winner" if winner is not None else "no_safe_selection",
        "tested_rank_count": len(rank_builder.RANKS),
        "selectable_rank_point_count": len(points),
        "safe_rank_point_count": len(safe),
        "winner": (
            _compact_point(
                winner,
                sample_indices,
                lr_slice=slice_by_id[winner["candidate_id"]],
            )
            if winner is not None
            else None
        ),
        "ordered_safe_candidate_ids": [row["candidate_id"] for row in safe],
        "rank_boundary": _rank_boundary(None if winner is None else int(winner["rank"])),
        "global_optimum_claim": False,
    }


def _comparison_views(
    *,
    row_map: dict[str, dict[str, Any]],
    scopes: dict[str, dict[str, Any]],
    sample_indices: Sequence[int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    comparisons: dict[str, list[dict[str, Any]]] = {view: [] for view in VIEWS}
    selected_rows: dict[str, list[dict[str, Any]]] = {view: [] for view in VIEWS}
    for mode in scopes:
        scope = scopes[mode]
        learning_rate_window = _required_dict(
            scope.get("learning_rate_window"), f"{mode} learning-rate window"
        )
        tuned_points: list[tuple[dict[str, Any], str]] = []
        per_rank = []
        for rank in rank_builder.RANKS:
            candidates = [
                (row_map[f"{mode}-r{rank}-lr-{slice_name}"], slice_name)
                for slice_name, _numerator, _denominator in rank_builder.LR_SLICES
            ]
            safe = sorted(
                (
                    (row, slice_name)
                    for row, slice_name in candidates
                    if _is_safe(row, sample_indices)
                ),
                key=lambda item: calibration_analysis._selection_order(item[0]),
            )
            winner = safe[0] if safe else None
            selected_slice = None if winner is None else winner[1]
            lower_boundary = selected_slice == "div3"
            upper_boundary = selected_slice == "times3"
            learning_rate_boundary = {
                "tested_learning_rate_bounds": {
                    "minimum": learning_rate_window["div3"],
                    "maximum": learning_rate_window["times3"],
                },
                "selected_slice": selected_slice,
                "at_lower_boundary": lower_boundary,
                "at_upper_boundary": upper_boundary,
                "requires_lr_grid_extension_before_optimum_claim": (
                    lower_boundary or upper_boundary
                ),
                "suggested_extension_learning_rates": (
                    [learning_rate_window["div3"] / 3.0]
                    if lower_boundary
                    else (
                        [learning_rate_window["times3"] * 3.0]
                        if upper_boundary
                        else []
                    )
                ),
                "global_optimum_claim": False,
            }
            per_rank.append(
                {
                    "rank": rank,
                    "status": "local_lr_winner" if winner else "no_safe_local_lr",
                    "candidate_count": len(candidates),
                    "safe_candidate_count": len(safe),
                    "winner": (
                        _compact_point(
                            winner[0], sample_indices, lr_slice=winner[1]
                        )
                        if winner
                        else None
                    ),
                    "ordered_safe_candidate_ids": [
                        row["candidate_id"] for row, _slice_name in safe
                    ],
                    "learning_rate_boundary": learning_rate_boundary,
                }
            )
            if winner:
                tuned_points.append(winner)
                selected_rows["tuned_envelope"].append(winner[0])
        tuned_decision = _cross_rank_decision(
            mode=mode,
            view="tuned_envelope",
            points=tuned_points,
            sample_indices=sample_indices,
        )
        tuned_decision["per_rank_local_lr_selection"] = per_rank
        winning_id = (
            None
            if tuned_decision["winner"] is None
            else tuned_decision["winner"]["candidate_id"]
        )
        winning_rank_record = next(
            (
                record
                for record in per_rank
                if record["winner"] is not None
                and record["winner"]["candidate_id"] == winning_id
            ),
            None,
        )
        tuned_decision["winner_learning_rate_boundary"] = (
            None
            if winning_rank_record is None
            else winning_rank_record["learning_rate_boundary"]
        )
        tuned_decision["requires_lr_grid_extension_before_optimum_claim"] = bool(
            winning_rank_record
            and winning_rank_record["learning_rate_boundary"][
                "requires_lr_grid_extension_before_optimum_claim"
            ]
        )
        tuned_decision["interpretation"] = (
            "joint local-LR-per-rank envelope; not a pure rank causal comparison"
        )
        comparisons["tuned_envelope"].append(tuned_decision)

        center_id = _required_dict(
            scope.get("fixed_center_control"),
            f"{mode} fixed-center provenance",
        ).get("candidate_id")
        source_rank = _required_dict(
            scope.get("source_winner"), f"{mode} source winner"
        ).get("rank")
        _expect(
            center_id,
            f"{mode}-r{source_rank}-lr-center",
            f"{mode} source fixed-center candidate",
        )
        fixed_points = [
            (row_map[f"{mode}-r{rank}-lr-center"], "center")
            for rank in rank_builder.RANKS
        ]
        fixed_decision = _cross_rank_decision(
            mode=mode,
            view="fixed_center_control",
            points=fixed_points,
            sample_indices=sample_indices,
        )
        fixed_decision["per_rank_points"] = [
            {
                "rank": rank,
                "safe_for_selection": _is_safe(row, sample_indices),
                "point": _compact_point(row, sample_indices, lr_slice="center"),
            }
            for (row, _slice), rank in zip(fixed_points, rank_builder.RANKS)
        ]
        fixed_decision["stage1_winner_control_candidate_id"] = center_id
        fixed_decision["learning_rate_tuned"] = False
        fixed_decision["interpretation"] = (
            "rank-only control at the exact Stage-1 winner optimizer/LR/WD"
        )
        comparisons["fixed_center_control"].append(fixed_decision)
        selected_rows["fixed_center_control"].extend(
            row for row, _slice in fixed_points if _is_safe(row, sample_indices)
        )
    return comparisons, selected_rows


def _pareto_payload(
    *,
    candidate_rows: list[dict[str, Any]],
    selected_rows: dict[str, list[dict[str, Any]]],
    sample_indices: Sequence[int],
) -> dict[str, Any]:
    def axis_rows(
        source: list[dict[str, Any]], *, reserved: bool
    ) -> list[dict[str, Any]]:
        if not reserved:
            return calibration_analysis._pareto(source)
        transformed = copy.deepcopy(source)
        for row in transformed:
            aggregate = row["aggregate"]
            aggregate["max_whole_process_peak_hbm_bytes"] = aggregate[
                "max_whole_process_peak_hbm_reserved_bytes"
            ]
        output = calibration_analysis._pareto(transformed)
        for row in output:
            row["max_whole_process_peak_hbm_reserved_bytes"] = row.pop(
                "max_whole_process_peak_hbm_bytes"
            )
        return output

    def frontiers(source: list[dict[str, Any]]) -> dict[str, Any]:
        allocated = axis_rows(source, reserved=False)
        reserved = axis_rows(source, reserved=True)
        return {
            "allocated": {
                "rows": allocated,
                "rows_sha256": calibration.frozen._sha256_json(allocated),
            },
            "reserved": {
                "rows": reserved,
                "rows_sha256": calibration.frozen._sha256_json(reserved),
            },
        }

    safe_raw = [
        row
        for row in candidate_rows
        if row["mode"] != "static" and _is_safe(row, sample_indices)
    ]
    active_modes = [
        mode
        for mode in rank_builder.MODES
        if any(row["mode"] == mode for row in candidate_rows)
    ]
    raw_by_mode = {
        mode: [row for row in safe_raw if row["mode"] == mode]
        for mode in active_modes
    }
    return {
        "axes": {
            "maximize": "mean_paired_delta_A",
            "minimize": [
                (
                    "max_whole_process_peak_hbm_bytes or "
                    "max_whole_process_peak_hbm_reserved_bytes"
                ),
                "trainable_parameter_count",
            ],
        },
        "safety_scope": "prompt_paired_delta_A_nonnegative_only",
        "raw_safe_candidates": {
            "overall_cross_mode": frontiers(safe_raw),
            "by_mode": {
                mode: frontiers(rows) for mode, rows in raw_by_mode.items()
            },
        },
        "views": {
            view: {
                "overall_cross_mode": frontiers(rows),
                "by_mode": {
                    mode: frontiers(
                        [row for row in rows if row["mode"] == mode]
                    )
                    for mode in active_modes
                },
            }
            for view, rows in selected_rows.items()
        },
    }


def build_analysis(*, candidate_spec: Path, output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    specification_path = candidate_spec.expanduser().resolve()
    if specification_path.parent != root:
        raise ValueError(
            "Stage-2 candidate spec must be inside its output-root bundle: "
            f"expected parent {root}, observed {specification_path.parent}"
        )
    expected_spec, provenance, sweep = _verify_stage2_bundle(candidate_spec)
    base = calibration_analysis.build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    rows = _portable_candidate_rows(
        _required_list(base.get("candidate_rows"), "Stage-2 candidate rows")
    )
    _enrich_candidate_measurements(
        rows=rows,
        sweep=sweep,
        output_root=output_root,
        base_analysis=base,
        provenance=provenance,
    )
    row_map = _validate_candidate_contract(
        expected_spec=expected_spec,
        sweep=sweep,
        rows=rows,
        provenance=provenance,
    )
    sample_indices = [int(sample["sample_index"]) for sample in sweep.samples]
    scopes = _scope_map(provenance)
    mode_omissions = provenance["derivation"]["omissions"]
    comparisons, selected_rows = _comparison_views(
        row_map=row_map,
        scopes=scopes,
        sample_indices=sample_indices,
    )
    pareto = _pareto_payload(
        candidate_rows=rows,
        selected_rows=selected_rows,
        sample_indices=sample_indices,
    )

    candidate_identity = _required_dict(
        provenance.get("stage2_candidate_specification"),
        "Stage-2 candidate specification provenance",
    )
    candidate_specification_identity = {
        key: candidate_identity[key]
        for key in (
            "file_sha256",
            "content_sha256",
            "study_id",
            "schema_version",
            "kind",
            "candidate_count",
            "sample_count",
            "planned_run_count",
        )
    }
    portable_evidence_core = {
        "stage2_candidate_specification": candidate_specification_identity,
        "normalized_candidate_rows_sha256": calibration.frozen._sha256_json(rows),
        "source_artifact_set_sha256": base["source_artifact_set_sha256"],
        "source_run_count": base["source_run_count"],
        "source_artifact_count": base["source_artifact_count"],
        "artifact_identity_lock": base["artifact_identity_lock"],
        # source_runs contain only logical sample/candidate keys and content
        # hashes.  They deliberately contain no run_root or filesystem path.
        "source_runs": base["source_runs"],
        "parameterization_group_sha256": base[
            "parameterization_group_sha256"
        ],
    }
    locator_bound_provenance = {
        "stage2_provenance_file_sha256": calibration.frozen._sha256_file(
            rank_builder.provenance_path(candidate_spec.expanduser().resolve())
        ),
        "stage2_provenance_sha256": provenance["provenance_sha256"],
        "stage2_derivation_sha256": provenance["derivation_sha256"],
        "stage1_binding": provenance["derivation"]["stage1_binding"],
        "builder_sha256": provenance["builder"]["sha256"],
        # The current Stage-1 analysis hash includes its spec/output locators.
        # Preserve it as strict provenance, but never label it portable.
        "base_analysis_sha256": base["analysis_sha256"],
        "base_candidate_rows_sha256": base["candidate_rows_sha256"],
        "common_control_identity_sha256": base[
            "common_control_identity_sha256"
        ],
    }
    source_attestation = {
        "portable_evidence_core": portable_evidence_core,
        "portable_evidence_core_sha256": calibration.frozen._sha256_json(
            portable_evidence_core
        ),
        "locator_bound_provenance": locator_bound_provenance,
    }
    implementation = {
        "stage2_analyzer": {
            "file": Path(__file__).name,
            "sha256": calibration.frozen._sha256_file(Path(__file__)),
        },
        "stage1_analyzer": base["analysis_implementation"]["analyzer"],
        "metric_aggregator": base["analysis_implementation"][
            "metric_aggregator"
        ],
        "calibration_orchestrator": base["analysis_implementation"][
            "calibration_orchestrator"
        ],
        "rank_candidate_builder": {
            "file": Path(rank_builder.__file__).name,
            "sha256": calibration.frozen._sha256_file(Path(rank_builder.__file__)),
        },
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete",
        "study_id": sweep.study_id,
        "evidence_scope": sweep.evidence_scope,
        "sample_indices": sample_indices,
        "selection_rule": SELECTION_RULE,
        "selection_rule_sha256": calibration.frozen._sha256_json(SELECTION_RULE),
        "analysis_implementation": implementation,
        "analysis_implementation_sha256": calibration.frozen._sha256_json(
            implementation
        ),
        "source_attestation": source_attestation,
        "source_attestation_sha256": calibration.frozen._sha256_json(
            source_attestation
        ),
        "candidate_rows": rows,
        "candidate_rows_sha256": calibration.frozen._sha256_json(rows),
        "comparisons": comparisons,
        "comparisons_sha256": calibration.frozen._sha256_json(comparisons),
        "mode_omissions": mode_omissions,
        "mode_omissions_sha256": calibration.frozen._sha256_json(
            mode_omissions
        ),
        "pareto": pareto,
        "pareto_sha256": calibration.frozen._sha256_json(pareto),
        "ablation_measurement_contract": {
            "acceptance": (
                "candidate_rows[*].sample_results[*].accepted_drafts_per_verify_A "
                "and paired_delta_A"
            ),
            "memory": (
                "candidate_rows[*].sample_results[*].memory records whole-process "
                "allocated/reserved HBM, paired Static deltas, phase snapshots, "
                "and the exact optimizer tensor ledger"
            ),
            "trainable_parameters": (
                "candidate_rows[*].aggregate.trainable_parameter_count; must "
                "be constant across the two prompts"
            ),
            "loss_by_context_summary": (
                "candidate_rows[*].sample_results[*].update records prefix range, "
                "loss first/final/mean/min/max, and gradient summaries"
            ),
            "loss_by_context_raw": (
                "candidate_rows[*].sample_results[*].loss_context stores the "
                "portable relative rounds locator/hash and the derived per-update "
                "prefix_len_before, loss, grad_norm, and optimizer_step points"
            ),
        },
        "limitations": [
            "Tuned-envelope is a joint rank-and-local-LR comparison, not a pure rank effect.",
            "Fixed-center-control isolates rank locally but may under-tune non-center ranks.",
            "Two locked development prompts do not establish statistical significance.",
            "A boundary winner requires the recorded rank-grid extension before any optimum claim.",
            "Locator-bound upstream provenance is separated from the portable evidence core.",
        ],
        "analysis_hash_scheme": "canonical_json_without_analysis_sha256_v1",
    }
    payload["analysis_sha256"] = calibration.frozen._sha256_json(payload)
    return payload


def verify_published_analysis(
    *,
    candidate_spec: Path,
    output_root: Path,
    analysis_path: Path,
) -> tuple[dict[str, Any], str]:
    path = analysis_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Stage-2 rank analysis is not a file: {path}")
    payload = build_analysis(candidate_spec=candidate_spec, output_root=output_root)
    if path.read_bytes() != _render(payload).encode("utf-8"):
        raise ValueError(f"Stage-2 rank analysis is stale or tampered: {path}")
    return payload, calibration.frozen._sha256_file(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.output_root.expanduser().resolve()
    published_path = args.check if args.check is not None else args.output
    if (
        published_path is not None
        and published_path.expanduser().resolve().parent != root
    ):
        raise ValueError(
            "Stage-2 analysis output/check must be inside its output-root bundle"
        )
    if args.check is not None:
        _payload, file_sha256 = verify_published_analysis(
            candidate_spec=args.candidate_spec,
            output_root=args.output_root,
            analysis_path=args.check,
        )
        print(file_sha256)
        return 0
    if args.output is not None and args.output.expanduser().resolve().exists():
        raise FileExistsError(
            f"refusing to overwrite immutable output: "
            f"{args.output.expanduser().resolve()}"
        )
    payload = build_analysis(
        candidate_spec=args.candidate_spec,
        output_root=args.output_root,
    )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        calibration.frozen._write_json_exclusive(output, payload)
        print(calibration.frozen._sha256_file(output))
        return 0
    print(_render(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
