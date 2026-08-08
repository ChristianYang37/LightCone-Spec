#!/usr/bin/env python3
"""Derive an explicit Stage-2 LoRA rank calibration from Stage-1 evidence.

The builder consumes the immutable schema-v3 Stage-1 analysis, verifies its
internal digest, and then re-runs the bound analyzer against the recorded
candidate specification and output root.  Only safe local-grid winners for
both DFlash LoRA scopes are accepted.

The calibration candidate schema intentionally has no provenance extension
point.  Consequently this command emits two no-clobber files:

* the ordinary calibration-schema-v3 candidate list; and
* ``<stem>.provenance.json``, which binds the Stage-1 analysis/spec/artifacts,
  the deterministic derivation, and the Stage-2 candidate-list hashes.

Use ``--check`` to re-run the complete derivation and verify both files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import analyze_dflash_tts_calibration as stage1_analyzer  # noqa: E402
import run_dflash_tts_calibration_sweep as calibration  # noqa: E402


PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_KIND = "dflash_tts_lora_rank_stage2_provenance"
DERIVATION_KIND = "dflash_tts_lora_rank_stage2_derivation"
RANKS = (4, 8, 16, 32, 64)
MODES = ("drafter-lora", "tail-lora")
LR_SLICES = (
    ("div3", 1, 3),
    ("center", 1, 1),
    ("times3", 3, 1),
)


@dataclass(frozen=True)
class VerifiedRankBundle:
    """One deterministically rebuilt and byte-verified Stage-2 bundle."""

    path: Path
    provenance_path: Path
    candidate_spec: dict[str, Any]
    provenance: dict[str, Any]
    sweep: calibration.CandidateSweep
    file_sha256: str
    provenance_file_sha256: str


def _expect(observed: Any, expected: Any, label: str) -> None:
    if calibration.frozen._canonical_json(observed) != calibration.frozen._canonical_json(
        expected
    ):
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, observed {observed!r}"
        )


def _dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _render(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def provenance_path(candidate_spec_path: Path) -> Path:
    return candidate_spec_path.with_name(
        f"{candidate_spec_path.stem}.provenance{candidate_spec_path.suffix}"
    )


def _verify_stage1_analysis(path_value: Path) -> tuple[
    Path,
    dict[str, Any],
    calibration.CandidateSweep,
]:
    try:
        verified = calibration.verify_stage1_published_analysis(path_value)
    except ValueError as exc:
        raise ValueError(
            "Stage-1 analysis is stale or tampered relative to its bound "
            f"candidate specification and output root: {exc}"
        ) from exc
    return verified.path, verified.payload, verified.sweep


def _winner_adapter_seed(
    analysis: dict[str, Any],
    sweep: calibration.CandidateSweep,
    *,
    analysis_path: Path,
    candidate_id: str,
) -> int:
    """Recover the attested Stage-1 seed without expanding its public schema."""

    output_root = _resolved_stage1_output_root(analysis, analysis_path)
    records = {
        (record.get("sample_index"), record.get("candidate_id")): record
        for record in _list(analysis.get("source_runs"), "Stage-1 source_runs")
        if isinstance(record, dict)
    }
    seeds: set[int] = set()
    for sample in sweep.samples:
        sample_index = int(sample["sample_index"])
        record = records.get((sample_index, candidate_id))
        if not isinstance(record, dict):
            raise ValueError(
                f"Stage-1 {candidate_id} lacks source evidence for sample {sample_index}"
            )
        identity_path = (
            output_root
            / f"sample-{sample_index:04d}"
            / candidate_id
            / "run_identity.json"
        )
        if calibration.frozen._sha256_file(identity_path) != record.get(
            "run_identity_sha256"
        ):
            raise ValueError(f"Stage-1 {candidate_id} run identity changed")
        stored = calibration.frozen._read_json(identity_path)
        identity = _dict(stored.get("identity"), f"Stage-1 {candidate_id} identity")
        _expect(
            stored.get("identity_sha256"),
            record.get("identity_sha256"),
            f"Stage-1 {candidate_id} identity_sha256",
        )
        _expect(
            calibration.frozen._sha256_json(identity),
            record.get("identity_sha256"),
            f"Stage-1 {candidate_id} identity content",
        )
        optimization = _dict(
            identity.get("optimization"), f"Stage-1 {candidate_id} optimization"
        )
        seed = optimization.get("adapter_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Stage-1 {candidate_id} adapter_seed is invalid")
        seeds.add(seed)
    if len(seeds) != 1:
        raise ValueError(f"Stage-1 {candidate_id} adapter_seed differs by prompt")
    return next(iter(seeds))


def _resolved_stage1_output_root(
    analysis: dict[str, Any], analysis_path: Path
) -> Path:
    """Resolve the portable bundle root, with legacy payload compatibility."""

    value = analysis.get("output_root")
    if value is None:
        return analysis_path.parent.resolve()
    if not isinstance(value, str) or not value:
        raise ValueError("Stage-1 output_root is invalid")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = analysis_path.parent / path
    return path.resolve()


def _safe_winner(
    analysis: dict[str, Any],
    sweep: calibration.CandidateSweep,
    *,
    analysis_path: Path,
    mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    decisions = [
        decision
        for decision in _list(
            analysis.get("selection_decisions"), "Stage-1 selection_decisions"
        )
        if isinstance(decision, dict) and decision.get("mode") == mode
    ]
    if not decisions:
        return None, {
            "mode": mode,
            "status": "omitted",
            "reason": "stage1_mode_absent",
            "stage1_selection_decisions_sha256": analysis[
                "selection_decisions_sha256"
            ],
        }
    if len(decisions) != 1:
        raise ValueError(
            f"Stage-1 requires exactly one {mode} selection decision, got "
            f"{len(decisions)}"
        )
    decision = decisions[0]
    if decision.get("status") == "no_safe_selection":
        if decision.get("winner") is not None or decision.get("safe_candidate_count") != 0:
            raise ValueError(f"Stage-1 {mode} no-safe decision is malformed")
        return None, {
            "mode": mode,
            "status": "omitted",
            "reason": "stage1_no_safe_selection",
            "candidate_count": decision.get("candidate_count"),
            "evidence_eligible_count": decision.get("evidence_eligible_count"),
            "safe_candidate_count": 0,
            "stage1_selection_decision_sha256": calibration.frozen._sha256_json(
                decision
            ),
            "stage1_selection_decisions_sha256": analysis[
                "selection_decisions_sha256"
            ],
        }
    if decision.get("status") != "local_grid_winner":
        raise ValueError(f"Stage-1 {mode} selection status is unsupported")
    winner = _dict(decision.get("winner"), f"Stage-1 {mode} winner")
    aggregate = _dict(
        winner.get("aggregate"), f"Stage-1 {mode} winner aggregate"
    )
    if aggregate.get("safe_for_selection") is not True:
        raise ValueError(f"Stage-1 {mode} winner is not safe_for_selection")
    if aggregate.get("evidence_eligible") is not True:
        raise ValueError(f"Stage-1 {mode} winner is not evidence eligible")
    if aggregate.get("all_outputs_exact_static") is not True:
        raise ValueError(f"Stage-1 {mode} winner is not exact to Static")
    if aggregate.get("all_losses_and_gradients_finite") is not True:
        raise ValueError(f"Stage-1 {mode} winner has non-finite update evidence")
    if aggregate.get("ineligibility_reasons") != []:
        raise ValueError(f"Stage-1 {mode} winner has ineligibility reasons")
    ordered_safe = _list(
        decision.get("ordered_safe_candidate_ids"),
        f"Stage-1 {mode} ordered safe candidates",
    )
    if not ordered_safe or ordered_safe[0] != winner.get("candidate_id"):
        raise ValueError(f"Stage-1 {mode} winner is not first in the safe ordering")

    candidate_id = winner.get("candidate_id")
    candidates = [
        candidate
        for candidate in sweep.candidates
        if candidate.candidate_id == candidate_id
    ]
    if len(candidates) != 1:
        raise ValueError(f"Stage-1 {mode} winner candidate is missing from its spec")
    candidate = candidates[0]
    if candidate.mode != mode or not candidate.selection_eligible:
        raise ValueError(f"Stage-1 {mode} winner is not an eligible {mode} candidate")
    if candidate.draft_cache_policy != "stale" or candidate.parameter_audit_stride != 0:
        raise ValueError(f"Stage-1 {mode} winner violates the selection hot-path contract")

    for key, expected in (
        ("optimizer", candidate.config.optimizer),
        ("learning_rate", candidate.config.learning_rate),
        ("weight_decay", candidate.config.weight_decay),
        ("rank", candidate.config.rank),
    ):
        _expect(winner.get(key), expected, f"Stage-1 {mode} winner {key}")
    optimizer = winner["optimizer"]
    learning_rate = winner["learning_rate"]
    weight_decay = winner["weight_decay"]
    rank = winner["rank"]
    if optimizer not in {"adam", "adamw"}:
        raise ValueError(f"Stage-1 {mode} winner optimizer is invalid")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise ValueError(f"Stage-1 {mode} winner learning rate is invalid")
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or not math.isfinite(float(weight_decay))
        or float(weight_decay) < 0.0
    ):
        raise ValueError(f"Stage-1 {mode} winner weight decay is invalid")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in RANKS:
        raise ValueError(
            f"Stage-1 {mode} winner rank must be one of {list(RANKS)} to "
            "provide a fixed-center control"
        )

    boundary = _dict(
        winner.get("learning_rate_boundary"),
        f"Stage-1 {mode} learning-rate boundary",
    )
    for key in (
        "at_group_boundary",
        "at_optimizer_weight_decay_boundary",
        "requires_grid_extension_before_optimum_claim",
    ):
        if not isinstance(boundary.get(key), bool):
            raise ValueError(f"Stage-1 {mode} boundary {key} must be boolean")
    bounds = _dict(
        boundary.get("optimizer_weight_decay_bounds"),
        f"Stage-1 {mode} optimizer/WD bounds",
    )
    for key in ("minimum", "maximum"):
        if not isinstance(bounds.get(key), (int, float)) or isinstance(
            bounds.get(key), bool
        ):
            raise ValueError(f"Stage-1 {mode} boundary {key} is invalid")

    row_matches = [
        row
        for row in _list(analysis.get("candidate_rows"), "Stage-1 candidate_rows")
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(row_matches) != 1:
        raise ValueError(f"Stage-1 {mode} winner row is missing or duplicated")
    row = row_matches[0]
    _expect(row.get("mode"), mode, f"Stage-1 {mode} winner-row mode")
    for key, expected in (
        ("rank", candidate.config.rank),
        ("optimizer", candidate.config.optimizer),
        ("learning_rate", candidate.config.learning_rate),
        ("weight_decay", candidate.config.weight_decay),
        ("draft_cache_policy", "stale"),
        ("diagnostic_kind", "selection"),
        ("candidate_spec_selection_eligible", True),
    ):
        _expect(row.get(key), expected, f"Stage-1 {mode} winner-row {key}")
    _expect(row.get("aggregate"), aggregate, f"Stage-1 {mode} winner-row aggregate")
    samples = _list(row.get("sample_results"), f"Stage-1 {mode} sample results")
    if len(samples) != len(sweep.samples):
        raise ValueError(f"Stage-1 {mode} winner lacks the complete sample pair")
    if any(
        not isinstance(sample, dict)
        or not isinstance(sample.get("paired_delta_A"), (int, float))
        or isinstance(sample.get("paired_delta_A"), bool)
        or not math.isfinite(float(sample["paired_delta_A"]))
        or float(sample["paired_delta_A"]) < 0.0
        for sample in samples
    ):
        raise ValueError(f"Stage-1 {mode} winner violates the paired safety gate")
    adapter_seed = _winner_adapter_seed(
        analysis,
        sweep,
        analysis_path=analysis_path,
        candidate_id=str(candidate_id),
    )
    _expect(
        aggregate.get("adapter_seed"),
        adapter_seed,
        f"Stage-1 {mode} winner aggregate adapter_seed",
    )
    return {
        "mode": mode,
        "candidate_id": str(candidate_id),
        "optimizer": str(optimizer),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "rank": int(rank),
        "adapter_seed": adapter_seed,
        "aggregate_sha256": calibration.frozen._sha256_json(aggregate),
        "learning_rate_boundary": boundary,
    }, None


def _learning_rates(center: float) -> dict[str, float]:
    decimal_center = Decimal(str(center))
    values = {
        name: float(decimal_center * Decimal(numerator) / Decimal(denominator))
        for name, numerator, denominator in LR_SLICES
    }
    if not (
        math.isfinite(values["div3"])
        and values["div3"] < values["center"] < values["times3"]
    ):
        raise ValueError("Stage-2 learning-rate window is not finite and ordered")
    if values["center"] != center:
        raise ValueError("Stage-2 center learning rate did not preserve the winner")
    return values


def _static_candidate(sweep: calibration.CandidateSweep) -> dict[str, Any]:
    values = [
        candidate
        for candidate in sweep.candidates
        if candidate.mode == "static" and candidate.selection_eligible
    ]
    if len(values) != 1:
        raise ValueError("Stage-1 must contain exactly one selection Static")
    candidate = values[0]
    return {
        "candidate_id": "static",
        "mode": "static",
        "optimizer": candidate.config.optimizer,
        "learning_rate": candidate.config.learning_rate,
        "weight_decay": candidate.config.weight_decay,
        "rank": None,
        "draft_cache_policy": "stale",
        "diagnostic_kind": "selection",
        "parameter_audit_stride": 0,
    }


def _candidate_rows(
    sweep: calibration.CandidateSweep,
    winners: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [_static_candidate(sweep)]
    scope_metadata = []
    for winner in winners:
        mode = winner["mode"]
        rates = _learning_rates(float(winner["learning_rate"]))
        fixed_center_id = f"{mode}-r{winner['rank']}-lr-center"
        for rank in RANKS:
            for slice_name, _numerator, _denominator in LR_SLICES:
                candidates.append(
                    {
                        "candidate_id": f"{mode}-r{rank}-lr-{slice_name}",
                        "mode": mode,
                        "optimizer": winner["optimizer"],
                        "learning_rate": rates[slice_name],
                        "weight_decay": winner["weight_decay"],
                        "rank": rank,
                        "draft_cache_policy": "stale",
                        "diagnostic_kind": "selection",
                        "parameter_audit_stride": 0,
                    }
                )
        fixed = next(
            row for row in candidates if row["candidate_id"] == fixed_center_id
        )
        fixed_matches = all(
            fixed[key] == winner[key]
            for key in ("mode", "optimizer", "learning_rate", "weight_decay", "rank")
        )
        if not fixed_matches:
            raise AssertionError(f"{mode} fixed-center control drifted")
        scope_metadata.append(
            {
                "mode": mode,
                "source_winner": winner,
                "learning_rate_window": rates,
                "fixed_center_control": {
                    "candidate_id": fixed_center_id,
                    "matches_stage1_winner_config": True,
                    "adapter_seed": winner["adapter_seed"],
                },
                "boundary_metadata": {
                    "stage1_learning_rate": winner["learning_rate_boundary"],
                    "stage2_rank_grid": {
                        "minimum": min(RANKS),
                        "maximum": max(RANKS),
                        "source_winner_rank": winner["rank"],
                        "source_winner_at_rank_boundary": winner["rank"]
                        in {min(RANKS), max(RANKS)},
                        "post_stage2_rule": (
                            "extend one octave only if the selected rank is 4 or 64; "
                            "never claim a global rank optimum from this bounded grid"
                        ),
                    },
                },
            }
        )
    expected_count = 1 + 15 * len(winners)
    if len(candidates) != expected_count:
        raise AssertionError(f"Stage-2 candidate count drifted: {len(candidates)}")
    semantic = {
        (
            row["mode"],
            row["optimizer"],
            row["learning_rate"],
            row["weight_decay"],
            row["rank"],
            row["draft_cache_policy"],
            row["diagnostic_kind"],
            row["parameter_audit_stride"],
        )
        for row in candidates
    }
    if len(semantic) != len(candidates):
        raise ValueError("Stage-2 derivation produced duplicate semantic candidates")
    return candidates, scope_metadata


def build_bundle(
    *,
    stage1_analysis_path: Path,
    candidate_spec_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis_path, analysis, sweep = _verify_stage1_analysis(stage1_analysis_path)
    outcomes = [
        _safe_winner(
            analysis,
            sweep,
            analysis_path=analysis_path,
            mode=mode,
        )
        for mode in MODES
    ]
    winners = [winner for winner, _omission in outcomes if winner is not None]
    omissions = [
        omission for _winner, omission in outcomes if omission is not None
    ]
    if not winners:
        raise ValueError(
            "Stage-1 has no safe LoRA scope; refusing to build an empty "
            "Stage-2 rank study"
        )
    adapter_seeds = {winner["adapter_seed"] for winner in winners}
    if len(adapter_seeds) != 1:
        raise ValueError(
            "Stage-1 LoRA winners use different adapter seeds; one Stage-2 "
            "rank-only sweep cannot preserve both controls"
        )
    [adapter_seed] = adapter_seeds
    candidates, scope_metadata = _candidate_rows(sweep, winners)

    source_specification = _dict(
        analysis.get("candidate_specification"),
        "Stage-1 candidate_specification",
    )
    stage1_binding = {
        "analysis_file_sha256": calibration.frozen._sha256_file(analysis_path),
        "analysis_sha256": analysis["analysis_sha256"],
        "selection_decisions_sha256": analysis["selection_decisions_sha256"],
        "source_artifact_set_sha256": analysis["source_artifact_set_sha256"],
        "candidate_specification_file_sha256": source_specification[
            "file_sha256"
        ],
        "candidate_specification_content_sha256": source_specification[
            "content_sha256"
        ],
        "source_study_id": source_specification["study_id"],
        "artifact_identity_lock_file_sha256": analysis[
            "artifact_identity_lock"
        ]["file_sha256"],
        "artifact_identity_lock_content_sha256": analysis[
            "artifact_identity_lock"
        ]["content_sha256"],
    }

    derivation = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": DERIVATION_KIND,
        "stage1_binding": stage1_binding,
        "rank_values": list(RANKS),
        "learning_rate_slices": [
            {
                "name": name,
                "numerator": numerator,
                "denominator": denominator,
            }
            for name, numerator, denominator in LR_SLICES
        ],
        "candidate_order": (
            "Static, then each non-omitted scope in drafter-lora/tail-lora "
            "order; within each scope rank ascending, then LR "
            "div3/center/times3"
        ),
        "requested_modes": list(MODES),
        "active_modes": [winner["mode"] for winner in winners],
        "omissions": omissions,
        "selection_hot_path": {
            "draft_cache_policy": "stale",
            "diagnostic_kind": "selection",
            "parameter_audit_stride": 0,
            "adapter_seed": adapter_seed,
        },
        "parameterization_control": (
            "the same attested Stage-1 adapter_seed is required for every "
            "non-Static Stage-2 rank"
        ),
        "scopes": scope_metadata,
        "claim_scope": "bounded_rank_and_local_lr_sensitivity_not_global_optimum",
    }
    derivation_sha256 = calibration.frozen._sha256_json(derivation)
    study_id = f"dflash-lora-rank-stage2-{derivation_sha256[:24]}"
    candidate_spec = {
        "schema_version": calibration.SCHEMA_VERSION,
        "kind": calibration.SPEC_KIND,
        "study_id": study_id,
        "evidence_scope": calibration.EVIDENCE_SCOPE,
        "max_new_tokens": calibration.MAX_NEW_TOKENS,
        "samples": [dict(sample) for sample in sweep.samples],
        "candidates": candidates,
    }
    candidate_body = _render(candidate_spec).encode("utf-8")
    output_path = candidate_spec_path.expanduser().resolve()
    sidecar_path = provenance_path(output_path)
    stage2_bundle_root = output_path.parent
    stage1_bundle_root = _resolved_stage1_output_root(analysis, analysis_path)
    stage1_analysis_locator = Path(
        os.path.relpath(analysis_path, stage2_bundle_root)
    ).as_posix()
    stage1_bundle_root_locator = Path(
        os.path.relpath(stage1_bundle_root, stage2_bundle_root)
    ).as_posix()
    stage1_candidate_locator = Path(
        os.path.relpath(sweep.path, analysis_path.parent)
    ).as_posix()
    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": PROVENANCE_KIND,
        "status": "complete",
        "stage1": {
            "analysis": {
                "locator": stage1_analysis_locator,
                "file_sha256": stage1_binding["analysis_file_sha256"],
                "analysis_sha256": stage1_binding["analysis_sha256"],
                "selection_decisions_sha256": stage1_binding[
                    "selection_decisions_sha256"
                ],
                "source_artifact_set_sha256": stage1_binding[
                    "source_artifact_set_sha256"
                ],
                "source_run_count": analysis["source_run_count"],
                "artifact_identity_lock": analysis["artifact_identity_lock"],
            },
            "candidate_specification": {
                "locator": stage1_candidate_locator,
                **{
                    key: source_specification[key]
                    for key in (
                        "file_sha256",
                        "content_sha256",
                        "study_id",
                        "schema_version",
                        "kind",
                        "evidence_scope",
                    )
                },
            },
            "bundle_root_locator": stage1_bundle_root_locator,
            "analyzer": {
                "file": Path(stage1_analyzer.__file__).name,
                "sha256": calibration.frozen._sha256_file(
                    Path(stage1_analyzer.__file__).resolve()
                ),
            },
        },
        "derivation": derivation,
        "derivation_sha256": derivation_sha256,
        "stage2_candidate_specification": {
            "locator": output_path.name,
            "file_sha256": _sha256_bytes(candidate_body),
            "content_sha256": calibration.frozen._sha256_json(candidate_spec),
            "study_id": study_id,
            "schema_version": calibration.SCHEMA_VERSION,
            "kind": calibration.SPEC_KIND,
            "candidate_count": len(candidates),
            "sample_count": len(sweep.samples),
            "planned_run_count": len(candidates) * len(sweep.samples),
        },
        "companion_locator": sidecar_path.name,
        "builder": {
            "file": Path(__file__).name,
            "sha256": calibration.frozen._sha256_file(Path(__file__).resolve()),
        },
        "provenance_hash_scheme": "canonical_json_without_provenance_sha256_v1",
    }
    provenance["provenance_sha256"] = calibration.frozen._sha256_json(provenance)
    return candidate_spec, provenance


def verify_published_bundle(candidate_spec_path: Path) -> VerifiedRankBundle:
    """Rebuild and byte-verify a published candidate/provenance pair."""

    path = candidate_spec_path.expanduser().resolve()
    sidecar = provenance_path(path)
    if not path.is_file():
        raise ValueError(f"Stage-2 candidate specification is not a file: {path}")
    if not sidecar.is_file():
        raise ValueError(f"Stage-2 provenance is not a file: {sidecar}")
    observed = calibration.frozen._read_json(sidecar)
    _expect(
        observed.get("schema_version"),
        PROVENANCE_SCHEMA_VERSION,
        "Stage-2 provenance schema",
    )
    _expect(observed.get("kind"), PROVENANCE_KIND, "Stage-2 provenance kind")
    _expect(observed.get("status"), "complete", "Stage-2 provenance status")
    unsigned = dict(observed)
    provenance_sha256 = unsigned.pop("provenance_sha256", None)
    if not calibration.frozen._is_sha256(provenance_sha256):
        raise ValueError("Stage-2 provenance_sha256 is invalid")
    _expect(
        provenance_sha256,
        calibration.frozen._sha256_json(unsigned),
        "Stage-2 provenance_sha256",
    )
    stage1 = _dict(observed.get("stage1"), "Stage-2 provenance.stage1")
    analysis = _dict(stage1.get("analysis"), "Stage-2 provenance.stage1.analysis")
    stage1_locator = analysis.get("locator", analysis.get("path"))
    if not isinstance(stage1_locator, str) or not stage1_locator:
        raise ValueError("Stage-2 provenance lacks the Stage-1 analysis locator")
    stage1_path = Path(stage1_locator).expanduser()
    if not stage1_path.is_absolute():
        stage1_path = path.parent / stage1_path
    stage1_path = stage1_path.resolve()
    candidate_spec, provenance = build_bundle(
        stage1_analysis_path=stage1_path,
        candidate_spec_path=path,
    )
    _check_bundle(path, candidate_spec, provenance)
    sweep = calibration.load_candidate_sweep(path)
    specification = _dict(
        provenance.get("stage2_candidate_specification"),
        "Stage-2 candidate specification identity",
    )
    _expect(
        specification.get("locator"),
        path.name,
        "Stage-2 candidate specification locator",
    )
    _expect(
        provenance.get("companion_locator"),
        sidecar.name,
        "Stage-2 companion locator",
    )
    for key, observed, expected in (
        ("file_sha256", sweep.file_sha256, specification.get("file_sha256")),
        (
            "content_sha256",
            sweep.content_sha256,
            specification.get("content_sha256"),
        ),
        ("study_id", sweep.study_id, specification.get("study_id")),
        ("kind", sweep.kind, specification.get("kind")),
        ("schema_version", calibration.SCHEMA_VERSION, specification.get("schema_version")),
        ("candidate_count", len(sweep.candidates), specification.get("candidate_count")),
        ("sample_count", len(sweep.samples), specification.get("sample_count")),
        (
            "planned_run_count",
            len(sweep.candidates) * len(sweep.samples),
            specification.get("planned_run_count"),
        ),
    ):
        _expect(observed, expected, f"Stage-2 actual candidate specification {key}")
    return VerifiedRankBundle(
        path=path,
        provenance_path=sidecar,
        candidate_spec=candidate_spec,
        provenance=provenance,
        sweep=sweep,
        file_sha256=calibration.frozen._sha256_file(path),
        provenance_file_sha256=calibration.frozen._sha256_file(sidecar),
    )


def _write_bundle(
    candidate_spec_path: Path,
    candidate_spec: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    output = candidate_spec_path.expanduser().resolve()
    sidecar = provenance_path(output)
    for path in (output, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration.frozen._write_json_exclusive(output, candidate_spec)
    try:
        calibration.frozen._write_json_exclusive(sidecar, provenance)
    except Exception:
        # The candidate path was created by this call and has no valid companion.
        # Remove only that just-created inode so a retry cannot mistake a partial
        # bundle for completed evidence.
        output.unlink(missing_ok=True)
        raise


def _check_bundle(
    candidate_spec_path: Path,
    candidate_spec: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    output = candidate_spec_path.expanduser().resolve()
    sidecar = provenance_path(output)
    expected = ((output, _render(candidate_spec)), (sidecar, _render(provenance)))
    for path, body in expected:
        if not path.is_file() or path.read_bytes() != body.encode("utf-8"):
            raise ValueError(f"Stage-2 rank bundle is stale or tampered: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-analysis", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.output if args.output is not None else args.check
    candidate_spec, provenance = build_bundle(
        stage1_analysis_path=args.stage1_analysis,
        candidate_spec_path=destination,
    )
    if args.check is not None:
        _check_bundle(args.check, candidate_spec, provenance)
    else:
        _write_bundle(args.output, candidate_spec, provenance)
    output = destination.expanduser().resolve()
    print(
        json.dumps(
            {
                "candidate_specification": {
                    "path": str(output),
                    "sha256": calibration.frozen._sha256_file(output),
                },
                "provenance": {
                    "path": str(provenance_path(output)),
                    "sha256": calibration.frozen._sha256_file(
                        provenance_path(output)
                    ),
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
