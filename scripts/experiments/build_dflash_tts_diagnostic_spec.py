#!/usr/bin/env python3
"""Derive an isolated schema-v3 diagnostic spec from verified Stage-1 evidence.

The builder re-runs the Stage-1 analysis before using any winner.  It emits
the selection Static-stale control, one non-selection Static-rebuild comparator,
and explicitly non-selection adaptive cache-policy and parameter-audit
candidates.  Publication is exclusive; ``--check`` rebuilds the spec from the
still-valid Stage-1 artifacts and compares it byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import aggregate_dflash_tts_ablations as aggregation  # noqa: E402
import run_dflash_tts_calibration_sweep as calibration  # noqa: E402


SCHEMA_VERSION = 1
PROVENANCE_KIND = calibration.DIAGNOSTIC_PROVENANCE_KIND
CACHE_MODES = calibration.DIAGNOSTIC_CACHE_MODES
AUDIT_STRIDES = calibration.DIAGNOSTIC_AUDIT_STRIDES
MODE_ORDER = tuple(AUDIT_STRIDES)


def _required_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _candidate_by_id(
    sweep: calibration.CandidateSweep,
) -> dict[str, calibration.Candidate]:
    return {candidate.candidate_id: candidate for candidate in sweep.candidates}


def _source_candidate_row(
    sweep: calibration.CandidateSweep, candidate_id: str
) -> dict[str, Any]:
    payload = calibration.frozen._read_json(sweep.path)
    rows = _required_list(payload.get("candidates"), "Stage-1 candidates")
    matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"Stage-1 winner {candidate_id!r} is not unique in its spec")
    return dict(matches[0])


def _safe_winners(
    payload: dict[str, Any],
    sweep: calibration.CandidateSweep,
) -> tuple[
    dict[str, tuple[calibration.Candidate, dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    decisions = _required_list(
        payload.get("selection_decisions"), "Stage-1 selection_decisions"
    )
    recorded_hash = payload.get("selection_decisions_sha256")
    if (
        not aggregation._is_sha256(recorded_hash)
        or aggregation._sha256_json(decisions) != recorded_hash
    ):
        raise ValueError("Stage-1 selection decision hash mismatch")
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for value in decisions:
        decision = _required_dict(value, "Stage-1 selection decision")
        mode = decision.get("mode")
        if isinstance(mode, str):
            by_mode.setdefault(mode, []).append(decision)

    source_candidates = _candidate_by_id(sweep)
    candidate_rows = {
        row["candidate_id"]: row
        for row in _required_list(payload.get("candidate_rows"), "Stage-1 candidate_rows")
        if isinstance(row, dict) and isinstance(row.get("candidate_id"), str)
    }
    winners: dict[str, tuple[calibration.Candidate, dict[str, Any]]] = {}
    status: dict[str, dict[str, Any]] = {}
    for mode in MODE_ORDER:
        matches = by_mode.get(mode, [])
        if not matches:
            status[mode] = {
                "reason": "stage1_selection_decision_missing",
                "decision_status": "missing",
            }
            continue
        if len(matches) != 1:
            raise ValueError(f"Stage-1 has ambiguous selection decisions for {mode}")
        decision = matches[0]
        decision_status = decision.get("status")
        winner = decision.get("winner")
        if decision_status == "no_safe_selection" and winner is None:
            status[mode] = {
                "reason": "stage1_no_safe_selection",
                "decision_status": decision_status,
            }
            continue
        if decision_status != "local_grid_winner" or not isinstance(winner, dict):
            raise ValueError(f"Stage-1 decision for {mode} is internally inconsistent")
        aggregate = _required_dict(
            winner.get("aggregate"), f"Stage-1 winner aggregate for {mode}"
        )
        if aggregate.get("safe_for_selection") is not True:
            raise ValueError(f"Stage-1 winner for {mode} is not marked safe")
        candidate_id = winner.get("candidate_id")
        candidate = source_candidates.get(candidate_id)
        row = candidate_rows.get(candidate_id)
        if candidate is None or row is None:
            raise ValueError(f"Stage-1 winner for {mode} lacks source evidence")
        row_aggregate = _required_dict(
            row.get("aggregate"), f"Stage-1 candidate aggregate for {mode}"
        )
        if row_aggregate.get("safe_for_selection") is not True:
            raise ValueError(f"Stage-1 winner row for {mode} is not safe")
        if (
            candidate.mode != mode
            or not candidate.selection_eligible
            or candidate.draft_cache_policy != "stale"
            or candidate.config.optimizer != winner.get("optimizer")
            or candidate.config.learning_rate != winner.get("learning_rate")
            or candidate.config.weight_decay != winner.get("weight_decay")
            or candidate.config.rank != winner.get("rank")
        ):
            raise ValueError(f"Stage-1 winner configuration mismatch for {mode}")
        if mode == "output-residual" and candidate.config.rank != 16:
            raise ValueError("diagnostic output-residual winner must use rank 16")
        source_row = _source_candidate_row(sweep, candidate.candidate_id)
        winners[mode] = (candidate, source_row)
        status[mode] = {
            "reason": None,
            "decision_status": decision_status,
        }
    return winners, status


def _diagnostic_candidate(
    *,
    candidate_id: str,
    source: calibration.Candidate,
    cache_policy: str,
    diagnostic_kind: str,
    audit_stride: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "mode": source.mode,
        "optimizer": source.config.optimizer,
        "learning_rate": source.config.learning_rate,
        "weight_decay": source.config.weight_decay,
        "rank": source.config.rank,
        "draft_cache_policy": cache_policy,
        "diagnostic_kind": diagnostic_kind,
        "parameter_audit_stride": audit_stride,
    }


def build_diagnostic_spec(
    *,
    stage1_analysis: Path,
    diagnostic_output_root: Path,
) -> dict[str, Any]:
    return derive_diagnostic_spec_from_verified_stage1(
        verified=calibration.verify_stage1_published_analysis(stage1_analysis),
        diagnostic_output_root=diagnostic_output_root,
    )


def derive_diagnostic_spec_from_verified_stage1(
    *,
    verified: calibration.VerifiedStage1Analysis,
    diagnostic_output_root: Path,
) -> dict[str, Any]:
    """Pure winner-to-diagnostic derivation after Stage-1 attestation."""

    analysis_path = verified.path
    source = verified.payload
    sweep = verified.sweep
    source_root_value = source.get("output_root")
    if source_root_value is None:
        source_output_root = analysis_path.parent
    else:
        source_output_root = Path(str(source_root_value)).expanduser()
        if not source_output_root.is_absolute():
            source_output_root = analysis_path.parent / source_output_root
        source_output_root = source_output_root.resolve()
    diagnostic_root = diagnostic_output_root.expanduser().resolve()
    if diagnostic_root == source_output_root:
        raise ValueError("diagnostic output root must differ from Stage-1 output root")

    winners, winner_status = _safe_winners(source, sweep)
    static = [candidate for candidate in sweep.candidates if candidate.mode == "static"]
    if len(static) != 1 or not static[0].selection_eligible:
        raise ValueError("Stage-1 must contain one selection-eligible Static")
    candidates = [
        _diagnostic_candidate(
            candidate_id="static",
            source=static[0],
            cache_policy="stale",
            diagnostic_kind="selection",
            audit_stride=0,
        ),
        # Rebuild changes the numerical execution path even for a frozen
        # drafter.  Give every rebuild diagnostic a policy-matched Static
        # comparator instead of attributing that path shift to adaptation.
        _diagnostic_candidate(
            candidate_id="static-rebuild",
            source=static[0],
            cache_policy="rebuild",
            diagnostic_kind="cache-policy-diagnostic",
            audit_stride=0,
        ),
    ]
    omissions: list[dict[str, Any]] = []

    for mode in CACHE_MODES:
        if mode not in winners:
            omissions.append(
                {
                    "diagnostic_kind": "cache-policy-diagnostic",
                    "mode": mode,
                    **winner_status[mode],
                }
            )
            continue
        source_candidate = winners[mode][0]
        for policy in ("stale", "rebuild"):
            candidates.append(
                _diagnostic_candidate(
                    candidate_id=f"cache-{mode}-{policy}",
                    source=source_candidate,
                    cache_policy=policy,
                    diagnostic_kind="cache-policy-diagnostic",
                    audit_stride=0,
                )
            )

    for mode, stride in AUDIT_STRIDES.items():
        if mode not in winners:
            omissions.append(
                {
                    "diagnostic_kind": "parameter-audit",
                    "mode": mode,
                    **winner_status[mode],
                }
            )
            continue
        candidates.append(
            _diagnostic_candidate(
                candidate_id=f"audit-{mode}-stride{stride}",
                source=winners[mode][0],
                cache_policy="stale",
                diagnostic_kind="parameter-audit",
                audit_stride=stride,
            )
        )

    diagnostic_study_id = f"{sweep.study_id}-diagnostics-v1"
    selected_winners = [
        {
            "mode": mode,
            "candidate_id": candidate.candidate_id,
            "rank": candidate.config.rank,
            "decision_status": winner_status[mode]["decision_status"],
            "source_candidate_sha256": aggregation._sha256_json(source_row),
        }
        for mode in MODE_ORDER
        if mode in winners
        for candidate, source_row in (winners[mode],)
    ]
    residual_ids = sorted(
        candidate["candidate_id"]
        for candidate in candidates
        if candidate["mode"] == "output-residual"
    )
    builder_path = Path(__file__).resolve()
    source_spec = _required_dict(
        source["candidate_specification"], "Stage-1 candidate specification"
    )
    payload = {
        "schema_version": calibration.SCHEMA_VERSION,
        "kind": calibration.DIAGNOSTIC_SPEC_KIND,
        "study_id": diagnostic_study_id,
        "evidence_scope": calibration.DIAGNOSTIC_EVIDENCE_SCOPE,
        "max_new_tokens": calibration.MAX_NEW_TOKENS,
        "samples": [dict(sample) for sample in sweep.samples],
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "kind": PROVENANCE_KIND,
            "builder": {
                "path": str(builder_path),
                "sha256": aggregation._sha256_file(builder_path),
            },
            "source_stage1_analysis": {
                "path": str(analysis_path),
                "file_sha256": aggregation._sha256_file(analysis_path),
                "analysis_sha256": source["analysis_sha256"],
                "kind": source["kind"],
                "study_id": source["study_id"],
                "candidate_specification_file_sha256": source_spec[
                    "file_sha256"
                ],
                "candidate_specification_content_sha256": source_spec[
                    "content_sha256"
                ],
                "output_root": str(source_output_root),
                "source_artifact_set_sha256": source[
                    "source_artifact_set_sha256"
                ],
                "selection_decisions_sha256": source[
                    "selection_decisions_sha256"
                ],
            },
            "selection_isolation": {
                "source_study_id": source["study_id"],
                "diagnostic_study_id": diagnostic_study_id,
                "source_output_root": str(source_output_root),
                "diagnostic_output_root": str(diagnostic_root),
                "required_distinct_output_roots": True,
                "adaptive_candidates_selection_eligible": False,
                "static_control_selection_eligible": True,
            },
            "winner_derivation": {
                "required_decision_status": "local_grid_winner",
                "required_safe_for_selection": True,
                "selected_winners": selected_winners,
                "omitted_diagnostics": omissions,
            },
            "residual_projection_requirement": {
                "required": bool(residual_ids),
                "rank": 16 if residual_ids else None,
                "candidate_ids": residual_ids,
                "runner_binding": (
                    "single_identity_bound_projection_artifact_at_run_plan_v1"
                ),
            },
        },
        "candidates": candidates,
    }
    # Reuse the execution-time parser as the final schema and pair-completeness
    # validator before anything is published.
    return payload


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _validate_rendered(payload: dict[str, Any], temporary_path: Path) -> None:
    temporary_path.write_text(_render(payload), encoding="utf-8")
    try:
        calibration.load_candidate_sweep(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validated_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_diagnostic_spec(
        stage1_analysis=args.stage1_analysis,
        diagnostic_output_root=args.diagnostic_output_root,
    )
    # Validation needs a file because the shared loader intentionally binds the
    # exact bytes.  Keep it adjacent to neither source nor destination.
    with tempfile.TemporaryDirectory(prefix="dflash-diagnostic-spec-") as directory:
        _validate_rendered(payload, Path(directory) / "candidate-spec.json")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-analysis", required=True, type=Path)
    parser.add_argument("--diagnostic-output-root", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _validated_payload(args)
    body = _render(payload)
    if args.check is not None:
        if not args.check.is_file() or args.check.read_text(encoding="utf-8") != body:
            raise ValueError(f"diagnostic candidate specification is stale: {args.check}")
        print(aggregation._sha256_file(args.check))
        return 0
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        calibration.frozen._write_json_exclusive(output, payload)
        print(aggregation._sha256_file(output))
        return 0
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
