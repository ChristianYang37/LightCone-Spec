#!/usr/bin/env python3
"""Analyze a Static/candidate pair or gate a common-prefix trajectory.

The script consumes the auditable ``summary.json``/``rounds.jsonl`` artifacts
written by :mod:`dflash_tts_reference`.  It validates paired-run identity and
reports both the paper-style acceptance length (committed tokens per verify)
and the stricter accepted-draft prefix metric.  Buckets use the real
``prefix_length_before`` value, including the prompt.

``--common-prefix`` instead compares two same-mode, fully verified schema-v3
runs.  Different ``max_new_tokens`` values select a common-prefix comparison;
equal values select a fresh-process exact-repeat comparison over every round
and the complete output token sequence.  Both exclude timing/HBM observations,
emit the first semantic divergence, and return a non-zero exit status when the
trajectory gate fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

# The analysis scripts are intentionally runnable without installing a package.
# Reuse the strict artifact loader instead of maintaining a second schema and
# final-round implementation here.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from aggregate_dflash_tts_ablations import (  # noqa: E402
    FULLY_VERIFIED_IDENTITY,
    _canonical_json,
    _effective_output_by_round,
    _load_run as _load_validated_run,
    _round_prefix,
    _sha256_json,
)


_TRAJECTORY_TOP_LEVEL_EXCLUSIONS = frozenset(
    {"timing_seconds", "hbm_bytes"}
)
_TRAJECTORY_UPDATE_EXCLUSIONS = frozenset(
    {"backward_cuda_us", "optimizer_cuda_us", "update_cuda_us"}
)
_LENGTH_DERIVED_PARAMETER_FIELDS = frozenset(
    {"max_new_tokens", "required_prefix_plus_block"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    index = round(probability * (len(ordered) - 1))
    return ordered[index]


def _run_metrics(
    summary: dict[str, Any],
    rounds: list[dict[str, Any]],
    *,
    bucket_size: int,
) -> dict[str, Any]:
    generation = summary["generation"]
    output_tokens = int(generation["num_output_tokens"])
    effective_output = _effective_output_by_round(summary, rounds)
    acceptance = [int(row["acceptance_length"]) for row in rounds]
    accepted_drafts = [int(row["accepted_draft_tokens"]) for row in rounds]
    timing_names = ("draft_forward", "target_verify", "update", "round_total")
    timing = {
        name: [float(row["timing_seconds"][name]) * 1e3 for row in rounds]
        for name in timing_names
    }
    bucket_starts = sorted(
        {
            (_round_prefix(row) // bucket_size) * bucket_size
            for row in rounds
        }
    )
    buckets: list[dict[str, Any]] = []
    for start in bucket_starts:
        selected = [
            row
            for row in rounds
            if start <= _round_prefix(row) < start + bucket_size
        ]
        committed = sum(
            effective_output[int(row["round_index"])] for row in selected
        )
        buckets.append(
            {
                "prefix_start": start,
                "prefix_end_exclusive": start + bucket_size,
                "rounds": len(selected),
                "paper_acceptance_length": statistics.mean(
                    int(row["acceptance_length"]) for row in selected
                ),
                "accepted_drafts_per_verify": statistics.mean(
                    int(row["accepted_draft_tokens"]) for row in selected
                ),
                "committed_tokens_per_verify": committed / len(selected),
                "target_calls_per_committed_token": len(selected) / committed,
                "comparison_status": (
                    "descriptive_round_composition_not_prefix_matched"
                ),
            }
        )
    decode_seconds = float(generation["decode_seconds"])
    return {
        "mode": summary["mode"],
        "trainable_scope": summary["trainable_scope"],
        "num_input_tokens": int(generation["num_input_tokens"]),
        "num_output_tokens": output_tokens,
        "rounds": len(rounds),
        "paper_acceptance_length": statistics.mean(acceptance),
        "accepted_drafts_per_verify": statistics.mean(accepted_drafts),
        "committed_tokens_per_verify": output_tokens / len(rounds),
        "algorithmic_committed_tokens_per_verify": sum(acceptance) / len(rounds),
        "target_calls_per_output_token": len(rounds) / output_tokens,
        "reference_decode_seconds": decode_seconds,
        "reference_tokens_per_second": output_tokens / decode_seconds,
        "timing_ms": {
            name: {
                "mean": statistics.mean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for name, values in timing.items()
        },
        "buckets": buckets,
    }


def compare_runs(
    baseline_dir: Path, candidate_dir: Path, *, bucket_size: int
) -> dict[str, Any]:
    baseline_run = _load_validated_run(baseline_dir)
    candidate_run = _load_validated_run(candidate_dir)
    baseline_summary, baseline_rounds = (
        baseline_run["summary"],
        baseline_run["rounds"],
    )
    candidate_summary, candidate_rounds = (
        candidate_run["summary"],
        candidate_run["rounds"],
    )
    if baseline_summary["mode"] != "static":
        raise ValueError("baseline run must have mode=static")
    baseline_identity = baseline_run["pair_identity"]
    candidate_identity = candidate_run["pair_identity"]
    for label, identity in (
        ("baseline", baseline_identity),
        ("candidate", candidate_identity),
    ):
        if (
            identity["identity_verification_status"]
            == FULLY_VERIFIED_IDENTITY
            and identity["numeric_runtime_status"]
            != "complete_numeric_runtime_v1"
        ):
            raise ValueError(
                f"{label} verified artifact has incomplete numerical runtime "
                f"identity: {identity['numeric_runtime_status']}"
            )
    if baseline_identity != candidate_identity:
        raise ValueError(
            "paired identity mismatch:\n"
            + json.dumps(
                {"baseline": baseline_identity, "candidate": candidate_identity},
                indent=2,
                sort_keys=True,
            )
        )

    baseline = _run_metrics(
        baseline_summary, baseline_rounds, bucket_size=bucket_size
    )
    candidate = _run_metrics(
        candidate_summary, candidate_rounds, bucket_size=bucket_size
    )
    exact = (
        baseline_summary["output"]["token_ids"]
        == candidate_summary["output"]["token_ids"]
    )
    baseline_buckets = {
        int(row["prefix_start"]): row for row in baseline["buckets"]
    }
    candidate_buckets = {
        int(row["prefix_start"]): row for row in candidate["buckets"]
    }
    bucket_comparisons = []
    for start in sorted(baseline_buckets.keys() & candidate_buckets.keys()):
        base = baseline_buckets[start]
        cand = candidate_buckets[start]
        bucket_comparisons.append(
            {
                "prefix_start": start,
                "prefix_end_exclusive": start + bucket_size,
                "comparison_status": (
                    "descriptive_round_composition_not_prefix_matched"
                ),
                "baseline_accepted_drafts_per_verify": base[
                    "accepted_drafts_per_verify"
                ],
                "candidate_accepted_drafts_per_verify": cand[
                    "accepted_drafts_per_verify"
                ],
                "accepted_draft_gain": cand["accepted_drafts_per_verify"]
                - base["accepted_drafts_per_verify"],
                "paper_acceptance_length_gain": cand[
                    "paper_acceptance_length"
                ]
                - base["paper_acceptance_length"],
            }
        )
    onset = None
    for current, following in zip(
        bucket_comparisons, bucket_comparisons[1:]
    ):
        contiguous = int(following["prefix_start"]) == int(
            current["prefix_end_exclusive"]
        )
        if (
            contiguous
            and float(current["accepted_draft_gain"]) > 0.0
            and float(following["accepted_draft_gain"]) > 0.0
        ):
            onset = int(current["prefix_start"])
            break

    accepted_gain = (
        candidate["accepted_drafts_per_verify"]
        - baseline["accepted_drafts_per_verify"]
    )
    target_call_change = (
        candidate["target_calls_per_output_token"]
        / baseline["target_calls_per_output_token"]
        - 1.0
    )
    speed_change = (
        candidate["reference_tokens_per_second"]
        / baseline["reference_tokens_per_second"]
        - 1.0
    )
    return {
        "schema_version": 1,
        "classification": (
            "single-prompt-exact-reference-pair"
            if baseline_identity["identity_verification_status"]
            == FULLY_VERIFIED_IDENTITY
            else "single-prompt-legacy-unverified-pilot"
        ),
        "identity": baseline_identity,
        "artifact_sha256": {
            "baseline_summary": _sha256(baseline_dir / "summary.json"),
            "baseline_rounds": _sha256(baseline_dir / "rounds.jsonl"),
            "candidate_summary": _sha256(candidate_dir / "summary.json"),
            "candidate_rounds": _sha256(candidate_dir / "rounds.jsonl"),
        },
        "exact_output_token_ids": exact,
        "baseline": baseline,
        "candidate": candidate,
        "gain": {
            "paper_acceptance_length_absolute": candidate[
                "paper_acceptance_length"
            ]
            - baseline["paper_acceptance_length"],
            "paper_acceptance_length_relative": candidate[
                "paper_acceptance_length"
            ]
            / baseline["paper_acceptance_length"]
            - 1.0,
            "accepted_drafts_per_verify_absolute": accepted_gain,
            "accepted_drafts_per_verify_relative": candidate[
                "accepted_drafts_per_verify"
            ]
            / baseline["accepted_drafts_per_verify"]
            - 1.0,
            "target_calls_per_output_token_relative": target_call_change,
            "reference_tokens_per_second_relative": speed_change,
        },
        "descriptive_bucket_comparisons": bucket_comparisons,
        "benefit_onset": {
            "status": "candidate_no_confidence_interval" if onset is not None else "none",
            "prefix_length": onset,
            "rule": (
                "two consecutive descriptive real-prefix buckets with positive "
                "gain; not a prefix-checkpoint-matched estimate"
            ),
        },
        "algorithmic_pass_exploratory": bool(
            exact and accepted_gain > 0.0 and target_call_change < 0.0
        ),
        "engineering_pass_reference": bool(exact and speed_change >= 0.0),
        "limitations": [
            "One prompt has no sampling uncertainty or confidence interval.",
            "Context-bucket differences are descriptive because proposal rounds "
            "do not share identical prefix checkpoints.",
            "Synchronous reference timings contain explicit CUDA synchronizations "
            "and are not serving-throughput measurements.",
        ],
    }


def _first_difference(
    shorter: Any, longer: Any, *, path: str = ""
) -> tuple[str, Any, Any] | None:
    """Return the first deterministic structural difference."""

    if isinstance(shorter, dict) and isinstance(longer, dict):
        keys = sorted(set(shorter) | set(longer))
        for key in keys:
            child = f"{path}.{key}" if path else key
            if key not in shorter:
                return child, "<missing>", longer[key]
            if key not in longer:
                return child, shorter[key], "<missing>"
            difference = _first_difference(
                shorter[key], longer[key], path=child
            )
            if difference is not None:
                return difference
        return None
    if isinstance(shorter, list) and isinstance(longer, list):
        common = min(len(shorter), len(longer))
        for index in range(common):
            difference = _first_difference(
                shorter[index], longer[index], path=f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        if len(shorter) != len(longer):
            return f"{path}.length", len(shorter), len(longer)
        return None
    if type(shorter) is not type(longer) or shorter != longer:
        return path, shorter, longer
    return None


def _trajectory_identity(run: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Build a strict numerical identity excluding only length-derived fields."""

    summary = run["summary"]
    root = run["root"]
    if summary.get("schema_version") != 3:
        raise ValueError(f"{root}: common-prefix gate requires schema_version=3")
    pair_identity = run["pair_identity"]
    if pair_identity.get("identity_verification_status") != FULLY_VERIFIED_IDENTITY:
        raise ValueError(
            f"{root}: common-prefix gate requires {FULLY_VERIFIED_IDENTITY}"
        )
    if pair_identity.get("numeric_runtime_status") != (
        "complete_numeric_runtime_v1"
    ):
        raise ValueError(
            f"{root}: common-prefix gate requires complete numerical runtime "
            "identity"
        )
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{root}: parameters must be an object")
    max_new_tokens = parameters.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError(f"{root}: parameters.max_new_tokens must be positive")
    normalized_pair_identity = dict(pair_identity)
    normalized_pair_identity.pop("max_new_tokens", None)
    normalized_parameters = {
        key: value
        for key, value in parameters.items()
        if key not in _LENGTH_DERIVED_PARAMETER_FIELDS
    }
    reference = summary.get("reference")
    if not isinstance(reference, dict):
        raise ValueError(f"{root}: reference must be an object")
    normalized_reference = {
        key: value
        for key, value in reference.items()
        if key != "official_static_parity"
    }
    identity = {
        "schema_version": 1,
        "mode": summary.get("mode"),
        "method": summary.get("method"),
        "trainable_scope": summary.get("trainable_scope"),
        "pair_identity_without_max_new_tokens": normalized_pair_identity,
        "runtime_fingerprint": summary.get("runtime_fingerprint"),
        "harness": summary.get("harness"),
        "reference_without_parity_result": normalized_reference,
        "models": summary.get("models"),
        "trainable_layout": summary.get("trainable_layout"),
        "parameters_without_length": normalized_parameters,
        "reconstruction_status": summary.get("reconstruction_status"),
        "explicit_non_equivalence": summary.get("explicit_non_equivalence"),
    }
    _canonical_json(identity)
    return max_new_tokens, identity


def _exact_repeat_identity(
    run: dict[str, Any],
    *,
    max_new_tokens: int,
    common_prefix_identity: dict[str, Any],
) -> dict[str, Any]:
    """Restore every field normalized away by the common-prefix identity."""

    summary = run["summary"]
    identity = {
        "schema_version": 1,
        "common_prefix_identity": common_prefix_identity,
        "artifact_identity": summary.get("artifact_identity"),
        "pair_identity": run["pair_identity"],
        "reference": summary.get("reference"),
        "parameters": summary.get("parameters"),
        "max_new_tokens": max_new_tokens,
    }
    _canonical_json(identity)
    return identity


def _trajectory_round_semantics(
    row: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    """Strip only timing/HBM evidence from one schema-v3 round."""

    if "prefix_len_before" not in row:
        raise ValueError(f"{root}: schema-v3 round lacks prefix_len_before")
    for name in (
        "draft_block_token_ids",
        "target_posterior_token_ids",
        "committed_token_ids",
    ):
        values = row.get(name)
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise ValueError(f"{root}: round field {name} must be integer IDs")
    bonus = row.get("bonus_token_id")
    if isinstance(bonus, bool) or not isinstance(bonus, int):
        raise ValueError(f"{root}: round bonus_token_id must be an integer")
    update = row.get("update")
    if not isinstance(update, dict):
        raise ValueError(f"{root}: round update must be an object")
    required_update = {
        "applied",
        "optimizer_step",
        "loss",
        "distillation_kl",
        "proximal_kl",
        "grad_norm",
    }
    missing_update = sorted(required_update - set(update))
    if missing_update:
        raise ValueError(
            f"{root}: round update lacks trajectory fields {missing_update}"
        )
    semantic = {
        key: value
        for key, value in row.items()
        if key not in _TRAJECTORY_TOP_LEVEL_EXCLUSIONS
        and key not in {"prefix_len_before", "prefix_length_before", "update"}
    }
    semantic["prefix_len_before"] = _round_prefix(row)
    semantic["update"] = {
        key: value
        for key, value in update.items()
        if key not in _TRAJECTORY_UPDATE_EXCLUSIONS
    }
    _canonical_json(semantic)
    return semantic


def _resolve_run_artifact_root(path: Path) -> Path:
    """Accept either a harness artifact directory or its runner run root."""

    root = path.expanduser().resolve()
    if (root / "summary.json").is_file() and (root / "rounds.jsonl").is_file():
        return root
    artifact = root / "artifact"
    if (artifact / "summary.json").is_file() and (
        artifact / "rounds.jsonl"
    ).is_file():
        return artifact
    return root


def compare_common_prefix_trajectory(
    run_a_dir: Path, run_b_dir: Path
) -> dict[str, Any]:
    """Certify common-prefix or exact-repeat schema-v3 trajectories."""

    run_a = _load_validated_run(_resolve_run_artifact_root(run_a_dir))
    run_b = _load_validated_run(_resolve_run_artifact_root(run_b_dir))
    mode_a = run_a["summary"].get("mode")
    mode_b = run_b["summary"].get("mode")
    if mode_a != mode_b:
        raise ValueError(
            f"trajectory runs must have the same mode: {mode_a!r} != {mode_b!r}"
        )
    max_a, identity_a = _trajectory_identity(run_a)
    max_b, identity_b = _trajectory_identity(run_b)
    exact_repeat = max_a == max_b
    comparison_kind = "exact_repeat" if exact_repeat else "common_prefix"
    if exact_repeat:
        left, right = run_a, run_b
        left_max, right_max = max_a, max_b
        left_role, right_role = "run_a", "run_b"
        left_identity = _exact_repeat_identity(
            run_a,
            max_new_tokens=max_a,
            common_prefix_identity=identity_a,
        )
        right_identity = _exact_repeat_identity(
            run_b,
            max_new_tokens=max_b,
            common_prefix_identity=identity_b,
        )
    elif max_a < max_b:
        shorter, longer = run_a, run_b
        shorter_max, longer_max = max_a, max_b
        shorter_identity, longer_identity = identity_a, identity_b
    else:
        shorter, longer = run_b, run_a
        shorter_max, longer_max = max_b, max_a
        shorter_identity, longer_identity = identity_b, identity_a

    if not exact_repeat:
        left, right = shorter, longer
        left_max, right_max = shorter_max, longer_max
        left_role, right_role = "shorter", "longer"
        left_identity, right_identity = shorter_identity, longer_identity

    identity_difference = _first_difference(left_identity, right_identity)
    if identity_difference is not None:
        field, left_value, right_value = identity_difference
        raise ValueError(
            "trajectory identity mismatch at "
            f"{field}: {left_role}={left_value!r}, "
            f"{right_role}={right_value!r}"
        )

    left_rounds = left["rounds"]
    right_rounds = right["rounds"]
    common_count = min(len(left_rounds), len(right_rounds))
    exact_common_rounds = 0
    exact_common_semantics: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    for index in range(common_count):
        left_semantic = _trajectory_round_semantics(
            left_rounds[index], root=left["root"]
        )
        right_semantic = _trajectory_round_semantics(
            right_rounds[index], root=right["root"]
        )
        difference = _first_difference(left_semantic, right_semantic)
        if difference is not None:
            field, left_value, right_value = difference
            first_mismatch = {
                "round_index": index,
                "field": field,
                f"{left_role}_value": left_value,
                f"{right_role}_value": right_value,
                f"{left_role}_round_semantics_sha256": _sha256_json(
                    left_semantic
                ),
                f"{right_role}_round_semantics_sha256": _sha256_json(
                    right_semantic
                ),
            }
            break
        exact_common_rounds += 1
        exact_common_semantics.append(left_semantic)

    unexpected_round_count = (
        len(left_rounds) != len(right_rounds)
        if exact_repeat
        else len(left_rounds) > len(right_rounds)
    )
    if first_mismatch is None and unexpected_round_count:
        first_mismatch = {
            "round_index": common_count,
            "field": "$round_presence",
            f"{left_role}_value": (
                "present" if len(left_rounds) > common_count else "missing"
            ),
            f"{right_role}_value": (
                "present" if len(right_rounds) > common_count else "missing"
            ),
        }

    output_prefix_exact: bool | None = None
    output_ids_exact: bool | None = None
    left_output = left["summary"].get("output", {}).get("token_ids")
    right_output = right["summary"].get("output", {}).get("token_ids")
    if not isinstance(left_output, list) or not isinstance(right_output, list):
        raise ValueError("trajectory summaries must contain output.token_ids")
    if exact_repeat:
        output_ids_exact = left_output == right_output
        output_matches = output_ids_exact
    elif first_mismatch is None:
        output_prefix_exact = left_output == right_output[: len(left_output)]
        output_matches = output_prefix_exact
    else:
        output_matches = None
    if first_mismatch is None and output_matches is False:
        mismatch_index = next(
            (
                index
                for index, (left_token, right_token) in enumerate(
                    zip(left_output, right_output)
                )
                if left_token != right_token
            ),
            min(len(left_output), len(right_output)),
        )
        first_mismatch = {
            "round_index": None,
            "field": f"output.token_ids[{mismatch_index}]",
            f"{left_role}_value": (
                left_output[mismatch_index]
                if mismatch_index < len(left_output)
                else "<missing>"
            ),
            f"{right_role}_value": (
                right_output[mismatch_index]
                if mismatch_index < len(right_output)
                else "<missing>"
            ),
        }

    artifacts = {
        left_role: {
            "root": str(left["root"]),
            "summary_sha256": left["summary_sha256"],
            "rounds_sha256": left["rounds_sha256"],
        },
        right_role: {
            "root": str(right["root"]),
            "summary_sha256": right["summary_sha256"],
            "rounds_sha256": right["rounds_sha256"],
        },
    }
    artifact_set = [
        {
            "role": role,
            "summary_sha256": value["summary_sha256"],
            "rounds_sha256": value["rounds_sha256"],
        }
        for role, value in sorted(artifacts.items())
    ]
    return {
        "schema_version": 1,
        "kind": "dflash_tts_common_prefix_trajectory",
        "comparison_kind": comparison_kind,
        "status": (
            ("exact_repeat" if exact_repeat else "exact_common_prefix")
            if first_mismatch is None
            else "trajectory_mismatch"
        ),
        "mode": mode_a,
        "comparison_identity": left_identity,
        "comparison_identity_sha256": _sha256_json(left_identity),
        "max_new_tokens": {
            left_role: left_max,
            right_role: right_max,
        },
        "round_counts": {
            left_role: len(left_rounds),
            right_role: len(right_rounds),
        },
        "exact_common_rounds": exact_common_rounds,
        "exact_common_trajectory_sha256": _sha256_json(
            exact_common_semantics
        ),
        "output_token_prefix_exact": output_prefix_exact,
        "output_token_ids_exact": output_ids_exact,
        "first_mismatch": first_mismatch,
        "excluded_performance_fields": [
            "timing_seconds",
            "hbm_bytes",
            "update.backward_cuda_us",
            "update.optimizer_cuda_us",
            "update.update_cuda_us",
        ],
        "artifact_sha256": artifacts,
        "artifact_set_sha256": _sha256_json(artifact_set),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--common-prefix",
        type=Path,
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help=(
            "validate the non-performance trajectory of two same-mode "
            "schema-v3 runs: common prefix for different max_new_tokens or "
            "an exact fresh-process repeat for equal max_new_tokens"
        ),
    )
    parser.add_argument("--bucket-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")
    paired = args.baseline is not None or args.candidate is not None
    trajectory = args.common_prefix is not None
    if paired and trajectory:
        raise ValueError(
            "use either --baseline/--candidate or --common-prefix, not both"
        )
    if trajectory:
        run_a, run_b = args.common_prefix
        result = compare_common_prefix_trajectory(
            run_a.resolve(), run_b.resolve()
        )
    else:
        if args.baseline is None or args.candidate is None:
            raise ValueError(
                "--baseline and --candidate must be provided together, or use "
                "--common-prefix RUN_A RUN_B"
            )
        result = compare_runs(
            args.baseline.resolve(),
            args.candidate.resolve(),
            bucket_size=args.bucket_size,
        )
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(body)
    print(body, end="")
    if trajectory and result["status"] == "trajectory_mismatch":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
