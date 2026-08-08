#!/usr/bin/env python3
"""Validate and select from an explicit schema-v3 DFlash calibration sweep.

This is deliberately a thin analysis layer.  Run/completion attestation is
delegated to ``run_dflash_tts_frozen_sweep`` and metric reconstruction is
delegated to ``aggregate_dflash_tts_ablations``.  The additional work here is
limited to binding every expected run to the explicit candidate specification,
pairing both calibration prompts with Static, and applying the preregistered
safe local-grid selection rule.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import aggregate_dflash_tts_ablations as aggregation  # noqa: E402
import run_dflash_tts_calibration_sweep as calibration  # noqa: E402


SCHEMA_VERSION = 1
KIND = "dflash_tts_schema_v3_calibration_analysis"
BUCKET_SIZE = 1 << 30
SELECTION_RULE = {
    "scope": "separate_within_each_fixed_mode_and_rank",
    "eligibility": (
        "selection candidate; fully attested schema-v3 pair on both locked "
        "samples; exact Static output; finite A, loss, gradient, HBM, and "
        "target-calls/output metrics; comparable runtime fingerprint"
    ),
    "safety_gate": "paired_delta_A_greater_than_or_equal_to_zero_on_both_samples",
    "ordering": [
        "higher mean paired delta A",
        "higher worst-sample paired delta A",
        "lower maximum whole-run PyTorch allocator allocated HBM",
        "fewer trainable parameters",
        "lower rank",
        "lexicographically smaller canonical config",
    ],
    "A_semantics": (
        "mean accepted draft tokens per target verification; paper acceptance "
        "length is A + 1 and is reported separately"
    ),
    "claim_scope": "bounded_local_grid_selection_not_global_optimization",
}
TARGET_CALL_METRIC_SCOPE = {
    "headline": "logical_block_target_calls_per_output_token",
    "legacy_alias": "target_calls_per_output_token",
    "reference_audit_metrics": [
        "physical_target_calls_per_output_token",
        "canonical_audit_target_calls_per_output_token",
    ],
    "interpretation": (
        "selection uses logical DFlash block verifications; physical and "
        "canonical calls measure exact reference-audit overhead and are not "
        "deployment throughput metrics"
    ),
}


@dataclass(frozen=True)
class BoundRun:
    sample_index: int
    candidate_id: str
    run_dir: Path
    artifact_dir: Path
    identity: dict[str, Any]
    summary: dict[str, Any]
    rounds: tuple[dict[str, Any], ...]
    evidence_hashes: dict[str, str]


@dataclass(frozen=True)
class BoundArtifactIdentityLock:
    path: Path
    relative_path: str
    payload: dict[str, Any]
    file_sha256: str
    content_sha256: str


def _expect(observed: Any, expected: Any, label: str) -> None:
    if aggregation._canonical_json(observed) != aggregation._canonical_json(expected):
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


def _relative_file_under_root(path: Path, root: Path, label: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must live under output_root: {resolved}") from exc
    return resolved, relative.as_posix()


def _load_artifact_identity_lock(root: Path) -> BoundArtifactIdentityLock:
    path = root / "artifact_identity_lock.json"
    if not path.is_file():
        raise ValueError(f"missing artifact identity lock: {path}")
    payload = calibration.frozen._read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: artifact identity lock must be an object")
    _expect(payload.get("schema_version"), 1, f"{path}: lock schema")
    _expect(
        payload.get("kind"),
        "dflash_tts_artifact_identity_lock",
        f"{path}: lock kind",
    )
    for role in ("target", "draft", "tokenizer"):
        _required_dict(payload.get(role), f"{path}: lock {role}")
    return BoundArtifactIdentityLock(
        path=path,
        relative_path="artifact_identity_lock.json",
        payload=payload,
        file_sha256=aggregation._sha256_file(path),
        content_sha256=aggregation._sha256_json(payload),
    )


def _validate_artifact_identity_lock_binding(
    identity: dict[str, Any],
    *,
    lock: BoundArtifactIdentityLock,
    run_dir: Path,
) -> None:
    runtime = _required_dict(identity.get("runtime"), f"{run_dir}: runtime")
    recorded = _required_dict(
        runtime.get("artifact_identity_lock"),
        f"{run_dir}: artifact identity lock",
    )
    # Old remote identities record an absolute source-machine path.  Relocation
    # is authorized by the immutable file hash and content identities, not by
    # requiring that historical locator to exist on the checking machine.
    if not isinstance(recorded.get("path"), str) or not recorded["path"]:
        raise ValueError(f"{run_dir}: artifact identity lock path is missing")
    _expect(
        recorded.get("sha256"),
        lock.file_sha256,
        f"{run_dir}: artifact identity lock file hash",
    )
    for role in ("target", "draft", "tokenizer"):
        _expect(
            identity.get(role),
            lock.payload[role],
            f"{run_dir}: artifact identity lock {role} content",
        )


def _output_token_ids(summary: dict[str, Any], label: str) -> list[int]:
    output = _required_dict(summary.get("output"), f"{label}.output")
    values = _required_list(output.get("token_ids"), f"{label}.output.token_ids")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{label}.output.token_ids must contain only integers")
    return [int(value) for value in values]


def _spec_identity(sweep: calibration.CandidateSweep) -> dict[str, Any]:
    return {
        "file_sha256": sweep.file_sha256,
        "content_sha256": sweep.content_sha256,
        "study_id": sweep.study_id,
        "schema_version": calibration.SCHEMA_VERSION,
        "kind": sweep.kind,
        "evidence_scope": sweep.evidence_scope,
    }


def _validate_spec_binding(
    identity: dict[str, Any],
    *,
    sweep: calibration.CandidateSweep,
    candidate: calibration.Candidate,
    candidate_index: int,
    sample: dict[str, Any],
    run_dir: Path,
) -> None:
    _expect(identity.get("schema_version"), 3, f"{run_dir}: identity schema")
    _expect(
        identity.get("sweep"),
        "dflash_tts_explicit_calibration_rank_v3",
        f"{run_dir}: sweep",
    )
    recorded_spec = _required_dict(
        identity.get("candidate_specification"),
        f"{run_dir}: candidate_specification",
    )
    if not isinstance(recorded_spec.get("path"), str) or not recorded_spec["path"]:
        raise ValueError(f"{run_dir}: candidate specification path is missing")
    for key, expected in _spec_identity(sweep).items():
        _expect(
            recorded_spec.get(key),
            expected,
            f"{run_dir}: candidate specification {key}",
        )

    _expect(
        identity.get("calibration_candidate"),
        {
            "candidate_id": candidate.candidate_id,
            "candidate_index": candidate_index,
            "diagnostic_kind": candidate.diagnostic_kind,
            "selection_eligible": candidate.selection_eligible,
        },
        f"{run_dir}: calibration candidate",
    )
    _expect(identity.get("mode"), candidate.mode, f"{run_dir}: mode")
    dataset = _required_dict(identity.get("dataset"), f"{run_dir}: dataset")
    for key in (
        "sample_index",
        "input_tokens",
        "rendered_input_token_ids_sha256",
    ):
        _expect(dataset.get(key), sample[key], f"{run_dir}: dataset {key}")
    _expect(
        dataset.get("input_token_source"),
        {
            "kind": "schema_v3_candidate_specification",
            "candidate_specification_sha256": sweep.file_sha256,
            "value": sample["input_tokens"],
        },
        f"{run_dir}: input token source",
    )

    generation = _required_dict(
        identity.get("generation"), f"{run_dir}: generation"
    )
    block_size = generation.get("draft_block_size")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 1:
        raise ValueError(f"{run_dir}: invalid draft block size")
    expected_context = sample["input_tokens"] + calibration.MAX_NEW_TOKENS + block_size - 1
    for key, expected in (
        ("input_tokens", sample["input_tokens"]),
        ("max_new_tokens", calibration.MAX_NEW_TOKENS),
        ("pending_draft_tokens", block_size - 1),
        ("requested_total_context", expected_context),
        ("required_prefix_plus_block", expected_context),
    ):
        _expect(generation.get(key), expected, f"{run_dir}: generation {key}")

    optimization = _required_dict(
        identity.get("optimization"), f"{run_dir}: optimization"
    )
    for key, expected in (
        ("optimizer", candidate.config.optimizer),
        ("learning_rate", candidate.config.learning_rate),
        ("weight_decay", candidate.config.weight_decay),
        ("rank", candidate.config.rank),
        ("draft_cache_policy", candidate.draft_cache_policy),
    ):
        _expect(optimization.get(key), expected, f"{run_dir}: optimization {key}")
    audit = _required_dict(identity.get("audit"), f"{run_dir}: audit")
    _expect(
        audit.get("parameter_audit_stride"),
        candidate.parameter_audit_stride,
        f"{run_dir}: parameter audit stride",
    )
    if candidate.selection_eligible:
        _expect(audit.get("cuda_timing"), False, f"{run_dir}: selection CUDA timing")
        _expect(
            audit.get("skip_static_parity_preflight"),
            False,
            f"{run_dir}: selection Static parity",
        )


def _reconstruct_plan(
    run_dir: Path,
    stored: dict[str, Any],
    *,
    artifact_identity_lock: BoundArtifactIdentityLock,
) -> Any:
    frozen = calibration.frozen
    identity = _required_dict(stored.get("identity"), f"{run_dir}: identity")
    command = _required_list(stored.get("command"), f"{run_dir}: command")
    if any(not isinstance(value, str) for value in command):
        raise ValueError(f"{run_dir}: command must contain only strings")
    environment = _required_dict(
        stored.get("environment"), f"{run_dir}: environment"
    )
    pythonpath = _required_list(
        environment.get("PYTHONPATH"), f"{run_dir}: environment.PYTHONPATH"
    )
    if any(not isinstance(value, str) for value in pythonpath):
        raise ValueError(f"{run_dir}: PYTHONPATH must contain only strings")
    identity_sha256 = stored.get("identity_sha256")
    if not aggregation._is_sha256(identity_sha256):
        raise ValueError(f"{run_dir}: invalid identity_sha256")
    return frozen.RunPlan(
        run_dir=run_dir,
        artifact_dir=run_dir / "artifact",
        log_path=run_dir / "run.log",
        identity_path=run_dir / "run_identity.json",
        completion_path=run_dir / "completion.json",
        identity=identity,
        identity_sha256=str(identity_sha256),
        command=tuple(command),
        pythonpath=tuple(pythonpath),
        artifact_identity_lock_path=artifact_identity_lock.path,
        artifact_identity_lock_payload=artifact_identity_lock.payload,
    )


def _load_bound_run(
    output_root: Path,
    *,
    artifact_identity_lock: BoundArtifactIdentityLock,
    sweep: calibration.CandidateSweep,
    candidate: calibration.Candidate,
    candidate_index: int,
    sample: dict[str, Any],
) -> BoundRun:
    sample_index = int(sample["sample_index"])
    run_dir = output_root / f"sample-{sample_index:04d}" / candidate.candidate_id
    identity_path = run_dir / "run_identity.json"
    completion_path = run_dir / "completion.json"
    if not identity_path.is_file():
        raise ValueError(f"{run_dir}: missing run_identity.json")
    if not completion_path.is_file():
        raise ValueError(f"{run_dir}: missing completion.json")
    stored = calibration.frozen._read_json(identity_path)
    identity = _required_dict(stored.get("identity"), f"{run_dir}: identity")
    _validate_artifact_identity_lock_binding(
        identity,
        lock=artifact_identity_lock,
        run_dir=run_dir,
    )
    _validate_spec_binding(
        identity,
        sweep=sweep,
        candidate=candidate,
        candidate_index=candidate_index,
        sample=sample,
        run_dir=run_dir,
    )
    plan = _reconstruct_plan(
        run_dir,
        stored,
        artifact_identity_lock=artifact_identity_lock,
    )
    if not calibration.frozen.completed_run_matches(plan):
        raise ValueError(f"{run_dir}: expected a completed run")
    summary_path = plan.artifact_dir / "summary.json"
    rounds_path = plan.artifact_dir / "rounds.jsonl"
    summary = calibration.frozen._read_json(summary_path)
    rounds = tuple(aggregation._read_jsonl(rounds_path))
    _output_token_ids(summary, str(run_dir))
    return BoundRun(
        sample_index=sample_index,
        candidate_id=candidate.candidate_id,
        run_dir=run_dir,
        artifact_dir=plan.artifact_dir,
        identity=identity,
        summary=summary,
        rounds=rounds,
        evidence_hashes={
            "run_identity_sha256": aggregation._sha256_file(identity_path),
            "identity_sha256": plan.identity_sha256,
            "completion_sha256": aggregation._sha256_file(completion_path),
            "summary_sha256": aggregation._sha256_file(summary_path),
            "rounds_sha256": aggregation._sha256_file(rounds_path),
            "command_sha256": calibration.frozen._signed_command_sha256(
                plan.command
            ),
        },
    )


def _common_control_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Remove only the sample and explicitly ablated candidate axes."""

    dataset = _required_dict(identity.get("dataset"), "identity.dataset")
    generation = _required_dict(identity.get("generation"), "identity.generation")
    optimization = _required_dict(
        identity.get("optimization"), "identity.optimization"
    )
    audit = _required_dict(identity.get("audit"), "identity.audit")
    return {
        "candidate_specification_recorded_path": identity[
            "candidate_specification"
        ]["path"],
        "runtime": identity["runtime"],
        "reference": identity["reference"],
        "target": identity["target"],
        "draft": identity["draft"],
        "tokenizer": identity["tokenizer"],
        "dataset": {
            key: dataset[key]
            for key in (
                "path",
                "revision",
                "sha256",
                "prompt_field",
                "messages_field",
                "turns_field",
                "enable_thinking",
            )
        },
        "generation": {
            key: generation[key]
            for key in (
                "max_new_tokens",
                "draft_block_size",
                "pending_draft_tokens",
                "stop_token_ids",
                "temperature",
                "seed",
                "mask_token_id",
            )
        },
        "optimization": {
            key: optimization[key]
            for key in (
                "proximal_lambda",
                "update_stride",
                "position_weighting",
                "position_decay_gamma",
                "loss_reduction",
                "adam_betas",
                "adam_eps",
            )
        },
        "audit": {
            key: audit[key]
            for key in (
                "cuda_timing",
                "parity_max_new_tokens",
                "skip_static_parity_preflight",
            )
        },
    }


def _parameterization_group(identity: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    optimization = identity["optimization"]
    rank = optimization["rank"]
    key = f"{identity['mode']}:rank={'none' if rank is None else rank}"
    return key, {
        "adapter_seed": optimization["adapter_seed"],
        "projection": identity["projection"],
    }


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _update_stats(run: BoundRun, *, mode: str) -> dict[str, Any]:
    applied: list[tuple[int, float, float]] = []
    missing_loss = 0
    missing_grad = 0
    nonfinite_loss = 0
    nonfinite_grad = 0
    for row in run.rounds:
        update = row.get("update")
        if not isinstance(update, dict) or not bool(update.get("applied")):
            continue
        prefix = aggregation._round_prefix(row)
        loss = update.get("loss")
        grad = update.get("grad_norm")
        if loss is None:
            missing_loss += 1
        elif not _finite_number(loss):
            nonfinite_loss += 1
        if grad is None:
            missing_grad += 1
        elif not _finite_number(grad):
            nonfinite_grad += 1
        if _finite_number(loss) and _finite_number(grad):
            applied.append((prefix, float(loss), float(grad)))
    applied_count = sum(
        1
        for row in run.rounds
        if isinstance(row.get("update"), dict)
        and bool(row["update"].get("applied"))
    )
    expected = mode != "static"
    all_finite = (
        (not expected and applied_count == 0)
        or (
            expected
            and applied_count > 0
            and len(applied) == applied_count
            and missing_loss == missing_grad == nonfinite_loss == nonfinite_grad == 0
        )
    )
    losses = [value[1] for value in applied]
    grads = [value[2] for value in applied]
    prefixes = [value[0] for value in applied]
    return {
        "status": "not_applicable_static" if not expected else "observed_updates",
        "applied_steps": applied_count,
        "all_finite": all_finite,
        "missing_loss_count": missing_loss,
        "missing_grad_norm_count": missing_grad,
        "nonfinite_loss_count": nonfinite_loss,
        "nonfinite_grad_norm_count": nonfinite_grad,
        "prefix_len_min": min(prefixes) if prefixes else None,
        "prefix_len_max": max(prefixes) if prefixes else None,
        "loss_first": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_mean": statistics.fmean(losses) if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "grad_norm_mean": statistics.fmean(grads) if grads else None,
        "grad_norm_max": max(grads) if grads else None,
    }


def _sample_metrics(
    run: BoundRun,
    row: dict[str, Any],
    *,
    static: BoundRun,
    static_row: dict[str, Any],
) -> dict[str, Any]:
    candidate_tokens = _output_token_ids(run.summary, str(run.run_dir))
    static_tokens = _output_token_ids(static.summary, str(static.run_dir))
    exact = candidate_tokens == static_tokens
    candidate_a = row.get("run_accepted_drafts_per_verify")
    static_a = static_row.get("run_accepted_drafts_per_verify")
    candidate_paper_al = row.get("run_paper_acceptance_length")
    static_paper_al = static_row.get("run_paper_acceptance_length")
    paired_gain = (
        float(candidate_a) - float(static_a)
        if _finite_number(candidate_a) and _finite_number(static_a)
        else None
    )
    hbm = row.get("whole_process_peak_hbm_bytes")
    static_hbm = static_row.get("whole_process_peak_hbm_bytes")
    reserved_hbm = row.get("whole_process_peak_hbm_reserved_bytes")
    static_reserved_hbm = static_row.get("whole_process_peak_hbm_reserved_bytes")
    runtime_match = (
        row.get("runtime_fingerprint_status") == "complete_hbm_comparable_v1"
        and static_row.get("runtime_fingerprint_status")
        == "complete_hbm_comparable_v1"
        and row.get("runtime_fingerprint_sha256")
        == static_row.get("runtime_fingerprint_sha256")
    )
    hbm_delta = (
        int(hbm) - int(static_hbm)
        if runtime_match and hbm is not None and static_hbm is not None
        else None
    )
    reserved_hbm_delta = (
        int(reserved_hbm) - int(static_reserved_hbm)
        if runtime_match
        and isinstance(reserved_hbm, int)
        and not isinstance(reserved_hbm, bool)
        and isinstance(static_reserved_hbm, int)
        and not isinstance(static_reserved_hbm, bool)
        else None
    )
    updates = _update_stats(run, mode=str(run.summary["mode"]))
    logical_target_calls = row.get(
        "run_logical_block_target_calls_per_output_token",
        row.get("run_target_calls_per_output_token"),
    )
    physical_target_calls = row.get(
        "run_physical_target_calls_per_output_token"
    )
    canonical_audit_target_calls = row.get(
        "run_canonical_audit_target_calls_per_output_token"
    )
    trainable = row.get("trainable_parameter_count")
    optimization = _required_dict(
        run.identity.get("optimization"),
        f"{run.run_dir}: optimization",
    )
    rank = optimization.get("rank")
    adapter_seed = optimization.get("adapter_seed")
    adapter_seed_valid = (
        adapter_seed is None
        if rank is None
        else isinstance(adapter_seed, int) and not isinstance(adapter_seed, bool)
    )
    optimizer_persistent = row.get("optimizer_resident_bytes")
    optimizer_update_peak = row.get("optimizer_update_peak_bytes")
    optimizer_persistent_per_parameter = (
        float(optimizer_persistent) / int(trainable)
        if isinstance(optimizer_persistent, int)
        and not isinstance(optimizer_persistent, bool)
        and isinstance(trainable, int)
        and not isinstance(trainable, bool)
        and trainable > 0
        else None
    )
    optimizer_update_peak_per_parameter = (
        float(optimizer_update_peak) / int(trainable)
        if isinstance(optimizer_update_peak, int)
        and not isinstance(optimizer_update_peak, bool)
        and isinstance(trainable, int)
        and not isinstance(trainable, bool)
        and trainable > 0
        else None
    )
    metric_finite = all(
        _finite_number(value)
        for value in (
            candidate_a,
            static_a,
            candidate_paper_al,
            static_paper_al,
            logical_target_calls,
        )
    )
    memory_complete = (
        isinstance(hbm, int)
        and not isinstance(hbm, bool)
        and hbm >= 0
        and isinstance(static_hbm, int)
        and not isinstance(static_hbm, bool)
        and static_hbm >= 0
        and isinstance(reserved_hbm, int)
        and not isinstance(reserved_hbm, bool)
        and reserved_hbm >= hbm
        and isinstance(static_reserved_hbm, int)
        and not isinstance(static_reserved_hbm, bool)
        and static_reserved_hbm >= static_hbm
        and runtime_match
    )
    parameter_complete = (
        isinstance(trainable, int)
        and trainable >= 0
        and (run.summary["mode"] == "static" or trainable > 0)
    )
    optimizer_memory_complete = (
        isinstance(optimizer_persistent, int)
        and not isinstance(optimizer_persistent, bool)
        and optimizer_persistent >= 0
        and isinstance(optimizer_update_peak, int)
        and not isinstance(optimizer_update_peak, bool)
        and optimizer_update_peak >= optimizer_persistent
        and (
            (
                run.summary["mode"] == "static"
                and optimizer_persistent == optimizer_update_peak == 0
            )
            or (
                run.summary["mode"] != "static"
                and optimizer_persistent_per_parameter is not None
                and optimizer_update_peak_per_parameter is not None
            )
        )
    )
    rounds_path = (
        Path(f"sample-{run.sample_index:04d}")
        / run.candidate_id
        / "artifact"
        / "rounds.jsonl"
    ).as_posix()
    return {
        "sample_index": run.sample_index,
        "rounds_path": rounds_path,
        "rounds_sha256": run.evidence_hashes["rounds_sha256"],
        "pair_identity_sha256": row.get("pair_key"),
        "static_pair_identity_sha256": static_row.get("pair_key"),
        "pair_identity_matches_static": row.get("pair_key")
        == static_row.get("pair_key"),
        "exact_output_token_ids": exact,
        "candidate_output_token_ids_sha256": aggregation._sha256_json(
            candidate_tokens
        ),
        "static_output_token_ids_sha256": aggregation._sha256_json(static_tokens),
        "accepted_drafts_per_verify_A": candidate_a,
        "static_accepted_drafts_per_verify_A": static_a,
        "paper_acceptance_length": candidate_paper_al,
        "static_paper_acceptance_length": static_paper_al,
        "paired_delta_A": paired_gain,
        # Keep the generic name as a backwards-compatible alias.  Selection
        # and paper-facing analysis always use the explicit logical metric.
        "target_calls_per_output_token": logical_target_calls,
        "logical_block_target_calls_per_output_token": logical_target_calls,
        "physical_target_calls_per_output_token": physical_target_calls,
        "canonical_audit_target_calls_per_output_token": (
            canonical_audit_target_calls
        ),
        "target_call_metric_scope": TARGET_CALL_METRIC_SCOPE,
        "static_target_calls_per_output_token": static_row.get(
            "run_logical_block_target_calls_per_output_token",
            static_row.get("run_target_calls_per_output_token"),
        ),
        "static_logical_block_target_calls_per_output_token": static_row.get(
            "run_logical_block_target_calls_per_output_token",
            static_row.get("run_target_calls_per_output_token"),
        ),
        "static_physical_target_calls_per_output_token": static_row.get(
            "run_physical_target_calls_per_output_token"
        ),
        "static_canonical_audit_target_calls_per_output_token": static_row.get(
            "run_canonical_audit_target_calls_per_output_token"
        ),
        "target_calls_per_output_token_change": (
            float(logical_target_calls)
            - float(
                static_row.get(
                    "run_logical_block_target_calls_per_output_token",
                    static_row.get("run_target_calls_per_output_token"),
                )
            )
            if _finite_number(logical_target_calls)
            and _finite_number(
                static_row.get(
                    "run_logical_block_target_calls_per_output_token",
                    static_row.get("run_target_calls_per_output_token"),
                )
            )
            else None
        ),
        "decode_window_peak_hbm_bytes": row.get("peak_hbm_bytes"),
        "decode_window_peak_hbm_reserved_bytes": row.get(
            "peak_hbm_reserved_bytes"
        ),
        "whole_process_peak_hbm_bytes": hbm,
        "static_whole_process_peak_hbm_bytes": static_hbm,
        "whole_process_peak_hbm_over_static_bytes": hbm_delta,
        "whole_process_peak_hbm_reserved_bytes": reserved_hbm,
        "static_whole_process_peak_hbm_reserved_bytes": static_reserved_hbm,
        "whole_process_peak_hbm_reserved_over_static_bytes": reserved_hbm_delta,
        "runtime_fingerprint_matches_static": runtime_match,
        "trainable_parameter_count": trainable,
        "adapter_seed": adapter_seed,
        "adapter_seed_valid_for_parameterization": adapter_seed_valid,
        "optimizer_persistent_bytes": optimizer_persistent,
        "optimizer_update_peak_bytes": optimizer_update_peak,
        "optimizer_persistent_bytes_per_trainable_parameter": (
            optimizer_persistent_per_parameter
        ),
        "optimizer_update_peak_bytes_per_trainable_parameter": (
            optimizer_update_peak_per_parameter
        ),
        "optimizer_memory_evidence": row.get("optimizer_memory_evidence"),
        "update": updates,
        "metric_finite": metric_finite,
        "memory_complete_and_comparable": memory_complete,
        "optimizer_memory_complete": optimizer_memory_complete,
        "trainable_parameter_count_valid": parameter_complete,
        "evidence_hashes": run.evidence_hashes,
    }


def _candidate_row(
    candidate: calibration.Candidate,
    sample_results: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not candidate.selection_eligible:
        reasons.append("candidate_spec_marks_diagnostic")
    if candidate.mode == "static":
        reasons.append("static_is_baseline_not_selectable")
    for result in sample_results:
        sample = result["sample_index"]
        if not result["pair_identity_matches_static"]:
            reasons.append(f"sample_{sample}_pair_identity_mismatch")
        if not result["exact_output_token_ids"]:
            reasons.append(f"sample_{sample}_output_not_exact_static")
        if not result["metric_finite"]:
            reasons.append(f"sample_{sample}_nonfinite_metric")
        if not result["memory_complete_and_comparable"]:
            reasons.append(f"sample_{sample}_hbm_not_comparable")
        if not result["optimizer_memory_complete"]:
            reasons.append(f"sample_{sample}_optimizer_memory_incomplete")
        if not result["trainable_parameter_count_valid"]:
            reasons.append(f"sample_{sample}_invalid_trainable_parameter_count")
        if not result["adapter_seed_valid_for_parameterization"]:
            reasons.append(f"sample_{sample}_invalid_adapter_seed")
        if not result["update"]["all_finite"]:
            reasons.append(f"sample_{sample}_loss_or_grad_not_finite")
    parameter_counts = {
        result["trainable_parameter_count"] for result in sample_results
    }
    if len(parameter_counts) != 1:
        reasons.append("trainable_parameter_count_differs_by_sample")
    adapter_seeds = {result["adapter_seed"] for result in sample_results}
    if len(adapter_seeds) != 1:
        reasons.append("adapter_seed_differs_by_sample")
    adapter_seed = next(iter(adapter_seeds)) if len(adapter_seeds) == 1 else None
    paired_gains = [result["paired_delta_A"] for result in sample_results]
    logical_target_calls = [
        result["logical_block_target_calls_per_output_token"]
        for result in sample_results
    ]
    physical_target_calls = [
        result["physical_target_calls_per_output_token"]
        for result in sample_results
    ]
    canonical_audit_target_calls = [
        result["canonical_audit_target_calls_per_output_token"]
        for result in sample_results
    ]
    whole_process_hbm = [
        result["whole_process_peak_hbm_bytes"] for result in sample_results
    ]
    whole_process_reserved_hbm = [
        result["whole_process_peak_hbm_reserved_bytes"]
        for result in sample_results
    ]
    optimizer_persistent = [
        result["optimizer_persistent_bytes"] for result in sample_results
    ]
    optimizer_update_peak = [
        result["optimizer_update_peak_bytes"] for result in sample_results
    ]
    optimizer_persistent_per_parameter = [
        result["optimizer_persistent_bytes_per_trainable_parameter"]
        for result in sample_results
    ]
    optimizer_update_peak_per_parameter = [
        result["optimizer_update_peak_bytes_per_trainable_parameter"]
        for result in sample_results
    ]
    evidence_eligible = not reasons
    safe = evidence_eligible and all(float(value) >= 0.0 for value in paired_gains)
    return {
        "candidate_id": candidate.candidate_id,
        "mode": candidate.mode,
        "rank": candidate.config.rank,
        "adapter_seed": adapter_seed,
        "optimizer": candidate.config.optimizer,
        "learning_rate": candidate.config.learning_rate,
        "weight_decay": candidate.config.weight_decay,
        "draft_cache_policy": candidate.draft_cache_policy,
        "diagnostic_kind": candidate.diagnostic_kind,
        "candidate_spec_selection_eligible": candidate.selection_eligible,
        "sample_results": sorted(sample_results, key=lambda row: row["sample_index"]),
        "aggregate": {
            "evidence_eligible": evidence_eligible,
            "ineligibility_reasons": sorted(set(reasons)),
            "safe_for_selection": safe,
            "safety_status": (
                "safe_nonnegative_on_both_samples"
                if safe
                else (
                    "unsafe_negative_delta_A_on_at_least_one_sample"
                    if evidence_eligible
                    else "ineligible"
                )
            ),
            "mean_paired_delta_A": (
                statistics.fmean(float(value) for value in paired_gains)
                if all(_finite_number(value) for value in paired_gains)
                else None
            ),
            "worst_sample_paired_delta_A": (
                min(float(value) for value in paired_gains)
                if all(_finite_number(value) for value in paired_gains)
                else None
            ),
            "mean_target_calls_per_output_token": (
                statistics.fmean(float(value) for value in logical_target_calls)
                if all(_finite_number(value) for value in logical_target_calls)
                else None
            ),
            "mean_logical_block_target_calls_per_output_token": (
                statistics.fmean(float(value) for value in logical_target_calls)
                if all(_finite_number(value) for value in logical_target_calls)
                else None
            ),
            "mean_physical_target_calls_per_output_token": (
                statistics.fmean(float(value) for value in physical_target_calls)
                if all(_finite_number(value) for value in physical_target_calls)
                else None
            ),
            "mean_canonical_audit_target_calls_per_output_token": (
                statistics.fmean(
                    float(value) for value in canonical_audit_target_calls
                )
                if all(
                    _finite_number(value) for value in canonical_audit_target_calls
                )
                else None
            ),
            "target_call_metric_scope": TARGET_CALL_METRIC_SCOPE,
            "max_whole_process_peak_hbm_bytes": (
                max(int(value) for value in whole_process_hbm)
                if all(isinstance(value, int) and value >= 0 for value in whole_process_hbm)
                else None
            ),
            "max_whole_process_peak_hbm_over_static_bytes": max(
                (
                    int(result["whole_process_peak_hbm_over_static_bytes"])
                    for result in sample_results
                    if result["whole_process_peak_hbm_over_static_bytes"] is not None
                ),
                default=None,
            ),
            "max_whole_process_peak_hbm_reserved_bytes": (
                max(int(value) for value in whole_process_reserved_hbm)
                if all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in whole_process_reserved_hbm
                )
                else None
            ),
            "max_whole_process_peak_hbm_reserved_over_static_bytes": max(
                (
                    int(
                        result[
                            "whole_process_peak_hbm_reserved_over_static_bytes"
                        ]
                    )
                    for result in sample_results
                    if result[
                        "whole_process_peak_hbm_reserved_over_static_bytes"
                    ]
                    is not None
                ),
                default=None,
            ),
            "trainable_parameter_count": (
                next(iter(parameter_counts)) if len(parameter_counts) == 1 else None
            ),
            "adapter_seed": adapter_seed,
            "max_optimizer_persistent_bytes": (
                max(int(value) for value in optimizer_persistent)
                if all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in optimizer_persistent
                )
                else None
            ),
            "max_optimizer_update_peak_bytes": (
                max(int(value) for value in optimizer_update_peak)
                if all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in optimizer_update_peak
                )
                else None
            ),
            "max_optimizer_persistent_bytes_per_trainable_parameter": (
                max(float(value) for value in optimizer_persistent_per_parameter)
                if all(
                    _finite_number(value)
                    for value in optimizer_persistent_per_parameter
                )
                else None
            ),
            "max_optimizer_update_peak_bytes_per_trainable_parameter": (
                max(float(value) for value in optimizer_update_peak_per_parameter)
                if all(
                    _finite_number(value)
                    for value in optimizer_update_peak_per_parameter
                )
                else None
            ),
            "all_losses_and_gradients_finite": all(
                result["update"]["all_finite"] for result in sample_results
            ),
            "all_outputs_exact_static": all(
                result["exact_output_token_ids"] for result in sample_results
            ),
        },
    }


def _canonical_config(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["optimizer"]),
        float(row["learning_rate"]),
        float(row["weight_decay"]),
        -1 if row["rank"] is None else int(row["rank"]),
        str(row["candidate_id"]),
    )


def _selection_order(row: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = row["aggregate"]
    return (
        -float(aggregate["mean_paired_delta_A"]),
        -float(aggregate["worst_sample_paired_delta_A"]),
        int(aggregate["max_whole_process_peak_hbm_bytes"]),
        int(aggregate["trainable_parameter_count"]),
        -1 if row["rank"] is None else int(row["rank"]),
        _canonical_config(row),
    )


def _selection_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        if row["mode"] == "static" or not row["candidate_spec_selection_eligible"]:
            continue
        grouped.setdefault((row["mode"], row["rank"]), []).append(row)
    decisions = []
    for (mode, rank), candidates in sorted(
        grouped.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
    ):
        eligible = [row for row in candidates if row["aggregate"]["evidence_eligible"]]
        safe = sorted(
            (row for row in eligible if row["aggregate"]["safe_for_selection"]),
            key=_selection_order,
        )
        all_lrs = sorted({float(row["learning_rate"]) for row in candidates})
        decision: dict[str, Any] = {
            "mode": mode,
            "rank": rank,
            "status": "no_safe_selection" if not safe else "local_grid_winner",
            "candidate_count": len(candidates),
            "evidence_eligible_count": len(eligible),
            "safe_candidate_count": len(safe),
            "tested_learning_rate_bounds": {
                "minimum": min(all_lrs),
                "maximum": max(all_lrs),
            },
            "winner": None,
            "global_optimum_claim": False,
        }
        if safe:
            winner = safe[0]
            same_profile_lrs = sorted(
                {
                    float(row["learning_rate"])
                    for row in candidates
                    if row["optimizer"] == winner["optimizer"]
                    and float(row["weight_decay"])
                    == float(winner["weight_decay"])
                }
            )
            winner_lr = float(winner["learning_rate"])
            group_boundary = winner_lr in {min(all_lrs), max(all_lrs)}
            profile_boundary = winner_lr in {
                min(same_profile_lrs),
                max(same_profile_lrs),
            }
            decision["winner"] = {
                "candidate_id": winner["candidate_id"],
                "optimizer": winner["optimizer"],
                "learning_rate": winner_lr,
                "weight_decay": winner["weight_decay"],
                "rank": winner["rank"],
                "aggregate": winner["aggregate"],
                "learning_rate_boundary": {
                    "at_group_boundary": group_boundary,
                    "at_optimizer_weight_decay_boundary": profile_boundary,
                    "optimizer_weight_decay_bounds": {
                        "minimum": min(same_profile_lrs),
                        "maximum": max(same_profile_lrs),
                    },
                    "requires_grid_extension_before_optimum_claim": (
                        group_boundary or profile_boundary
                    ),
                },
                "claim": "selected_safe_local_grid_configuration_not_global_optimum",
            }
            decision["ordered_safe_candidate_ids"] = [
                row["candidate_id"] for row in safe
            ]
        decisions.append(decision)
    return decisions


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["mode"] != "static"
        and row["candidate_spec_selection_eligible"]
        and row["aggregate"]["evidence_eligible"]
    ]
    output = []
    for row in candidates:
        aggregate = row["aggregate"]
        gain = float(aggregate["mean_paired_delta_A"])
        hbm = int(aggregate["max_whole_process_peak_hbm_bytes"])
        parameters = int(aggregate["trainable_parameter_count"])
        dominated = any(
            float(other["aggregate"]["mean_paired_delta_A"]) >= gain
            and int(other["aggregate"]["max_whole_process_peak_hbm_bytes"]) <= hbm
            and int(other["aggregate"]["trainable_parameter_count"]) <= parameters
            and (
                float(other["aggregate"]["mean_paired_delta_A"]) > gain
                or int(other["aggregate"]["max_whole_process_peak_hbm_bytes"]) < hbm
                or int(other["aggregate"]["trainable_parameter_count"]) < parameters
            )
            for other in candidates
            if other is not row
        )
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "mode": row["mode"],
                "rank": row["rank"],
                "optimizer": row["optimizer"],
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "mean_paired_delta_A": gain,
                "max_whole_process_peak_hbm_bytes": hbm,
                "max_whole_process_peak_hbm_reserved_bytes": aggregate[
                    "max_whole_process_peak_hbm_reserved_bytes"
                ],
                "trainable_parameter_count": parameters,
                "max_optimizer_persistent_bytes": aggregate[
                    "max_optimizer_persistent_bytes"
                ],
                "max_optimizer_update_peak_bytes": aggregate[
                    "max_optimizer_update_peak_bytes"
                ],
                "max_optimizer_persistent_bytes_per_trainable_parameter": aggregate[
                    "max_optimizer_persistent_bytes_per_trainable_parameter"
                ],
                "max_optimizer_update_peak_bytes_per_trainable_parameter": aggregate[
                    "max_optimizer_update_peak_bytes_per_trainable_parameter"
                ],
                "mean_target_calls_per_output_token": aggregate[
                    "mean_target_calls_per_output_token"
                ],
                "mean_logical_block_target_calls_per_output_token": aggregate[
                    "mean_logical_block_target_calls_per_output_token"
                ],
                "mean_physical_target_calls_per_output_token": aggregate[
                    "mean_physical_target_calls_per_output_token"
                ],
                "mean_canonical_audit_target_calls_per_output_token": aggregate[
                    "mean_canonical_audit_target_calls_per_output_token"
                ],
                "target_call_metric_scope": TARGET_CALL_METRIC_SCOPE,
                "safe_for_selection": aggregate["safe_for_selection"],
                "pareto_optimal": not dominated,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["pareto_optimal"] is not True,
            int(row["max_whole_process_peak_hbm_bytes"]),
            -float(row["mean_paired_delta_A"]),
            str(row["candidate_id"]),
        ),
    )


def build_analysis(*, candidate_spec: Path, output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"output root is not a directory: {root}")
    candidate_spec_path, candidate_spec_relative = _relative_file_under_root(
        candidate_spec,
        root,
        "candidate specification copy",
    )
    sweep = calibration.load_candidate_sweep(candidate_spec_path)
    if sweep.kind != calibration.SPEC_KIND:
        raise ValueError("Stage-1 calibration analysis rejects diagnostic specifications")
    artifact_identity_lock = _load_artifact_identity_lock(root)
    bound: list[BoundRun] = []
    by_key: dict[tuple[int, str], BoundRun] = {}
    for sample in sweep.samples:
        for candidate_index, candidate in enumerate(sweep.candidates):
            run = _load_bound_run(
                root,
                artifact_identity_lock=artifact_identity_lock,
                sweep=sweep,
                candidate=candidate,
                candidate_index=candidate_index,
                sample=sample,
            )
            key = (run.sample_index, run.candidate_id)
            if key in by_key:
                raise ValueError(f"duplicate run evidence for {key}")
            by_key[key] = run
            bound.append(run)

    common_controls = {
        aggregation._sha256_json(_common_control_identity(run.identity))
        for run in bound
    }
    if len(common_controls) != 1:
        raise ValueError(
            "calibration runs disagree on a non-ablated global control identity"
        )
    parameterization_groups: dict[str, dict[str, Any]] = {}
    for run in bound:
        key, value = _parameterization_group(run.identity)
        previous = parameterization_groups.setdefault(key, value)
        if aggregation._canonical_json(previous) != aggregation._canonical_json(value):
            raise ValueError(
                f"calibration runs disagree on parameterization identity for {key}"
            )

    metric_rows = aggregation.build_long_table(
        [run.artifact_dir for run in bound], bucket_size=BUCKET_SIZE
    )
    by_root: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        run_root = str(Path(row["run_root"]).resolve())
        if run_root in by_root:
            raise ValueError(
                f"{run_root}: calibration run unexpectedly crossed the analysis bucket"
            )
        by_root[run_root] = row
    if len(by_root) != len(bound):
        raise ValueError("metric reconstruction did not return exactly one row per run")

    static_candidates = [
        candidate for candidate in sweep.candidates if candidate.mode == "static"
    ]
    if len(static_candidates) != 1:
        raise ValueError("candidate specification must contain exactly one Static")
    static_id = static_candidates[0].candidate_id
    candidate_rows = []
    for candidate in sweep.candidates:
        sample_results = []
        for sample in sweep.samples:
            sample_index = int(sample["sample_index"])
            run = by_key[(sample_index, candidate.candidate_id)]
            static = by_key[(sample_index, static_id)]
            row = by_root[str(run.artifact_dir.resolve())]
            static_row = by_root[str(static.artifact_dir.resolve())]
            sample_results.append(
                _sample_metrics(run, row, static=static, static_row=static_row)
            )
        candidate_rows.append(_candidate_row(candidate, sample_results))

    artifact_set = [
        {
            "sample_index": run.sample_index,
            "candidate_id": run.candidate_id,
            **run.evidence_hashes,
        }
        for run in sorted(bound, key=lambda item: (item.sample_index, item.candidate_id))
    ]
    artifact_identity_lock_record = {
        "path": artifact_identity_lock.relative_path,
        "file_sha256": artifact_identity_lock.file_sha256,
        "content_sha256": artifact_identity_lock.content_sha256,
    }
    source_artifact_set = {
        "artifact_identity_lock": artifact_identity_lock_record,
        "runs": artifact_set,
    }
    contract_sha256 = aggregation._sha256_json(SELECTION_RULE)
    decisions = _selection_groups(candidate_rows)
    pareto_rows = _pareto(candidate_rows)
    implementation = {
        "analyzer": {
            "file": Path(__file__).name,
            "sha256": aggregation._sha256_file(Path(__file__)),
        },
        "metric_aggregator": {
            "file": Path(str(aggregation.__file__)).name,
            "sha256": aggregation._sha256_file(Path(str(aggregation.__file__))),
        },
        "calibration_orchestrator": {
            "file": Path(str(calibration.__file__)).name,
            "sha256": aggregation._sha256_file(Path(str(calibration.__file__))),
        },
        "frozen_run_validator": {
            "file": Path(str(calibration.frozen.__file__)).name,
            "sha256": aggregation._sha256_file(
                Path(str(calibration.frozen.__file__))
            ),
        },
    }
    execution_runtime = _required_dict(
        bound[0].identity.get("runtime"), "identity.runtime"
    )
    execution_implementation = {
        key: _required_dict(
            execution_runtime.get(key), f"identity.runtime.{key}"
        )
        for key in ("calibration_orchestrator", "frozen_run_validator", "harness")
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete",
        "study_id": sweep.study_id,
        "evidence_scope": calibration.EVIDENCE_SCOPE,
        "candidate_specification": {
            "path": candidate_spec_relative,
            **_spec_identity(sweep),
        },
        "artifact_identity_lock": artifact_identity_lock_record,
        "sample_indices": [int(sample["sample_index"]) for sample in sweep.samples],
        "A_semantics": SELECTION_RULE["A_semantics"],
        "target_call_metric_scope": TARGET_CALL_METRIC_SCOPE,
        "selection_rule": SELECTION_RULE,
        "selection_rule_sha256": contract_sha256,
        "analysis_implementation": implementation,
        "analysis_implementation_sha256": aggregation._sha256_json(
            implementation
        ),
        "execution_implementation": execution_implementation,
        "execution_implementation_sha256": aggregation._sha256_json(
            execution_implementation
        ),
        "common_control_identity_sha256": next(iter(common_controls)),
        "parameterization_group_sha256": {
            key: aggregation._sha256_json(value)
            for key, value in sorted(parameterization_groups.items())
        },
        "candidate_rows": candidate_rows,
        "candidate_rows_sha256": aggregation._sha256_json(candidate_rows),
        "selection_decisions": decisions,
        "selection_decisions_sha256": aggregation._sha256_json(decisions),
        "pareto": {
            "axes": {
                "maximize": "mean_paired_delta_A",
                "minimize": [
                    "max_whole_process_peak_hbm_bytes",
                    "trainable_parameter_count",
                ],
            },
            "includes_unsafe_exact_candidates": True,
            "rows": pareto_rows,
            "rows_sha256": aggregation._sha256_json(pareto_rows),
        },
        "source_artifact_count": len(artifact_set) * 4 + 1,
        "source_run_count": len(artifact_set),
        "source_artifact_set_sha256": aggregation._sha256_json(
            source_artifact_set
        ),
        "source_runs": artifact_set,
        "limitations": [
            "This selects within an explicit bounded calibration grid only.",
            "Two locked development prompts do not establish statistical significance.",
            "Long-context and throughput claims require separate held-out evaluation.",
            "Canonical reference-verifier physical calls are audit overhead, not deployment throughput.",
        ],
        "analysis_hash_scheme": "canonical_json_without_analysis_sha256_v1",
    }
    payload["analysis_sha256"] = aggregation._sha256_json(payload)
    return payload


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        # Match frozen._write_json_exclusive's json.dump defaults exactly.
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _render_bytes(payload: dict[str, Any]) -> bytes:
    return _render(payload).encode("utf-8")


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable analysis artifact; use --check to revalidate it."""

    expected = _render_bytes(payload)
    calibration.frozen._write_json_exclusive(path, payload)
    if path.read_bytes() != expected:
        raise RuntimeError(
            "exclusive analysis writer serialization disagrees with _render"
        )


def _analysis_path_under_output_root(path: Path, output_root: Path) -> Path:
    root = output_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved.parent != root:
        raise ValueError(
            "analysis --output/--check must live directly under output_root"
        )
    return resolved


def verify_published_analysis(
    *,
    candidate_spec: Path,
    output_root: Path,
    analysis_path: Path,
) -> tuple[dict[str, Any], str]:
    """Rebuild and byte-verify one published schema-v3 analysis artifact."""

    path = _analysis_path_under_output_root(analysis_path, output_root)
    if not path.is_file():
        raise ValueError(f"calibration analysis is not a file: {path}")
    payload = build_analysis(
        candidate_spec=candidate_spec,
        output_root=output_root,
    )
    if path.read_bytes() != _render_bytes(payload):
        raise ValueError(f"calibration analysis is stale: {path}")
    return payload, aggregation._sha256_file(path)


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
    if args.check is not None:
        check_path = _analysis_path_under_output_root(
            args.check,
            args.output_root,
        )
        _verified, file_sha256 = verify_published_analysis(
            candidate_spec=args.candidate_spec,
            output_root=args.output_root,
            analysis_path=check_path,
        )
        print(file_sha256)
        return 0
    payload = build_analysis(
        candidate_spec=args.candidate_spec,
        output_root=args.output_root,
    )
    body = _render(payload)
    if args.output is not None:
        output_path = _analysis_path_under_output_root(
            args.output,
            args.output_root,
        )
        _write_exclusive(output_path, payload)
        print(aggregation._sha256_file(output_path))
        return 0
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
