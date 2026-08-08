#!/usr/bin/env python3
"""Run the frozen six-mode DFlash/TTS long-context sweep.

This is a deliberately thin orchestrator around ``dflash_tts_reference.py``.
It freezes the selected optimizer table, closes the context arithmetic exactly,
and resumes only byte-bound complete artifacts.  Incomplete attempts are moved
atomically into a monotonically numbered quarantine before the same logical run
is retried; completed evidence is never moved or overwritten.  Machine-specific
paths are CLI inputs; no credentials or remote-host assumptions live here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


SCHEMA_VERSION = 3
HARNESS_ARTIFACT_SCHEMA_VERSION = 3
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
COMMAND_SHA256_SCHEME = "canonical_json_harness_argv_without_digest_v1"
FAILED_ATTEMPT_SCHEMA_VERSION = 1
SUPPORTED_TOTAL_CONTEXTS = (8192, 16384, 32768, 40960)
MODE_ORDER = (
    "static",
    "full-drafter",
    "drafter-lora",
    "full-rank-tail",
    "tail-lora",
    "output-residual",
)
TOKENIZER_ARTIFACT_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
)
SCHEMA_V3_SELECTION_KIND = "schema_v3_stage1_stage2_frozen_selection"
SCHEMA_V3_STAGE1_KIND = "dflash_tts_schema_v3_calibration_analysis"
SCHEMA_V3_STAGE2_KIND = "dflash_tts_lora_rank_stage2_analysis"
MODE_PARAMETER_SCOPES = {
    "static": "frozen_static",
    "full-drafter": "drafter_all_trainable_parameters",
    "drafter-lora": "drafter_allowlisted_lora_parameters",
    "full-rank-tail": "cache_safe_full_rank_tail_only",
    "tail-lora": "cache_safe_tail_lora_only",
    "output-residual": "cache_safe_output_residual_only",
}


@dataclass(frozen=True)
class ModeConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    rank: int | None


@dataclass(frozen=True)
class RunPlan:
    run_dir: Path
    artifact_dir: Path
    log_path: Path
    identity_path: Path
    completion_path: Path
    identity: dict[str, Any]
    identity_sha256: str
    command: tuple[str, ...]
    pythonpath: tuple[str, ...]
    artifact_identity_lock_path: Path
    artifact_identity_lock_payload: dict[str, Any]

    def plan_payload(self) -> dict[str, Any]:
        determinism = self.identity["runtime"]["determinism"]
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "identity_sha256": self.identity_sha256,
            "command": list(self.command),
            "environment": {
                "PYTHONPATH": list(self.pythonpath),
                "CUBLAS_WORKSPACE_CONFIG": determinism[
                    "cublas_workspace_config"
                ],
            },
            "artifacts": {
                "summary": "artifact/summary.json",
                "rounds": "artifact/rounds.jsonl",
                "log": "run.log",
            },
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def determinism_contract(enabled: bool) -> dict[str, Any]:
    """Numerical settings frozen into every formal run identity."""

    return {
        "enabled": bool(enabled),
        "cublas_workspace_config": (
            DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG if enabled else None
        ),
        "torch_deterministic_algorithms": bool(enabled),
        "torch_deterministic_warn_only": False,
        "cuda_matmul_allow_tf32": not enabled,
        "cudnn_allow_tf32": not enabled,
        "float32_matmul_precision": "highest" if enabled else "high",
        "cudnn_benchmark": not enabled,
        "cudnn_deterministic": bool(enabled),
        "sdpa_backends": {
            "flash": not enabled,
            "memory_efficient": not enabled,
            "math": True,
            "cudnn": not enabled,
        },
    }


def _signed_command_sha256(command: Sequence[str]) -> str:
    """Validate and return the self-excluding harness argv digest."""

    positions = [
        index for index, value in enumerate(command) if value == "--command-sha256"
    ]
    if len(positions) != 1:
        raise ValueError("run command must contain one --command-sha256")
    position = positions[0]
    if position + 1 >= len(command):
        raise ValueError("run command --command-sha256 lacks a value")
    unsigned_harness_argv = [
        *command[1:position],
        *command[position + 2 :],
    ]
    observed = _sha256_json(unsigned_harness_argv)
    expected = command[position + 1]
    if expected != observed:
        raise ValueError(
            "run command sha256 mismatch: "
            f"expected {expected}, observed {observed}"
        )
    return observed


def _expected_run_attestation(plan: RunPlan) -> dict[str, str]:
    positions = [
        index
        for index, value in enumerate(plan.command)
        if value == "--run-identity-sha256"
    ]
    if len(positions) != 1 or positions[0] + 1 >= len(plan.command):
        raise ValueError("run command must contain one identity sha256")
    _expect(
        plan.command[positions[0] + 1],
        plan.identity_sha256,
        "run command identity sha256",
    )
    return {
        "status": "runner_bound",
        "scheme": COMMAND_SHA256_SCHEME,
        "run_identity_sha256": plan.identity_sha256,
        "command_sha256": _signed_command_sha256(plan.command),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exclusive_json_file_sha256(value: dict[str, Any]) -> str:
    body = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _mode_config(value: Any, *, label: str, mode: str) -> ModeConfig:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected = {"optimizer", "learning_rate", "weight_decay", "rank"}
    if set(value) != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}, got {sorted(value)}"
        )
    optimizer = value["optimizer"]
    if not isinstance(optimizer, str) or optimizer.lower() not in {"adam", "adamw"}:
        raise ValueError(f"{label}.optimizer must be adam or adamw")
    learning_rate = value["learning_rate"]
    weight_decay = value["weight_decay"]
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise ValueError(f"{label}.learning_rate must be numeric")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise ValueError(f"{label}.weight_decay must be numeric")
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError(f"{label} has invalid learning rate or weight decay")
    rank = value["rank"]
    requires_rank = mode in {"drafter-lora", "tail-lora", "output-residual"}
    if requires_rank:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"{label}.rank must be a positive integer")
    elif rank is not None:
        raise ValueError(f"{label}.rank must be null for {mode}")
    return ModeConfig(optimizer.lower(), learning_rate, weight_decay, rank)


def _selection_config_key(config: ModeConfig) -> tuple[Any, ...]:
    return (
        config.optimizer,
        config.learning_rate,
        config.weight_decay,
        -1 if config.rank is None else config.rank,
    )


def _validate_optimizer_selection_summary(
    summary_path: Path,
    *,
    selection_payload: dict[str, Any],
    configs: dict[str, ModeConfig],
) -> dict[str, Any]:
    """Validate that the locked table is the deterministic 2K winner table."""

    summary = _read_json(summary_path)
    if summary.get("schema_version") != 1:
        raise ValueError("optimizer selection summary schema_version must be 1")
    if summary.get("status") != "complete_local_calibration_selection":
        raise ValueError("optimizer selection summary is not complete")
    if summary.get("study_id") != selection_payload.get("study_id"):
        raise ValueError("optimizer selection summary study_id mismatch")
    if summary.get("evidence_classification") != selection_payload.get(
        "evidence_scope"
    ):
        raise ValueError("optimizer selection summary evidence scope mismatch")
    calibration = summary.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("sample_ids") != (
        selection_payload["calibration"]["sample_ids"]
    ):
        raise ValueError("optimizer selection summary calibration samples mismatch")
    candidate_grid = selection_payload["candidate_grid"]
    selection_rule = selection_payload["selection_rule"]
    if summary.get("candidate_grid") != candidate_grid:
        raise ValueError("optimizer selection summary candidate_grid mismatch")
    if summary.get("candidate_grid_sha256") != _sha256_json(candidate_grid):
        raise ValueError("optimizer selection summary candidate_grid hash mismatch")
    if summary.get("selection_rule") != selection_rule:
        raise ValueError("optimizer selection summary selection_rule mismatch")
    if summary.get("selection_rule_sha256") != _sha256_json(selection_rule):
        raise ValueError("optimizer selection summary selection_rule hash mismatch")

    selected = summary.get("selected_configs")
    if not isinstance(selected, dict) or set(selected) != set(MODE_ORDER):
        raise ValueError("optimizer selection summary selected_configs mismatch")
    for mode in MODE_ORDER:
        observed = _mode_config(
            selected[mode],
            label=f"optimizer selection summary selected_configs.{mode}",
            mode=mode,
        )
        if observed != configs[mode]:
            raise ValueError(
                f"optimizer selection summary selected_configs.{mode} mismatch"
            )

    stages = summary.get("stages")
    if not isinstance(stages, dict) or set(stages) != {
        "512_screening",
        "2048_selection",
    }:
        raise ValueError("optimizer selection summary stages mismatch")
    expected_samples = set(selection_payload["calibration"]["sample_ids"])
    artifact_hashes: dict[str, str] = {}

    def artifact_pair(value: Any, *, label: str) -> None:
        if not isinstance(value, dict) or set(value) != {
            "summary_path",
            "summary_sha256",
            "rounds_path",
            "rounds_sha256",
        }:
            raise ValueError(f"{label} must record summary/rounds paths and hashes")
        for kind in ("summary", "rounds"):
            path_value = value[f"{kind}_path"]
            digest = value[f"{kind}_sha256"]
            if not isinstance(path_value, str) or not path_value:
                raise ValueError(f"{label} has an invalid {kind} path")
            if not _is_sha256(digest):
                raise ValueError(f"{label} has an invalid {kind} SHA256")
            previous = artifact_hashes.setdefault(path_value, digest)
            if previous != digest:
                raise ValueError(f"{label} assigns conflicting hashes to {path_value}")

    observed_2048: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in MODE_ORDER[1:]
    }
    for stage_name, expected_context in (
        ("512_screening", 512),
        ("2048_selection", 2048),
    ):
        stage = stages[stage_name]
        if not isinstance(stage, dict) or stage.get("context_length") != expected_context:
            raise ValueError(f"optimizer selection summary {stage_name} is invalid")
        baselines = stage.get("baselines")
        candidates = stage.get("observed_candidates")
        if not isinstance(baselines, list) or not isinstance(candidates, list):
            raise ValueError(f"optimizer selection summary {stage_name} inventory is invalid")
        if {row.get("sample_id") for row in baselines if isinstance(row, dict)} != expected_samples:
            raise ValueError(f"optimizer selection summary {stage_name} baselines mismatch")
        for index, baseline in enumerate(baselines):
            if not isinstance(baseline, dict):
                raise ValueError(f"{stage_name} baseline {index} must be an object")
            artifact_pair(
                baseline.get("artifacts"), label=f"{stage_name} baseline {index}"
            )
        candidate_ids: set[str] = set()
        for index, candidate in enumerate(candidates):
            label = f"{stage_name} candidate {index}"
            if not isinstance(candidate, dict):
                raise ValueError(f"{label} must be an object")
            candidate_id = candidate.get("candidate_id")
            mode = candidate.get("mode")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"{label} has an invalid candidate_id")
            if candidate_id in candidate_ids:
                raise ValueError(f"{stage_name} has duplicate candidate_id {candidate_id}")
            candidate_ids.add(candidate_id)
            if mode not in MODE_ORDER[1:]:
                raise ValueError(f"{label} has invalid mode {mode!r}")
            candidate_config = _mode_config(
                candidate.get("config"), label=f"{label}.config", mode=mode
            )
            grid = candidate_grid[mode]
            if (
                candidate_config.optimizer not in grid["optimizers"]
                or candidate_config.learning_rate not in grid["learning_rates"]
                or candidate_config.weight_decay not in grid["weight_decays"]
                or candidate_config.rank not in grid["ranks"]
            ):
                raise ValueError(f"{label}.config is outside candidate_grid")
            sample_results = candidate.get("sample_results")
            if not isinstance(sample_results, list) or not sample_results:
                raise ValueError(f"{label}.sample_results must be non-empty")
            for result_index, result in enumerate(sample_results):
                if not isinstance(result, dict):
                    raise ValueError(f"{label}.sample_results[{result_index}] is invalid")
                artifact_pair(
                    result.get("artifacts"),
                    label=f"{label}.sample_results[{result_index}]",
                )
            if stage_name == "2048_selection":
                observed_2048[mode].append(candidate)

    if summary.get("source_artifact_count") != len(artifact_hashes):
        raise ValueError("optimizer selection summary source artifact count mismatch")
    artifact_set = [
        {"path": path, "sha256": artifact_hashes[path]}
        for path in sorted(artifact_hashes)
    ]
    if summary.get("source_artifact_set_sha256") != _sha256_json(artifact_set):
        raise ValueError("optimizer selection summary source artifact set hash mismatch")

    decisions = summary.get("selection_decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(MODE_ORDER[1:]):
        raise ValueError("optimizer selection summary selection_decisions mismatch")
    for mode in MODE_ORDER[1:]:
        eligible: list[tuple[tuple[Any, ...], dict[str, Any], ModeConfig]] = []
        for candidate in observed_2048[mode]:
            sample_results = candidate["sample_results"]
            if {row.get("sample_id") for row in sample_results} != expected_samples:
                raise ValueError(f"2048 candidate {candidate['candidate_id']} is not paired")
            gains: list[float] = []
            peak_hbm: list[int] = []
            parameter_counts: list[int] = []
            exact_and_finite = True
            for result in sample_results:
                numeric_names = (
                    "candidate_paper_acceptance_length",
                    "static_paper_acceptance_length",
                    "paired_paper_acceptance_length_gain",
                )
                numeric = [result.get(name) for name in numeric_names]
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in numeric
                ):
                    raise ValueError(
                        f"2048 candidate {candidate['candidate_id']} has nonfinite AL evidence"
                    )
                expected_gain = float(numeric[0]) - float(numeric[1])
                if not math.isclose(
                    float(numeric[2]), expected_gain, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"2048 candidate {candidate['candidate_id']} has inconsistent paired AL gain"
                    )
                exact = result.get("exact_output_token_ids") is True
                candidate_output = result.get("candidate_output_token_ids_sha256")
                static_output = result.get("static_output_token_ids_sha256")
                if not _is_sha256(candidate_output) or not _is_sha256(static_output):
                    raise ValueError(
                        f"2048 candidate {candidate['candidate_id']} has invalid output hash"
                    )
                if exact != (candidate_output == static_output):
                    raise ValueError(
                        f"2048 candidate {candidate['candidate_id']} exactness/hash mismatch"
                    )
                finite = result.get("finite_metrics") is True
                loss = result.get("update_loss")
                if not isinstance(loss, dict) or loss.get("all_finite") is not True:
                    finite = False
                hbm = result.get("peak_hbm_bytes")
                parameters = result.get("trainable_parameter_count")
                if (
                    isinstance(hbm, bool)
                    or not isinstance(hbm, int)
                    or hbm <= 0
                    or isinstance(parameters, bool)
                    or not isinstance(parameters, int)
                    or parameters < 0
                ):
                    raise ValueError(
                        f"2048 candidate {candidate['candidate_id']} has invalid memory evidence"
                    )
                exact_and_finite &= exact and finite
                gains.append(float(numeric[2]))
                peak_hbm.append(hbm)
                parameter_counts.append(parameters)
            aggregate = candidate.get("aggregate")
            if not isinstance(aggregate, dict):
                raise ValueError(
                    f"2048 candidate {candidate['candidate_id']} has no aggregate"
                )
            expected_aggregate = {
                "complete_two_prompt_pair": True,
                "eligible": exact_and_finite,
                "mean_paired_paper_acceptance_length_gain": sum(gains) / len(gains),
                "worst_prompt_paired_paper_acceptance_length_gain": min(gains),
                "max_peak_hbm_bytes": max(peak_hbm),
                "max_trainable_parameter_count": max(parameter_counts),
            }
            for key, expected in expected_aggregate.items():
                observed = aggregate.get(key)
                if isinstance(expected, float):
                    matches = isinstance(observed, (int, float)) and math.isclose(
                        float(observed), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                else:
                    matches = observed == expected
                if not matches:
                    raise ValueError(
                        f"2048 candidate {candidate['candidate_id']} aggregate {key} mismatch"
                    )
            if exact_and_finite:
                config = _mode_config(
                    candidate["config"],
                    label=f"2048 candidate {candidate['candidate_id']}.config",
                    mode=mode,
                )
                order = (
                    -expected_aggregate[
                        "mean_paired_paper_acceptance_length_gain"
                    ],
                    -expected_aggregate[
                        "worst_prompt_paired_paper_acceptance_length_gain"
                    ],
                    expected_aggregate["max_peak_hbm_bytes"],
                    expected_aggregate["max_trainable_parameter_count"],
                    *_selection_config_key(config),
                )
                eligible.append((order, candidate, config))
        if not eligible:
            raise ValueError(f"optimizer selection summary has no eligible {mode} candidate")
        eligible.sort(key=lambda value: value[0])
        _, winner, winner_config = eligible[0]
        decision = decisions[mode]
        if (
            not isinstance(decision, dict)
            or decision.get("winner_candidate_id") != winner["candidate_id"]
            or winner_config != configs[mode]
        ):
            raise ValueError(
                f"optimizer selection summary deterministic winner mismatch for {mode}"
            )
    return summary


def _load_optimizer_selection(path_value: str) -> dict[str, Any]:
    path = _required_path(
        path_value, directory=False, label="selected optimizer config"
    )
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("selected optimizer config schema_version must be 1")
    status = payload.get("status")
    if status not in {"locked", "provisional_pilot"}:
        raise ValueError(
            "selected optimizer config status must be locked or provisional_pilot"
        )
    evidence_scope = payload.get("evidence_scope")
    if evidence_scope != (
        "legacy_schema_v1_optimizer_screening_only_not_formal_identity_runtime_"
        "hbm_or_speed_evidence"
    ):
        raise ValueError(
            "selected optimizer config must label legacy optimizer-screening-only "
            "evidence scope"
        )
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("selected optimizer config calibration must be an object")
    sample_ids = calibration.get("sample_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != 2
        or len(set(sample_ids)) != 2
        or any(not isinstance(value, str) or not value for value in sample_ids)
    ):
        raise ValueError(
            "selected optimizer config requires exactly two distinct calibration sample IDs"
        )
    candidate_grid = payload.get("candidate_grid")
    locked = payload.get("locked_configs")
    if not isinstance(candidate_grid, dict) or set(candidate_grid) != set(MODE_ORDER):
        raise ValueError("candidate_grid must cover every frozen sweep mode exactly")
    if not isinstance(locked, dict) or set(locked) != set(MODE_ORDER):
        raise ValueError("locked_configs must cover every frozen sweep mode exactly")
    configs: dict[str, ModeConfig] = {}
    required_axes = {
        "optimizers",
        "learning_rates",
        "weight_decays",
        "ranks",
    }
    for mode in MODE_ORDER:
        grid = candidate_grid[mode]
        if not isinstance(grid, dict) or set(grid) != required_axes:
            raise ValueError(
                f"candidate_grid.{mode} keys must be exactly {sorted(required_axes)}"
            )
        if any(not isinstance(grid[axis], list) or not grid[axis] for axis in required_axes):
            raise ValueError(f"candidate_grid.{mode} axes must be non-empty lists")
        selected = _mode_config(
            locked[mode], label=f"locked_configs.{mode}", mode=mode
        )
        membership = {
            "optimizer": selected.optimizer in [str(item).lower() for item in grid["optimizers"]],
            "learning_rate": selected.learning_rate in [float(item) for item in grid["learning_rates"]],
            "weight_decay": selected.weight_decay in [float(item) for item in grid["weight_decays"]],
            "rank": selected.rank in grid["ranks"],
        }
        missing_axes = [axis for axis, present in membership.items() if not present]
        if missing_axes:
            raise ValueError(
                f"locked_configs.{mode} is outside candidate_grid axes: {missing_axes}"
            )
        configs[mode] = selected
    selection_rule = payload.get("selection_rule")
    required_rule = {"eligibility", "primary_metric", "aggregation", "tie_breakers"}
    if not isinstance(selection_rule, dict) or not required_rule.issubset(selection_rule):
        raise ValueError(
            "selection_rule must record eligibility, primary_metric, aggregation, and tie_breakers"
        )
    evidence = payload.get("evidence_artifacts")
    if not isinstance(evidence, list):
        raise ValueError("evidence_artifacts must be a list")
    if status == "locked" and not evidence:
        raise ValueError("locked optimizer selection requires evidence_artifacts")
    normalized_evidence: list[dict[str, str]] = []
    for index, artifact in enumerate(evidence):
        if not isinstance(artifact, dict):
            raise ValueError(
                f"evidence_artifacts[{index}] must be an object"
            )
        if set(artifact) != {"kind", "path", "sha256"}:
            raise ValueError(
                f"evidence_artifacts[{index}] keys must be kind, path, sha256"
            )
        kind = artifact.get("kind")
        if kind != "deterministic_selection_summary":
            raise ValueError(
                f"evidence_artifacts[{index}] has unsupported kind {kind!r}"
            )
        artifact_path_value = artifact.get("path")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(artifact_path_value, str) or not artifact_path_value:
            raise ValueError(f"evidence_artifacts[{index}] must contain a path")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
        ):
            raise ValueError(
                f"evidence_artifacts[{index}] must contain a lowercase SHA256"
            )
        requested = Path(artifact_path_value).expanduser()
        if not requested.is_absolute():
            requested = path.parent / requested
        artifact_path = _required_path(
            str(requested),
            directory=False,
            label=f"optimizer selection evidence artifact {index}",
        )
        observed_sha256 = _sha256_file(artifact_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"evidence_artifacts[{index}] SHA256 mismatch: expected "
                f"{expected_sha256}, got {observed_sha256}"
            )
        normalized_evidence.append(
            {
                "kind": kind,
                "path": str(artifact_path),
                "sha256": observed_sha256,
            }
        )
    if status == "locked":
        summaries = [
            artifact
            for artifact in normalized_evidence
            if artifact["kind"] == "deterministic_selection_summary"
        ]
        if len(summaries) != 1:
            raise ValueError(
                "locked optimizer selection requires exactly one deterministic "
                "selection summary"
            )
        _validate_optimizer_selection_summary(
            Path(summaries[0]["path"]),
            selection_payload=payload,
            configs=configs,
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "status": status,
        "study_id": payload.get("study_id"),
        "evidence_scope": evidence_scope,
        "calibration_sample_ids": list(sample_ids),
        "selection_rule_sha256": _sha256_json(selection_rule),
        "candidate_grid_sha256": _sha256_json(candidate_grid),
        "evidence_artifacts": normalized_evidence,
        "configs": configs,
    }


def _required_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _required_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _validate_analysis_envelope(
    payload: dict[str, Any], *, kind: str, label: str
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")
    if payload.get("kind") != kind or payload.get("status") != "complete":
        raise ValueError(f"{label} is not a complete {kind} artifact")
    if payload.get("analysis_hash_scheme") != (
        "canonical_json_without_analysis_sha256_v1"
    ):
        raise ValueError(f"{label} analysis hash scheme is unsupported")
    unsigned = dict(payload)
    observed = unsigned.pop("analysis_sha256", None)
    if not _is_sha256(observed) or _sha256_json(unsigned) != observed:
        raise ValueError(f"{label} analysis_sha256 mismatch")


def _validate_embedded_hash(
    payload: dict[str, Any], value_key: str, hash_key: str, *, label: str
) -> None:
    value = payload.get(value_key)
    digest = payload.get(hash_key)
    if not _is_sha256(digest) or _sha256_json(value) != digest:
        raise ValueError(f"{label} {hash_key} mismatch")


def _find_stage2_candidate_spec(
    analysis_path: Path, payload: dict[str, Any]
) -> Path:
    core = _required_object(
        _get(payload, "source_attestation.portable_evidence_core"),
        "Stage-2 portable evidence core",
    )
    identity = _required_object(
        core.get("stage2_candidate_specification"),
        "Stage-2 candidate specification identity",
    )
    expected_file = identity.get("file_sha256")
    expected_content = identity.get("content_sha256")
    if not _is_sha256(expected_file) or not _is_sha256(expected_content):
        raise ValueError("Stage-2 candidate specification hashes are invalid")
    matches = []
    for candidate in sorted(analysis_path.parent.iterdir()):
        if not candidate.is_file() or candidate == analysis_path:
            continue
        if _sha256_file(candidate) != expected_file:
            continue
        document = _read_json(candidate)
        if _sha256_json(document) != expected_content:
            raise ValueError(
                "Stage-2 candidate specification file/content hash mismatch"
            )
        matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ValueError(
            "Stage-2 analysis must resolve exactly one adjacent candidate "
            f"specification, observed {len(matches)}"
        )
    return matches[0]


def _verify_schema_v3_analysis_closure(
    *,
    stage1_path: Path,
    stage1: dict[str, Any],
    stage2_path: Path,
    stage2: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild both analyses, including every source run and hash closure."""

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import analyze_dflash_tts_calibration as stage1_analyzer
    import analyze_dflash_tts_lora_rank as stage2_analyzer

    stage1_spec_record = _required_object(
        stage1.get("candidate_specification"),
        "Stage-1 candidate specification",
    )
    locator = stage1_spec_record.get("path")
    if not isinstance(locator, str) or not locator:
        raise ValueError("Stage-1 candidate specification path is missing")
    stage1_spec = Path(locator).expanduser()
    if not stage1_spec.is_absolute():
        stage1_spec = stage1_path.parent / stage1_spec
    stage1_spec = stage1_spec.resolve()
    verified_stage1, observed_stage1_file = (
        stage1_analyzer.verify_published_analysis(
            candidate_spec=stage1_spec,
            output_root=stage1_path.parent,
            analysis_path=stage1_path,
        )
    )
    _expect(verified_stage1, stage1, "rebuilt Stage-1 analysis")
    _expect(
        observed_stage1_file,
        _sha256_file(stage1_path),
        "rebuilt Stage-1 analysis file hash",
    )

    stage2_spec = _find_stage2_candidate_spec(stage2_path, stage2)
    verified_stage2, observed_stage2_file = (
        stage2_analyzer.verify_published_analysis(
            candidate_spec=stage2_spec,
            output_root=stage2_path.parent,
            analysis_path=stage2_path,
        )
    )
    _expect(verified_stage2, stage2, "rebuilt Stage-2 analysis")
    _expect(
        observed_stage2_file,
        _sha256_file(stage2_path),
        "rebuilt Stage-2 analysis file hash",
    )
    return {
        "stage1_candidate_specification": {
            "path": str(stage1_spec),
            "file_sha256": _sha256_file(stage1_spec),
            "content_sha256": _sha256_json(_read_json(stage1_spec)),
        },
        "stage2_candidate_specification": {
            "path": str(stage2_spec),
            "file_sha256": _sha256_file(stage2_spec),
            "content_sha256": _sha256_json(_read_json(stage2_spec)),
        },
    }


def _candidate_row(
    payload: dict[str, Any], candidate_id: str, *, label: str
) -> dict[str, Any]:
    matches = [
        row
        for row in _required_list(payload.get("candidate_rows"), f"{label} rows")
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label} candidate {candidate_id!r} is missing or duplicated"
        )
    return matches[0]


def _config_from_selection_row(
    row: dict[str, Any], *, mode: str, label: str
) -> ModeConfig:
    if row.get("mode") != mode:
        raise ValueError(f"{label} mode mismatch for {mode}")
    return _mode_config(
        {
            "optimizer": row.get("optimizer"),
            "learning_rate": row.get("learning_rate"),
            "weight_decay": row.get("weight_decay"),
            "rank": row.get("rank"),
        },
        label=label,
        mode=mode,
    )


def _safe_stage1_winner(
    payload: dict[str, Any], *, mode: str
) -> tuple[ModeConfig, dict[str, Any], int | None]:
    decisions = [
        value
        for value in _required_list(
            payload.get("selection_decisions"), "Stage-1 selection decisions"
        )
        if isinstance(value, dict) and value.get("mode") == mode
    ]
    if len(decisions) != 1:
        raise ValueError(
            f"Stage-1 must contain exactly one decision for {mode}, "
            f"observed {len(decisions)}"
        )
    decision = decisions[0]
    winner = decision.get("winner")
    if (
        decision.get("status") != "local_grid_winner"
        or not isinstance(winner, dict)
        or not isinstance(decision.get("safe_candidate_count"), int)
        or decision["safe_candidate_count"] <= 0
    ):
        raise ValueError(f"Stage-1 has no safe winner for {mode}")
    candidate_id = winner.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"Stage-1 {mode} winner candidate_id is invalid")
    row = _candidate_row(payload, candidate_id, label="Stage-1")
    aggregate = _required_object(row.get("aggregate"), f"Stage-1 {mode} aggregate")
    if (
        aggregate.get("evidence_eligible") is not True
        or aggregate.get("safe_for_selection") is not True
        or aggregate.get("all_outputs_exact_static") is not True
        or aggregate.get("all_losses_and_gradients_finite") is not True
        or aggregate.get("ineligibility_reasons") != []
    ):
        raise ValueError(f"Stage-1 winner for {mode} fails the safety gate")
    for key in ("optimizer", "learning_rate", "weight_decay", "rank"):
        _expect(row.get(key), winner.get(key), f"Stage-1 {mode} winner {key}")
    boundary = _required_object(
        winner.get("learning_rate_boundary"),
        f"Stage-1 {mode} learning-rate boundary",
    )
    if boundary.get("requires_grid_extension_before_optimum_claim") is not False:
        raise ValueError(
            f"Stage-1 {mode} winner is on a learning-rate boundary; extend the "
            "grid before freezing long-context runs"
        )
    config = _config_from_selection_row(
        row, mode=mode, label=f"Stage-1 winner {mode}"
    )
    adapter_seed = row.get("adapter_seed")
    if config.rank is not None and (
        isinstance(adapter_seed, bool) or not isinstance(adapter_seed, int)
    ):
        raise ValueError(f"Stage-1 {mode} adapter seed is invalid")
    return config, {
        "source": "stage1",
        "candidate_id": candidate_id,
        "boundary": boundary,
        "parameter_scope": MODE_PARAMETER_SCOPES[mode],
    }, adapter_seed if config.rank is not None else None


def _safe_stage2_winner(
    payload: dict[str, Any], *, mode: str
) -> tuple[ModeConfig, dict[str, Any], int]:
    decisions = [
        value
        for value in _required_list(
            _get(payload, "comparisons.tuned_envelope"),
            "Stage-2 tuned-envelope decisions",
        )
        if isinstance(value, dict) and value.get("mode") == mode
    ]
    if len(decisions) != 1:
        raise ValueError(
            f"Stage-2 must contain exactly one tuned-envelope decision for "
            f"{mode}, observed {len(decisions)}"
        )
    decision = decisions[0]
    winner = decision.get("winner")
    if decision.get("status") != "bounded_rank_winner" or not isinstance(
        winner, dict
    ):
        raise ValueError(f"Stage-2 has no safe winner for {mode}")
    prompts = _required_list(
        winner.get("prompt_safety"), f"Stage-2 {mode} prompt safety"
    )
    if not prompts or any(
        not isinstance(item, dict) or item.get("safe_nonnegative") is not True
        for item in prompts
    ):
        raise ValueError(f"Stage-2 winner for {mode} fails the prompt safety gate")
    aggregate = _required_object(
        winner.get("aggregate"), f"Stage-2 {mode} aggregate"
    )
    if (
        aggregate.get("evidence_eligible") is not True
        or aggregate.get("safe_for_selection") is not True
        or aggregate.get("all_outputs_exact_static") is not True
        or aggregate.get("all_losses_and_gradients_finite") is not True
        or aggregate.get("ineligibility_reasons") != []
    ):
        raise ValueError(f"Stage-2 winner for {mode} fails the safety gate")
    rank_boundary = _required_object(
        decision.get("rank_boundary"), f"Stage-2 {mode} rank boundary"
    )
    if rank_boundary.get(
        "requires_rank_grid_extension_before_optimum_claim"
    ) is not False:
        raise ValueError(
            f"Stage-2 {mode} winner is on a rank boundary; extend the grid "
            "before freezing long-context runs"
        )
    if decision.get("requires_lr_grid_extension_before_optimum_claim") is not False:
        raise ValueError(
            f"Stage-2 {mode} winner is on a learning-rate boundary; extend the "
            "grid before freezing long-context runs"
        )
    candidate_id = winner.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"Stage-2 {mode} winner candidate_id is invalid")
    row = _candidate_row(payload, candidate_id, label="Stage-2")
    for key in ("mode", "optimizer", "learning_rate", "weight_decay", "rank"):
        _expect(row.get(key), winner.get(key), f"Stage-2 {mode} winner {key}")
    _expect(row.get("aggregate"), aggregate, f"Stage-2 {mode} winner aggregate")
    config = _config_from_selection_row(
        row, mode=mode, label=f"Stage-2 winner {mode}"
    )
    adapter_seed = winner.get("adapter_seed")
    if isinstance(adapter_seed, bool) or not isinstance(adapter_seed, int):
        raise ValueError(f"Stage-2 {mode} adapter seed is invalid")
    return config, {
        "source": "stage2_tuned_envelope",
        "candidate_id": candidate_id,
        "rank_boundary": rank_boundary,
        "learning_rate_boundary": decision.get(
            "winner_learning_rate_boundary"
        ),
        "parameter_scope": MODE_PARAMETER_SCOPES[mode],
    }, adapter_seed


def _load_schema_v3_selection(
    stage1_path_value: str, stage2_path_value: str
) -> dict[str, Any]:
    stage1_path = _required_path(
        stage1_path_value, directory=False, label="Stage-1 analysis"
    )
    stage2_path = _required_path(
        stage2_path_value, directory=False, label="Stage-2 analysis"
    )
    stage1 = _read_json(stage1_path)
    stage2 = _read_json(stage2_path)
    _validate_analysis_envelope(
        stage1, kind=SCHEMA_V3_STAGE1_KIND, label="Stage-1 analysis"
    )
    _validate_analysis_envelope(
        stage2, kind=SCHEMA_V3_STAGE2_KIND, label="Stage-2 analysis"
    )
    for value_key, hash_key in (
        ("candidate_rows", "candidate_rows_sha256"),
        ("selection_decisions", "selection_decisions_sha256"),
    ):
        _validate_embedded_hash(
            stage1, value_key, hash_key, label="Stage-1 analysis"
        )
    for value_key, hash_key in (
        ("candidate_rows", "candidate_rows_sha256"),
        ("comparisons", "comparisons_sha256"),
        ("mode_omissions", "mode_omissions_sha256"),
        ("pareto", "pareto_sha256"),
        ("source_attestation", "source_attestation_sha256"),
    ):
        _validate_embedded_hash(
            stage2, value_key, hash_key, label="Stage-2 analysis"
        )
    if stage1.get("sample_indices") != stage2.get("sample_indices"):
        raise ValueError("Stage-1 and Stage-2 calibration samples mismatch")
    if stage2.get("mode_omissions") != []:
        raise ValueError(
            "Stage-2 must cover both drafter-lora and tail-lora; omissions "
            "cannot produce a six-mode frozen sweep"
        )

    stage1_spec = _required_object(
        stage1.get("candidate_specification"),
        "Stage-1 candidate specification",
    )
    stage1_lock = _required_object(
        stage1.get("artifact_identity_lock"), "Stage-1 artifact identity lock"
    )
    binding = _required_object(
        _get(
            stage2,
            "source_attestation.locator_bound_provenance.stage1_binding",
        ),
        "Stage-2 Stage-1 binding",
    )
    expected_binding = {
        "analysis_file_sha256": _sha256_file(stage1_path),
        "analysis_sha256": stage1["analysis_sha256"],
        "selection_decisions_sha256": stage1["selection_decisions_sha256"],
        "source_artifact_set_sha256": stage1.get(
            "source_artifact_set_sha256"
        ),
        "candidate_specification_file_sha256": stage1_spec.get("file_sha256"),
        "candidate_specification_content_sha256": stage1_spec.get(
            "content_sha256"
        ),
        "source_study_id": stage1_spec.get("study_id"),
        "artifact_identity_lock_file_sha256": stage1_lock.get("file_sha256"),
        "artifact_identity_lock_content_sha256": stage1_lock.get(
            "content_sha256"
        ),
    }
    for key, expected in expected_binding.items():
        _expect(binding.get(key), expected, f"Stage-2 Stage-1 binding {key}")

    closure = _verify_schema_v3_analysis_closure(
        stage1_path=stage1_path,
        stage1=stage1,
        stage2_path=stage2_path,
        stage2=stage2,
    )
    static_rows = [
        row
        for row in _required_list(stage1.get("candidate_rows"), "Stage-1 rows")
        if isinstance(row, dict) and row.get("mode") == "static"
    ]
    if len(static_rows) != 1:
        raise ValueError("Stage-1 must contain exactly one Static candidate")
    configs: dict[str, ModeConfig] = {
        "static": _config_from_selection_row(
            static_rows[0], mode="static", label="Stage-1 Static"
        )
    }
    selected_candidates: dict[str, dict[str, Any]] = {
        "static": {
            "source": "stage1_baseline",
            "candidate_id": static_rows[0].get("candidate_id"),
            "parameter_scope": MODE_PARAMETER_SCOPES["static"],
        }
    }
    adapter_seeds: dict[str, int | None] = {"static": None}
    for mode in ("full-drafter", "full-rank-tail", "output-residual"):
        config, source, adapter_seed = _safe_stage1_winner(stage1, mode=mode)
        configs[mode] = config
        selected_candidates[mode] = source
        adapter_seeds[mode] = adapter_seed
    for mode in ("drafter-lora", "tail-lora"):
        config, source, adapter_seed = _safe_stage2_winner(stage2, mode=mode)
        configs[mode] = config
        selected_candidates[mode] = source
        adapter_seeds[mode] = adapter_seed
    configs = {mode: configs[mode] for mode in MODE_ORDER}
    selected_candidates = {
        mode: selected_candidates[mode] for mode in MODE_ORDER
    }
    adapter_seeds = {mode: adapter_seeds[mode] for mode in MODE_ORDER}
    source_identity = {
        "kind": SCHEMA_V3_SELECTION_KIND,
        "stage1_analysis": {
            "path": str(stage1_path),
            "file_sha256": _sha256_file(stage1_path),
            "analysis_sha256": stage1["analysis_sha256"],
            "selection_decisions_sha256": stage1[
                "selection_decisions_sha256"
            ],
            "candidate_rows_sha256": stage1["candidate_rows_sha256"],
            "source_artifact_set_sha256": stage1.get(
                "source_artifact_set_sha256"
            ),
        },
        "stage2_analysis": {
            "path": str(stage2_path),
            "file_sha256": _sha256_file(stage2_path),
            "analysis_sha256": stage2["analysis_sha256"],
            "comparisons_sha256": stage2["comparisons_sha256"],
            "candidate_rows_sha256": stage2["candidate_rows_sha256"],
            "source_attestation_sha256": stage2[
                "source_attestation_sha256"
            ],
        },
        "analysis_closure": closure,
        "stage1_to_stage2_binding_sha256": _sha256_json(binding),
        "selected_candidates": selected_candidates,
        "adapter_seeds": adapter_seeds,
        "boundary_gate": {
            "status": "passed",
            "rule": (
                "every frozen Stage-1 LR and Stage-2 tuned-envelope LR/rank "
                "winner is safe, exact, finite, and interior to its tested grid"
            ),
        },
        "mode_parameter_scopes": MODE_PARAMETER_SCOPES,
    }
    return {
        "path": str(stage1_path),
        "sha256": _sha256_file(stage1_path),
        "status": "locked",
        "kind": SCHEMA_V3_SELECTION_KIND,
        "study_id": _sha256_json(
            {
                "stage1": stage1["analysis_sha256"],
                "stage2": stage2["analysis_sha256"],
            }
        ),
        "evidence_scope": (
            "schema_v3_exact_paired_optimizer_and_rank_selection_for_frozen_"
            "long_context_evaluation"
        ),
        "calibration_sample_ids": list(stage1.get("sample_indices", [])),
        "selection_rule_sha256": _sha256_json(
            {
                "stage1": stage1.get("selection_rule_sha256"),
                "stage2": stage2.get("selection_rule_sha256"),
                "boundary_gate": source_identity["boundary_gate"],
            }
        ),
        "candidate_grid_sha256": _sha256_json(
            {
                "stage1": stage1["candidate_rows_sha256"],
                "stage2": stage2["candidate_rows_sha256"],
            }
        ),
        "evidence_artifacts": [
            {
                "kind": "schema_v3_stage1_analysis",
                "path": str(stage1_path),
                "sha256": _sha256_file(stage1_path),
            },
            {
                "kind": "schema_v3_stage2_rank_analysis",
                "path": str(stage2_path),
                "sha256": _sha256_file(stage2_path),
            },
        ],
        "schema_v3_selection": source_identity,
        "adapter_seeds": adapter_seeds,
        "configs": configs,
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Publish complete JSON atomically without replacing an existing file."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link is both atomic and no-clobber: readers
        # see either no final path or the complete fsynced inode.  Unlike
        # os.replace(), it preserves the immutable-artifact contract under a
        # concurrent writer.
        os.link(temporary_path, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class ImmutableCompletionError(ValueError):
    """A completion claim exists but no longer validates byte-for-byte."""


class RetryableRunError(RuntimeError):
    """A failed attempt was preserved and the logical run may be retried."""


def _lexists(path: Path) -> bool:
    """Like ``Path.exists`` but do not hide a dangling symlink collision."""

    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_lock_path(plan: RunPlan) -> Path:
    return plan.run_dir.with_name(f".{plan.run_dir.name}.run.lock")


@contextmanager
def _exclusive_run_lock(plan: RunPlan) -> Iterator[int]:
    """Hold one advisory lock for the entire logical run attempt.

    The descriptor is inherited by the harness.  If the orchestrator is killed
    while the harness survives, a resumed queue therefore cannot quarantine a
    directory that the surviving child is still writing.
    """

    lock_path = _run_lock_path(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise RuntimeError(
                f"logical run is already active: {plan.run_dir}"
            ) from exc
        yield descriptor
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _completion_claim_exists(plan: RunPlan) -> bool:
    return _lexists(plan.completion_path)


def _quarantine_root(plan: RunPlan) -> Path:
    return plan.run_dir.with_name(f"{plan.run_dir.name}.quarantine")


def _archive_failed_attempt(plan: RunPlan, *, reason: str) -> Path:
    """Atomically preserve an incomplete run under ``quarantine/attempt-N``.

    Numbered attempt directories are created exclusively and never reused.  A
    completion claim, even an invalid one, is deliberately not moved: callers
    must investigate it instead of risking replacement of successful evidence.
    """

    if not _lexists(plan.run_dir):
        raise FileNotFoundError(
            f"no failed run directory to archive: {plan.run_dir}"
        )
    if _completion_claim_exists(plan):
        raise ImmutableCompletionError(
            f"{plan.run_dir}: completion evidence exists; refusing to archive "
            "or overwrite it"
        )

    root = _quarantine_root(plan)
    if _lexists(root):
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"quarantine root is not a real directory: {root}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(root.parent)

    attempt_dir: Path | None = None
    for attempt_number in range(1, 1_000_000):
        candidate = root / f"attempt-{attempt_number:04d}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        attempt_dir = candidate
        break
    if attempt_dir is None:  # pragma: no cover - operational exhaustion guard
        raise RuntimeError(f"quarantine attempt namespace exhausted: {root}")

    metadata = {
        "schema_version": FAILED_ATTEMPT_SCHEMA_VERSION,
        "archived_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "reason": reason,
        "logical_run_dir": str(plan.run_dir),
        "expected_run_identity_sha256": plan.identity_sha256,
        "evidence_path": "run",
    }
    evidence_path = attempt_dir / "run"
    try:
        os.rename(plan.run_dir, evidence_path)
    except Exception:
        attempt_dir.rmdir()
        raise
    _fsync_directory(attempt_dir)
    _fsync_directory(plan.run_dir.parent)
    # The directory rename is the preservation primitive.  Metadata is useful
    # provenance, but a disk-full condition must not undo or mask an already
    # successful archive.  Its absence is therefore an explicit, recoverable
    # state: ``run`` still contains the byte-for-byte failed evidence.
    try:
        _write_json_exclusive(attempt_dir / "attempt.json", metadata)
    except OSError:
        pass
    return evidence_path


def _prepare_existing_run(plan: RunPlan) -> bool:
    """Resume a valid final artifact or archive a retryable partial attempt."""

    if not _lexists(plan.run_dir):
        return False
    try:
        return completed_run_matches(plan)
    except (OSError, ValueError) as exc:
        if _completion_claim_exists(plan):
            raise ImmutableCompletionError(
                f"{plan.run_dir}: immutable completion evidence failed "
                "validation; refusing to archive or overwrite it"
            ) from exc
        _archive_failed_attempt(
            plan,
            reason="incomplete_or_invalid_attempt_detected_before_retry",
        )
        return False


def _preserve_retryable_failure(
    plan: RunPlan,
    error: Exception,
    *,
    reason: str,
) -> RetryableRunError:
    """Archive the current failed attempt and build an actionable exception."""

    try:
        evidence_path = _archive_failed_attempt(plan, reason=reason)
    except Exception as archive_error:
        return RetryableRunError(
            f"{error}; additionally failed to preserve the partial attempt: "
            f"{archive_error}"
        )
    return RetryableRunError(
        f"{error}; failed evidence preserved at {evidence_path}; rerun the same "
        "logical plan to create the next numbered attempt"
    )


def _required_path(
    value: str,
    *,
    directory: bool,
    label: str,
    resolve_symlinks: bool = True,
) -> Path:
    absolute = Path(os.path.abspath(Path(value).expanduser()))
    checked = absolute.resolve() if resolve_symlinks else absolute
    if not (checked.is_dir() if directory else checked.is_file()):
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} is not a local {kind}: {checked}")
    return checked


def _get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing field {path}")
        current = current[key]
    return current


def _expect(observed: Any, expected: Any, label: str) -> None:
    if _canonical_json(observed) != _canonical_json(expected):
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, got {observed!r}"
        )


def _check_fields(
    document: dict[str, Any], checks: Sequence[tuple[str, Any]], *, label: str
) -> None:
    for path, expected in checks:
        _expect(_get(document, path), expected, f"{label} {path}")


def _validate_runtime_fingerprint(
    summary: dict[str, Any], identity: dict[str, Any]
) -> str:
    fingerprint = _get(summary, "runtime_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("summary runtime_fingerprint must be an object")
    required = {
        "schema_version",
        "python_version",
        "python_implementation",
        "platform",
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "attention_implementation",
        "dtype",
        "device",
        "resolved_device",
        "allocator_config",
        "cuda_visible_devices",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "allow_tf32",
        "float32_matmul_precision",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cublas_workspace_config",
        "determinism_contract",
        "sdpa_backends",
        "gpu",
    }
    missing = sorted(required - set(fingerprint))
    if missing:
        raise ValueError(f"summary runtime_fingerprint lacks {missing}")
    _expect(fingerprint["schema_version"], 1, "runtime fingerprint schema")
    for key in (
        "python_version",
        "python_implementation",
        "platform",
        "torch_version",
        "float32_matmul_precision",
    ):
        if not isinstance(fingerprint[key], str) or not fingerprint[key]:
            raise ValueError(f"runtime fingerprint {key} must be a non-empty string")
    for key in ("attention_implementation", "dtype", "device"):
        _expect(
            fingerprint[key],
            {
                "attention_implementation": identity["runtime"][
                    "attention_implementation"
                ],
                "dtype": identity["runtime"]["dtype"],
                "device": identity["runtime"]["device"],
            }[key],
            f"runtime fingerprint {key}",
        )
    requested_device = str(fingerprint["device"])
    resolved_device = fingerprint["resolved_device"]
    if not isinstance(resolved_device, str) or not resolved_device:
        raise ValueError("runtime fingerprint resolved_device is invalid")
    if requested_device.split(":", 1)[0] != resolved_device.split(":", 1)[0]:
        raise ValueError("runtime fingerprint requested/resolved device type mismatch")
    if ":" in requested_device and resolved_device != requested_device:
        raise ValueError("runtime fingerprint resolved device index mismatch")
    for key in (
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    ):
        if not isinstance(fingerprint[key], bool):
            raise ValueError(f"runtime fingerprint {key} must be boolean")
    if fingerprint["deterministic_warn_only"] is not None and not isinstance(
        fingerprint["deterministic_warn_only"], bool
    ):
        raise ValueError(
            "runtime fingerprint deterministic_warn_only must be boolean or null"
        )
    allow_tf32 = fingerprint["allow_tf32"]
    if (
        not isinstance(allow_tf32, dict)
        or set(allow_tf32) != {"matmul", "cudnn"}
        or any(not isinstance(value, bool) for value in allow_tf32.values())
    ):
        raise ValueError("runtime fingerprint allow_tf32 is invalid")
    allocator = fingerprint["allocator_config"]
    if not isinstance(allocator, dict) or set(allocator) != {
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
    }:
        raise ValueError("runtime fingerprint allocator_config is invalid")
    _expect(
        allocator,
        identity["runtime"].get(
            "allocator_config",
            {
                "PYTORCH_CUDA_ALLOC_CONF": None,
                "PYTORCH_ALLOC_CONF": None,
            },
        ),
        "runtime fingerprint allocator config",
    )
    planned_determinism = identity["runtime"]["determinism"]
    _expect(
        fingerprint["determinism_contract"],
        planned_determinism,
        "runtime fingerprint determinism contract",
    )
    _expect(
        fingerprint["cublas_workspace_config"],
        planned_determinism["cublas_workspace_config"],
        "runtime fingerprint CUBLAS_WORKSPACE_CONFIG",
    )
    _expect(
        fingerprint["deterministic_algorithms"],
        planned_determinism["torch_deterministic_algorithms"],
        "runtime fingerprint deterministic algorithms",
    )
    _expect(
        fingerprint["deterministic_warn_only"],
        planned_determinism["torch_deterministic_warn_only"],
        "runtime fingerprint deterministic warn-only",
    )
    _expect(
        fingerprint["allow_tf32"],
        {
            "matmul": planned_determinism["cuda_matmul_allow_tf32"],
            "cudnn": planned_determinism["cudnn_allow_tf32"],
        },
        "runtime fingerprint TF32",
    )
    _expect(
        fingerprint["sdpa_backends"],
        planned_determinism["sdpa_backends"],
        "runtime fingerprint SDPA backends",
    )
    for fingerprint_key, contract_key in (
        ("float32_matmul_precision", "float32_matmul_precision"),
        ("cudnn_benchmark", "cudnn_benchmark"),
        ("cudnn_deterministic", "cudnn_deterministic"),
    ):
        _expect(
            fingerprint[fingerprint_key],
            planned_determinism[contract_key],
            f"runtime fingerprint {fingerprint_key}",
        )
    device_type = resolved_device.split(":", 1)[0]
    gpu = fingerprint["gpu"]
    if device_type == "cuda":
        if not isinstance(fingerprint["cuda_runtime_version"], str):
            raise ValueError("CUDA run lacks CUDA runtime version")
        if fingerprint["cuda_driver_version"] is not None and not isinstance(
            fingerprint["cuda_driver_version"], (int, str)
        ):
            raise ValueError("CUDA driver version must be integer, string, or null")
        if not isinstance(gpu, dict):
            raise ValueError("CUDA run lacks GPU fingerprint")
        for key in ("name", "compute_capability"):
            if not isinstance(gpu.get(key), str) or not gpu[key]:
                raise ValueError(f"runtime fingerprint gpu.{key} is invalid")
        if (
            isinstance(gpu.get("total_memory_bytes"), bool)
            or not isinstance(gpu.get("total_memory_bytes"), int)
            or gpu["total_memory_bytes"] <= 0
        ):
            raise ValueError("runtime fingerprint GPU memory is invalid")
    elif gpu is not None:
        raise ValueError("non-CUDA run must record gpu as null")
    return _sha256_json(fingerprint)


def _validate_parity_result(
    summary: dict[str, Any], identity: dict[str, Any]
) -> None:
    parity = _get(summary, "reference.official_static_parity")
    if not isinstance(parity, dict):
        raise ValueError("summary official static parity must be an object")
    audit = identity["audit"]
    if audit["skip_static_parity_preflight"]:
        _expect(
            parity,
            {"status": "skipped_by_explicit_cli"},
            "summary skipped static parity result",
        )
        return
    legacy_keys = {
        "status",
        "max_new_tokens",
        "official_acceptance_lengths",
        "policies",
    }
    current_keys = legacy_keys | {"classification", "official_policy"}
    if set(parity) not in (legacy_keys, current_keys) or (
        parity.get("status") != "passed"
    ):
        raise ValueError("summary static parity result is not a complete pass")
    expected_tokens = min(
        identity["generation"]["max_new_tokens"],
        audit["parity_max_new_tokens"],
    )
    _expect(
        parity["max_new_tokens"],
        expected_tokens,
        "summary static parity max_new_tokens",
    )
    official = parity["official_acceptance_lengths"]
    if not isinstance(official, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in official
    ):
        raise ValueError("summary official parity acceptance lengths are invalid")
    policies = parity["policies"]
    is_current = "classification" in parity
    expected_policies = {"stale"} if is_current else {"stale", "rebuild"}
    if is_current and (
        parity["classification"]
        != "official_stale_cache_block_verifier_reconstruction"
        or parity["official_policy"] != "stale"
    ):
        raise ValueError("summary static parity classification is invalid")
    if not isinstance(policies, dict) or set(policies) != expected_policies:
        raise ValueError("summary static parity policies are incomplete")
    for policy_name, policy in policies.items():
        if not isinstance(policy, dict) or set(policy) != {
            "output_ids_match",
            "acceptance_lengths_match",
            "acceptance_lengths",
        }:
            raise ValueError(f"summary static parity {policy_name} is invalid")
        if policy["output_ids_match"] is not True or (
            policy["acceptance_lengths_match"] is not True
        ):
            raise ValueError(f"summary static parity {policy_name} did not pass")
        _expect(
            policy["acceptance_lengths"],
            official,
            f"summary static parity {policy_name} acceptance lengths",
        )


def max_new_tokens_for_total_context(
    *, total_context: int, input_tokens: int, draft_block_size: int
) -> int:
    """Close ``input + max_new + (block - 1) == total`` exactly."""

    if total_context <= 0 or input_tokens <= 0:
        raise ValueError("total context and input token count must be positive")
    if draft_block_size <= 1:
        raise ValueError("draft block size must be greater than one")
    pending = draft_block_size - 1  # verifier seed is already in the prefix
    max_new_tokens = total_context - input_tokens - pending
    if max_new_tokens <= 0:
        raise ValueError(
            "requested context leaves no generation budget: "
            f"total={total_context}, input={input_tokens}, pending_draft={pending}"
        )
    if input_tokens + max_new_tokens + pending != total_context:
        raise AssertionError("context arithmetic did not close exactly")
    return max_new_tokens


def _hashed_file_identity(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _model_identity(path: Path, revision: str) -> dict[str, Any]:
    identity: dict[str, Any] = {"path": str(path), "revision": revision}
    for name in ("config.json", "model.safetensors.index.json"):
        artifact = path / name
        if artifact.is_file():
            identity[f"{name}_sha256"] = _sha256_file(artifact)
    identity["weight_files"] = [
        _hashed_file_identity(item)
        for item in sorted(path.glob("*.safetensors"))
    ]
    if not identity["weight_files"]:
        raise ValueError(f"model has no local safetensors weights: {path}")
    identity["content_identity_sha256"] = _sha256_json(
        {
            key: value
            for key, value in identity.items()
            if key not in {"path", "revision"}
        }
    )
    return identity


def _tokenizer_identity(path: Path) -> dict[str, Any]:
    files = [
        _hashed_file_identity(path / name)
        for name in TOKENIZER_ARTIFACT_FILES
        if (path / name).is_file()
    ]
    if not files:
        raise ValueError(f"target model has no recognized tokenizer files: {path}")
    return {
        "path": str(path),
        "files": files,
        "content_identity_sha256": _sha256_json(files),
    }


def _artifact_file_stats(
    *, target: Path, draft: Path
) -> dict[str, list[dict[str, Any]]]:
    def collect(root: Path, names: Sequence[str]) -> list[dict[str, Any]]:
        rows = []
        for name in names:
            candidate = root / name
            if candidate.is_file():
                stat = candidate.stat()
                rows.append(
                    {
                        "name": name,
                        "bytes": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
        return rows

    return {
        "target": collect(
            target,
            (
                "config.json",
                "model.safetensors.index.json",
                *(item.name for item in sorted(target.glob("*.safetensors"))),
            ),
        ),
        "draft": collect(
            draft,
            (
                "config.json",
                "model.safetensors.index.json",
                *(item.name for item in sorted(draft.glob("*.safetensors"))),
            ),
        ),
        "tokenizer": collect(target, TOKENIZER_ARTIFACT_FILES),
    }


def _build_or_load_artifact_identity_lock(
    path: Path,
    *,
    target: Path,
    target_revision: str,
    draft: Path,
    draft_revision: str,
) -> dict[str, Any]:
    """Hash once per sweep, then reuse only while immutable stat guards match."""

    current_stats = _artifact_file_stats(target=target, draft=draft)
    if path.is_file():
        lock = _read_json(path)
        if lock.get("schema_version") != 1 or lock.get("kind") != (
            "dflash_tts_artifact_identity_lock"
        ):
            raise ValueError(f"{path}: unsupported artifact identity lock")
        if lock.get("file_stats") != current_stats:
            # Do not silently bless mutated files or rewrite evidence used by
            # completed runs. Hashing is intentionally skipped on the normal
            # unchanged resume path.
            raise ValueError(
                "model/tokenizer files changed since the immutable artifact "
                f"identity lock was created: {path}"
            )
        if (
            _get(lock, "target.path") != str(target)
            or _get(lock, "target.revision") != target_revision
            or _get(lock, "draft.path") != str(draft)
            or _get(lock, "draft.revision") != draft_revision
            or _get(lock, "tokenizer.path") != str(target)
        ):
            raise ValueError(f"{path}: artifact identity lock selection mismatch")
        for role in ("target", "draft"):
            model_identity = lock[role]
            weights = model_identity.get("weight_files")
            if not isinstance(weights, list) or not weights or any(
                not isinstance(item, dict)
                or not _is_sha256(item.get("sha256"))
                for item in weights
            ):
                raise ValueError(f"{path}: {role} shard identity is incomplete")
            expected_content_sha256 = _sha256_json(
                {
                    key: value
                    for key, value in model_identity.items()
                    if key
                    not in {"path", "revision", "content_identity_sha256"}
                }
            )
            _expect(
                model_identity.get("content_identity_sha256"),
                expected_content_sha256,
                f"{role} content identity",
            )
        tokenizer_identity = lock["tokenizer"]
        tokenizer_files = tokenizer_identity.get("files")
        if not isinstance(tokenizer_files, list) or not tokenizer_files or any(
            not isinstance(item, dict)
            or not _is_sha256(item.get("sha256"))
            for item in tokenizer_files
        ):
            raise ValueError(f"{path}: tokenizer identity is incomplete")
        _expect(
            tokenizer_identity.get("content_identity_sha256"),
            _sha256_json(tokenizer_files),
            "tokenizer content identity",
        )
        return lock
    if path.exists():
        raise ValueError(f"artifact identity lock is not a file: {path}")
    return {
        "schema_version": 1,
        "kind": "dflash_tts_artifact_identity_lock",
        "target": _model_identity(target, target_revision),
        "draft": _model_identity(draft, draft_revision),
        "tokenizer": _tokenizer_identity(target),
        "file_stats": current_stats,
    }


def _projection_identity(
    value: str, *, target_identity: dict[str, Any]
) -> dict[str, str]:
    requested = Path(value).expanduser()
    payload_name = requested if str(requested).endswith(".npz") else Path(
        str(requested) + ".npz"
    )
    payload = _required_path(
        str(payload_name), directory=False, label="projection payload"
    )
    metadata = _required_path(
        str(payload) + ".meta.json",
        directory=False,
        label="projection metadata sidecar",
    )
    metadata_payload = _read_json(metadata)
    bound_target = _get(metadata_payload, "binding.target_head_artifact")
    if not isinstance(bound_target, dict):
        raise ValueError("projection metadata lacks target LM-head binding")
    _expect(
        bound_target.get("weight_files"),
        target_identity["weight_files"],
        "projection target weight shard identity",
    )
    if any(
        not isinstance(item, dict) or len(str(item.get("sha256", ""))) != 64
        for item in target_identity["weight_files"]
    ):
        raise ValueError("projection target shard identity is not content-bound")
    return {
        "path": str(payload),
        "sha256": _sha256_file(payload),
        "metadata_path": str(metadata),
        "metadata_sha256": _sha256_file(metadata),
    }


def _preflight_input_tokens(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset_sha256: str,
    tokenizer_identity: dict[str, Any],
) -> tuple[int, str]:
    summary = _read_json(path)
    _check_fields(
        summary,
        (
            ("dataset.sample_index", args.sample_index),
            ("dataset.sha256", dataset_sha256),
            ("dataset.declared_revision", args.dataset_revision),
            ("models.target.declared_revision", args.target_revision),
            ("models.draft.declared_revision", args.draft_revision),
            (
                "tokenizer.content_identity_sha256",
                tokenizer_identity["content_identity_sha256"],
            ),
            ("reference.declared_revision", args.reference_revision),
            ("parameters.block_size", args.draft_block_size),
            ("parameters.enable_thinking", args.enable_thinking),
            ("parameters.prompt_field", args.prompt_field),
            ("parameters.messages_field", args.messages_field),
            ("parameters.turns_field", args.turns_field),
        ),
        label="preflight",
    )
    count = _get(summary, "generation.num_input_tokens")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("preflight input token count must be a positive integer")
    rendered_sha256 = _get(
        summary, "dataset.rendered_input_token_ids.sha256"
    )
    if not isinstance(rendered_sha256, str) or len(rendered_sha256) != 64:
        raise ValueError("preflight rendered input token ids sha256 is invalid")
    return count, rendered_sha256


def _command(
    identity: dict[str, Any],
    artifact_dir: Path,
    *,
    run_identity_sha256: str,
) -> tuple[str, ...]:
    runtime = identity["runtime"]
    reference = identity["reference"]
    dataset = identity["dataset"]
    generation = identity["generation"]
    optimization = identity["optimization"]
    command = [runtime["python"], runtime["harness"]["path"]]
    options = (
        ("mode", identity["mode"]),
        ("reference-root", reference["root"]),
        ("reference-module", reference["module"]),
        ("reference-revision", reference["revision"]),
        ("target-model", identity["target"]["path"]),
        ("target-revision", identity["target"]["revision"]),
        ("draft-model", identity["draft"]["path"]),
        ("draft-revision", identity["draft"]["revision"]),
        (
            "artifact-identity-lock",
            runtime["artifact_identity_lock"]["path"],
        ),
        (
            "artifact-identity-lock-sha256",
            runtime["artifact_identity_lock"]["sha256"],
        ),
        ("dataset", dataset["path"]),
        ("dataset-revision", dataset["revision"]),
        ("sample-index", dataset["sample_index"]),
        ("prompt-field", dataset["prompt_field"]),
        ("messages-field", dataset["messages_field"]),
        ("turns-field", dataset["turns_field"]),
        ("output-dir", artifact_dir),
        ("run-identity-sha256", run_identity_sha256),
        ("device", runtime["device"]),
        ("dtype", runtime["dtype"]),
        ("attn-implementation", runtime["attention_implementation"]),
        ("max-new-tokens", generation["max_new_tokens"]),
        ("block-size", generation["draft_block_size"]),
        ("stop-token-ids", "none"),
        ("temperature", generation["temperature"]),
        ("seed", generation["seed"]),
        ("lr", optimization["learning_rate"]),
        ("proximal-lambda", optimization["proximal_lambda"]),
        ("update-stride", optimization["update_stride"]),
        ("position-weighting", optimization["position_weighting"]),
        ("position-decay-gamma", optimization["position_decay_gamma"]),
        ("loss-reduction", optimization["loss_reduction"]),
        ("adam-beta1", optimization["adam_betas"][0]),
        ("adam-beta2", optimization["adam_betas"][1]),
        ("adam-eps", optimization["adam_eps"]),
        ("optimizer", optimization["optimizer"]),
        ("weight-decay", optimization["weight_decay"]),
        ("draft-cache-policy", optimization["draft_cache_policy"]),
        ("parity-max-new-tokens", identity["audit"]["parity_max_new_tokens"]),
        ("parameter-audit-stride", identity["audit"]["parameter_audit_stride"]),
        ("mask-token-id", generation["mask_token_id"]),
        ("rank", optimization["rank"]),
        ("adapter-seed", optimization["adapter_seed"]),
    )
    for name, value in options:
        if value is not None:
            command.extend((f"--{name}", str(value)))
    if identity["projection"] is not None:
        command.extend(("--projection-artifact", identity["projection"]["path"]))
    boolean_flags = (
        (dataset["enable_thinking"], "--enable-thinking"),
        (identity["audit"]["cuda_timing"], "--audit-cuda-timing"),
        (
            identity["audit"]["skip_static_parity_preflight"],
            "--skip-static-parity-preflight",
        ),
    )
    command.extend(flag for enabled, flag in boolean_flags if enabled)
    command.append(
        "--deterministic"
        if runtime["determinism"]["enabled"]
        else "--no-deterministic"
    )
    command_sha256 = _sha256_json(command[1:])
    command.extend(("--command-sha256", command_sha256))
    return tuple(command)


def build_run_plans(args: argparse.Namespace) -> list[RunPlan]:
    if (args.stage1_analysis is None) != (args.stage2_analysis is None):
        raise ValueError(
            "--stage1-analysis and --stage2-analysis must be supplied together"
        )
    if args.stage1_analysis is not None and args.selected_optimizer_config is not None:
        raise ValueError(
            "schema-v3 analyses and --selected-optimizer-config are mutually "
            "exclusive selection sources"
        )
    if args.stage1_analysis is not None:
        optimizer_selection = _load_schema_v3_selection(
            args.stage1_analysis, args.stage2_analysis
        )
    else:
        if args.selected_optimizer_config is None:
            raise ValueError(
                "provide --selected-optimizer-config or both schema-v3 "
                "analysis artifacts; result-derived defaults are forbidden"
            )
        optimizer_selection = _load_optimizer_selection(
            args.selected_optimizer_config
        )
    optimizer_selection_identity = {
        key: value
        for key, value in optimizer_selection.items()
        if key != "configs"
    }
    harness = _required_path(args.harness, directory=False, label="harness")
    # Resolving a venv's bin/python symlink can silently select global Python.
    python = _required_path(
        args.python,
        directory=False,
        label="Python executable",
        resolve_symlinks=False,
    )
    if not os.access(python, os.X_OK):
        raise ValueError(f"Python executable is not executable: {python}")
    reference_root = _required_path(
        args.reference_root, directory=True, label="reference root"
    )
    reference_source = reference_root.joinpath(
        *args.reference_module.split(".")
    ).with_suffix(".py")
    reference_source = _required_path(
        str(reference_source), directory=False, label="reference module source"
    )
    target = _required_path(args.target_model, directory=True, label="target model")
    draft = _required_path(args.draft_model, directory=True, label="draft model")
    dataset = _required_path(args.dataset, directory=False, label="dataset")
    dataset_sha256 = _sha256_file(dataset)
    output_root = Path(args.output_root).expanduser().resolve()
    pythonpath = tuple(
        str(_required_path(item, directory=True, label="PYTHONPATH entry"))
        for item in args.pythonpath
    )
    artifact_identity_lock_path = output_root / "artifact_identity_lock.json"
    artifact_identity_lock = _build_or_load_artifact_identity_lock(
        artifact_identity_lock_path,
        target=target,
        target_revision=args.target_revision,
        draft=draft,
        draft_revision=args.draft_revision,
    )
    target_identity = artifact_identity_lock["target"]
    draft_identity = artifact_identity_lock["draft"]
    tokenizer_identity = artifact_identity_lock["tokenizer"]
    artifact_identity_lock_sha256 = _exclusive_json_file_sha256(
        artifact_identity_lock
    )
    orchestrator_path = Path(__file__).resolve()
    frozen_implementation = {
        "path": str(orchestrator_path),
        "sha256": _sha256_file(orchestrator_path),
    }

    if len(args.total_contexts) != len(set(args.total_contexts)):
        raise ValueError("total context list contains duplicates")
    unsupported = sorted(set(args.total_contexts) - set(SUPPORTED_TOTAL_CONTEXTS))
    if unsupported:
        raise ValueError(
            f"unsupported total contexts {unsupported}; locked sweep supports "
            f"{list(SUPPORTED_TOTAL_CONTEXTS)}"
        )
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("mode list contains duplicates")
    if "output-residual" in args.modes and args.projection_artifact is None:
        raise ValueError(
            "output-residual requires an existing --projection-artifact and "
            "metadata sidecar so both hashes can be frozen"
        )

    projection = (
        _projection_identity(
            args.projection_artifact, target_identity=target_identity
        )
        if args.projection_artifact is not None
        and "output-residual" in args.modes
        else None
    )

    if args.input_tokens is not None:
        input_tokens = args.input_tokens
        if input_tokens <= 0:
            raise ValueError("--input-tokens must be positive")
        rendered_input_token_ids_sha256 = None
        token_source = {"kind": "explicit_cli", "value": input_tokens}
    else:
        preflight = _required_path(
            args.preflight_summary, directory=False, label="preflight summary"
        )
        input_tokens, rendered_input_token_ids_sha256 = _preflight_input_tokens(
            preflight,
            args=args,
            dataset_sha256=dataset_sha256,
            tokenizer_identity=tokenizer_identity,
        )
        token_source = {
            "kind": "validated_preflight_summary",
            "path": str(preflight),
            "sha256": _sha256_file(preflight),
            "value": input_tokens,
        }

    common = {
        "schema_version": SCHEMA_VERSION,
        "sweep": "dflash_tts_frozen_long_context_v3",
        "optimizer_selection": optimizer_selection_identity,
        "runtime": {
            "python": str(python),
            # Validation is intentionally implemented in this frozen
            # orchestrator.  Keep the two semantic roles explicit so a future
            # split remains an identity change rather than an invisible one.
            "frozen_orchestrator": dict(frozen_implementation),
            "frozen_run_validator": dict(frozen_implementation),
            "harness": {"path": str(harness), "sha256": _sha256_file(harness)},
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "allocator_config": {
                name: os.environ.get(name)
                for name in (
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "PYTORCH_ALLOC_CONF",
                )
            },
            "determinism": determinism_contract(args.deterministic),
            "pythonpath": list(pythonpath),
            "artifact_identity_lock": {
                "path": str(artifact_identity_lock_path),
                "sha256": artifact_identity_lock_sha256,
            },
        },
        "reference": {
            "root": str(reference_root),
            "module": args.reference_module,
            "revision": args.reference_revision,
            "source_path": str(reference_source),
            "source_sha256": _sha256_file(reference_source),
        },
        "target": target_identity,
        "draft": draft_identity,
        "tokenizer": tokenizer_identity,
        "dataset": {
            "path": str(dataset),
            "revision": args.dataset_revision,
            "sha256": dataset_sha256,
            "sample_index": args.sample_index,
            "prompt_field": args.prompt_field,
            "messages_field": args.messages_field,
            "turns_field": args.turns_field,
            "enable_thinking": args.enable_thinking,
            "input_tokens": input_tokens,
            "input_token_source": token_source,
            "rendered_input_token_ids_sha256": (
                rendered_input_token_ids_sha256
            ),
        },
        "audit": {
            "cuda_timing": args.audit_cuda_timing,
            "parameter_audit_stride": args.parameter_audit_stride,
            "parity_max_new_tokens": args.parity_max_new_tokens,
            "skip_static_parity_preflight": args.skip_static_parity_preflight,
        },
    }
    plans: list[RunPlan] = []
    for total_context in args.total_contexts:
        max_new = max_new_tokens_for_total_context(
            total_context=total_context,
            input_tokens=input_tokens,
            draft_block_size=args.draft_block_size,
        )
        for mode in args.modes:
            selected = optimizer_selection["configs"][mode]
            run_dir = (
                output_root
                / f"sample-{args.sample_index:04d}"
                / f"context-{total_context}"
                / mode
            )
            identity = {
                **common,
                "mode": mode,
                "generation": {
                    "requested_total_context": total_context,
                    "input_tokens": input_tokens,
                    "max_new_tokens": max_new,
                    "draft_block_size": args.draft_block_size,
                    "pending_draft_tokens": args.draft_block_size - 1,
                    "required_prefix_plus_block": total_context,
                    "stop_token_ids": None,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "mask_token_id": args.mask_token_id,
                },
                "optimization": {
                    "optimizer": selected.optimizer,
                    "learning_rate": selected.learning_rate,
                    "weight_decay": selected.weight_decay,
                    "rank": selected.rank,
                    "adapter_seed": (
                        optimizer_selection.get("adapter_seeds", {}).get(
                            mode, args.adapter_seed
                        )
                        if selected.rank is not None
                        else None
                    ),
                    "proximal_lambda": args.proximal_lambda,
                    "update_stride": args.update_stride,
                    "position_weighting": args.position_weighting,
                    "position_decay_gamma": args.position_decay_gamma,
                    "loss_reduction": args.loss_reduction,
                    "adam_betas": [args.adam_beta1, args.adam_beta2],
                    "adam_eps": args.adam_eps,
                    "draft_cache_policy": args.draft_cache_policy,
                },
                "projection": projection if mode == "output-residual" else None,
            }
            artifact_dir = run_dir / "artifact"
            identity_sha256 = _sha256_json(identity)
            plans.append(
                RunPlan(
                    run_dir=run_dir,
                    artifact_dir=artifact_dir,
                    log_path=run_dir / "run.log",
                    identity_path=run_dir / "run_identity.json",
                    completion_path=run_dir / "completion.json",
                    identity=identity,
                    identity_sha256=identity_sha256,
                    command=_command(
                        identity,
                        artifact_dir,
                        run_identity_sha256=identity_sha256,
                    ),
                    pythonpath=pythonpath,
                    artifact_identity_lock_path=artifact_identity_lock_path,
                    artifact_identity_lock_payload=artifact_identity_lock,
                )
            )
    return plans


def _validate_summary(plan: RunPlan) -> tuple[str, str, str, str]:
    summary_path = plan.artifact_dir / "summary.json"
    rounds_path = plan.artifact_dir / "rounds.jsonl"
    if not summary_path.is_file() or not rounds_path.is_file():
        raise ValueError(f"{plan.run_dir}: missing summary.json or rounds.jsonl")
    summary = _read_json(summary_path)
    identity = plan.identity
    generation = identity["generation"]
    optimization = identity["optimization"]
    effective_optimizer = (
        None
        if identity["mode"] == "static"
        else optimization["optimizer"].upper()
    )
    checks = (
        ("schema_version", HARNESS_ARTIFACT_SCHEMA_VERSION),
        ("status", "complete_reference_run"),
        ("mode", identity["mode"]),
        ("run_attestation", _expected_run_attestation(plan)),
        ("harness.source_sha256", identity["runtime"]["harness"]["sha256"]),
        ("reference.declared_revision", identity["reference"]["revision"]),
        ("reference.source_sha256", identity["reference"]["source_sha256"]),
        ("models.target.declared_revision", identity["target"]["revision"]),
        ("models.draft.declared_revision", identity["draft"]["revision"]),
        (
            "artifact_identity.verification_status",
            "fully_verified_content_sha256_v1",
        ),
        (
            "artifact_identity.lock.sha256",
            identity["runtime"]["artifact_identity_lock"]["sha256"],
        ),
        ("tokenizer", identity["tokenizer"]),
        ("dataset.declared_revision", identity["dataset"]["revision"]),
        ("dataset.sha256", identity["dataset"]["sha256"]),
        ("dataset.sample_index", identity["dataset"]["sample_index"]),
        ("generation.num_input_tokens", generation["input_tokens"]),
        ("generation.num_output_tokens", generation["max_new_tokens"]),
        ("parameters.max_new_tokens", generation["max_new_tokens"]),
        (
            "parameters.required_prefix_plus_block",
            generation["requested_total_context"],
        ),
        ("parameters.stop_token_ids", None),
        ("parameters.seed", generation["seed"]),
        (
            "parameters.deterministic",
            identity["runtime"]["determinism"]["enabled"],
        ),
        ("parameters.temperature", generation["temperature"]),
        ("parameters.block_size", generation["draft_block_size"]),
        ("parameters.mask_token_id", generation["mask_token_id"]),
        ("parameters.lr", optimization["learning_rate"]),
        ("parameters.optimizer", effective_optimizer),
        ("parameters.weight_decay", optimization["weight_decay"]),
        ("parameters.rank", optimization["rank"]),
        ("parameters.adapter_seed", optimization["adapter_seed"]),
        ("parameters.proximal_lambda", optimization["proximal_lambda"]),
        ("parameters.update_stride", optimization["update_stride"]),
        ("parameters.position_weighting", optimization["position_weighting"]),
        (
            "parameters.position_decay_gamma",
            optimization["position_decay_gamma"],
        ),
        ("parameters.loss_reduction", optimization["loss_reduction"]),
        ("parameters.adam_betas", optimization["adam_betas"]),
        ("parameters.adam_eps", optimization["adam_eps"]),
        ("parameters.draft_cache_policy", optimization["draft_cache_policy"]),
        ("parameters.dtype", identity["runtime"]["dtype"]),
        ("parameters.device", identity["runtime"]["device"]),
        ("parameters.enable_thinking", identity["dataset"]["enable_thinking"]),
        ("parameters.prompt_field", identity["dataset"]["prompt_field"]),
        ("parameters.messages_field", identity["dataset"]["messages_field"]),
        ("parameters.turns_field", identity["dataset"]["turns_field"]),
        (
            "parameters.projection_artifact",
            None
            if identity["projection"] is None
            else identity["projection"]["path"],
        ),
        ("parameters.audit_cuda_timing", identity["audit"]["cuda_timing"]),
        (
            "parameters.parameter_audit_stride",
            identity["audit"]["parameter_audit_stride"],
        ),
        (
            "parameters.parity_max_new_tokens",
            identity["audit"]["parity_max_new_tokens"],
        ),
        (
            "parameters.skip_static_parity_preflight",
            identity["audit"]["skip_static_parity_preflight"],
        ),
    )
    _check_fields(summary, checks, label="summary")
    _validate_parity_result(summary, identity)
    runtime_fingerprint_sha256 = _validate_runtime_fingerprint(
        summary, identity
    )
    rendered_input = _get(summary, "dataset.rendered_input_token_ids")
    if not isinstance(rendered_input, dict):
        raise ValueError("summary rendered input token identity must be an object")
    _expect(
        rendered_input.get("serialization"),
        "int64_le_c_order_v1",
        "summary rendered input serialization",
    )
    rendered_sha256 = rendered_input.get("sha256")
    if (
        not isinstance(rendered_sha256, str)
        or len(rendered_sha256) != 64
        or any(character not in "0123456789abcdef" for character in rendered_sha256)
    ):
        raise ValueError("summary rendered input token ids sha256 is invalid")
    _expect(
        rendered_input.get("shape"),
        [1, generation["input_tokens"]],
        "summary rendered input token shape",
    )
    expected_rendered_sha256 = identity["dataset"].get(
        "rendered_input_token_ids_sha256"
    )
    if expected_rendered_sha256 is not None:
        _expect(
            rendered_sha256,
            expected_rendered_sha256,
            "summary rendered input token ids sha256",
        )
    if identity["projection"] is not None:
        _expect(
            _get(
                summary,
                "trainable_layout.projection_identity.artifact_file_sha256",
            ),
            identity["projection"]["sha256"],
            "summary projection artifact sha256",
        )
    for role in ("target", "draft"):
        observed_model = _get(summary, f"models.{role}")
        if not isinstance(observed_model, dict):
            raise ValueError(f"summary models.{role} must be an object")
        for key, expected in identity[role].items():
            if key == "revision":
                continue
            if key not in observed_model:
                raise ValueError(f"missing field models.{role}[{key!r}]")
            # Artifact keys such as ``config.json_sha256`` contain literal
            # dots, so they must not pass through the dotted-path helper.
            _expect(
                observed_model[key],
                expected,
                f"summary {role} artifact {key}",
            )
    rounds_sha256 = _sha256_file(rounds_path)
    _expect(
        _get(summary, "output.rounds_sha256"),
        rounds_sha256,
        "summary rounds sha256",
    )
    return (
        _sha256_file(summary_path),
        rounds_sha256,
        rendered_sha256,
        runtime_fingerprint_sha256,
    )


def _completion(
    plan: RunPlan,
    summary_sha256: str,
    rounds_sha256: str,
    rendered_input_token_ids_sha256: str,
    runtime_fingerprint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_identity_sha256": plan.identity_sha256,
        "summary_path": "artifact/summary.json",
        "summary_sha256": summary_sha256,
        "rounds_path": "artifact/rounds.jsonl",
        "rounds_sha256": rounds_sha256,
        "rendered_input_token_ids_sha256": (
            rendered_input_token_ids_sha256
        ),
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
    }


def _ensure_artifact_identity_lock(plan: RunPlan) -> None:
    path = plan.artifact_identity_lock_path
    if path.is_file():
        _expect(
            _read_json(path),
            plan.artifact_identity_lock_payload,
            "artifact identity lock",
        )
        _expect(
            _sha256_file(path),
            plan.identity["runtime"]["artifact_identity_lock"]["sha256"],
            "artifact identity lock file hash",
        )
        return
    if path.exists():
        raise ValueError(f"artifact identity lock is not a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json_exclusive(path, plan.artifact_identity_lock_payload)
    except FileExistsError:
        # Another process may have won the exclusive creation race.
        _expect(
            _read_json(path),
            plan.artifact_identity_lock_payload,
            "concurrent artifact identity lock",
        )
    _expect(
        _sha256_file(path),
        plan.identity["runtime"]["artifact_identity_lock"]["sha256"],
        "artifact identity lock file hash",
    )


def completed_run_matches(plan: RunPlan) -> bool:
    """Return true only for a complete matching run; reject all partials."""

    if not _lexists(plan.run_dir):
        return False
    if not plan.run_dir.is_dir() or not plan.identity_path.is_file():
        raise ValueError(f"{plan.run_dir}: partial run; refusing to overwrite")
    stored_plan = _read_json(plan.identity_path)
    _expect(stored_plan, plan.plan_payload(), "stored run plan")
    _expect(
        _sha256_json(stored_plan["identity"]),
        stored_plan["identity_sha256"],
        "stored identity hash",
    )
    if not (plan.artifact_dir / "summary.json").is_file() or not (
        plan.artifact_dir / "rounds.jsonl"
    ).is_file():
        raise ValueError(
            f"{plan.run_dir}: partial run directory; refusing to overwrite"
        )
    (
        summary_sha256,
        rounds_sha256,
        rendered_sha256,
        runtime_fingerprint_sha256,
    ) = _validate_summary(plan)
    # A valid summary is the harness's final artifact.  Missing completion.json
    # means only that the orchestrator died in the tiny post-run window.
    if not _completion_claim_exists(plan):
        return True
    if not plan.completion_path.is_file():
        raise ValueError(f"completion path is not a file: {plan.completion_path}")
    _expect(
        _read_json(plan.completion_path),
        _completion(
            plan,
            summary_sha256,
            rounds_sha256,
            rendered_sha256,
            runtime_fingerprint_sha256,
        ),
        "completion record",
    )
    return True


def _subprocess_environment(plan: RunPlan) -> dict[str, str]:
    """Build the exact child environment represented by the run plan."""

    environment = os.environ.copy()
    if plan.pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(plan.pythonpath)
    else:
        environment.pop("PYTHONPATH", None)
    workspace = plan.identity["runtime"]["determinism"][
        "cublas_workspace_config"
    ]
    if workspace is None:
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        environment["CUBLAS_WORKSPACE_CONFIG"] = workspace
    allocator_config = plan.identity["runtime"].get(
        "allocator_config",
        {
            "PYTORCH_CUDA_ALLOC_CONF": None,
            "PYTORCH_ALLOC_CONF": None,
        },
    )
    for name, value in allocator_config.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _execute_resumable_plan(plan: RunPlan) -> str:
    """Execute one plan with crash-safe retry and immutable completion rules."""

    with _exclusive_run_lock(plan) as lock_descriptor:
        if _prepare_existing_run(plan):
            if not _completion_claim_exists(plan):
                hashes = _validate_summary(plan)
                try:
                    _write_json_exclusive(
                        plan.completion_path,
                        _completion(plan, *hashes),
                    )
                except FileExistsError:
                    # An uncooperative external writer may not honor the flock.
                    # Accept only an identical, fully validated completion.
                    if not completed_run_matches(plan):  # pragma: no cover
                        raise ImmutableCompletionError(
                            f"{plan.run_dir}: concurrent completion is invalid"
                        )
            return "resumed_complete"

        try:
            plan.run_dir.mkdir(parents=True, exist_ok=False)
            _write_json_exclusive(plan.identity_path, plan.plan_payload())
            environment = _subprocess_environment(plan)
            with plan.log_path.open("xb") as log:
                result = subprocess.run(
                    plan.command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                    # Keep the logical-run lock alive if this orchestrator is
                    # killed while the model process continues writing.
                    pass_fds=(lock_descriptor,),
                )
                log.flush()
                os.fsync(log.fileno())
        except Exception as exc:
            if _lexists(plan.run_dir) and not _completion_claim_exists(plan):
                raise _preserve_retryable_failure(
                    plan,
                    exc,
                    reason="orchestrator_or_child_launch_failure",
                ) from exc
            raise

        if result.returncode != 0:
            failure = RuntimeError(
                f"run failed with exit code {result.returncode}; "
                f"log={plan.log_path}"
            )
            raise _preserve_retryable_failure(
                plan,
                failure,
                reason=f"child_exit_{result.returncode}",
            ) from failure

        try:
            hashes = _validate_summary(plan)
        except Exception as exc:
            raise _preserve_retryable_failure(
                plan,
                exc,
                reason="child_returned_success_with_invalid_artifact",
            ) from exc

        # Once a valid harness artifact exists, leave it in place if the tiny
        # completion publication step fails.  A subsequent resume validates
        # the bytes again and repairs only the missing completion record.
        try:
            _write_json_exclusive(
                plan.completion_path,
                _completion(plan, *hashes),
            )
        except FileExistsError as exc:
            try:
                completed_run_matches(plan)
            except (OSError, ValueError) as validation_error:
                raise ImmutableCompletionError(
                    f"{plan.run_dir}: completion was published concurrently "
                    "but failed validation; refusing to overwrite it"
                ) from validation_error
            if not plan.completion_path.is_file():  # pragma: no cover
                raise ImmutableCompletionError(
                    f"completion path is not a file: {plan.completion_path}"
                ) from exc
        return "completed"


def _validate_optimizer_selection_evidence(plan: RunPlan) -> None:
    selection = plan.identity["optimizer_selection"]
    path = Path(selection["path"])
    if not path.is_file() or _sha256_file(path) != selection["sha256"]:
        raise ValueError(
            "selected optimizer config changed after run-plan construction"
        )
    for artifact in selection["evidence_artifacts"]:
        evidence_path = Path(artifact["path"])
        if (
            not evidence_path.is_file()
            or _sha256_file(evidence_path) != artifact["sha256"]
        ):
            raise ValueError(
                "optimizer selection evidence artifact changed after run-plan "
                f"construction: {evidence_path}"
            )


def execute_plan(plan: RunPlan) -> str:
    if plan.identity["optimizer_selection"]["status"] != "locked":
        raise ValueError(
            "optimizer selection is provisional; localize and hash calibration "
            "artifacts, set selected_optimizer_config status=locked, and rebuild "
            "the run plan before execution"
        )
    _validate_optimizer_selection_evidence(plan)
    _ensure_artifact_identity_lock(plan)
    return _execute_resumable_plan(plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--harness",
        default=str(Path(__file__).with_name("dflash_tts_reference.py")),
    )
    parser.add_argument(
        "--selected-optimizer-config",
        default=None,
        help=(
            "read-only selection evidence containing the candidate grid, two "
            "calibration samples, selection rule, and locked mode configs; "
            "legacy schema-v1 path"
        ),
    )
    parser.add_argument(
        "--stage1-analysis",
        help=(
            "complete schema-v3 optimizer/LR selection-analysis.json; must be "
            "paired with --stage2-analysis"
        ),
    )
    parser.add_argument(
        "--stage2-analysis",
        help=(
            "complete schema-v3 LoRA rank-analysis.json; must be paired with "
            "--stage1-analysis"
        ),
    )
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--reference-module", default="dflash.model")
    parser.add_argument("--reference-revision", default="94e4abc")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-tokens", type=int)
    source.add_argument("--preflight-summary")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--total-contexts",
        nargs="+",
        type=int,
        default=list(SUPPORTED_TOTAL_CONTEXTS),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODE_ORDER, default=list(MODE_ORDER)
    )
    parser.add_argument("--draft-block-size", type=int, default=16)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--turns-field", default="turns")
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable the formal deterministic numerical contract; use "
            "--no-deterministic only for explicitly labelled performance runs"
        ),
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        help=(
            "required explicit DFlash mask token id; formal identities may not "
            "defer this behavior-affecting value to checkpoint runtime defaults"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--proximal-lambda", type=float, default=0.0)
    parser.add_argument("--update-stride", type=int, default=1)
    parser.add_argument(
        "--position-weighting",
        choices=("uniform", "linear", "exponential"),
        default="exponential",
    )
    parser.add_argument("--position-decay-gamma", type=float, default=7.0)
    parser.add_argument(
        "--loss-reduction", choices=("weighted-mean", "sum"), default="weighted-mean"
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--draft-cache-policy", choices=("stale", "rebuild"), default="stale"
    )
    parser.add_argument("--projection-artifact")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--audit-cuda-timing", action="store_true")
    parser.add_argument("--parameter-audit-stride", type=int, default=0)
    parser.add_argument("--parity-max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-static-parity-preflight", action="store_true")
    parser.add_argument(
        "--prepare-identity-lock",
        action="store_true",
        help=(
            "atomically create or validate only artifact_identity_lock.json, "
            "then exit without launching the harness or creating run directories; "
            "requires --modes static"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if (args.stage1_analysis is None) != (args.stage2_analysis is None):
        raise ValueError(
            "--stage1-analysis and --stage2-analysis must be supplied together"
        )
    if args.stage1_analysis is not None and args.selected_optimizer_config is not None:
        raise ValueError(
            "schema-v3 analyses and --selected-optimizer-config are mutually "
            "exclusive selection sources"
        )
    if args.sample_index < 0:
        raise ValueError("--sample-index must be non-negative")
    if args.mask_token_id is None or args.mask_token_id < 0:
        raise ValueError("--mask-token-id must be an explicit non-negative integer")
    if args.draft_block_size <= 1:
        raise ValueError("--draft-block-size must be greater than one")
    if args.temperature < 0 or args.proximal_lambda < 0:
        raise ValueError("temperature and proximal lambda must be non-negative")
    if args.update_stride <= 0:
        raise ValueError("--update-stride must be positive")
    if args.position_decay_gamma is not None and args.position_decay_gamma <= 0:
        raise ValueError("--position-decay-gamma must be positive")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be positive")
    if args.parameter_audit_stride < 0 or args.parity_max_new_tokens <= 0:
        raise ValueError("audit stride/count values are invalid")
    if args.prepare_identity_lock and args.dry_run:
        raise ValueError(
            "--prepare-identity-lock and --dry-run are mutually exclusive"
        )
    if args.prepare_identity_lock and args.modes != ["static"]:
        raise ValueError(
            "--prepare-identity-lock requires exactly --modes static so "
            "projection requirements cannot be silently bypassed"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    plans = build_run_plans(args)
    if args.prepare_identity_lock:
        if not plans:
            raise ValueError("identity-lock preparation produced no run plan")
        if plans[0].identity["optimizer_selection"]["status"] != "locked":
            raise ValueError(
                "selected optimizer config is provisional_pilot; bind localized "
                "calibration evidence artifacts and set status=locked before "
                "preparing the identity lock"
            )
        _validate_optimizer_selection_evidence(plans[0])
        _ensure_artifact_identity_lock(plans[0])
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "identity_lock_ready",
                    "path": str(plans[0].artifact_identity_lock_path),
                    "sha256": plans[0].identity["runtime"][
                        "artifact_identity_lock"
                    ]["sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "dry_run",
                    "optimizer_selection": (
                        plans[0].identity["optimizer_selection"] if plans else None
                    ),
                    "runs": [
                        {
                            "run_dir": str(plan.run_dir),
                            "identity_sha256": plan.identity_sha256,
                            "command": list(plan.command),
                        }
                        for plan in plans
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if plans and plans[0].identity["optimizer_selection"]["status"] != "locked":
        raise ValueError(
            "selected optimizer config is provisional_pilot; bind localized "
            "calibration evidence artifacts and set status=locked before execution"
        )

    counts = {"completed": 0, "resumed_complete": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    for plan in plans:
        try:
            status = execute_plan(plan)
            counts[status] += 1
            print(
                json.dumps(
                    {
                        "status": status,
                        "run_dir": str(plan.run_dir),
                        "identity_sha256": plan.identity_sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            counts["failed"] += 1
            failures.append({"run_dir": str(plan.run_dir), "error": str(exc)})
            print(
                json.dumps({**failures[-1], "status": "failed"}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            if not args.keep_going:
                break
    print(
        json.dumps(
            {"counts": counts, "failures": failures}, indent=2, sort_keys=True
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
