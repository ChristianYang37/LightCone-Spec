#!/usr/bin/env python3
"""Run an explicit schema-v3 DFlash calibration/rank candidate list.

The candidate JSON is a list, never a Cartesian grid.  It must bind exactly
Math500 samples 0 and 419, ``max_new_tokens=2048``, each rendered-input hash,
and one row per optimizer/rank/LR/weight-decay choice.  Diagnostic candidates
are explicitly marked and are never selection-eligible.

This selection entry point always uses the deterministic contract, mandatory
Static parity, and lightweight timing.  Nondeterministic performance profiling,
parity skips, and synchronized CUDA timing belong in a separate diagnostic run.

Example candidate row::

    {
      "candidate_id": "tail-lora-r16-adamw-lr1e-4-wd1e-2",
      "mode": "tail-lora",
      "optimizer": "adamw",
      "learning_rate": 0.0001,
      "weight_decay": 0.01,
      "rank": 16,
      "draft_cache_policy": "stale",
      "diagnostic_kind": "selection",
      "parameter_audit_stride": 0
    }

Model/content identity, command construction, summary validation, completion
records, deterministic child environment, and exclusive JSON publication are
delegated to :mod:`run_dflash_tts_frozen_sweep`.  The harness is not copied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import run_dflash_tts_frozen_sweep as frozen  # noqa: E402


SCHEMA_VERSION = 3
SPEC_KIND = "dflash_tts_explicit_calibration_candidates"
EVIDENCE_SCOPE = (
    "schema_v3_calibration_selection_only_not_long_context_evaluation"
)
DIAGNOSTIC_SPEC_KIND = "dflash_tts_explicit_diagnostic_candidates"
DIAGNOSTIC_EVIDENCE_SCOPE = (
    "schema_v3_diagnostics_only_not_selection_or_long_context_evaluation"
)
DIAGNOSTIC_PROVENANCE_KIND = (
    "dflash_tts_schema_v3_diagnostic_spec_provenance"
)
SAMPLE_INDICES = (0, 419)
MAX_NEW_TOKENS = 2048
DIAGNOSTIC_KINDS = (
    "selection",
    "cache-policy-diagnostic",
    "parameter-audit",
)
DIAGNOSTIC_CACHE_MODES = ("full-drafter", "drafter-lora", "tail-lora")
DIAGNOSTIC_AUDIT_STRIDES = {
    "full-drafter": 256,
    "drafter-lora": 32,
    "full-rank-tail": 32,
    "tail-lora": 32,
    "output-residual": 32,
}
_CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    mode: str
    config: frozen.ModeConfig
    draft_cache_policy: str
    diagnostic_kind: str
    parameter_audit_stride: int

    @property
    def selection_eligible(self) -> bool:
        return self.diagnostic_kind == "selection"


@dataclass(frozen=True)
class CandidateSweep:
    path: Path
    file_sha256: str
    content_sha256: str
    study_id: str
    kind: str
    evidence_scope: str
    samples: tuple[dict[str, Any], ...]
    candidates: tuple[Candidate, ...]
    provenance: dict[str, Any] | None


@dataclass(frozen=True)
class VerifiedStage1Analysis:
    """One byte- and evidence-verified published Stage-1 analysis."""

    path: Path
    payload: dict[str, Any]
    sweep: CandidateSweep
    file_sha256: str


def _verified_hashed_field(
    payload: dict[str, Any],
    *,
    value_key: str,
    hash_key: str,
    expected_type: type,
) -> Any:
    value = payload.get(value_key)
    if not isinstance(value, expected_type):
        raise ValueError(f"Stage-1 analysis {value_key} has an invalid type")
    observed = payload.get(hash_key)
    if not frozen._is_sha256(observed):
        raise ValueError(f"Stage-1 analysis {hash_key} is invalid")
    if frozen._sha256_json(value) != observed:
        raise ValueError(f"Stage-1 analysis {value_key} hash mismatch")
    return value


def verify_stage1_published_analysis(
    path_value: str | Path,
) -> VerifiedStage1Analysis:
    """Close every published Stage-1 hash before diagnostics can be planned.

    Shape checks alone are intentionally insufficient.  After checking the
    embedded hash graph, the canonical Stage-1 verifier re-reads every bound
    run artifact, rebuilds the analysis, and requires byte-identical output.
    """

    # Local import avoids the analyzer -> calibration runner dependency cycle.
    import analyze_dflash_tts_calibration as stage1

    path = frozen._required_path(
        str(path_value), directory=False, label="Stage-1 analysis"
    )
    payload = frozen._read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Stage-1 analysis must be an object")
    if (
        payload.get("schema_version") != stage1.SCHEMA_VERSION
        or payload.get("kind") != stage1.KIND
        or payload.get("status") != "complete"
        or payload.get("evidence_scope") != EVIDENCE_SCOPE
    ):
        raise ValueError("Stage-1 analysis schema/kind/status/scope mismatch")

    analysis_sha256 = payload.get("analysis_sha256")
    if not frozen._is_sha256(analysis_sha256):
        raise ValueError("Stage-1 analysis analysis_sha256 is invalid")
    if payload.get("analysis_hash_scheme") != (
        "canonical_json_without_analysis_sha256_v1"
    ):
        raise ValueError("Stage-1 analysis hash scheme mismatch")
    unsigned = dict(payload)
    unsigned.pop("analysis_sha256")
    if frozen._sha256_json(unsigned) != analysis_sha256:
        raise ValueError("Stage-1 analysis self-hash mismatch")

    specification = payload.get("candidate_specification")
    if not isinstance(specification, dict):
        raise ValueError("Stage-1 analysis candidate_specification must be an object")
    candidate_path = specification.get("path")
    if not isinstance(candidate_path, str) or not candidate_path:
        raise ValueError("Stage-1 analysis candidate specification path is invalid")
    candidate_locator = Path(candidate_path).expanduser()
    if not candidate_locator.is_absolute():
        candidate_locator = path.parent / candidate_locator
    sweep = load_candidate_sweep(candidate_locator)
    if sweep.kind != SPEC_KIND:
        raise ValueError("diagnostics require a selection-only Stage-1 candidate spec")
    expected_specification = {
        "file_sha256": sweep.file_sha256,
        "content_sha256": sweep.content_sha256,
        "study_id": sweep.study_id,
        "schema_version": SCHEMA_VERSION,
        "kind": sweep.kind,
        "evidence_scope": sweep.evidence_scope,
    }
    recorded_specification = dict(specification)
    recorded_specification.pop("path", None)
    if candidate_locator.resolve() != sweep.path or frozen._sha256_json(
        recorded_specification
    ) != frozen._sha256_json(
        expected_specification
    ):
        raise ValueError("Stage-1 analysis candidate specification binding mismatch")
    if payload.get("study_id") != sweep.study_id:
        raise ValueError("Stage-1 analysis study_id mismatch")
    expected_samples = [int(sample["sample_index"]) for sample in sweep.samples]
    if payload.get("sample_indices") != expected_samples:
        raise ValueError("Stage-1 analysis sample_indices mismatch")

    _verified_hashed_field(
        payload,
        value_key="selection_rule",
        hash_key="selection_rule_sha256",
        expected_type=dict,
    )
    candidate_rows = _verified_hashed_field(
        payload,
        value_key="candidate_rows",
        hash_key="candidate_rows_sha256",
        expected_type=list,
    )
    expected_candidate_ids = [candidate.candidate_id for candidate in sweep.candidates]
    observed_candidate_ids = [
        row.get("candidate_id") if isinstance(row, dict) else None
        for row in candidate_rows
    ]
    if observed_candidate_ids != expected_candidate_ids:
        raise ValueError("Stage-1 analysis candidate_rows do not bind the candidate spec")
    _verified_hashed_field(
        payload,
        value_key="selection_decisions",
        hash_key="selection_decisions_sha256",
        expected_type=list,
    )

    implementation = _verified_hashed_field(
        payload,
        value_key="analysis_implementation",
        hash_key="analysis_implementation_sha256",
        expected_type=dict,
    )
    expected_implementation = {
        "analyzer",
        "metric_aggregator",
        "calibration_orchestrator",
        "frozen_run_validator",
    }
    if set(implementation) != expected_implementation:
        raise ValueError("Stage-1 analysis implementation source set mismatch")
    for name, record in implementation.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"file", "sha256"}
            or not isinstance(record["file"], str)
            or not record["file"]
            or not frozen._is_sha256(record["sha256"])
        ):
            raise ValueError(
                f"Stage-1 analysis implementation record {name!r} is invalid"
            )

    pareto = payload.get("pareto")
    if not isinstance(pareto, dict):
        raise ValueError("Stage-1 analysis pareto must be an object")
    pareto_rows = pareto.get("rows")
    pareto_rows_sha256 = pareto.get("rows_sha256")
    if (
        not isinstance(pareto_rows, list)
        or not frozen._is_sha256(pareto_rows_sha256)
        or frozen._sha256_json(pareto_rows) != pareto_rows_sha256
    ):
        raise ValueError("Stage-1 analysis pareto rows hash mismatch")

    source_runs = payload.get("source_runs")
    if not isinstance(source_runs, list):
        raise ValueError("Stage-1 analysis source_runs must be a list")
    if payload.get("source_run_count") != len(source_runs):
        raise ValueError("Stage-1 analysis source_run_count mismatch")
    expected_run_count = len(sweep.samples) * len(sweep.candidates)
    if len(source_runs) != expected_run_count:
        raise ValueError("Stage-1 analysis source run coverage mismatch")
    artifact_identity_lock = payload.get("artifact_identity_lock")
    if artifact_identity_lock is None:
        source_hash_payload: Any = source_runs
        expected_artifact_count = len(source_runs) * 4
    else:
        if (
            not isinstance(artifact_identity_lock, dict)
            or set(artifact_identity_lock)
            != {"path", "file_sha256", "content_sha256"}
            or not isinstance(artifact_identity_lock["path"], str)
            or not artifact_identity_lock["path"]
            or not frozen._is_sha256(artifact_identity_lock["file_sha256"])
            or not frozen._is_sha256(artifact_identity_lock["content_sha256"])
        ):
            raise ValueError("Stage-1 analysis artifact_identity_lock is invalid")
        lock_locator = Path(artifact_identity_lock["path"]).expanduser()
        if not lock_locator.is_absolute():
            lock_locator = path.parent / lock_locator
        lock_path = frozen._required_path(
            str(lock_locator),
            directory=False,
            label="Stage-1 artifact identity lock",
        )
        if (
            frozen._sha256_file(lock_path)
            != artifact_identity_lock["file_sha256"]
            or frozen._sha256_json(frozen._read_json(lock_path))
            != artifact_identity_lock["content_sha256"]
        ):
            raise ValueError("Stage-1 artifact identity lock hash mismatch")
        source_hash_payload = {
            "artifact_identity_lock": artifact_identity_lock,
            "runs": source_runs,
        }
        expected_artifact_count = len(source_runs) * 4 + 1
    if payload.get("source_artifact_count") != expected_artifact_count:
        raise ValueError("Stage-1 analysis source_artifact_count mismatch")
    source_set_sha256 = payload.get("source_artifact_set_sha256")
    if (
        not frozen._is_sha256(source_set_sha256)
        or frozen._sha256_json(source_hash_payload) != source_set_sha256
    ):
        raise ValueError("Stage-1 analysis source artifact-set hash mismatch")
    required_source_keys = {
        "sample_index",
        "candidate_id",
        "run_identity_sha256",
        "identity_sha256",
        "completion_sha256",
        "summary_sha256",
        "rounds_sha256",
        "command_sha256",
    }
    expected_run_keys = {
        (int(sample["sample_index"]), candidate.candidate_id)
        for sample in sweep.samples
        for candidate in sweep.candidates
    }
    observed_run_keys: set[tuple[int, str]] = set()
    for index, record in enumerate(source_runs):
        if not isinstance(record, dict) or set(record) != required_source_keys:
            raise ValueError(f"Stage-1 analysis source_runs[{index}] is invalid")
        key = (record["sample_index"], record["candidate_id"])
        if (
            isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or not isinstance(key[1], str)
            or key in observed_run_keys
        ):
            raise ValueError(f"Stage-1 analysis source_runs[{index}] key is invalid")
        if any(
            not frozen._is_sha256(record[field])
            for field in required_source_keys - {"sample_index", "candidate_id"}
        ):
            raise ValueError(f"Stage-1 analysis source_runs[{index}] hash is invalid")
        observed_run_keys.add(key)
    if observed_run_keys != expected_run_keys:
        raise ValueError("Stage-1 analysis source_runs do not cover the candidate spec")

    output_root = payload.get("output_root")
    if output_root is None:
        output_root_locator = path.parent
    else:
        if not isinstance(output_root, str) or not output_root:
            raise ValueError("Stage-1 analysis output_root is invalid")
        output_root_locator = Path(output_root).expanduser()
        if not output_root_locator.is_absolute():
            output_root_locator = path.parent / output_root_locator
    rebuilt, file_sha256 = stage1.verify_published_analysis(
        candidate_spec=sweep.path,
        output_root=output_root_locator,
        analysis_path=path,
    )
    if frozen._sha256_json(rebuilt) != frozen._sha256_json(payload):
        raise ValueError("Stage-1 analysis verifier returned different evidence")
    observed_file_sha256 = frozen._sha256_file(path)
    if file_sha256 != observed_file_sha256:
        raise ValueError("Stage-1 analysis file hash changed during verification")
    return VerifiedStage1Analysis(
        path=path,
        payload=dict(payload),
        sweep=sweep,
        file_sha256=observed_file_sha256,
    )


def _expect_keys(value: Any, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} keys must be exactly {sorted(expected)}, got {observed}")
    return value


def _candidate(value: Any, *, index: int) -> Candidate:
    label = f"candidates[{index}]"
    row = _expect_keys(
        value,
        {
            "candidate_id",
            "mode",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "rank",
            "draft_cache_policy",
            "diagnostic_kind",
            "parameter_audit_stride",
        },
        label=label,
    )
    candidate_id = row["candidate_id"]
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError(
            f"{label}.candidate_id must match [a-z0-9][a-z0-9._-]*"
        )
    mode = row["mode"]
    if mode not in frozen.MODE_ORDER:
        raise ValueError(f"{label}.mode must be one of {list(frozen.MODE_ORDER)}")
    config = frozen._mode_config(
        {
            "optimizer": row["optimizer"],
            "learning_rate": row["learning_rate"],
            "weight_decay": row["weight_decay"],
            "rank": row["rank"],
        },
        label=label,
        mode=mode,
    )
    cache_policy = row["draft_cache_policy"]
    if cache_policy not in {"stale", "rebuild"}:
        raise ValueError(f"{label}.draft_cache_policy must be stale or rebuild")
    diagnostic = row["diagnostic_kind"]
    if diagnostic not in DIAGNOSTIC_KINDS:
        raise ValueError(
            f"{label}.diagnostic_kind must be one of {list(DIAGNOSTIC_KINDS)}"
        )
    audit_stride = row["parameter_audit_stride"]
    if isinstance(audit_stride, bool) or not isinstance(audit_stride, int):
        raise ValueError(f"{label}.parameter_audit_stride must be an integer")
    if diagnostic == "selection" and (cache_policy != "stale" or audit_stride != 0):
        raise ValueError(
            f"{label}: selection candidates require stale cache and zero audit stride"
        )
    if diagnostic == "cache-policy-diagnostic" and audit_stride != 0:
        raise ValueError(
            f"{label}: cache-policy diagnostics require zero audit stride"
        )
    if diagnostic == "parameter-audit" and audit_stride <= 0:
        raise ValueError(
            f"{label}: parameter-audit candidates require a positive audit stride"
        )
    if audit_stride < 0:
        raise ValueError(f"{label}.parameter_audit_stride cannot be negative")
    return Candidate(
        candidate_id=candidate_id,
        mode=mode,
        config=config,
        draft_cache_policy=cache_policy,
        diagnostic_kind=diagnostic,
        parameter_audit_stride=audit_stride,
    )


def _diagnostic_provenance(
    value: Any,
    *,
    base_dir: Path,
    study_id: str,
    candidates: tuple[Candidate, ...],
) -> dict[str, Any]:
    provenance = _expect_keys(
        value,
        {
            "schema_version",
            "kind",
            "builder",
            "source_stage1_analysis",
            "selection_isolation",
            "winner_derivation",
            "residual_projection_requirement",
        },
        label="candidate specification provenance",
    )
    if (
        provenance["schema_version"] != 1
        or provenance["kind"] != DIAGNOSTIC_PROVENANCE_KIND
    ):
        raise ValueError("diagnostic candidate provenance schema/kind mismatch")

    builder = _expect_keys(
        provenance["builder"],
        {"path", "sha256"},
        label="candidate specification provenance.builder",
    )
    if not isinstance(builder["path"], str) or not builder["path"]:
        raise ValueError("diagnostic provenance builder path is invalid")
    if not frozen._is_sha256(builder["sha256"]):
        raise ValueError("diagnostic provenance builder sha256 is invalid")
    builder_locator = Path(builder["path"]).expanduser()
    if not builder_locator.is_absolute():
        builder_locator = base_dir / builder_locator
    builder_path = frozen._required_path(
        str(builder_locator), directory=False, label="diagnostic provenance builder"
    )
    if frozen._sha256_file(builder_path) != builder["sha256"]:
        raise ValueError("diagnostic provenance builder content changed")

    source = _expect_keys(
        provenance["source_stage1_analysis"],
        {
            "path",
            "file_sha256",
            "analysis_sha256",
            "kind",
            "study_id",
            "candidate_specification_file_sha256",
            "candidate_specification_content_sha256",
            "output_root",
            "source_artifact_set_sha256",
            "selection_decisions_sha256",
        },
        label="candidate specification provenance.source_stage1_analysis",
    )
    for key in (
        "file_sha256",
        "analysis_sha256",
        "candidate_specification_file_sha256",
        "candidate_specification_content_sha256",
        "source_artifact_set_sha256",
        "selection_decisions_sha256",
    ):
        if not frozen._is_sha256(source[key]):
            raise ValueError(f"diagnostic provenance source {key} is invalid")
    for key in ("path", "kind", "study_id", "output_root"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"diagnostic provenance source {key} is invalid")
    source_locator = Path(source["path"]).expanduser()
    if not source_locator.is_absolute():
        source_locator = base_dir / source_locator
    source_path = frozen._required_path(
        str(source_locator),
        directory=False,
        label="diagnostic provenance Stage-1 analysis",
    )
    if frozen._sha256_file(source_path) != source["file_sha256"]:
        raise ValueError("diagnostic provenance Stage-1 analysis content changed")

    isolation = _expect_keys(
        provenance["selection_isolation"],
        {
            "source_study_id",
            "diagnostic_study_id",
            "source_output_root",
            "diagnostic_output_root",
            "required_distinct_output_roots",
            "adaptive_candidates_selection_eligible",
            "static_control_selection_eligible",
        },
        label="candidate specification provenance.selection_isolation",
    )
    if isolation["diagnostic_study_id"] != study_id:
        raise ValueError("diagnostic provenance study_id mismatch")
    if isolation["source_study_id"] != source["study_id"]:
        raise ValueError("diagnostic provenance source study_id mismatch")
    if study_id == source["study_id"]:
        raise ValueError("diagnostic and Stage-1 study IDs must differ")
    if isolation["source_output_root"] != source["output_root"]:
        raise ValueError("diagnostic provenance source output root mismatch")
    for key in ("source_output_root", "diagnostic_output_root"):
        path = isolation[key]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"diagnostic provenance {key} must be absolute")
    if Path(isolation["source_output_root"]).resolve() == Path(
        isolation["diagnostic_output_root"]
    ).resolve():
        raise ValueError("diagnostic output root must differ from Stage-1")
    if (
        isolation["required_distinct_output_roots"] is not True
        or isolation["adaptive_candidates_selection_eligible"] is not False
        or isolation["static_control_selection_eligible"] is not True
    ):
        raise ValueError("diagnostic provenance selection isolation is invalid")

    derivation = _expect_keys(
        provenance["winner_derivation"],
        {
            "required_decision_status",
            "required_safe_for_selection",
            "selected_winners",
            "omitted_diagnostics",
        },
        label="candidate specification provenance.winner_derivation",
    )
    if (
        derivation["required_decision_status"] != "local_grid_winner"
        or derivation["required_safe_for_selection"] is not True
        or not isinstance(derivation["selected_winners"], list)
        or not isinstance(derivation["omitted_diagnostics"], list)
    ):
        raise ValueError("diagnostic provenance winner derivation is invalid")

    static = [candidate for candidate in candidates if candidate.mode == "static"]
    static_selection = [
        candidate for candidate in static if candidate.selection_eligible
    ]
    static_rebuild = [
        candidate
        for candidate in static
        if candidate.diagnostic_kind == "cache-policy-diagnostic"
        and candidate.draft_cache_policy == "rebuild"
        and candidate.parameter_audit_stride == 0
    ]
    if (
        len(static) != 2
        or len(static_selection) != 1
        or static_selection[0].draft_cache_policy != "stale"
        or static_selection[0].parameter_audit_stride != 0
        or len(static_rebuild) != 1
    ):
        raise ValueError(
            "diagnostic specifications require one selection Static-stale and "
            "one non-selection Static-rebuild comparator"
        )
    adaptive = [candidate for candidate in candidates if candidate.mode != "static"]
    if any(candidate.selection_eligible for candidate in adaptive):
        raise ValueError(
            "diagnostic specifications cannot contain selection-eligible adaptive candidates"
        )

    selected_modes: set[str] = set()
    for index, value in enumerate(derivation["selected_winners"]):
        winner = _expect_keys(
            value,
            {
                "mode",
                "candidate_id",
                "rank",
                "decision_status",
                "source_candidate_sha256",
            },
            label=f"diagnostic provenance selected_winners[{index}]",
        )
        mode = winner["mode"]
        if mode not in DIAGNOSTIC_AUDIT_STRIDES or mode in selected_modes:
            raise ValueError("diagnostic provenance selected winner mode is invalid")
        if (
            not isinstance(winner["candidate_id"], str)
            or not winner["candidate_id"]
            or winner["decision_status"] != "local_grid_winner"
            or not frozen._is_sha256(winner["source_candidate_sha256"])
        ):
            raise ValueError("diagnostic provenance selected winner is invalid")
        selected_modes.add(mode)
    adaptive_modes = {candidate.mode for candidate in adaptive}
    if selected_modes != adaptive_modes:
        raise ValueError(
            "diagnostic provenance selected winners do not match adaptive candidates"
        )

    observed_omissions: set[tuple[str, str]] = set()
    for index, value in enumerate(derivation["omitted_diagnostics"]):
        omission = _expect_keys(
            value,
            {"diagnostic_kind", "mode", "reason", "decision_status"},
            label=f"diagnostic provenance omitted_diagnostics[{index}]",
        )
        key = (omission["diagnostic_kind"], omission["mode"])
        reason = omission["reason"]
        expected_status = {
            "stage1_no_safe_selection": "no_safe_selection",
            "stage1_selection_decision_missing": "missing",
        }.get(reason)
        if (
            key in observed_omissions
            or key[0] not in {"cache-policy-diagnostic", "parameter-audit"}
            or key[1] not in DIAGNOSTIC_AUDIT_STRIDES
            or expected_status is None
            or omission["decision_status"] != expected_status
        ):
            raise ValueError("diagnostic provenance omission is invalid")
        observed_omissions.add(key)
    expected_omissions = {
        ("cache-policy-diagnostic", mode)
        for mode in DIAGNOSTIC_CACHE_MODES
        if mode not in selected_modes
    } | {
        ("parameter-audit", mode)
        for mode in DIAGNOSTIC_AUDIT_STRIDES
        if mode not in selected_modes
    }
    if observed_omissions != expected_omissions:
        raise ValueError("diagnostic provenance omission coverage mismatch")

    for candidate in adaptive:
        if candidate.diagnostic_kind == "cache-policy-diagnostic":
            if (
                candidate.mode not in DIAGNOSTIC_CACHE_MODES
                or candidate.parameter_audit_stride != 0
            ):
                raise ValueError("diagnostic cache-policy candidate is invalid")
        elif candidate.diagnostic_kind == "parameter-audit":
            if (
                candidate.mode not in DIAGNOSTIC_AUDIT_STRIDES
                or candidate.parameter_audit_stride
                != DIAGNOSTIC_AUDIT_STRIDES[candidate.mode]
                or candidate.draft_cache_policy != "stale"
            ):
                raise ValueError("diagnostic parameter-audit candidate is invalid")
        else:
            raise ValueError("diagnostic adaptive candidate kind is invalid")
    diagnostic_counts: dict[tuple[str, str], int] = {}
    for candidate in adaptive:
        key = (candidate.diagnostic_kind, candidate.mode)
        diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1
    expected_counts = {
        ("cache-policy-diagnostic", mode): 2
        for mode in DIAGNOSTIC_CACHE_MODES
        if mode in selected_modes
    } | {
        ("parameter-audit", mode): 1
        for mode in DIAGNOSTIC_AUDIT_STRIDES
        if mode in selected_modes
    }
    if diagnostic_counts != expected_counts:
        raise ValueError("diagnostic candidate coverage does not match safe winners")

    residual_ids = sorted(
        candidate.candidate_id
        for candidate in adaptive
        if candidate.mode == "output-residual"
    )
    residual = _expect_keys(
        provenance["residual_projection_requirement"],
        {
            "required",
            "rank",
            "candidate_ids",
            "runner_binding",
        },
        label="candidate specification provenance.residual_projection_requirement",
    )
    if (
        residual["required"] is not bool(residual_ids)
        or residual["rank"] != (16 if residual_ids else None)
        or residual["candidate_ids"] != residual_ids
        or residual["runner_binding"]
        != "single_identity_bound_projection_artifact_at_run_plan_v1"
    ):
        raise ValueError("diagnostic residual projection provenance mismatch")
    if any(
        candidate.mode == "output-residual" and candidate.config.rank != 16
        for candidate in adaptive
    ):
        raise ValueError("diagnostic output-residual candidates require rank 16")
    return dict(provenance)


def _verify_diagnostic_derivation(
    payload: dict[str, Any], provenance: dict[str, Any], *, base_dir: Path
) -> None:
    """Prove that the diagnostic spec is the deterministic Stage-1 derivative."""

    source = provenance["source_stage1_analysis"]
    source_locator = Path(source["path"]).expanduser()
    if not source_locator.is_absolute():
        source_locator = base_dir / source_locator
    verified = verify_stage1_published_analysis(source_locator)
    source_payload = verified.payload
    source_spec = source_payload["candidate_specification"]
    source_root_value = source_payload.get("output_root")
    if source_root_value is None:
        source_output_root = verified.path.parent
    else:
        source_output_root = Path(str(source_root_value)).expanduser()
        if not source_output_root.is_absolute():
            source_output_root = verified.path.parent / source_output_root
        source_output_root = source_output_root.resolve()
    expected_source = {
        "file_sha256": verified.file_sha256,
        "analysis_sha256": source_payload["analysis_sha256"],
        "kind": source_payload["kind"],
        "study_id": source_payload["study_id"],
        "candidate_specification_file_sha256": source_spec["file_sha256"],
        "candidate_specification_content_sha256": source_spec["content_sha256"],
        "output_root": str(source_output_root),
        "source_artifact_set_sha256": source_payload[
            "source_artifact_set_sha256"
        ],
        "selection_decisions_sha256": source_payload[
            "selection_decisions_sha256"
        ],
    }
    observed_source = dict(source)
    observed_source.pop("path", None)
    if frozen._sha256_json(observed_source) != frozen._sha256_json(expected_source):
        raise ValueError(
            "diagnostic provenance does not bind the verified Stage-1 analysis"
        )

    # The builder owns the only winner-to-diagnostic mapping.  Reusing its
    # pure derivation here prevents a second, weaker interpretation in the
    # execution path while avoiding a second Stage-1 evidence traversal.
    import build_dflash_tts_diagnostic_spec as diagnostic_builder

    expected = diagnostic_builder.derive_diagnostic_spec_from_verified_stage1(
        verified=verified,
        diagnostic_output_root=Path(
            provenance["selection_isolation"]["diagnostic_output_root"]
        ),
    )
    if frozen._sha256_json(payload["samples"]) != frozen._sha256_json(
        expected["samples"]
    ):
        raise ValueError("diagnostic samples are not derived from Stage-1")
    if frozen._sha256_json(payload["candidates"]) != frozen._sha256_json(
        expected["candidates"]
    ):
        raise ValueError(
            "diagnostic candidates are not the exact Stage-1 winner derivation"
        )
    if frozen._sha256_json(provenance["winner_derivation"]) != frozen._sha256_json(
        expected["provenance"]["winner_derivation"]
    ):
        raise ValueError("diagnostic winner derivation does not match Stage-1")
    normalized = dict(payload)
    normalized_provenance = dict(provenance)
    normalized_provenance["builder"] = {
        **provenance["builder"],
        "path": expected["provenance"]["builder"]["path"],
    }
    normalized_provenance["source_stage1_analysis"] = {
        **source,
        "path": expected["provenance"]["source_stage1_analysis"]["path"],
    }
    normalized["provenance"] = normalized_provenance
    if frozen._sha256_json(normalized) != frozen._sha256_json(expected):
        raise ValueError(
            "diagnostic specification is not the canonical Stage-1 derivative"
        )


def load_candidate_sweep(path_value: str | Path) -> CandidateSweep:
    path = frozen._required_path(
        str(path_value), directory=False, label="candidate specification"
    )
    payload = frozen._read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("candidate specification must be an object")
    kind = payload.get("kind")
    expected_keys = {
        "schema_version",
        "kind",
        "study_id",
        "evidence_scope",
        "max_new_tokens",
        "samples",
        "candidates",
    }
    if kind == DIAGNOSTIC_SPEC_KIND:
        expected_keys.add("provenance")
    _expect_keys(payload, expected_keys, label="candidate specification")
    if payload["schema_version"] != SCHEMA_VERSION or kind not in {
        SPEC_KIND,
        DIAGNOSTIC_SPEC_KIND,
    }:
        raise ValueError(
            "candidate specification must use a supported calibration/diagnostic schema-v3 kind"
        )
    evidence_scope = (
        EVIDENCE_SCOPE
        if kind == SPEC_KIND
        else DIAGNOSTIC_EVIDENCE_SCOPE
    )
    if payload["evidence_scope"] != evidence_scope:
        raise ValueError("candidate specification evidence_scope mismatch")
    study_id = payload["study_id"]
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("candidate specification study_id must be non-empty")
    if payload["max_new_tokens"] != MAX_NEW_TOKENS:
        raise ValueError(f"candidate specification max_new_tokens must be {MAX_NEW_TOKENS}")

    samples_value = payload["samples"]
    if not isinstance(samples_value, list) or len(samples_value) != 2:
        raise ValueError("candidate specification must contain exactly two samples")
    samples = []
    for index, value in enumerate(samples_value):
        row = _expect_keys(
            value,
            {
                "sample_index",
                "input_tokens",
                "rendered_input_token_ids_sha256",
            },
            label=f"samples[{index}]",
        )
        sample_index = row["sample_index"]
        input_tokens = row["input_tokens"]
        rendered = row["rendered_input_token_ids_sha256"]
        if sample_index != SAMPLE_INDICES[index]:
            raise ValueError(
                f"candidate samples must appear in fixed order {list(SAMPLE_INDICES)}"
            )
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens <= 0
        ):
            raise ValueError(f"samples[{index}].input_tokens must be positive")
        if not frozen._is_sha256(rendered):
            raise ValueError(
                f"samples[{index}].rendered_input_token_ids_sha256 is invalid"
            )
        samples.append(dict(row))

    candidates_value = payload["candidates"]
    if not isinstance(candidates_value, list) or not candidates_value:
        raise ValueError("candidate specification candidates must be a non-empty list")
    candidates = tuple(
        _candidate(value, index=index)
        for index, value in enumerate(candidates_value)
    )
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate specification contains duplicate candidate_id values")
    semantic_keys = [
        (
            candidate.mode,
            candidate.config,
            candidate.draft_cache_policy,
            candidate.diagnostic_kind,
            candidate.parameter_audit_stride,
        )
        for candidate in candidates
    ]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError("candidate specification contains duplicate semantic candidates")
    static_selection = [
        candidate
        for candidate in candidates
        if candidate.mode == "static" and candidate.selection_eligible
    ]
    if len(static_selection) != 1:
        raise ValueError("candidate specification requires exactly one selection Static")

    diagnostic_groups: dict[tuple[Any, ...], set[str]] = {}
    for candidate in candidates:
        if candidate.diagnostic_kind != "cache-policy-diagnostic":
            continue
        key = (candidate.mode, candidate.config)
        diagnostic_groups.setdefault(key, set()).add(candidate.draft_cache_policy)
    # The selection Static-stale run is immutable and can serve as the stale
    # half of the Static cache-policy pair without spending a duplicate GPU run.
    for candidate in static_selection:
        key = (candidate.mode, candidate.config)
        if key in diagnostic_groups:
            diagnostic_groups[key].add(candidate.draft_cache_policy)
    incomplete = [
        key for key, policies in diagnostic_groups.items() if policies != {"stale", "rebuild"}
    ]
    if incomplete:
        raise ValueError(
            "each cache-policy diagnostic configuration requires explicit stale "
            "and rebuild candidates"
        )
    provenance = None
    if kind == DIAGNOSTIC_SPEC_KIND:
        provenance = _diagnostic_provenance(
            payload["provenance"],
            base_dir=path.parent,
            study_id=study_id,
            candidates=candidates,
        )
        _verify_diagnostic_derivation(payload, provenance, base_dir=path.parent)
    return CandidateSweep(
        path=path,
        file_sha256=frozen._sha256_file(path),
        content_sha256=frozen._sha256_json(payload),
        study_id=study_id,
        kind=kind,
        evidence_scope=evidence_scope,
        samples=tuple(samples),
        candidates=candidates,
        provenance=provenance,
    )


def _common_identities(
    args: argparse.Namespace,
    *,
    sweep: CandidateSweep,
) -> dict[str, Any]:
    harness = frozen._required_path(args.harness, directory=False, label="harness")
    python = frozen._required_path(
        args.python,
        directory=False,
        label="Python executable",
        resolve_symlinks=False,
    )
    if not os.access(python, os.X_OK):
        raise ValueError(f"Python executable is not executable: {python}")
    reference_root = frozen._required_path(
        args.reference_root, directory=True, label="reference root"
    )
    reference_source = frozen._required_path(
        str(reference_root.joinpath(*args.reference_module.split(".")).with_suffix(".py")),
        directory=False,
        label="reference module source",
    )
    target = frozen._required_path(args.target_model, directory=True, label="target model")
    draft = frozen._required_path(args.draft_model, directory=True, label="draft model")
    dataset = frozen._required_path(args.dataset, directory=False, label="dataset")
    pythonpath = tuple(
        str(frozen._required_path(path, directory=True, label="PYTHONPATH entry"))
        for path in args.pythonpath
    )
    output_root = Path(args.output_root).expanduser().resolve()
    if sweep.kind == DIAGNOSTIC_SPEC_KIND:
        if sweep.provenance is None:
            raise ValueError("diagnostic candidate specification lost provenance")
        isolation = sweep.provenance["selection_isolation"]
        expected_output_root = Path(
            isolation["diagnostic_output_root"]
        ).expanduser().resolve()
        if output_root != expected_output_root:
            raise ValueError(
                "diagnostic --output-root does not match the provenance-bound "
                f"path: expected={expected_output_root}, observed={output_root}"
            )
        if output_root == Path(isolation["source_output_root"]).expanduser().resolve():
            raise ValueError("diagnostic output root cannot reuse Stage-1 output root")
    lock_path = output_root / "artifact_identity_lock.json"
    lock = frozen._build_or_load_artifact_identity_lock(
        lock_path,
        target=target,
        target_revision=args.target_revision,
        draft=draft,
        draft_revision=args.draft_revision,
    )
    target_identity = lock["target"]
    projection = None
    has_residual = any(
        candidate.mode == "output-residual" for candidate in sweep.candidates
    )
    if has_residual:
        if args.projection_artifact is None:
            raise ValueError("output-residual candidates require --projection-artifact")
        projection = frozen._projection_identity(
            args.projection_artifact, target_identity=target_identity
        )
    elif args.projection_artifact is not None:
        raise ValueError(
            "--projection-artifact was provided but no output-residual candidate uses it"
        )
    return {
        "python": python,
        "harness": harness,
        "reference_root": reference_root,
        "reference_source": reference_source,
        "target_identity": target_identity,
        "draft_identity": lock["draft"],
        "tokenizer_identity": lock["tokenizer"],
        "dataset": dataset,
        "dataset_sha256": frozen._sha256_file(dataset),
        "pythonpath": pythonpath,
        "output_root": output_root,
        "lock_path": lock_path,
        "lock": lock,
        "lock_sha256": frozen._exclusive_json_file_sha256(lock),
        "projection": projection,
    }


def build_run_plans(args: argparse.Namespace) -> list[frozen.RunPlan]:
    _validate_args(args)
    sweep = load_candidate_sweep(args.candidate_spec)
    common = _common_identities(args, sweep=sweep)
    spec_identity = {
        "path": str(sweep.path),
        "file_sha256": sweep.file_sha256,
        "content_sha256": sweep.content_sha256,
        "study_id": sweep.study_id,
        "schema_version": SCHEMA_VERSION,
        "kind": sweep.kind,
        "evidence_scope": sweep.evidence_scope,
    }
    plans = []
    for sample in sweep.samples:
        sample_index = sample["sample_index"]
        input_tokens = sample["input_tokens"]
        for candidate_index, candidate in enumerate(sweep.candidates):
            run_dir = (
                common["output_root"]
                / f"sample-{sample_index:04d}"
                / candidate.candidate_id
            )
            identity = {
                "schema_version": SCHEMA_VERSION,
                "sweep": (
                    "dflash_tts_explicit_calibration_rank_v3"
                    if sweep.kind == SPEC_KIND
                    else "dflash_tts_explicit_diagnostics_v3"
                ),
                "candidate_specification": spec_identity,
                "calibration_candidate": {
                    "candidate_id": candidate.candidate_id,
                    "candidate_index": candidate_index,
                    "diagnostic_kind": candidate.diagnostic_kind,
                    "selection_eligible": candidate.selection_eligible,
                },
                "runtime": {
                    "python": str(common["python"]),
                    "calibration_orchestrator": {
                        "path": str(Path(__file__).resolve()),
                        "sha256": frozen._sha256_file(Path(__file__).resolve()),
                    },
                    "frozen_run_validator": {
                        "path": str(Path(frozen.__file__).resolve()),
                        "sha256": frozen._sha256_file(
                            Path(frozen.__file__).resolve()
                        ),
                    },
                    "harness": {
                        "path": str(common["harness"]),
                        "sha256": frozen._sha256_file(common["harness"]),
                    },
                    "device": args.device,
                    "dtype": args.dtype,
                    "attention_implementation": args.attn_implementation,
                    "determinism": frozen.determinism_contract(args.deterministic),
                    "pythonpath": list(common["pythonpath"]),
                    "artifact_identity_lock": {
                        "path": str(common["lock_path"]),
                        "sha256": common["lock_sha256"],
                    },
                },
                "reference": {
                    "root": str(common["reference_root"]),
                    "module": args.reference_module,
                    "revision": args.reference_revision,
                    "source_path": str(common["reference_source"]),
                    "source_sha256": frozen._sha256_file(common["reference_source"]),
                },
                "target": common["target_identity"],
                "draft": common["draft_identity"],
                "tokenizer": common["tokenizer_identity"],
                "dataset": {
                    "path": str(common["dataset"]),
                    "revision": args.dataset_revision,
                    "sha256": common["dataset_sha256"],
                    "sample_index": sample_index,
                    "prompt_field": args.prompt_field,
                    "messages_field": args.messages_field,
                    "turns_field": args.turns_field,
                    "enable_thinking": args.enable_thinking,
                    "input_tokens": input_tokens,
                    "input_token_source": {
                        "kind": "schema_v3_candidate_specification",
                        "candidate_specification_sha256": sweep.file_sha256,
                        "value": input_tokens,
                    },
                    "rendered_input_token_ids_sha256": sample[
                        "rendered_input_token_ids_sha256"
                    ],
                },
                "mode": candidate.mode,
                "generation": {
                    "requested_total_context": (
                        input_tokens + MAX_NEW_TOKENS + args.draft_block_size - 1
                    ),
                    "input_tokens": input_tokens,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "draft_block_size": args.draft_block_size,
                    "pending_draft_tokens": args.draft_block_size - 1,
                    "required_prefix_plus_block": (
                        input_tokens + MAX_NEW_TOKENS + args.draft_block_size - 1
                    ),
                    "stop_token_ids": None,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "mask_token_id": args.mask_token_id,
                },
                "optimization": {
                    "optimizer": candidate.config.optimizer,
                    "learning_rate": candidate.config.learning_rate,
                    "weight_decay": candidate.config.weight_decay,
                    "rank": candidate.config.rank,
                    "adapter_seed": (
                        args.adapter_seed if candidate.config.rank is not None else None
                    ),
                    "proximal_lambda": args.proximal_lambda,
                    "update_stride": args.update_stride,
                    "position_weighting": args.position_weighting,
                    "position_decay_gamma": args.position_decay_gamma,
                    "loss_reduction": args.loss_reduction,
                    "adam_betas": [args.adam_beta1, args.adam_beta2],
                    "adam_eps": args.adam_eps,
                    "draft_cache_policy": candidate.draft_cache_policy,
                },
                "projection": (
                    common["projection"]
                    if candidate.mode == "output-residual"
                    else None
                ),
                "audit": {
                    "cuda_timing": args.audit_cuda_timing,
                    "parameter_audit_stride": candidate.parameter_audit_stride,
                    "parity_max_new_tokens": args.parity_max_new_tokens,
                    "skip_static_parity_preflight": args.skip_static_parity_preflight,
                },
            }
            artifact_dir = run_dir / "artifact"
            identity_sha256 = frozen._sha256_json(identity)
            plans.append(
                frozen.RunPlan(
                    run_dir=run_dir,
                    artifact_dir=artifact_dir,
                    log_path=run_dir / "run.log",
                    identity_path=run_dir / "run_identity.json",
                    completion_path=run_dir / "completion.json",
                    identity=identity,
                    identity_sha256=identity_sha256,
                    command=frozen._command(
                        identity,
                        artifact_dir,
                        run_identity_sha256=identity_sha256,
                    ),
                    pythonpath=common["pythonpath"],
                    artifact_identity_lock_path=common["lock_path"],
                    artifact_identity_lock_payload=common["lock"],
                )
            )
    return plans


def _validate_specification_unchanged(plan: frozen.RunPlan) -> None:
    identity = plan.identity["candidate_specification"]
    path = Path(identity["path"])
    if not path.is_file() or frozen._sha256_file(path) != identity["file_sha256"]:
        raise ValueError("candidate specification changed after run-plan construction")
    if frozen._sha256_json(frozen._read_json(path)) != identity["content_sha256"]:
        raise ValueError("candidate specification canonical content changed")


def _completed_run_matches(plan: frozen.RunPlan) -> bool:
    return frozen.completed_run_matches(plan)


def execute_plan(plan: frozen.RunPlan) -> str:
    _validate_specification_unchanged(plan)
    frozen._ensure_artifact_identity_lock(plan)
    return frozen._execute_resumable_plan(plan)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--harness",
        default=str(Path(__file__).with_name("dflash_tts_reference.py")),
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
    parser.add_argument("--output-root", required=True)
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
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mask-token-id", type=int)
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
    parser.add_argument("--projection-artifact")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--audit-cuda-timing", action="store_true")
    parser.add_argument("--parity-max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-static-parity-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.draft_block_size <= 1:
        raise ValueError("--draft-block-size must be greater than one")
    if args.temperature < 0 or args.proximal_lambda < 0:
        raise ValueError("temperature and proximal lambda must be non-negative")
    if args.update_stride <= 0 or args.position_decay_gamma <= 0:
        raise ValueError("update stride and position decay gamma must be positive")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if args.adam_eps <= 0 or args.parity_max_new_tokens <= 0:
        raise ValueError("Adam epsilon and parity max tokens must be positive")
    if args.mask_token_id is None or args.mask_token_id < 0:
        raise ValueError("--mask-token-id must be explicitly bound and non-negative")
    if not args.deterministic:
        raise ValueError(
            "calibration selection requires --deterministic; nondeterministic "
            "runs must use a separate diagnostic entry point"
        )
    if args.skip_static_parity_preflight:
        raise ValueError(
            "calibration selection cannot skip Static parity; parity-skip runs "
            "must use a separate diagnostic entry point"
        )
    if args.audit_cuda_timing:
        raise ValueError(
            "calibration selection cannot enable synchronized CUDA timing; "
            "profiling must use a separate diagnostic entry point"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    plans = build_run_plans(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "dry_run",
                    "candidate_specification": (
                        plans[0].identity["candidate_specification"] if plans else None
                    ),
                    "runs": [
                        {
                            "run_dir": str(plan.run_dir),
                            "identity_sha256": plan.identity_sha256,
                            "candidate": plan.identity["calibration_candidate"],
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

    counts = {"completed": 0, "resumed_complete": 0, "failed": 0}
    failures = []
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
                        "candidate": plan.identity["calibration_candidate"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            counts["failed"] += 1
            failure = {"run_dir": str(plan.run_dir), "error": str(exc)}
            failures.append(failure)
            print(
                json.dumps({**failure, "status": "failed"}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            if not args.keep_going:
                break
    print(json.dumps({"counts": counts, "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
