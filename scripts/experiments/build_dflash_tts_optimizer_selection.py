#!/usr/bin/env python3
"""Deterministically rebuild the frozen DFlash optimizer-selection evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from aggregate_dflash_tts_ablations import (  # noqa: E402
    _canonical_json,
    _read_json,
    _read_jsonl,
    _sha256_file,
    _sha256_json,
)


MODE_ORDER = (
    "static",
    "full-drafter",
    "drafter-lora",
    "full-rank-tail",
    "tail-lora",
    "output-residual",
)
SAMPLE_IDS = (
    "math500:sample_index=0",
    "math500:sample_index=419",
)
STAGE_SPECS = {
    "512_screening": {
        "context_length": 512,
        "purpose": "full_grid_screening_only",
        "sample_directories": {
            SAMPLE_IDS[0]: "calib-a-512-s0",
            SAMPLE_IDS[1]: "calib-a-512-s419",
        },
    },
    "2048_selection": {
        "context_length": 2048,
        "purpose": "deterministic_optimizer_selection",
        "sample_directories": {
            SAMPLE_IDS[0]: "calib-b-2048-s0",
            SAMPLE_IDS[1]: "calib-b-2048-s419",
        },
    },
}
CANDIDATE_GRID = {
    "static": {
        "optimizers": ["adam"],
        "learning_rates": [0.0001],
        "weight_decays": [0.0],
        "ranks": [None],
    },
    "full-drafter": {
        "optimizers": ["adam", "adamw"],
        "learning_rates": [1e-05, 3e-05, 0.0001, 0.0003],
        "weight_decays": [0.0, 0.001, 0.01],
        "ranks": [None],
    },
    "drafter-lora": {
        "optimizers": ["adam", "adamw"],
        "learning_rates": [3e-05, 0.0001, 0.0003, 0.001, 0.003],
        "weight_decays": [0.0, 0.001, 0.01],
        "ranks": [8, 16, 32],
    },
    "full-rank-tail": {
        "optimizers": ["adam", "adamw"],
        "learning_rates": [3e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.001],
        "weight_decays": [0.0, 0.001, 0.01],
        "ranks": [None],
    },
    "tail-lora": {
        "optimizers": ["adam", "adamw"],
        "learning_rates": [0.0001, 0.0003, 0.001, 0.003],
        "weight_decays": [0.0, 0.001, 0.01],
        "ranks": [8, 16, 32],
    },
    "output-residual": {
        "optimizers": ["adam", "adamw"],
        "learning_rates": [0.0003, 0.001, 0.003, 0.01, 0.03],
        "weight_decays": [0.0, 0.001, 0.01],
        "ranks": [8, 16, 32],
    },
}
SELECTION_RULE = {
    "eligibility": (
        "both calibration samples have exact output tokens and finite "
        "acceptance/loss/memory metrics"
    ),
    "primary_metric": (
        "mean paired paper acceptance length gain at context 2048"
    ),
    "aggregation": "unweighted mean over the two locked calibration samples",
    "tie_breakers": [
        "higher worst-prompt paired acceptance-length gain",
        "lower maximum run peak HBM over the two calibration samples",
        "fewer trainable parameters",
        (
            "lexicographically smaller canonical (optimizer, learning_rate, "
            "weight_decay, rank) tuple only when every preregistered scientific "
            "criterion is exactly tied"
        ),
    ],
    "leakage_control": (
        "512-token artifacts screen candidates; only the locked 2048-token "
        "calibration pairs select; long-context evaluation artifacts cannot "
        "change this selection"
    ),
    "deterministic_order_version": (
        "paired_al_mean_worst_hbm_params_canonical_v1"
    ),
}


def _render(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _token_ids_sha256(values: list[int]) -> str:
    return _sha256_json([int(value) for value in values])


def _mode_config(summary: dict[str, Any]) -> dict[str, Any]:
    parameters = summary["parameters"]
    optimizer = parameters["optimizer"]
    return {
        "optimizer": "adam" if optimizer is None else str(optimizer).lower(),
        "learning_rate": float(parameters["lr"]),
        "weight_decay": float(parameters["weight_decay"]),
        "rank": parameters["rank"],
    }


def _loss_stats(rounds_path: Path) -> dict[str, Any]:
    values = [
        float(row["update"]["loss"])
        for row in _read_jsonl(rounds_path)
        if (row.get("update") or {}).get("applied")
    ]
    return {
        "applied_steps": len(values),
        "all_finite": all(math.isfinite(value) for value in values),
        "first": values[0] if values else None,
        "final": values[-1] if values else None,
        "mean": statistics.fmean(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _artifact_pair(
    run_dir: Path,
    *,
    project_root: Path,
    artifact_records: list[dict[str, str]],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for kind, filename in (("summary", "summary.json"), ("rounds", "rounds.jsonl")):
        path = run_dir / filename
        relative = str(path.relative_to(project_root))
        digest = _sha256_file(path)
        output[f"{kind}_path"] = relative
        output[f"{kind}_sha256"] = digest
        artifact_records.append({"path": relative, "sha256": digest})
    return output


def _sample_result(
    run_dir: Path,
    *,
    sample_id: str,
    baseline: dict[str, Any],
    project_root: Path,
    artifact_records: list[dict[str, str]],
) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    candidate_al = statistics.fmean(
        float(value) for value in summary["generation"]["acceptance_lengths"]
    )
    static_al = statistics.fmean(
        float(value) for value in baseline["generation"]["acceptance_lengths"]
    )
    candidate_output = _token_ids_sha256(summary["output"]["token_ids"])
    static_output = _token_ids_sha256(baseline["output"]["token_ids"])
    loss = _loss_stats(run_dir / "rounds.jsonl")
    return {
        "sample_id": sample_id,
        "candidate_paper_acceptance_length": candidate_al,
        "static_paper_acceptance_length": static_al,
        "paired_paper_acceptance_length_gain": candidate_al - static_al,
        "exact_output_token_ids": (
            summary["output"]["token_ids"] == baseline["output"]["token_ids"]
        ),
        "candidate_output_token_ids_sha256": candidate_output,
        "static_output_token_ids_sha256": static_output,
        "finite_metrics": (
            math.isfinite(candidate_al)
            and math.isfinite(static_al)
            and loss["all_finite"]
        ),
        "peak_hbm_bytes": int(summary["generation"]["peak_hbm_bytes"]),
        "trainable_parameter_count": int(
            summary["generation"]["trainable_parameter_count"]
        ),
        "update_loss": loss,
        "artifacts": _artifact_pair(
            run_dir,
            project_root=project_root,
            artifact_records=artifact_records,
        ),
    }


def _build_stage(
    calibration_root: Path,
    *,
    spec: dict[str, Any],
    project_root: Path,
    artifact_records: list[dict[str, str]],
) -> dict[str, Any]:
    sample_directories = spec["sample_directories"]
    baseline_summaries: dict[str, dict[str, Any]] = {}
    baselines = []
    for sample_id, directory in sample_directories.items():
        run_dir = calibration_root / directory / "static"
        summary = _read_json(run_dir / "summary.json")
        baseline_summaries[sample_id] = summary
        baselines.append(
            {
                "sample_id": sample_id,
                "config": _mode_config(summary),
                "paper_acceptance_length": statistics.fmean(
                    float(value)
                    for value in summary["generation"]["acceptance_lengths"]
                ),
                "output_token_ids_sha256": _token_ids_sha256(
                    summary["output"]["token_ids"]
                ),
                "peak_hbm_bytes": int(summary["generation"]["peak_hbm_bytes"]),
                "trainable_parameter_count": int(
                    summary["generation"]["trainable_parameter_count"]
                ),
                "update_loss": _loss_stats(run_dir / "rounds.jsonl"),
                "artifacts": _artifact_pair(
                    run_dir,
                    project_root=project_root,
                    artifact_records=artifact_records,
                ),
            }
        )

    candidate_ids = sorted(
        {
            path.name
            for directory in sample_directories.values()
            for path in (calibration_root / directory).iterdir()
            if path.is_dir() and path.name not in {"logs", "static"}
        }
    )
    candidates = []
    for candidate_id in candidate_ids:
        mode = None
        config = None
        sample_results = []
        for sample_id, directory in sample_directories.items():
            run_dir = calibration_root / directory / candidate_id
            if not (run_dir / "summary.json").is_file():
                continue
            summary = _read_json(run_dir / "summary.json")
            current_mode = summary["mode"]
            current_config = _mode_config(summary)
            if mode is not None and (mode, config) != (current_mode, current_config):
                raise ValueError(f"{candidate_id}: configuration differs by sample")
            mode, config = current_mode, current_config
            sample_results.append(
                _sample_result(
                    run_dir,
                    sample_id=sample_id,
                    baseline=baseline_summaries[sample_id],
                    project_root=project_root,
                    artifact_records=artifact_records,
                )
            )
        if mode is None or config is None:
            raise ValueError(f"{candidate_id}: no complete artifact")
        complete = len(sample_results) == len(sample_directories)
        eligible = complete and all(
            result["exact_output_token_ids"] and result["finite_metrics"]
            for result in sample_results
        )
        gains = [
            result["paired_paper_acceptance_length_gain"]
            for result in sample_results
        ]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "mode": mode,
                "config": config,
                "sample_results": sample_results,
                "aggregate": {
                    "complete_two_prompt_pair": complete,
                    "eligible": eligible,
                    "mean_paired_paper_acceptance_length_gain": (
                        statistics.fmean(gains) if complete else None
                    ),
                    "worst_prompt_paired_paper_acceptance_length_gain": (
                        min(gains) if complete else None
                    ),
                    "max_peak_hbm_bytes": (
                        max(result["peak_hbm_bytes"] for result in sample_results)
                        if complete
                        else None
                    ),
                    "max_trainable_parameter_count": (
                        max(
                            result["trainable_parameter_count"]
                            for result in sample_results
                        )
                        if complete
                        else None
                    ),
                },
            }
        )
    return {
        "context_length": spec["context_length"],
        "purpose": spec["purpose"],
        "sample_directories": sample_directories,
        "baselines": baselines,
        "observed_candidates": sorted(
            candidates,
            key=lambda row: (MODE_ORDER.index(row["mode"]), row["candidate_id"]),
        ),
    }


def _selection_order(candidate: dict[str, Any]) -> tuple[Any, ...]:
    aggregate = candidate["aggregate"]
    config = candidate["config"]
    return (
        -float(aggregate["mean_paired_paper_acceptance_length_gain"]),
        -float(aggregate["worst_prompt_paired_paper_acceptance_length_gain"]),
        int(aggregate["max_peak_hbm_bytes"]),
        int(aggregate["max_trainable_parameter_count"]),
        config["optimizer"],
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        -1 if config["rank"] is None else int(config["rank"]),
    )


def build_selection_summary(
    *, calibration_root: Path, project_root: Path
) -> dict[str, Any]:
    project_root = project_root.resolve()
    calibration_root = calibration_root.resolve()
    artifact_records: list[dict[str, str]] = []
    stages = {
        name: _build_stage(
            calibration_root,
            spec=spec,
            project_root=project_root,
            artifact_records=artifact_records,
        )
        for name, spec in STAGE_SPECS.items()
    }
    selection_candidates = stages["2048_selection"]["observed_candidates"]
    selected_configs = {
        "static": stages["2048_selection"]["baselines"][0]["config"]
    }
    decisions = {}
    for mode in MODE_ORDER[1:]:
        eligible = sorted(
            (
                candidate
                for candidate in selection_candidates
                if candidate["mode"] == mode and candidate["aggregate"]["eligible"]
            ),
            key=_selection_order,
        )
        if not eligible:
            raise ValueError(f"no eligible 2048 candidate for {mode}")
        winner = eligible[0]
        selected_configs[mode] = winner["config"]
        decisions[mode] = {
            "winner_candidate_id": winner["candidate_id"],
            "winner_config": winner["config"],
            "winner_aggregate": winner["aggregate"],
            "eligible_candidate_order": [
                candidate["candidate_id"] for candidate in eligible
            ],
        }

    artifact_set = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(
            {(row["path"], row["sha256"]) for row in artifact_records}
        )
    ]
    return {
        "schema_version": 1,
        "status": "complete_local_calibration_selection",
        "study_id": "dflash_tts_two_prompt_512_grid_2k_selection_v1",
        "evidence_classification": (
            "legacy_schema_v1_optimizer_screening_only_not_formal_identity_"
            "runtime_hbm_or_speed_evidence"
        ),
        "artifact_root": str(calibration_root.relative_to(project_root)),
        "calibration": {
            "sample_ids": list(SAMPLE_IDS),
            "seed": 0,
            "enable_thinking": True,
            "stop_token_ids": None,
        },
        "candidate_grid": CANDIDATE_GRID,
        "candidate_grid_sha256": _sha256_json(CANDIDATE_GRID),
        "selection_rule": SELECTION_RULE,
        "selection_rule_sha256": _sha256_json(SELECTION_RULE),
        "stages": stages,
        "selected_configs": selected_configs,
        "selection_decisions": decisions,
        "source_artifact_count": len(artifact_set),
        "source_artifact_set_sha256": _sha256_json(artifact_set),
        "limitations": [
            (
                "The source runs use legacy schema-v1 identities and therefore "
                "lock optimizer selection only; they are not eligible for formal "
                "model-identity, runtime, HBM Pareto, or speed claims."
            ),
            "The selection uses two fixed Math500 prompts and has no confidence interval.",
            (
                "The 512-token stage is screening evidence only; deterministic "
                "selection uses only exact paired 2048-token results."
            ),
            "Long-context evaluation artifacts are excluded from optimizer selection.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    default_project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--calibration-root", type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    calibration_root = (
        args.calibration_root.resolve()
        if args.calibration_root is not None
        else project_root
        / "artifacts/tts-dflash-repro/2026-08-03/calibration-512-2k"
    )
    payload = build_selection_summary(
        calibration_root=calibration_root,
        project_root=project_root,
    )
    body = _render(payload)
    if args.check is not None:
        expected = args.check.resolve()
        if expected.read_text(encoding="utf-8") != body:
            raise ValueError(f"selection summary is stale: {expected}")
        print(_sha256_file(expected))
        return 0
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, output)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        print(_sha256_file(output))
        return 0
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
