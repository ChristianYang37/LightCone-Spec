from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "build_dflash_tts_optimizer_selection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_dflash_tts_optimizer_selection", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(
    run_dir: Path,
    *,
    mode: str,
    optimizer: str | None,
    learning_rate: float,
    weight_decay: float,
    rank: int | None,
    acceptance_lengths: list[int],
    peak_hbm_bytes: int,
    trainable_parameter_count: int,
) -> None:
    run_dir.mkdir(parents=True)
    summary = {
        "mode": mode,
        "parameters": {
            "optimizer": optimizer,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "rank": rank,
        },
        "generation": {
            "acceptance_lengths": acceptance_lengths,
            "peak_hbm_bytes": peak_hbm_bytes,
            "trainable_parameter_count": trainable_parameter_count,
        },
        "output": {"token_ids": [10, 20, 30]},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    update = (
        {"applied": False}
        if mode == "static"
        else {"applied": True, "loss": 1.25}
    )
    (run_dir / "rounds.jsonl").write_text(
        json.dumps({"update": update}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_calibration(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    calibration_root = project_root / "artifacts" / "calibration"
    configs = {
        "full-drafter-adam": ("full-drafter", "ADAM", 1e-5, 0.0, None, [2, 2]),
        "drafter-lora-adam": ("drafter-lora", "ADAM", 3e-4, 0.0, 8, [2, 2]),
        "full-rank-tail-adam": (
            "full-rank-tail",
            "ADAM",
            3e-6,
            0.0,
            None,
            [2, 2],
        ),
        "tail-lora-adam": ("tail-lora", "ADAM", 1e-4, 0.0, 16, [2, 1]),
        "tail-lora-adamw": (
            "tail-lora",
            "ADAMW",
            1e-4,
            0.01,
            16,
            [3, 1],
        ),
        "output-residual-adam": (
            "output-residual",
            "ADAM",
            3e-4,
            0.0,
            16,
            [2, 2],
        ),
    }
    for spec in builder.STAGE_SPECS.values():
        for sample_directory in spec["sample_directories"].values():
            sample_root = calibration_root / sample_directory
            _write_run(
                sample_root / "static",
                mode="static",
                optimizer=None,
                learning_rate=1e-4,
                weight_decay=0.0,
                rank=None,
                acceptance_lengths=[1, 1],
                peak_hbm_bytes=100,
                trainable_parameter_count=0,
            )
            for candidate_id, values in configs.items():
                mode, optimizer, lr, weight_decay, rank, acceptance = values
                _write_run(
                    sample_root / candidate_id,
                    mode=mode,
                    optimizer=optimizer,
                    learning_rate=lr,
                    weight_decay=weight_decay,
                    rank=rank,
                    acceptance_lengths=acceptance,
                    peak_hbm_bytes=200,
                    trainable_parameter_count=10,
                )
    return project_root, calibration_root


def test_generator_is_byte_stable_and_checkable(tmp_path: Path, capsys):
    project_root, calibration_root = _synthetic_calibration(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    common = [
        "--project-root",
        str(project_root),
        "--calibration-root",
        str(calibration_root),
    ]
    assert builder.main([*common, "--output", str(first)]) == 0
    capsys.readouterr()
    assert builder.main([*common, "--output", str(second)]) == 0
    capsys.readouterr()
    assert first.read_bytes() == second.read_bytes()
    assert _sha256(first) == _sha256(second)
    assert builder.main([*common, "--check", str(first)]) == 0

    payload = json.loads(first.read_text())
    assert payload["selection_decisions"]["tail-lora"][
        "winner_candidate_id"
    ] == "tail-lora-adamw"
    first.write_text(first.read_text() + "\n")
    with pytest.raises(ValueError, match="selection summary is stale"):
        builder.main([*common, "--check", str(first)])


def test_result_derived_selection_is_not_checked_in():
    summary_path = SCRIPT.with_name("selection_summary.json")
    config_path = SCRIPT.with_name("selected_optimizer_config.json")
    assert not summary_path.exists()
    assert not config_path.exists()
