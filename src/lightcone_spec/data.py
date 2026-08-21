"""Local benchmark prompt loading."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

PROMPT_FIELDS = ("prompt", "problem", "question", "text", "instruction", "input")
ID_FIELDS = ("problem_id", "question_id", "id", "task_id")


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        raise ValueError(f"dataset path must name one explicit file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "examples", "rows", "items"):
            if isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
    raise ValueError(f"dataset {path} does not contain row objects")


def load_prompt_records(
    path: Path, *, limit: int, selection_seed: int = 0
) -> tuple[dict[str, Any], ...]:
    available = load_prompt_pool(path)
    if len(available) < limit:
        raise ValueError(f"dataset {path} supplied {len(available)} prompts; {limit} required")
    indexes = list(range(len(available)))
    random.Random(selection_seed).shuffle(indexes)
    return tuple(available[index] for index in indexes[:limit])


def load_prompt_pool(path: Path) -> tuple[dict[str, Any], ...]:
    available: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(path)):
        value = next((row[field] for field in PROMPT_FIELDS if isinstance(row.get(field), str) and row[field].strip()), None)
        if value is not None:
            identity = next(
                (str(row[field]) for field in ID_FIELDS if row.get(field) is not None),
                f"row-{index:06d}",
            )
            available.append(
                {
                    "problem_id": identity,
                    "split": row.get("split"),
                    "prompt": value,
                    "template": row.get("template"),
                    "reference": row.get("reference", row.get("answer")),
                    "test_metadata": row.get("test_metadata", row.get("tests")),
                }
            )
    return tuple(available)


def load_prompts(path: Path, *, limit: int, offset: int = 0) -> tuple[str, ...]:
    return tuple(
        row["prompt"]
        for row in load_prompt_records(path, limit=limit, selection_seed=offset)
    )


def load_tts_calibration(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    identified: list[tuple[str, str, str]] = []
    for index, row in enumerate(_rows(path)):
        prompt = next(
            (
                row[field]
                for field in PROMPT_FIELDS
                if isinstance(row.get(field), str) and row[field].strip()
            ),
            None,
        )
        if prompt is None:
            continue
        identity = next(
            (str(row[field]) for field in ID_FIELDS if row.get(field) is not None),
            None,
        )
        split = row.get("split")
        if identity is None or split not in {"tuning", "holdout"}:
            continue
        identified.append((identity, prompt, split))
    tuning_rows = sorted(
        ((identity, prompt) for identity, prompt, split in identified if split == "tuning")
    )
    holdout_rows = sorted(
        ((identity, prompt) for identity, prompt, split in identified if split == "holdout")
    )
    if len(tuning_rows) != 76 or len(holdout_rows) != 4:
        raise ValueError("TTS-Cal requires explicit 76 tuning and 4 holdout rows")
    holdout = tuple(identity for identity, _ in holdout_rows)
    tuning = tuple(prompt for _, prompt in tuning_rows)
    return tuning, holdout


def load_arrival_offsets(path: Path, *, limit: int, offset: int = 0) -> tuple[float, ...]:
    offsets, _ = load_arrival_trace(path, limit=limit, offset=offset)
    return offsets


def load_arrival_trace(
    path: Path, *, limit: int, offset: int = 0
) -> tuple[tuple[float, ...], tuple[tuple[int, int], ...]]:
    rows = _rows(path)
    values: list[tuple[float, int, int]] = []
    for row in rows:
        raw = next(
            (
                value
                for name, value in row.items()
                if name.lower()
                in {"arrival_time", "arrival_time_seconds", "timestamp", "time", "arrival"}
                and isinstance(value, (int, float))
            ),
            None,
        )
        input_length = next(
            (
                value
                for name, value in row.items()
                if name.lower() in {"input_length", "prompt_length", "input_tokens"}
                and isinstance(value, int)
                and value > 0
            ),
            None,
        )
        output_length = next(
            (
                value
                for name, value in row.items()
                if name.lower() in {"output_length", "completion_length", "output_tokens"}
                and isinstance(value, int)
                and value > 0
            ),
            None,
        )
        if raw is not None and input_length is not None and output_length is not None:
            values.append((float(raw), input_length, output_length))
    if len(values) < limit:
        raise ValueError(
            f"arrival trace {path} supplied {len(values)} complete rows; {limit} required"
        )
    start = random.Random(offset).randrange(len(values) - limit + 1)
    selected = values[start : start + limit]
    origin = selected[0][0]
    offsets = tuple(value[0] - origin for value in selected)
    if offsets[0] != 0 or any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"arrival trace {path} is not monotone")
    lengths = tuple((value[1], value[2]) for value in selected)
    return offsets, lengths
