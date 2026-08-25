"""Local prompt-pool loading for efficiency experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

PROMPT_FIELDS = (
    "prompt",
    "problem",
    "question",
    "question_content",
    "text",
    "instruction",
    "input",
)
ID_FIELDS = ("problem_id", "question_id", "unique_id", "uid", "id", "task_id")
CALIBRATION_SOURCES = {
    "APPS": 24,
    "OpenR1-Math": 24,
    "UltraChat": 24,
    "controlled_synthetic": 4,
}


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
    path: Path,
    *,
    limit: int,
    selection_seed: int = 0,
    allow_repeat: bool = False,
) -> tuple[dict[str, Any], ...]:
    available = load_prompt_pool(path)
    if not available:
        raise ValueError(f"dataset {path} supplied no prompts")
    if len(available) < limit and not allow_repeat:
        raise ValueError(f"dataset {path} supplied {len(available)} prompts; {limit} required")
    selected: list[dict[str, Any]] = []
    cycle = 0
    while len(selected) < limit:
        indexes = list(range(len(available)))
        random.Random(selection_seed + cycle).shuffle(indexes)
        for index in indexes:
            row = dict(available[index])
            row["repeat_index"] = cycle
            selected.append(row)
            if len(selected) == limit:
                break
        cycle += 1
    return tuple(selected)


def load_prompt_pool(path: Path) -> tuple[dict[str, Any], ...]:
    available: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, row in enumerate(_rows(path)):
        turns = row.get("turns")
        if turns is not None and (
            not isinstance(turns, list)
            or not all(isinstance(turn, str) and turn.strip() for turn in turns)
        ):
            raise ValueError(f"dataset row {index} turns must be non-empty text")
        value = next(
            (
                row[field]
                for field in PROMPT_FIELDS
                if isinstance(row.get(field), str) and row[field].strip()
            ),
            None,
        )
        if value is None and turns:
            value = "\n".join(turns)
        identity = next(
            (str(row[field]) for field in ID_FIELDS if row.get(field) is not None),
            None,
        )
        if identity is None:
            raise ValueError(f"dataset row {index} requires problem_id")
        if identity in identities:
            raise ValueError(f"dataset {path} repeats problem_id {identity}")
        identities.add(identity)
        if value is None:
            raise ValueError(f"dataset row {index} requires a prompt or template input")
        template = row.get("template")
        if template is not None:
            if not isinstance(template, str):
                raise ValueError(f"dataset row {index} template must be text")
            try:
                value = template.format(**row)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"dataset row {index} template cannot be rendered: {error}"
                ) from error
        if turns is not None:
            value = "\n".join(turns)
        available.append(
            {
                "problem_id": identity,
                "prompt": value,
                "template": template,
                "turns": turns,
                "source": row.get("source"),
            }
        )
    return tuple(available)


def load_prompts(path: Path, *, limit: int, offset: int = 0) -> tuple[str, ...]:
    return tuple(
        row["prompt"]
        for row in load_prompt_records(
            path, limit=limit, selection_seed=offset
        )
    )


def load_calibration_mix(path: Path) -> tuple[str, ...]:
    rows = load_prompt_pool(path)
    counts = {
        source: sum(row["source"] == source for row in rows)
        for source in CALIBRATION_SOURCES
    }
    if len(rows) != 76 or counts != CALIBRATION_SOURCES:
        raise ValueError(
            "CalibrationMix requires 24 APPS, 24 OpenR1-Math, "
            "24 UltraChat, and 4 controlled_synthetic prompts"
        )
    return tuple(row["prompt"] for row in rows)


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
