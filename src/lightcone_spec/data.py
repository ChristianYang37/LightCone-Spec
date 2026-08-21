"""Local benchmark prompt loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROMPT_FIELDS = ("prompt", "problem", "question", "text", "instruction", "input")
ID_FIELDS = ("problem_id", "question_id", "id", "task_id")


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        candidates = sorted(
            item
            for pattern in ("*.jsonl", "*.json", "*.parquet", "*.csv")
            for item in path.rglob(pattern)
        )
        if not candidates:
            raise FileNotFoundError(f"no JSON, JSONL, or Parquet data under {path}")
        path = candidates[0]
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


def load_prompts(path: Path, *, limit: int, offset: int = 0) -> tuple[str, ...]:
    rows = _rows(path)
    available: list[str] = []
    for row in rows:
        value = next((row[field] for field in PROMPT_FIELDS if isinstance(row.get(field), str) and row[field].strip()), None)
        if value is not None:
            available.append(value)
    if len(available) < limit:
        raise ValueError(f"dataset {path} supplied {len(available)} prompts; {limit} required")
    start = offset % len(available)
    return tuple(available[(start + index) % len(available)] for index in range(limit))


def load_tts_calibration(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    identified: list[tuple[str, str]] = []
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
            f"row-{index:06d}",
        )
        identified.append((identity, prompt))
    identified.sort(key=lambda row: row[0])
    if len(identified) < 80:
        raise ValueError("TTS-Cal requires at least 80 identified LCB-v6-hard problems")
    domain = identified[:80]
    holdout = tuple(identity for identity, _ in domain[:4])
    tuning = tuple(prompt for _, prompt in domain[4:])
    return tuning, holdout


def load_arrival_offsets(path: Path, *, limit: int, offset: int = 0) -> tuple[float, ...]:
    rows = _rows(path)
    values: list[float] = []
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
        if raw is not None:
            values.append(float(raw))
    if len(values) < limit:
        raise ValueError(
            f"arrival trace {path} supplied {len(values)} timestamps; {limit} required"
        )
    start = offset % (len(values) - limit + 1)
    selected = values[start : start + limit]
    origin = selected[0]
    offsets = tuple(value - origin for value in selected)
    if offsets[0] != 0 or any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"arrival trace {path} is not monotone")
    return offsets
