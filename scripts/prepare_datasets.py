#!/usr/bin/env python3
"""Normalize prompt pools and build the fixed 76-row CalibrationMix."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from lightcone_spec.data import (
    CALIBRATION_SOURCES,
    ID_FIELDS,
    PROMPT_FIELDS,
    _rows,
    load_calibration_mix,
    load_prompt_pool,
)


def _source(value: str) -> tuple[str, Path]:
    task, separator, path = value.partition("=")
    if not separator or not task or not path:
        raise argparse.ArgumentTypeError("--task requires NAME=/absolute/source")
    source = Path(path)
    if not source.is_absolute():
        raise argparse.ArgumentTypeError("dataset source paths must be absolute")
    return task, source


def _first(row: dict, fields: tuple[str, ...]):
    return next(
        (
            row[name]
            for name in fields
            if row.get(name) is not None and row.get(name) != ""
        ),
        None,
    )


def _turns(row: dict) -> list[str] | None:
    turns = row.get("turns")
    if isinstance(turns, list) and all(isinstance(turn, str) for turn in turns):
        return turns
    messages = row.get("messages")
    if isinstance(messages, list):
        values = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if values and all(isinstance(value, str) for value in values):
            return values
    return None


def _normalize(task: str, source: Path) -> list[dict[str, object]]:
    normalized = []
    for index, row in enumerate(_rows(source)):
        turns = _turns(row)
        prompt = _first(row, PROMPT_FIELDS)
        if prompt is None and turns:
            prompt = "\n".join(turns)
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        template = row.get("template")
        if isinstance(template, str):
            prompt = template.format(**row)
        problem_id = _first(row, ID_FIELDS)
        normalized.append(
            {
                "problem_id": str(problem_id or f"{task}-{index:08d}"),
                "prompt": prompt,
                "turns": turns,
                "source": task,
            }
        )
    if not normalized:
        raise ValueError(f"{task} supplied no usable prompts")
    return normalized


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_calibration(path: Path, output: Path) -> None:
    selected: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    counts = {row.get("source"): int(row.get("count", 0)) for row in manifest}
    if counts != CALIBRATION_SOURCES:
        raise ValueError(
            "calibration manifest must request 24 APPS, 24 OpenR1-Math, "
            "24 UltraChat, and 4 controlled_synthetic prompts"
        )
    by_source = {str(row["source"]): row for row in manifest}
    for source_name in CALIBRATION_SOURCES:
        row = by_source[source_name]
        source_path = Path(str(row["path"]))
        if not source_path.is_absolute():
            raise ValueError("calibration source paths must be absolute")
        candidates = _normalize(source_name, source_path)
        random.Random(0).shuffle(candidates)
        count = int(row["count"])
        if len(candidates) < count:
            raise ValueError(f"{source_name} supplied {len(candidates)} prompts; {count} required")
        selected.extend(
            {
                **candidate,
                "problem_id": f"{source_name}:{candidate['problem_id']}",
                "source": source_name,
            }
            for candidate in candidates[:count]
        )
    _write(output, selected)
    load_calibration_mix(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", type=_source, default=[])
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for task, source in args.task:
        output = args.output_root / f"{task}.jsonl"
        _write(output, _normalize(task, source))
        load_prompt_pool(output)
    if args.calibration_manifest is not None:
        _build_calibration(
            args.calibration_manifest,
            args.output_root / "CalibrationMix.jsonl",
        )


if __name__ == "__main__":
    main()
