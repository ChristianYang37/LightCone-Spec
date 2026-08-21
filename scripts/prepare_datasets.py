#!/usr/bin/env python3
"""Convert explicitly selected benchmark rows to LightCone JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lightcone_spec.data import ID_FIELDS, PROMPT_FIELDS, _rows, load_prompt_pool


def _assignments(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["task"], row["problem_id"])
            if key in result:
                raise ValueError(f"duplicate split assignment {key}")
            result[key] = row["split"]
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", type=_source, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    split_by_id = _assignments(args.splits)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for task, source in args.task:
        output = args.output_root / f"{task}.jsonl"
        found: set[str] = set()
        with output.open("w", encoding="utf-8") as stream:
            for row in _rows(source):
                problem_id = _first(row, ID_FIELDS)
                prompt = _first(row, PROMPT_FIELDS)
                turns = row.get("turns")
                if prompt is None and isinstance(turns, list) and turns:
                    prompt = "\n".join(str(turn) for turn in turns)
                if problem_id is None or prompt is None:
                    continue
                problem_id = str(problem_id)
                split = split_by_id.get((task, problem_id))
                if split is None:
                    continue
                found.add(problem_id)
                reference = _first(
                    row,
                    (
                        "reference",
                        "answer",
                        "solution",
                        "canonical_solution",
                        "output",
                        "code",
                    ),
                )
                test_metadata = _first(
                    row,
                    (
                        "test_metadata",
                        "tests",
                        "test",
                        "test_code",
                        "test_list",
                    ),
                )
                if reference is None and test_metadata is None:
                    if task not in {"controlled_baseline", "TTS-Cal"}:
                        raise ValueError(f"{task}:{problem_id} lacks reference or tests")
                    test_metadata = {
                        "scorer": "N/A",
                        "reason": "protocol control task is not accuracy-scored",
                    }
                normalized = {
                    "problem_id": problem_id,
                    "split": split,
                    "prompt": prompt,
                    "template": row.get("template"),
                    "turns": turns,
                    "reference": reference,
                    "test_metadata": test_metadata,
                }
                stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        expected = {
            problem_id
            for candidate, problem_id in split_by_id
            if candidate == task
        }
        missing = expected - found
        if missing:
            raise ValueError(f"{task} is missing selected IDs: {sorted(missing)[:10]}")
        rows = load_prompt_pool(output)
        if task == "TTS-Cal":
            counts = {name: sum(row["split"] == name for row in rows) for name in ("tuning", "holdout")}
            if counts != {"tuning": 76, "holdout": 4}:
                raise ValueError("TTS-Cal needs exactly 76 tuning and 4 holdout IDs")


if __name__ == "__main__":
    main()
