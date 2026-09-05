#!/usr/bin/env python3
"""Fetch official evaluation files unchanged; emit additive dataset/draft mappings."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from lightcone_spec.data import load_source_prompt_records
from lightcone_spec.protocol import (
    E0_BACKENDS,
    E0_MODELS,
    SOURCE_EVALUATION_TASKS,
    source_checkpoint_id,
)


def prepare(output: Path, revision: str = "main") -> dict:
    if not revision or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for char in revision
    ):
        raise ValueError("source revision must be a Git ref without path components")
    output.mkdir(parents=True, exist_ok=True)
    datasets, sources = {}, {}
    for _, dataset, count in SOURCE_EVALUATION_TASKS:
        url = f"https://raw.githubusercontent.com/deepseek-ai/DeepSpec/{revision}/eval_datasets/{dataset}.jsonl"
        destination = output / f"{dataset}.jsonl"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        # Never silently replace a prior official file on a moving branch.
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(
                f"official source changed; retain old evidence and use a new directory: {dataset}"
            )
        if not destination.exists():
            destination.write_bytes(payload)
        rows = load_source_prompt_records(destination, max_samples=count, seed=980406)
        datasets[f"DeepSpec-source|{dataset}"] = str(destination.resolve())
        sources[dataset] = {"url": url, "registered_requests": count, "loaded_requests": len(rows)}
    manifest = {
        "datasets": datasets,
        "official_files": sources,
        "checkpoint_repositories": {
            f"DeepSpec-source|{model}|{backend}": source_checkpoint_id(model, backend)
            for model in E0_MODELS
            for backend in E0_BACKENDS
        },
        "scope": "prompt files only; checkpoint paths must be downloaded or verified separately",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    result = prepare(args.output, args.revision)
    print(
        json.dumps(
            {"datasets": len(result["datasets"]), "manifest": str(args.output / "manifest.json")}
        )
    )
