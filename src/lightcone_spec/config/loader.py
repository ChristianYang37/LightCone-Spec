"""Load strict JSON or YAML configuration without eager model dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from lightcone_spec.config.schema import RunConfig


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return RunConfig.model_validate(data)


def run_config_sha256(config: RunConfig) -> str:
    body = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(body).hexdigest()
