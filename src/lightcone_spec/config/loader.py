"""Config loading with strict validation and path canonicalization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from lightcone_spec.config.schema import AdaptationConfig
from lightcone_spec.exit_codes import ConfigError

_PATH_FIELDS = (
    ("trace", "artifact_root"),
    ("controller", "artifact_path"),
    ("transport", "basis_path"),
    ("trace", "telemetry_path"),
    ("model", "projection_artifact_path"),
)


def _canonicalize_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Paths must be absolute, or resolved relative to the manifest/config
    root and normalized (spec 10.2)."""
    for section, field in _PATH_FIELDS:
        sec = raw.get(section)
        if not isinstance(sec, dict):
            continue
        val = sec.get(field)
        if val is None:
            continue
        p = Path(os.path.expanduser(str(val)))
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        sec[field] = str(p)
    return raw


def load_adaptation_config(path: str | Path) -> AdaptationConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"config is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    raw = _canonicalize_paths(raw, path.parent.resolve())
    try:
        return AdaptationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid AdaptationConfig ({path}): {exc}") from exc


def validate_adaptation_config_dict(raw: dict[str, Any]) -> AdaptationConfig:
    try:
        return AdaptationConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid AdaptationConfig: {exc}") from exc
