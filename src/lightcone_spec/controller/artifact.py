"""Frozen controller artifact (spec 7.7-7.9, 13.2).

One pooled artifact per model-pair/update-mode/parameter-layout identity,
fitted only on calibration-pool sequences disjoint from every test prompt,
containing: distance weights and normalization, the predictor coefficients (utility,
mismatch, harmful + isotonic calibration), the L1 threshold, the L2
calibration radius, the transport basis/map, the z-vectorizer, and the
train-group hashes. Written once with a SHA-256 sidecar; runtime loads
fail closed on drift.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.locking.hashing import canonical_json, sha256_bytes
from lightcone_spec.trajectory.distance import DistanceWeights
from lightcone_spec.trajectory.predictors import HarmfulClassifier, RidgePredictor
from lightcone_spec.trajectory.zvector import ZVectorizer
from lightcone_spec.transport.fit import TransportMap

ARTIFACT_SCHEMA_VERSION = 1
_SAFE_ARTIFACT_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def controller_artifact_filename(
    model_pair_id: str,
    weight_update_mode: str,
    parameter_layout_sha256: str,
) -> str:
    """Canonical controller filename bound to pair, tier and tail layout."""
    from lightcone_spec.config.schema import canonical_tail_layout_mode

    mode = canonical_tail_layout_mode(weight_update_mode)
    if not isinstance(model_pair_id, str) or not _SAFE_ARTIFACT_COMPONENT.fullmatch(
        model_pair_id
    ):
        raise ConfigError(f"unsafe controller model-pair id: {model_pair_id!r}")
    if not isinstance(parameter_layout_sha256, str) or not _SHA256_HEX.fullmatch(
        parameter_layout_sha256
    ):
        raise ConfigError(
            "controller parameter-layout identity must be 64 lowercase hex "
            "characters"
        )
    return f"{model_pair_id}.{mode}.{parameter_layout_sha256}.controller.json"


@dataclass
class ControllerArtifact:
    model_pair_id: str
    clock_variant: str
    feature_set: str
    distance_weights: DistanceWeights
    utility_predictor: RidgePredictor
    mismatch_predictor: RidgePredictor
    harmful_classifier: HarmfulClassifier
    gate_threshold: float
    gate_discard_all: bool
    damping_radius: float
    damping_kernel: str
    zvectorizer: Optional[ZVectorizer] = None
    transport_map: Optional[TransportMap] = None
    train_group_hash: str = ""
    calibration_group_hash: str = ""
    controller_version: int = 1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_pair_id": self.model_pair_id,
            "clock_variant": self.clock_variant,
            "feature_set": self.feature_set,
            "distance_weights": self.distance_weights.to_dict(),
            "utility_predictor": self.utility_predictor.to_dict(),
            "mismatch_predictor": self.mismatch_predictor.to_dict(),
            "harmful_classifier": self.harmful_classifier.to_dict(),
            "gate_threshold": self.gate_threshold,
            "gate_discard_all": self.gate_discard_all,
            "damping_radius": self.damping_radius,
            "damping_kernel": self.damping_kernel,
            "zvectorizer": None if self.zvectorizer is None else self.zvectorizer.artifact_dict(),
            "transport_map": None if self.transport_map is None else self.transport_map.to_dict(),
            "train_group_hash": self.train_group_hash,
            "calibration_group_hash": self.calibration_group_hash,
            "controller_version": self.controller_version,
            "extra": self.extra,
        }

    def freeze(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = canonical_json(self.to_dict())
        path.write_text(body)
        digest = sha256_bytes(body.encode("utf-8"))
        Path(str(path) + ".sha256").write_text(digest + "\n")
        return digest

    @classmethod
    def load(cls, path: str | Path, verify_hash: bool = True) -> "ControllerArtifact":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"controller artifact missing: {path}")
        try:
            body = path.read_text()
            d = json.loads(body)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"controller artifact is unreadable or invalid: {path}") from exc
        if not isinstance(d, dict):
            raise ConfigError(f"controller artifact root is not an object: {path}")
        if verify_hash:
            sha_path = Path(str(path) + ".sha256")
            if not sha_path.is_file():
                raise ConfigError(f"controller artifact hash sidecar missing: {sha_path}")
            expected = sha_path.read_text().strip()
            if not _SHA256_HEX.fullmatch(expected):
                raise ConfigError(f"controller artifact hash sidecar is invalid: {sha_path}")
            actual = sha256_bytes(canonical_json(d).encode("utf-8"))
            if actual != expected:
                raise ConfigError(f"controller artifact hash drift: {path}")
        if d.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ConfigError(
                f"controller artifact schema_version {d.get('schema_version')} "
                f"!= {ARTIFACT_SCHEMA_VERSION}"
            )
        try:
            return cls(
                model_pair_id=d["model_pair_id"],
                clock_variant=d["clock_variant"],
                feature_set=d["feature_set"],
                distance_weights=DistanceWeights.from_dict(d["distance_weights"]),
                utility_predictor=RidgePredictor.from_dict(d["utility_predictor"]),
                mismatch_predictor=RidgePredictor.from_dict(d["mismatch_predictor"]),
                harmful_classifier=HarmfulClassifier.from_dict(d["harmful_classifier"]),
                gate_threshold=d["gate_threshold"],
                gate_discard_all=d["gate_discard_all"],
                damping_radius=d["damping_radius"],
                damping_kernel=d["damping_kernel"],
                zvectorizer=(
                    None
                    if d["zvectorizer"] is None
                    else ZVectorizer.from_artifact(d["zvectorizer"])
                ),
                transport_map=(
                    None
                    if d["transport_map"] is None
                    else TransportMap.from_dict(d["transport_map"])
                ),
                train_group_hash=d["train_group_hash"],
                calibration_group_hash=d["calibration_group_hash"],
                controller_version=d["controller_version"],
                extra=d.get("extra", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"controller artifact content is incomplete or incompatible: {path}"
            ) from exc


def load_bound_controller_artifact(
    path: str | Path,
    *,
    model_pair_id: str,
    weight_update_mode: str,
) -> ControllerArtifact:
    """Load an artifact only when its content and filename bind identically."""
    from lightcone_spec.config.schema import canonical_tail_layout_mode

    path = Path(path)
    mode = canonical_tail_layout_mode(weight_update_mode)
    artifact = ControllerArtifact.load(path)
    layout_sha = artifact.extra.get("parameter_layout_sha256")
    runtime_mode = (
        artifact.extra.get("controller_runtime_identity", {})
        .get("candidate", {})
        .get("weight_update_mode")
    )
    if artifact.model_pair_id != model_pair_id:
        raise ConfigError(
            "controller filename/content model pair mismatch: "
            f"{artifact.model_pair_id!r} != {model_pair_id!r}"
        )
    if runtime_mode != mode:
        raise ConfigError(
            "controller filename/content update mode mismatch: "
            f"{runtime_mode!r} != {mode!r}"
        )
    expected_name = controller_artifact_filename(model_pair_id, mode, layout_sha)
    if path.name != expected_name:
        raise ConfigError(
            "controller artifact filename is not bound to its pair, canonical "
            f"mode and parameter layout: {path.name!r} != {expected_name!r}"
        )
    return artifact


def resolve_controller_artifact(
    root: str | Path,
    *,
    model_pair_id: str,
    weight_update_mode: str,
) -> tuple[Path, ControllerArtifact]:
    """Resolve exactly one bound artifact from a mode-specific controller root."""
    from lightcone_spec.config.schema import canonical_tail_layout_mode

    root = Path(root)
    mode = canonical_tail_layout_mode(weight_update_mode)
    pattern = f"{model_pair_id}.{mode}.*.controller.json"
    matches = sorted(root.glob(pattern)) if root.is_dir() else []
    if not matches:
        raise ConfigError(
            f"controller artifact missing for {model_pair_id}/{mode} under "
            f"{root}; run the bounded p5_cross_backend_trace producer for this "
            "mode, then fit it with `lightcone-spec replay --trace-root ...`"
        )
    if len(matches) != 1:
        raise ConfigError(
            f"controller artifact selection is ambiguous for "
            f"{model_pair_id}/{mode}: {[str(path) for path in matches]}; use "
            "one immutable controller directory per fitted trace set"
        )
    path = matches[0]
    return path, load_bound_controller_artifact(
        path,
        model_pair_id=model_pair_id,
        weight_update_mode=mode,
    )
