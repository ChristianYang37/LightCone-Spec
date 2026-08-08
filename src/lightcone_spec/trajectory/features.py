"""Predictor feature sets (spec 7.5).

All clock predictors share one model class and the same source-quality
covariates (prefix length, pre-update accepted-prefix estimate, training loss
and gradient norm); only the clock feature differs:

  delay-only : log1p(round_delay), log1p(token_delay)
  wall-only  : log1p(wall_us)
  endpoint   : endpoint_distance
  path-length: rho_path
  hybrid     : delay + wall + endpoint + path + parameter displacement
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE_QUALITY_FEATURES = (
    "log1p_source_prefix_len",
    "source_acceptance",
    "log1p_source_training_loss",
    "log1p_source_grad_norm",
)

FEATURE_SETS = {
    "delay_only": SOURCE_QUALITY_FEATURES
    + ("log1p_round_delay", "log1p_token_delay"),
    "wall_only": SOURCE_QUALITY_FEATURES + ("log1p_wall_us",),
    "endpoint": SOURCE_QUALITY_FEATURES + ("endpoint_distance",),
    "path_length": SOURCE_QUALITY_FEATURES + ("rho_path",),
    "hybrid": SOURCE_QUALITY_FEATURES
    + (
        "log1p_round_delay",
        "log1p_token_delay",
        "log1p_wall_us",
        "endpoint_distance",
        "rho_path",
        "parameter_displacement",
    ),
}

ALL_FEATURE_NAMES = (
    "log1p_source_prefix_len",
    "source_acceptance",
    "log1p_source_training_loss",
    "log1p_source_grad_norm",
    "log1p_round_delay",
    "log1p_token_delay",
    "log1p_wall_us",
    "endpoint_distance",
    "rho_path",
    "parameter_displacement",
)


@dataclass
class UpdateFeatureRow:
    """One labeled stale update with publish-time observable features only.

    Controller artifacts fail closed on missing/non-finite inputs. Runtime
    inference has no separate missing-indicator inputs, so silently fitting a
    wider offline-only matrix would make training and publication semantics
    differ.
    """

    sequence_id: str
    update_id: str
    round_delay: float
    token_delay: float
    wall_us: float
    endpoint_distance: float
    rho_path: float
    parameter_displacement: float
    utility: float  # U_r(H) for the main horizon
    relative_gradient_mismatch: float
    harmful: int  # 1[U_r(8) < 0]
    source_prefix_len: float = 0.0
    source_acceptance: float = 0.0
    source_training_loss: float = 0.0
    source_grad_norm: float = 0.0

    def features(self) -> dict[str, float]:
        return {
            "log1p_source_prefix_len": float(
                np.log1p(max(self.source_prefix_len, 0.0))
            ),
            "source_acceptance": self.source_acceptance,
            "log1p_source_training_loss": float(
                np.log1p(max(self.source_training_loss, 0.0))
            ),
            "log1p_source_grad_norm": float(
                np.log1p(max(self.source_grad_norm, 0.0))
            ),
            "log1p_round_delay": float(np.log1p(max(self.round_delay, 0.0))),
            "log1p_token_delay": float(np.log1p(max(self.token_delay, 0.0))),
            "log1p_wall_us": float(np.log1p(max(self.wall_us, 0.0))),
            "endpoint_distance": self.endpoint_distance,
            "rho_path": self.rho_path,
            "parameter_displacement": self.parameter_displacement,
        }


def design_matrix(
    rows: list[UpdateFeatureRow], feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed runtime feature matrix and a zero missingness matrix.

    Missing values are rejected instead of imputed because the online
    controller consumes exactly ``FEATURE_SETS[feature_set]`` and has no
    offline-only missing-indicator channels.
    """
    names = FEATURE_SETS[feature_set]
    x = np.zeros((len(rows), len(names)), dtype=np.float64)
    miss = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, row in enumerate(rows):
        feats = row.features()
        for j, name in enumerate(names):
            v = feats[name]
            if not np.isfinite(v):
                raise ValueError(
                    f"controller feature {name!r} is non-finite for "
                    f"update {row.update_id!r}"
                )
            x[i, j] = v
    return x, miss
