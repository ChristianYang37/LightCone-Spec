"""L2 damping (spec 7.8).

``kappa_r = exp(-max(0, eps_hat_r / R))``. Real schema-v3 replay chooses
``R`` on calibration-only same-candidate kappa-utility grids; legacy/toy
replay falls back to ``Q_0.9(eps_hat | U_r(8) > 0)``. The main kernel is
exponential; clipped-linear exists only as an ablation. Scaling acts on the
optimizer-produced candidate delta, never on the gradient fed into AdamW.
"""

from __future__ import annotations

import numpy as np
import torch

R_FLOOR = 1e-8


def calibration_radius(
    predicted_mismatch: np.ndarray, utilities: np.ndarray
) -> float:
    predicted_mismatch = np.asarray(predicted_mismatch, dtype=np.float64)
    utilities = np.asarray(utilities, dtype=np.float64)
    positive = predicted_mismatch[utilities > 0.0]
    if positive.size == 0:
        return R_FLOOR
    r = float(np.quantile(positive, 0.9))
    return max(r, R_FLOOR)


def select_utility_calibrated_radius(
    predicted_mismatch: np.ndarray,
    utility_grids: list[dict[float, float]],
    *,
    kernel: str = "exponential",
) -> tuple[float, dict]:
    """Choose the L2 radius on calibration-only counterfactual utility.

    The old positive-utility mismatch quantile never used the captured oracle
    kappa labels.  This one-dimensional selector keeps the same online kernel
    but chooses its only parameter to maximize mean calibration utility over
    the exact same-candidate kappa grids.  Candidate radii include every grid
    breakpoint plus near-discard and near-identity limits; ties choose the
    smaller mean kappa.
    """
    mismatch = np.asarray(predicted_mismatch, dtype=np.float64)
    if len(mismatch) != len(utility_grids) or len(mismatch) == 0:
        raise ValueError("damping calibration arrays must be non-empty and aligned")
    if kernel not in ("exponential", "clipped_linear"):
        raise ValueError(f"unknown damping kernel {kernel!r}")
    if not np.isfinite(mismatch).all():
        raise ValueError("predicted mismatch must be finite")
    mismatch = np.maximum(mismatch, 0.0)

    candidates = {R_FLOOR}
    positive_mismatch = mismatch[mismatch > 0.0]
    if positive_mismatch.size:
        # Explicit near-identity endpoint; exactly one is already handled by
        # the runtime's zero-delay parity branch.
        candidates.add(float(positive_mismatch.max() / 1e-6))
    for eps, grid in zip(mismatch, utility_grids):
        if not grid or 0.0 not in grid or 1.0 not in grid:
            raise ValueError("each utility grid must include kappa 0 and 1")
        if any(
            not np.isfinite(float(kappa))
            or not np.isfinite(float(utility))
            or not 0.0 <= float(kappa) <= 1.0
            for kappa, utility in grid.items()
        ):
            raise ValueError("damping utility grid is non-finite or outside [0, 1]")
        if eps <= 0.0:
            continue
        for raw_kappa in grid:
            kappa = float(raw_kappa)
            if not 0.0 < kappa < 1.0:
                continue
            if kernel == "exponential":
                radius = -float(eps) / float(np.log(kappa))
            else:
                radius = float(eps) / (1.0 - kappa)
            if np.isfinite(radius) and radius >= R_FLOOR:
                candidates.add(radius)

    def utility_at(index: int, kappa: float) -> float:
        ordered = sorted(
            (float(key), float(value))
            for key, value in utility_grids[index].items()
        )
        return float(
            np.interp(
                kappa,
                [item[0] for item in ordered],
                [item[1] for item in ordered],
            )
        )

    best_radius = R_FLOOR
    best_utility = -np.inf
    best_mean_kappa = np.inf
    for radius in sorted(candidates):
        kappas = np.asarray(
            [damping_factor(value, radius, kernel) for value in mismatch],
            dtype=np.float64,
        )
        mean_utility = float(
            np.mean(
                [utility_at(index, float(kappa)) for index, kappa in enumerate(kappas)]
            )
        )
        mean_kappa = float(kappas.mean())
        if (
            mean_utility > best_utility + 1e-15
            or (
                abs(mean_utility - best_utility) <= 1e-15
                and mean_kappa < best_mean_kappa
            )
        ):
            best_radius = float(radius)
            best_utility = mean_utility
            best_mean_kappa = mean_kappa
    return best_radius, {
        "contract": "calibration_same_candidate_kappa_utility_v1",
        "candidate_count": len(candidates),
        "mean_calibration_utility": best_utility,
        "mean_kappa": best_mean_kappa,
    }


def damping_factor(
    predicted_mismatch: float | torch.Tensor,
    radius: float,
    kernel: str = "exponential",
) -> float | torch.Tensor:
    if isinstance(predicted_mismatch, torch.Tensor):
        x = (predicted_mismatch / max(radius, R_FLOOR)).clamp_min(0.0)
        if kernel == "exponential":
            return torch.exp(-x)
        if kernel == "clipped_linear":
            return (1.0 - x).clamp(0.0, 1.0)
        raise ValueError(f"unknown damping kernel {kernel!r}")
    x = max(0.0, float(predicted_mismatch) / max(radius, R_FLOOR))
    if kernel == "exponential":
        return float(np.exp(-x))
    if kernel == "clipped_linear":
        return float(np.clip(1.0 - x, 0.0, 1.0))
    raise ValueError(f"unknown damping kernel {kernel!r}")
