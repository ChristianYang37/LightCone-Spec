"""Future utility labels and gradient mismatch (spec 7.6).

U_r(H) = sum_{j=a}^{a+H-1} [ l_j(phi_before) - l_j(phi_after) ] with
teacher-forced fixed future prefixes; harmful = 1[U_r(8) < 0]. The pure
statistical utility never subtracts system cost; the system utility
U_net = U - lambda_sys * C is a separate, secondary quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MAIN_HORIZON = 8
AUX_HORIZONS = (1, 4, 16)


@dataclass
class UtilityLabel:
    update_id: str
    horizon: int
    utility: float
    harmful: int


def future_utility(
    losses_before: np.ndarray, losses_after: np.ndarray, horizon: int
) -> float:
    """losses_*: per-round common loss l_j evaluated on the fixed
    teacher-forced future prefixes starting at the arrival round."""
    h = min(horizon, len(losses_before), len(losses_after))
    if h == 0:
        return 0.0
    lb = np.asarray(losses_before[:h], dtype=np.float64)
    la = np.asarray(losses_after[:h], dtype=np.float64)
    return float((lb - la).sum())


def harmful_label(utility_h8: float) -> int:
    return int(utility_h8 < 0.0)


def system_utility(utility: float, cost: float, lambda_sys: float) -> float:
    return utility - lambda_sys * cost


def relative_gradient_mismatch(
    g_stale: torch.Tensor | np.ndarray, g_fresh: torch.Tensor | np.ndarray
) -> float:
    gs = np.asarray(
        g_stale.detach().numpy() if isinstance(g_stale, torch.Tensor) else g_stale,
        dtype=np.float64,
    )
    gf = np.asarray(
        g_fresh.detach().numpy() if isinstance(g_fresh, torch.Tensor) else g_fresh,
        dtype=np.float64,
    )
    return float(np.linalg.norm(gs - gf) / (np.linalg.norm(gf) + 1e-12))


def gradient_cosine(
    g_stale: torch.Tensor | np.ndarray, g_fresh: torch.Tensor | np.ndarray
) -> float:
    gs = np.asarray(
        g_stale.detach().numpy() if isinstance(g_stale, torch.Tensor) else g_stale,
        dtype=np.float64,
    )
    gf = np.asarray(
        g_fresh.detach().numpy() if isinstance(g_fresh, torch.Tensor) else g_fresh,
        dtype=np.float64,
    )
    denom = np.linalg.norm(gs) * np.linalg.norm(gf)
    if denom == 0:
        return 0.0
    return float(np.dot(gs, gf) / denom)
