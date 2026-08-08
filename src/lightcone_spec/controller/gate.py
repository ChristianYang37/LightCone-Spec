"""L1 gate threshold selection (spec 7.7).

apply <=> p_hat_harmful <= tau. Threshold selection on the calibration
split:
  1. enumerate all unique predicted scores,
  2. keep thresholds whose unsafe-apply rate Pr(apply | U < 0) <= 10%,
  3. among feasible thresholds pick max validation mean utility,
  4. ties -> the more conservative (fewer applies) threshold,
  5. include discard-all (utility 0) in the calibration objective, so a
     feasible but negative-utility non-empty policy cannot be selected.

`unsafe-apply rate` is the miss rate on harmful updates; logs and docs
must never blur it into a generic FPR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GateSelection:
    threshold: float
    unsafe_apply_rate: float
    mean_utility: float
    apply_fraction: float
    discard_all: bool


def select_gate_threshold(
    harm_probs: np.ndarray,
    utilities: np.ndarray,
    unsafe_apply_limit: float = 0.10,
) -> GateSelection:
    harm_probs = np.asarray(harm_probs, dtype=np.float64)
    utilities = np.asarray(utilities, dtype=np.float64)
    harmful = utilities < 0.0
    n_harmful = int(harmful.sum())

    candidates = np.unique(harm_probs)
    # Discard-all is a real L1 policy, not merely an error fallback.  Starting
    # from this baseline makes the stated max-utility calibration objective
    # exact and prevents a safe-but-net-harmful non-empty threshold from
    # winning only because zero was omitted from the candidate set.
    best = GateSelection(
        threshold=-np.inf,
        unsafe_apply_rate=0.0,
        mean_utility=0.0,
        apply_fraction=0.0,
        discard_all=True,
    )
    for tau in candidates:
        apply_mask = harm_probs <= tau
        if not apply_mask.any():
            continue
        if n_harmful > 0:
            unsafe = float((apply_mask & harmful).sum() / n_harmful)
        else:
            unsafe = 0.0
        if unsafe > unsafe_apply_limit:
            continue
        # Utility of gating: applied updates contribute their utility,
        # discarded ones contribute 0.
        mean_util = float(np.where(apply_mask, utilities, 0.0).mean())
        frac = float(apply_mask.mean())
        cand = GateSelection(
            threshold=float(tau),
            unsafe_apply_rate=unsafe,
            mean_utility=mean_util,
            apply_fraction=frac,
            discard_all=False,
        )
        if (
            cand.mean_utility > best.mean_utility + 1e-15
            or (
                abs(cand.mean_utility - best.mean_utility) <= 1e-15
                and cand.apply_fraction < best.apply_fraction
            )
        ):
            best = cand
    return best


def gate_decision(
    harm_probability: float | torch.Tensor, threshold: float
) -> bool | torch.Tensor:
    """True = apply the full candidate delta u_r; False = discard."""
    decision = harm_probability <= threshold
    return decision if isinstance(decision, torch.Tensor) else bool(decision)
