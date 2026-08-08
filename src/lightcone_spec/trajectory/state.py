"""Confirmatory trajectory state z_r (spec 7.1).

z_r = (ptilde_r, Pi h_r, e_r), indexed by the verified prefix at round
start and persisted once the round's teacher logits are ready. All
components reuse tensors the normal target verification already
produced; recording the trajectory never adds target calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOPK = 64


@dataclass
class TrajectoryState:
    round_id: int
    topk_token_ids: np.ndarray  # (<=64,) int32
    topk_probs: np.ndarray  # (<=64,) float32
    other_mass: float
    hidden_proj: np.ndarray  # (128,) float32
    event_sketch: np.ndarray  # variable-length float32; may be length 0

    def validate(self) -> None:
        s = float(self.topk_probs.sum() + self.other_mass)
        if not (abs(s - 1.0) < 1e-4):
            raise ValueError(f"top-k + other mass must sum to 1, got {s}")


def make_state(
    round_id: int,
    target_probs: np.ndarray,
    hidden_projected: np.ndarray,
    entropy: float | None = None,
    extra_events: np.ndarray | None = None,
    topk: int = TOPK,
) -> TrajectoryState:
    """Build z_r from the first-draft-position target distribution and the
    projected final-norm hidden state. The event sketch holds target-only
    scalars: entropy, max prob, top1-top2 margin, plus any target-only
    event features; when no target event exists the schema is an explicit
    length-0 vector (never draft features)."""
    p = np.asarray(target_probs, dtype=np.float64)
    k = min(topk, p.shape[0])
    order = np.argsort(-p, kind="stable")[:k]
    top_probs = p[order]
    other = max(0.0, 1.0 - float(top_probs.sum()))
    if entropy is None:
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
    srt = np.sort(p)[::-1]
    margin = float(srt[0] - srt[1]) if p.shape[0] >= 2 else float(srt[0])
    events = [entropy, float(p.max()), margin]
    if extra_events is not None:
        events.extend(np.asarray(extra_events, dtype=np.float64).tolist())
    return TrajectoryState(
        round_id=round_id,
        topk_token_ids=order.astype(np.int32),
        topk_probs=top_probs.astype(np.float32),
        other_mass=float(other),
        hidden_proj=np.asarray(hidden_projected, dtype=np.float32),
        event_sketch=np.asarray(events, dtype=np.float32),
    )


@dataclass
class SecondaryFeatures:
    """Draft-aware features (spec 7.3). Allowed only in secondary
    predictors or diagnostic oracles; never inside the target-only
    confirmatory clock."""

    target_draft_residual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    accept_pattern: np.ndarray = field(default_factory=lambda: np.zeros(0))
    draft_hidden: np.ndarray = field(default_factory=lambda: np.zeros(0))
    proposal_confidence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gradient_sketch: np.ndarray = field(default_factory=lambda: np.zeros(0))
    adapter_norm: float = 0.0
