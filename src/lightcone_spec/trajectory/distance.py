"""Trajectory distance d_z and path-length exposure rho (spec 7.2).

d_z(z, z')^2 = a_p * JS(ptilde, ptilde') + a_h * ||Pi h - Pi h'||^2/128
             + a_e * ||e - e'||^2

with a_p, a_h, a_e >= 0 summing to 1, fitted only on sequence-grouped
train/calibration splits and frozen afterwards. JSD is computed on the
union of the two top-k id sets plus an `other` bucket; probabilities are
renormalized and clipped at 1e-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lightcone_spec.trajectory.state import TrajectoryState

_CLIP = 1e-12


def js_divergence_topk(
    ids_a: np.ndarray,
    probs_a: np.ndarray,
    other_a: float,
    ids_b: np.ndarray,
    probs_b: np.ndarray,
    other_b: float,
) -> float:
    union = np.union1d(ids_a, ids_b)
    pa = np.zeros(union.shape[0] + 1, dtype=np.float64)
    pb = np.zeros(union.shape[0] + 1, dtype=np.float64)
    map_a = {int(t): float(p) for t, p in zip(ids_a, probs_a)}
    map_b = {int(t): float(p) for t, p in zip(ids_b, probs_b)}
    for i, tok in enumerate(union):
        pa[i] = map_a.get(int(tok), 0.0)
        pb[i] = map_b.get(int(tok), 0.0)
    pa[-1] = max(other_a, 0.0)
    pb[-1] = max(other_b, 0.0)
    pa = np.clip(pa, _CLIP, None)
    pb = np.clip(pb, _CLIP, None)
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kl_am = float((pa * np.log(pa / m)).sum())
    kl_bm = float((pb * np.log(pb / m)).sum())
    js = 0.5 * kl_am + 0.5 * kl_bm
    return max(js, 0.0)


@dataclass
class DistanceWeights:
    """Named a_p, a_h, a_e to avoid any confusion with the DSpark draft
    depth gamma (spec 7.2)."""

    a_p: float
    a_h: float
    a_e: float
    hidden_mean: np.ndarray | None = None
    hidden_std: np.ndarray | None = None
    event_mean: np.ndarray | None = None
    event_std: np.ndarray | None = None
    frozen: bool = False

    def __post_init__(self) -> None:
        if min(self.a_p, self.a_h, self.a_e) < 0:
            raise ValueError("distance weights must be nonnegative")
        total = self.a_p + self.a_h + self.a_e
        if total <= 0:
            raise ValueError("distance weights must not all be zero")
        self.a_p, self.a_h, self.a_e = (
            self.a_p / total,
            self.a_h / total,
            self.a_e / total,
        )

    def _norm_hidden(self, h: np.ndarray) -> np.ndarray:
        if self.hidden_mean is None or self.hidden_std is None:
            return np.asarray(h, dtype=np.float64)
        return (np.asarray(h, dtype=np.float64) - self.hidden_mean) / np.maximum(
            self.hidden_std, 1e-8
        )

    def _norm_event(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float64)
        if e.size == 0 or self.event_mean is None or self.event_std is None:
            return e
        n = min(e.size, self.event_mean.size)
        return (e[:n] - self.event_mean[:n]) / np.maximum(self.event_std[:n], 1e-8)

    def to_dict(self) -> dict:
        return {
            "a_p": self.a_p,
            "a_h": self.a_h,
            "a_e": self.a_e,
            "hidden_mean": None if self.hidden_mean is None else self.hidden_mean.tolist(),
            "hidden_std": None if self.hidden_std is None else self.hidden_std.tolist(),
            "event_mean": None if self.event_mean is None else self.event_mean.tolist(),
            "event_std": None if self.event_std is None else self.event_std.tolist(),
            "frozen": self.frozen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DistanceWeights":
        def arr(x):
            return None if x is None else np.asarray(x, dtype=np.float64)

        w = cls(
            a_p=d["a_p"],
            a_h=d["a_h"],
            a_e=d["a_e"],
            hidden_mean=arr(d.get("hidden_mean")),
            hidden_std=arr(d.get("hidden_std")),
            event_mean=arr(d.get("event_mean")),
            event_std=arr(d.get("event_std")),
        )
        w.frozen = bool(d.get("frozen", False))
        return w


def d_z(a: TrajectoryState, b: TrajectoryState, w: DistanceWeights) -> float:
    js = js_divergence_topk(
        a.topk_token_ids, a.topk_probs, a.other_mass,
        b.topk_token_ids, b.topk_probs, b.other_mass,
    )
    ha = w._norm_hidden(a.hidden_proj)
    hb = w._norm_hidden(b.hidden_proj)
    h_term = float(((ha - hb) ** 2).sum()) / 128.0
    ea = w._norm_event(a.event_sketch)
    eb = w._norm_event(b.event_sketch)
    n = min(ea.size, eb.size)
    e_term = float(((ea[:n] - eb[:n]) ** 2).sum()) if n > 0 else 0.0
    return float(np.sqrt(w.a_p * js + w.a_h * h_term + w.a_e * e_term))


def rho_path(
    states: list[TrajectoryState], source_round: int, exposure_round: int,
    w: DistanceWeights,
) -> float:
    """rho_r = sum_{j=r+1}^{c(r)} d_z(z_{j-1}, z_j). States are indexed by
    round id; idle insertion (repeated identical states) adds zero."""
    by_round = {s.round_id: s for s in states}
    total = 0.0
    for j in range(source_round + 1, exposure_round + 1):
        if j - 1 not in by_round or j not in by_round:
            raise KeyError(f"missing trajectory state for round {j - 1} or {j}")
        total += d_z(by_round[j - 1], by_round[j], w)
    return total


def endpoint_distance(
    states: list[TrajectoryState], source_round: int, exposure_round: int,
    w: DistanceWeights,
) -> float:
    by_round = {s.round_id: s for s in states}
    return d_z(by_round[source_round], by_round[exposure_round], w)
