"""Fixed-dimension transport state vector (spec 7.4).

z_vector = [count_sketch_256(top-64 probs); standardized 128-d hidden
projection; standardized event vector], standardized with train-split
statistics. Delta z = z_vector(c(r)) - z_vector(r).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lightcone_spec.adapters.projections import CountSketch
from lightcone_spec.trajectory.state import TrajectoryState

SKETCH_DIM = 256
EVENT_DIM = 3  # entropy, max prob, top1-top2 margin (fixed schema)


@dataclass
class ZVectorizer:
    sketch: CountSketch
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return SKETCH_DIM + 128 + EVENT_DIM

    def raw(self, state: TrajectoryState) -> np.ndarray:
        sk = self.sketch.project(state.topk_token_ids, state.topk_probs)
        ev = np.zeros(EVENT_DIM, dtype=np.float64)
        n = min(EVENT_DIM, state.event_sketch.size)
        ev[:n] = state.event_sketch[:n]
        return np.concatenate([sk, state.hidden_proj.astype(np.float64), ev])

    def fit_normalization(self, states: list[TrajectoryState]) -> None:
        raws = np.stack([self.raw(s) for s in states])
        self.mean = raws.mean(axis=0)
        self.std = np.maximum(raws.std(axis=0), 1e-8)

    def vector(self, state: TrajectoryState) -> np.ndarray:
        raw = self.raw(state)
        if self.mean is None or self.std is None:
            return raw
        return (raw - self.mean) / self.std

    def delta_z(self, source: TrajectoryState, arrival: TrajectoryState) -> np.ndarray:
        return self.vector(arrival) - self.vector(source)

    def artifact_dict(self) -> dict:
        return {
            "sketch": self.sketch.artifact_dict(),
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "dim": self.dim,
        }

    @classmethod
    def from_artifact(cls, d: dict) -> "ZVectorizer":
        z = cls(sketch=CountSketch(dim=d["sketch"]["dim"], seed=d["sketch"]["seed"]))
        z.mean = None if d["mean"] is None else np.asarray(d["mean"])
        z.std = None if d["std"] is None else np.asarray(d["std"])
        return z


def default_zvectorizer() -> ZVectorizer:
    return ZVectorizer(sketch=CountSketch(dim=SKETCH_DIM, seed=0))
