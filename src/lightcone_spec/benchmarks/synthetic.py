"""Synthetic worlds (spec 12.3, 13.1).

- Enumerable 4-symbol Markov world with hidden regimes A/B;
- phase-switch schedule;
- A -> B -> A recurrence;
- idle-insertion / wall-only / state-only twins (delay-injection
  profiles paired with identical base worlds).

Every generator's transition matrices, seeds and schedule parameters are
serializable into the experiment manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ALPHABET = 4


def _row_stochastic(rng: np.random.Generator, sharpness: float) -> np.ndarray:
    mat = rng.dirichlet(np.full(ALPHABET, sharpness), size=ALPHABET)
    return mat


@dataclass
class MarkovWorld:
    """Hidden-regime Markov world. The regime schedule maps a round index
    to a regime id; the target distribution depends on (last token,
    regime)."""

    transitions: dict[str, np.ndarray]
    initial: np.ndarray
    regime_schedule: Callable[[int], str]
    name: str = "markov4_world"
    params: dict = field(default_factory=dict)

    def target_dist(self, prefix: tuple[int, ...], round_id: int) -> np.ndarray:
        regime = self.regime_schedule(round_id)
        t = self.transitions[regime]
        if not prefix:
            return self.initial.copy()
        return t[prefix[-1]].copy()

    def manifest_dict(self) -> dict:
        return {
            "name": self.name,
            "alphabet": ALPHABET,
            "transitions": {k: v.tolist() for k, v in self.transitions.items()},
            "initial": self.initial.tolist(),
            "params": self.params,
        }


def make_two_regime_world(
    seed: int = 0, sharpness_a: float = 0.6, sharpness_b: float = 0.6
) -> MarkovWorld:
    rng = np.random.Generator(np.random.PCG64(seed))
    ta = _row_stochastic(rng, sharpness_a)
    tb = _row_stochastic(rng, sharpness_b)
    initial = rng.dirichlet(np.full(ALPHABET, 1.0))
    return MarkovWorld(
        transitions={"A": ta, "B": tb},
        initial=initial,
        regime_schedule=lambda r: "A",
        name="markov4_world",
        params={"seed": seed, "sharpness_a": sharpness_a, "sharpness_b": sharpness_b},
    )


def make_phase_switch_world(seed: int = 0, switch_round: int = 8) -> MarkovWorld:
    w = make_two_regime_world(seed)
    w.regime_schedule = lambda r: "A" if r < switch_round else "B"
    w.name = "phase_switch"
    w.params["switch_round"] = switch_round
    return w


def make_aba_world(
    seed: int = 0, first_switch: int = 6, second_switch: int = 12
) -> MarkovWorld:
    """A -> B -> A recurrence: endpoint distance returns to near zero
    while path length stays strictly positive."""
    w = make_two_regime_world(seed)

    def schedule(r: int) -> str:
        if r < first_switch:
            return "A"
        if r < second_switch:
            return "B"
        return "A"

    w.regime_schedule = schedule
    w.name = "aba_recurrence"
    w.params.update({"first_switch": first_switch, "second_switch": second_switch})
    return w


@dataclass
class DelayTwinProfile:
    """Delay-injection profile for the twin experiments (spec 6.14, 13.1).

    - idle_insertion: repeat each trajectory state `dilation` extra times
      without moving the world (rho must not change);
    - wall_only: add wall-clock delay to updates without state movement;
    - state_only: let the state move while wall delay stays at baseline.
    """

    kind: str  # idle_insertion | wall_only | state_only | none
    dilation: int = 0
    extra_wall_us: float = 0.0
    extra_state_rounds: int = 0

    def manifest_dict(self) -> dict:
        return {
            "kind": self.kind,
            "dilation": self.dilation,
            "extra_wall_us": self.extra_wall_us,
            "extra_state_rounds": self.extra_state_rounds,
        }


def make_twin_profiles() -> dict[str, DelayTwinProfile]:
    return {
        "none": DelayTwinProfile(kind="none"),
        "idle_insertion_twins": DelayTwinProfile(kind="idle_insertion", dilation=2),
        "wall_only_twins": DelayTwinProfile(kind="wall_only", extra_wall_us=5e5),
        "state_only_twins": DelayTwinProfile(kind="state_only", extra_state_rounds=4),
    }


def synthetic_world(name: str, seed: int = 0) -> MarkovWorld:
    if name == "markov4_world":
        return make_two_regime_world(seed)
    if name == "phase_switch":
        return make_phase_switch_world(seed)
    if name == "aba_recurrence":
        return make_aba_world(seed)
    if name in ("idle_insertion_twins", "wall_only_twins", "state_only_twins"):
        return make_two_regime_world(seed)
    raise KeyError(f"unknown synthetic world {name!r}")
