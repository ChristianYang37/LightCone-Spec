"""Toy DSpark model pair for the CPU reference engine, P0 toys, replay
unit tests and the exactness harness.

It mirrors the real integration surface: a frozen target (Markov world),
a frozen base drafter (perturbed transitions), frozen feature maps
(hidden states, markov_w1 embeddings, confidence head), the fixed
projection artifacts of spec 5.1, and a proposal that adds the linear
adapter residuals. All randomness in construction is PCG64-seeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterParams,
    AdapterShapes,
    HIDDEN_PROJ_DIM,
)
from lightcone_spec.adapters.projections import (
    build_hidden_projection,
    build_output_basis,
)
from lightcone_spec.benchmarks.synthetic import ALPHABET, MarkovWorld


@dataclass
class ToyModelPair:
    world: MarkovWorld
    drafter_transitions: dict[str, np.ndarray]
    hidden_map: np.ndarray  # (ALPHABET+1, d) fixed random target hiddens
    markov_w1: np.ndarray  # (ALPHABET+1, r_M) frozen drafter embedding
    conf_bias: np.ndarray  # (7,) frozen base confidence logits
    r_h: np.ndarray  # (d, 128)
    basis: torch.Tensor  # (V, rank)
    shapes: AdapterShapes
    hidden_dim: int
    seed: int

    # ---- frozen target side -------------------------------------------

    def target_dist(self, prefix: tuple[int, ...], round_id: int) -> np.ndarray:
        return self.world.target_dist(prefix, round_id)

    def target_logits(self, prefix: tuple[int, ...], round_id: int) -> np.ndarray:
        p = np.clip(self.target_dist(prefix, round_id), 1e-12, None)
        return np.log(p)

    def target_hidden(self, prefix: tuple[int, ...]) -> np.ndarray:
        last = prefix[-1] + 1 if prefix else 0
        return self.hidden_map[last]

    def projected_hidden(self, prefix: tuple[int, ...]) -> np.ndarray:
        return self.target_hidden(prefix) @ self.r_h

    # ---- frozen drafter side --------------------------------------------

    def base_drafter_logits(
        self, prefix: tuple[int, ...], round_id: int
    ) -> np.ndarray:
        regime = self.world.regime_schedule(round_id)
        t = self.drafter_transitions[regime]
        if not prefix:
            row = np.full(ALPHABET, 1.0 / ALPHABET)
        else:
            row = t[prefix[-1]]
        return np.log(np.clip(row, 1e-12, None))

    def markov_embedding(self, prefix: tuple[int, ...]) -> np.ndarray:
        last = prefix[-1] + 1 if prefix else 0
        return self.markov_w1[last]

    # ---- adapter-modulated proposal --------------------------------------

    def proposal_logits(
        self,
        prefix: tuple[int, ...],
        round_id: int,
        phi: torch.Tensor,
        draft_position: int,
    ) -> np.ndarray:
        base = self.base_drafter_logits(prefix, round_id)
        adapter = AdapterParams(self.shapes, self.basis)
        adapter.load_flat(phi)
        u = torch.from_numpy(
            self.projected_hidden(prefix).astype(np.float32)
        ).unsqueeze(0)
        m = torch.from_numpy(
            self.markov_embedding(prefix).astype(np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            resid = (
                adapter.draft_logit_residual(u) + adapter.markov_logit_residual(m)
            ).squeeze(0).numpy()
        return base + resid

    def proposal_dist(
        self,
        prefix: tuple[int, ...],
        round_id: int,
        phi: torch.Tensor,
        draft_position: int = 0,
    ) -> np.ndarray:
        logits = self.proposal_logits(prefix, round_id, phi, draft_position)
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()

    def base_confidence_logit(self, draft_position: int) -> float:
        return float(self.conf_bias[min(draft_position, 6)])


def make_toy_pair(
    world: MarkovWorld,
    seed: int = 0,
    hidden_dim: int = 160,
    markov_dim: int = 8,
    rank: int = 4,
    drafter_noise: float = 0.5,
) -> ToyModelPair:
    rng = np.random.Generator(np.random.PCG64(seed + 1000))
    drafter = {}
    for regime, t in world.transitions.items():
        noise = rng.dirichlet(np.full(ALPHABET, 1.0), size=ALPHABET)
        mixed = (1.0 - drafter_noise) * t + drafter_noise * noise
        drafter[regime] = mixed / mixed.sum(axis=1, keepdims=True)
    hidden_map = rng.standard_normal((ALPHABET + 1, hidden_dim))
    markov_w1 = rng.standard_normal((ALPHABET + 1, markov_dim))
    conf_bias = rng.standard_normal(7) * 0.1
    r_h = build_hidden_projection(hidden_dim, HIDDEN_PROJ_DIM, seed=0).astype(
        np.float64
    )
    # Toy stand-ins for lm_head (V x d) and markov_w2 (V x r_M).
    w_lm = rng.standard_normal((ALPHABET, hidden_dim))
    w_m2 = rng.standard_normal((ALPHABET, markov_dim))
    ob = build_output_basis(w_lm, w_m2, r_h, rank=min(rank, ALPHABET))
    shapes = AdapterShapes(
        rank=ob.basis.shape[1], markov_dim=markov_dim, vocab_size=ALPHABET
    )
    return ToyModelPair(
        world=world,
        drafter_transitions=drafter,
        hidden_map=hidden_map,
        markov_w1=markov_w1,
        conf_bias=conf_bias,
        r_h=r_h,
        basis=torch.from_numpy(ob.basis.astype(np.float32)),
        shapes=shapes,
        hidden_dim=hidden_dim,
        seed=seed,
    )
