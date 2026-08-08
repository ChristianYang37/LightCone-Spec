"""Sequence traces for counterfactual replay.

A trace freezes the teacher-forced token sequence, per-round trajectory
states and per-round wall times of one target trajectory of one base
prompt. Replay never lets a method change future inputs (spec 7.6):
every counterfactual evaluates losses along this frozen sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightcone_spec.runtime.exact_sampler import normalize_sampling_dist
from lightcone_spec.runtime.rng import DrawKind, categorical_draw, request_id_hash
from lightcone_spec.runtime.toy_model import ToyModelPair
from lightcone_spec.trajectory.state import TrajectoryState, make_state


@dataclass
class SequenceTrace:
    base_prompt_id: str  # sequence group / bootstrap cluster
    trajectory_id: str  # one sampled trajectory of the base prompt
    dataset: str
    tokens: list[int]
    rounds: list[dict] = field(default_factory=list)  # round_id -> prefix_len, wall_us
    states: list[TrajectoryState] = field(default_factory=list)

    @property
    def group_id(self) -> str:
        return self.base_prompt_id

    def prefix_at_round(self, round_id: int) -> tuple[int, ...]:
        for r in self.rounds:
            if r["round_id"] == round_id:
                return tuple(self.tokens[: r["prefix_len"]])
        raise KeyError(f"round {round_id} not in trace {self.trajectory_id}")

    def max_round(self) -> int:
        return self.rounds[-1]["round_id"] if self.rounds else -1


def generate_trace(
    pair: ToyModelPair,
    base_prompt_id: str,
    trajectory_id: str,
    dataset: str,
    num_rounds: int,
    tokens_per_round: int = 2,
    seed: int = 0,
    wall_us_per_round: float = 1000.0,
) -> SequenceTrace:
    """Sample one target trajectory (locked sampling trajectory of a base
    prompt) and record per-round states. Tokens come from the *target*
    law, so replay is teacher-forced by construction."""
    rid = request_id_hash(trajectory_id)
    tokens: list[int] = []
    rounds: list[dict] = []
    states: list[TrajectoryState] = []
    for round_id in range(num_rounds):
        prefix = tuple(tokens)
        p_first = normalize_sampling_dist(pair.target_logits(prefix, round_id))
        states.append(
            make_state(
                round_id=round_id,
                target_probs=p_first,
                hidden_projected=pair.projected_hidden(prefix),
                topk=min(64, p_first.shape[0]),
            )
        )
        rounds.append(
            {
                "round_id": round_id,
                "prefix_len": len(tokens),
                "wall_us": wall_us_per_round * (round_id + 1),
            }
        )
        cur = prefix
        for k in range(tokens_per_round):
            p = normalize_sampling_dist(pair.target_logits(cur, round_id))
            tok = categorical_draw(p, seed, rid, round_id, k, DrawKind.BONUS)
            tokens.append(tok)
            cur = cur + (tok,)
    return SequenceTrace(
        base_prompt_id=base_prompt_id,
        trajectory_id=trajectory_id,
        dataset=dataset,
        tokens=tokens,
        rounds=rounds,
        states=states,
    )
