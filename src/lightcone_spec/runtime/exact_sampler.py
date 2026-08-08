"""Target-exact speculative sampling (spec 3.1) plus the exact
enumeration machinery used by P0-B and the exactness harness.

The single-token invariant: draw y ~ q^v, accept with probability
min(1, p(y)/q^v(y)); on rejection draw from the normalized residual
[p - q^v]_+; when all draft tokens are accepted draw the bonus token
from the target distribution. p and q^v are the normalized sampling
distributions after locked logit processors / temperature / top-p.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from lightcone_spec.exit_codes import ExactnessViolation
from lightcone_spec.runtime.rng import DrawKind, categorical_draw, uniform_draw

Dist = np.ndarray  # 1-D probability vector over the alphabet/vocab
PrefixDist = Callable[[tuple[int, ...]], Dist]  # prefix -> distribution

_EPS = 1e-300


def normalize_sampling_dist(
    logits: np.ndarray, temperature: float = 1.0, top_p: float = 1.0
) -> Dist:
    """Locked sampling-distribution semantics. Main-table configs use
    temperature 1 / top-p 1 with no extra truncation; temperature 0 is
    greedy (a point mass), used only in the parity side table."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0.0:
        out = np.zeros_like(logits)
        out[int(np.argmax(logits))] = 1.0
        return out
    z = logits / temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        csum = np.cumsum(p[order])
        cut = int(np.searchsorted(csum, top_p) + 1)
        mask = np.zeros_like(p)
        mask[order[:cut]] = 1.0
        p = p * mask
        p /= p.sum()
    return p


def residual_distribution(p: Dist, q: Dist) -> Dist:
    """r(x) proportional to [p(x) - q(x)]_+ ; must be sampleable whenever a
    rejection can occur."""
    resid = np.maximum(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64), 0.0)
    mass = resid.sum()
    if mass <= 0.0:
        raise ExactnessViolation(
            "rejection occurred but residual mass is zero: p/q inconsistent"
        )
    return resid / mass


def acceptance_probability(p_y: float, q_y: float) -> float:
    if q_y <= 0.0:
        raise ExactnessViolation("proposal assigned zero probability to drawn token")
    return min(1.0, p_y / q_y)


@dataclass
class RoundOutcome:
    committed_tokens: list[int]
    accepted_drafts: int
    draft_tokens: list[int]
    rejected_at: int | None  # 0-based draft position, None if all accepted
    used_bonus: bool
    rng_substream_ids: list[str] = field(default_factory=list)


def run_speculative_round(
    prefix: tuple[int, ...],
    target: PrefixDist,
    proposal: PrefixDist,
    depth: int,
    seed: int,
    request_id_hash: str,
    round_id: int,
    denominator_proposal: PrefixDist | None = None,
) -> RoundOutcome:
    """One draft/verify/accept round with counter-based Philox substreams.

    denominator_proposal exists only for the deliberate race canary: the
    correct protocol always uses the same version for sampling and for the
    acceptance denominator/residual.
    """
    q_den = denominator_proposal or proposal
    drafts: list[int] = []
    cur = tuple(prefix)
    for k in range(depth):
        q = proposal(cur)
        tok = categorical_draw(
            q, seed, request_id_hash, round_id, k, DrawKind.PROPOSAL
        )
        drafts.append(tok)
        cur = cur + (tok,)

    committed: list[int] = []
    cur = tuple(prefix)
    rejected_at: int | None = None
    for k, tok in enumerate(drafts):
        p = target(cur)
        qd = q_den(cur)
        a = acceptance_probability(float(p[tok]), float(qd[tok]))
        u = uniform_draw(seed, request_id_hash, round_id, k, DrawKind.ACCEPTANCE)
        if u <= a:
            committed.append(tok)
            cur = cur + (tok,)
        else:
            r = residual_distribution(p, qd)
            rtok = categorical_draw(
                r, seed, request_id_hash, round_id, k, DrawKind.RESIDUAL
            )
            committed.append(rtok)
            rejected_at = k
            break
    used_bonus = False
    if rejected_at is None:
        p = target(cur)
        btok = categorical_draw(
            p, seed, request_id_hash, round_id, depth, DrawKind.BONUS
        )
        committed.append(btok)
        used_bonus = True
    accepted = (rejected_at if rejected_at is not None else depth)
    return RoundOutcome(
        committed_tokens=committed,
        accepted_drafts=accepted,
        draft_tokens=drafts,
        rejected_at=rejected_at,
        used_bonus=used_bonus,
    )


# ---------------------------------------------------------------------------
# Exact law enumeration (P0-B, spec 3.4 / 13.1)
# ---------------------------------------------------------------------------


def _enumerate_round(
    prefix: tuple[int, ...],
    target: PrefixDist,
    proposal: PrefixDist,
    q_den: PrefixDist,
    depth: int,
) -> list[tuple[tuple[int, ...], float]]:
    """All (committed_suffix, probability) outcomes for one round starting
    at prefix, enumerated exactly."""
    alphabet = len(target(prefix))
    outcomes: list[tuple[tuple[int, ...], float]] = []

    def recurse(
        pos: int,
        cur_prefix: tuple[int, ...],
        committed: tuple[int, ...],
        prob: float,
    ) -> None:
        if pos == depth:
            # All drafts accepted: bonus draw from target.
            p = target(cur_prefix)
            for b in range(alphabet):
                if p[b] > 0:
                    outcomes.append((committed + (b,), prob * float(p[b])))
            return
        q = proposal(cur_prefix)
        p = target(cur_prefix)
        qd = q_den(cur_prefix)
        for tok in range(alphabet):
            q_tok = float(q[tok])
            if q_tok <= 0.0:
                continue
            a = min(1.0, float(p[tok]) / max(float(qd[tok]), _EPS))
            # Accept branch.
            recurse(pos + 1, cur_prefix + (tok,), committed + (tok,), prob * q_tok * a)
            # Reject branch -> residual against the denominator version.
            rej = q_tok * (1.0 - a)
            if rej > 0.0:
                resid = np.maximum(p - qd, 0.0)
                mass = float(resid.sum())
                if mass <= 0.0:
                    # Correct protocol: rejection probability is exactly 0
                    # when p <= q everywhere; a wrong-version denominator can
                    # land here with leftover mass, which we spread over p
                    # (this branch is unreachable under version-correct runs).
                    resid = p.copy()
                    mass = float(resid.sum())
                for x in range(alphabet):
                    rx = float(resid[x]) / mass
                    if rx > 0.0:
                        outcomes.append((committed + (x,), prob * rej * rx))

    recurse(0, tuple(prefix), tuple(), 1.0)
    return outcomes


def enumerate_speculative_law(
    target: PrefixDist,
    proposal_for_round: Callable[[int], PrefixDist],
    horizon: int,
    depth: int,
    denominator_for_round: Callable[[int], PrefixDist] | None = None,
) -> dict[tuple[int, ...], float]:
    """Exact joint law over the first `horizon` tokens under the
    speculative protocol. proposal_for_round(r) is the (possibly
    round-varying) locked proposal version used by round r; the correct
    protocol uses the same version as acceptance denominator."""
    law: dict[tuple[int, ...], float] = {}
    stack: list[tuple[tuple[int, ...], float, int]] = [((), 1.0, 0)]
    while stack:
        prefix, prob, round_id = stack.pop()
        remaining = horizon - len(prefix)
        if remaining <= 0:
            key = prefix[:horizon]
            law[key] = law.get(key, 0.0) + prob
            continue
        gamma = min(depth, max(remaining - 1, 0))
        q = proposal_for_round(round_id)
        q_den = denominator_for_round(round_id) if denominator_for_round else q
        for suffix, sp in _enumerate_round(prefix, target, q, q_den, gamma):
            new_prefix = prefix + suffix
            stack.append((new_prefix[:horizon] if len(new_prefix) >= horizon else new_prefix,
                          prob * sp, round_id + 1))
            if len(new_prefix) > horizon:
                # Excess tokens beyond the horizon do not affect the law of
                # the first `horizon` tokens; mass already truncated above.
                pass
    return law


def enumerate_target_law(
    target: PrefixDist, horizon: int, alphabet: int
) -> dict[tuple[int, ...], float]:
    law: dict[tuple[int, ...], float] = {}

    def recurse(prefix: tuple[int, ...], prob: float) -> None:
        if len(prefix) == horizon:
            law[prefix] = law.get(prefix, 0.0) + prob
            return
        p = target(prefix)
        for tok in range(alphabet):
            if p[tok] > 0:
                recurse(prefix + (tok,), prob * float(p[tok]))

    recurse((), 1.0)
    return law


def total_variation_between_laws(
    a: dict[tuple[int, ...], float], b: dict[tuple[int, ...], float]
) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
