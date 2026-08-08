"""Counter-based Philox RNG substreams (spec 3.2).

Every random draw in the speculative pipeline comes from a Philox
substream keyed by (seed, request_id_hash, round_id, draft_position,
draw_kind). Proposal, acceptance, residual and bonus draws use distinct
draw kinds, so CUDA scheduling or async completion order can never
change which random numbers a decision consumes.
"""

from __future__ import annotations

import hashlib
from enum import Enum

import numpy as np


class DrawKind(str, Enum):
    PROPOSAL = "proposal"
    ACCEPTANCE = "acceptance"
    RESIDUAL = "residual"
    BONUS = "bonus"


def substream_id(
    seed: int,
    request_id_hash: str,
    round_id: int,
    draft_position: int,
    draw_kind: DrawKind,
) -> str:
    """Canonical, logged substream identifier."""
    return (
        f"philox/s{seed}/r{request_id_hash}/rd{round_id}"
        f"/p{draft_position}/{draw_kind.value}"
    )


def _philox_key(sid: str) -> np.ndarray:
    digest = hashlib.sha256(sid.encode("utf-8")).digest()
    return np.frombuffer(digest[:16], dtype=np.uint64)


def substream(
    seed: int,
    request_id_hash: str,
    round_id: int,
    draft_position: int,
    draw_kind: DrawKind,
) -> np.random.Generator:
    """A fresh, deterministic Philox generator for exactly one decision."""
    sid = substream_id(seed, request_id_hash, round_id, draft_position, draw_kind)
    return np.random.Generator(np.random.Philox(key=_philox_key(sid)))


def uniform_draw(
    seed: int,
    request_id_hash: str,
    round_id: int,
    draft_position: int,
    draw_kind: DrawKind,
) -> float:
    return float(
        substream(seed, request_id_hash, round_id, draft_position, draw_kind).random()
    )


def categorical_draw(
    probs: np.ndarray,
    seed: int,
    request_id_hash: str,
    round_id: int,
    draft_position: int,
    draw_kind: DrawKind,
) -> int:
    """Inverse-CDF categorical draw from a single uniform, so that the
    consumed randomness is exactly one counter tick per decision."""
    u = uniform_draw(seed, request_id_hash, round_id, draft_position, draw_kind)
    cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        raise ValueError("categorical_draw requires a positive finite mass")
    return int(np.searchsorted(cdf, u * total, side="right").clip(0, len(cdf) - 1))


def request_id_hash(request_id: str) -> str:
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
