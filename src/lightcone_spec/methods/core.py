"""Core method semantics.

TTS and L0 consume the same source-bound candidate. Their sole semantic
difference is the legal publication round: TTS waits for the next fixed
update boundary, whereas L0 publishes at the first legal boundary after the
side-stream event becomes ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MethodPolicy(str, Enum):
    STATIC = "static"
    FIXED_BARRIER = "tts"
    FIRST_READY_BOUNDARY = "naive_async"


@dataclass(frozen=True)
class CandidateUpdate:
    payload: tuple[Any, ...]
    source_round: int
    source_version: int
    cohort_epoch: int
    slot_generation: int
    ready_round: int

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("candidate payload must be non-empty")
        counters = (
            self.source_round,
            self.source_version,
            self.cohort_epoch,
            self.slot_generation,
            self.ready_round,
        )
        if any(value < 0 for value in counters):
            raise ValueError("candidate counters must be non-negative")
        if self.ready_round < self.source_round:
            raise ValueError("candidate cannot be ready before its source round")


def policy_for(method: str) -> MethodPolicy:
    try:
        return MethodPolicy(method)
    except ValueError as exc:
        raise ValueError(f"{method!r} is not a core method") from exc


def publication_round(
    policy: MethodPolicy, candidate: CandidateUpdate, stride: int
) -> int | None:
    """Return the first legal round, independent of candidate contents."""
    if policy is MethodPolicy.STATIC:
        return None
    if stride < 1:
        raise ValueError("stride must be positive")
    if policy is MethodPolicy.FIRST_READY_BOUNDARY:
        return candidate.ready_round
    next_boundary = ((candidate.source_round // stride) + 1) * stride
    return max(candidate.ready_round, next_boundary)


def assert_candidate_equivalence(
    left: CandidateUpdate, right: CandidateUpdate
) -> None:
    """Fail closed if a comparison changes anything except publication policy."""
    if left != right:
        raise ValueError("TTS and L0 must receive the identical candidate update")
