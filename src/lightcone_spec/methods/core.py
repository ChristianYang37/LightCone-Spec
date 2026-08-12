"""Core method semantics.

TTS and L0 consume the same source-bound candidate. Their sole semantic
difference is the legal publication round: TTS waits for the next fixed
update boundary, whereas L0 publishes at the first legal boundary after the
side-stream event becomes ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class MethodPolicy(str, Enum):
    STATIC = "static"
    FIXED_BARRIER = "tts"
    FIRST_READY_BOUNDARY = "l0"


@dataclass(frozen=True)
class CandidateUpdate:
    payload: tuple[Any, ...]
    source_round: int
    source_version: int
    cohort_epoch: int
    slot_generation: int
    ready_round: int
    buffer_generation: int = 0
    optimizer_state_generation: int = 0
    ready_event_id: str = "logical-ready"
    numerically_valid: bool = True
    memory_reservation_bytes: int = 0
    outcome: Literal["pending"] = "pending"

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("candidate payload must be non-empty")
        counters = (
            self.source_round,
            self.source_version,
            self.cohort_epoch,
            self.slot_generation,
            self.ready_round,
            self.buffer_generation,
            self.optimizer_state_generation,
            self.memory_reservation_bytes,
        )
        if any(value < 0 for value in counters):
            raise ValueError("candidate counters must be non-negative")
        if self.ready_round < self.source_round:
            raise ValueError("candidate cannot be ready before its source round")
        if not self.ready_event_id:
            raise ValueError("candidate readiness event identity must be non-empty")


@dataclass(frozen=True)
class CandidateTermination:
    """Exactly-once terminal outcome for a source-bound candidate."""

    candidate: CandidateUpdate
    outcome: Literal["published", "discarded"]
    reason: str
    published_version: int | None = None

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("candidate termination requires a reason code")
        if self.outcome == "published" and self.published_version is None:
            raise ValueError("published candidate requires its committed version")
        if self.outcome == "discarded" and self.published_version is not None:
            raise ValueError("discarded candidate cannot carry a published version")


@dataclass(frozen=True)
class PublicationDelay:
    """One definition shared by schema, simulator, runtime, and telemetry."""

    source_round: int
    intrinsic_ready_round: int
    publication_round: int
    extra_logical_rounds: int

    @property
    def intrinsic_pipeline_rounds(self) -> int:
        return self.intrinsic_ready_round - self.source_round


def policy_for(method: str) -> MethodPolicy:
    try:
        return MethodPolicy(method)
    except ValueError as exc:
        raise ValueError(f"{method!r} is not a core method") from exc


def publication_round(
    policy: MethodPolicy,
    candidate: CandidateUpdate,
    stride: int,
    extra_logical_delay: int = 0,
) -> int | None:
    """Return the first legal round, independent of candidate contents."""
    if policy is MethodPolicy.STATIC:
        return None
    if stride < 1:
        raise ValueError("stride must be positive")
    if extra_logical_delay < 0:
        raise ValueError("extra logical delay must be non-negative")
    delayed_ready = candidate.ready_round + extra_logical_delay
    if policy is MethodPolicy.FIRST_READY_BOUNDARY:
        return delayed_ready
    next_boundary = ((candidate.source_round // stride) + 1) * stride
    return max(delayed_ready, next_boundary)


def publication_delay(
    policy: MethodPolicy,
    candidate: CandidateUpdate,
    stride: int,
    extra_logical_delay: int = 0,
) -> PublicationDelay | None:
    published = publication_round(
        policy,
        candidate,
        stride,
        extra_logical_delay,
    )
    if published is None:
        return None
    return PublicationDelay(
        source_round=candidate.source_round,
        intrinsic_ready_round=candidate.ready_round,
        publication_round=published,
        extra_logical_rounds=published - candidate.ready_round,
    )


def assert_candidate_equivalence(left: CandidateUpdate, right: CandidateUpdate) -> None:
    """Fail closed if a comparison changes anything except publication policy."""
    if left != right:
        raise ValueError("TTS and L0 must receive the identical candidate update")
