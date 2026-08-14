"""Cohort isolation, latest-signal batching, and version-safe publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from lightcone_spec.methods import (
    CandidateTermination,
    CandidateUpdate,
    MethodPolicy,
    publication_round,
)


@dataclass(frozen=True)
class CohortIdentity:
    target_revision: str
    drafter_revision: str
    algorithm: str
    sampling_profile_sha256: str
    adaptation_group_id: str
    tenant_id: str
    update_mode: str
    parameter_scope: str
    parameter_layout_sha256: str
    optimizer_identity: str

    def __post_init__(self) -> None:
        for name, size in (
            ("target_revision", 40),
            ("drafter_revision", 40),
            ("sampling_profile_sha256", 64),
            ("parameter_layout_sha256", 64),
        ):
            value = getattr(self, name)
            if len(value) != size or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase immutable hash")
        for name in (
            "algorithm",
            "adaptation_group_id",
            "tenant_id",
            "update_mode",
            "parameter_scope",
            "optimizer_identity",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")

    @property
    def sha256(self) -> str:
        body = json.dumps(
            self.__dict__,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True)
class SupervisionSignal:
    cohort_sha256: str
    request_id: str
    sequence_number: int
    source_version: int
    slot_generation: int
    tensors: tuple[Any, ...]
    valid_positions: int

    def __post_init__(self) -> None:
        if len(self.cohort_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.cohort_sha256
        ):
            raise ValueError("cohort_sha256 must be a lowercase SHA-256")
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if (
            self.sequence_number < 0
            or self.source_version < 0
            or self.slot_generation < 0
            or self.valid_positions < 1
        ):
            raise ValueError("signal counters are outside their valid range")
        if not self.tensors:
            raise ValueError("supervision signal tensors must be non-empty")


@dataclass
class LatestSignalBatch:
    """One latest legal signal per request, normalized over the batch."""

    cohort_sha256: str
    _signals: dict[str, SupervisionSignal] = field(default_factory=dict)

    def offer(self, signal: SupervisionSignal) -> bool:
        if signal.cohort_sha256 != self.cohort_sha256:
            raise ValueError("cross-cohort supervision is forbidden")
        previous = self._signals.get(signal.request_id)
        if previous is not None and signal.sequence_number <= previous.sequence_number:
            return False
        self._signals[signal.request_id] = signal
        return True

    def drain(self) -> tuple[tuple[SupervisionSignal, ...], float]:
        if not self._signals:
            return (), 0.0
        ordered = tuple(
            sorted(
                self._signals.values(),
                key=lambda item: (item.request_id, item.sequence_number),
            )
        )
        self._signals.clear()
        return ordered, 1.0 / len(ordered)

    def discard_request(self, request_id: str) -> None:
        self._signals.pop(request_id, None)


@dataclass
class CohortRuntime:
    """Host-side protocol state; tensor payloads remain on their CUDA device."""

    identity: CohortIdentity
    epoch: int = 0
    active_version: int = 0
    slot_generation: int = 0
    buffer_generation: int = 0
    optimizer_state_generation: int = 0
    signals: LatestSignalBatch = field(init=False)
    in_flight: CandidateUpdate | None = None
    disabled_reason: str | None = None
    last_termination: CandidateTermination | None = None

    def __post_init__(self) -> None:
        self.signals = LatestSignalBatch(self.identity.sha256)

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def offer_signal(self, signal: SupervisionSignal) -> bool:
        if not self.enabled:
            return False
        if signal.source_version != self.active_version:
            return False
        if signal.slot_generation != self.slot_generation:
            return False
        return self.signals.offer(signal)

    def begin_candidate(
        self,
        *,
        payload: tuple[Any, ...],
        source_round: int,
        ready_round: int,
    ) -> CandidateUpdate:
        if not self.enabled:
            raise RuntimeError(f"cohort adaptation disabled: {self.disabled_reason}")
        if self.in_flight is not None:
            raise RuntimeError("max_in_flight=1")
        candidate = CandidateUpdate(
            payload=payload,
            source_round=source_round,
            source_version=self.active_version,
            cohort_epoch=self.epoch,
            slot_generation=self.slot_generation,
            ready_round=ready_round,
            buffer_generation=self.buffer_generation,
            optimizer_state_generation=self.optimizer_state_generation,
        )
        self.in_flight = candidate
        return candidate

    def can_publish(
        self,
        candidate: CandidateUpdate,
        *,
        policy: MethodPolicy,
        current_round: int,
        stride: int,
        extra_logical_delay: int = 0,
    ) -> tuple[bool, str]:
        if stride < 1:
            raise ValueError("stride must be positive")
        if candidate is not self.in_flight:
            return False, "not_active_candidate"
        if candidate.cohort_epoch != self.epoch:
            return False, "stale_epoch"
        if candidate.slot_generation != self.slot_generation:
            return False, "stale_slot_generation"
        if candidate.source_version != self.active_version:
            return False, "source_version_conflict"
        if candidate.buffer_generation != self.buffer_generation:
            return False, "stale_buffer_generation"
        if candidate.optimizer_state_generation != self.optimizer_state_generation:
            return False, "stale_optimizer_state_generation"
        if not candidate.numerically_valid:
            return False, "nonfinite_candidate"
        if current_round < candidate.ready_round:
            return False, "side_stream_not_ready"
        if policy is MethodPolicy.STATIC:
            return False, "static_disabled"
        boundary = publication_round(
            policy,
            candidate,
            stride,
            extra_logical_delay,
        )
        if boundary is None:
            return False, "static_disabled"
        if current_round < boundary:
            if current_round < candidate.ready_round + extra_logical_delay:
                return False, "waiting_extra_logical_delay"
            return False, "waiting_fixed_boundary"
        return True, "ready"

    def commit(
        self,
        candidate: CandidateUpdate,
        *,
        policy: MethodPolicy,
        current_round: int,
        stride: int,
        extra_logical_delay: int = 0,
    ) -> int:
        """Publish only after revalidating the complete boundary authority.

        Requiring the boundary inputs here prevents a caller from using
        ``commit`` as a back door around stale/version/numerical/readiness
        checks performed by :meth:`can_publish`.
        """

        allowed, reason = self.can_publish(
            candidate,
            policy=policy,
            current_round=current_round,
            stride=stride,
            extra_logical_delay=extra_logical_delay,
        )
        if not allowed:
            raise RuntimeError(f"candidate lacks publication authority: {reason}")
        self.active_version += 1
        self.optimizer_state_generation += 1
        self._terminate(
            candidate,
            outcome="published",
            reason="committed",
            published_version=self.active_version,
        )
        return self.active_version

    def _terminate(
        self,
        candidate: CandidateUpdate,
        *,
        outcome: Literal["published", "discarded"],
        reason: str,
        published_version: int | None = None,
    ) -> CandidateTermination:
        if (
            self.last_termination is not None
            and self.last_termination.candidate is candidate
        ):
            raise RuntimeError("candidate was already terminated")
        if candidate is not self.in_flight:
            raise RuntimeError("candidate is not active")
        termination = CandidateTermination(
            candidate=candidate,
            outcome=outcome,
            reason=reason,
            published_version=published_version,
        )
        self.in_flight = None
        self.last_termination = termination
        return termination

    def discard(
        self,
        candidate: CandidateUpdate,
        reason: str = "discarded",
    ) -> CandidateTermination:
        return self._terminate(
            candidate,
            outcome="discarded",
            reason=reason,
        )

    def cancel_request(self, request_id: str) -> None:
        self.signals.discard_request(request_id)
        if self.in_flight is not None:
            self.discard(self.in_flight, "request_cancelled")
        self.slot_generation += 1
        self.buffer_generation += 1

    def reset(self) -> None:
        if self.in_flight is not None:
            self.discard(self.in_flight, "cohort_reset")
        self.epoch += 1
        self.active_version = 0
        self.slot_generation += 1
        self.buffer_generation += 1
        self.optimizer_state_generation = 0
        self.signals = LatestSignalBatch(self.identity.sha256)
        self.disabled_reason = None

    def disable(self, reason: str) -> None:
        if not reason:
            raise ValueError("disabled cohorts require an evidence reason")
        if self.in_flight is not None:
            self.discard(self.in_flight, reason)
        self.disabled_reason = reason
