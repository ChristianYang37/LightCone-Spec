"""Cohort isolation, latest-signal batching, and version-safe publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from lightcone_spec.methods.core import CandidateUpdate, MethodPolicy


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
            character not in "0123456789abcdef"
            for character in self.cohort_sha256
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
    signals: LatestSignalBatch = field(init=False)
    in_flight: CandidateUpdate | None = None
    disabled_reason: str | None = None

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
        if current_round < candidate.ready_round:
            return False, "side_stream_not_ready"
        if policy is MethodPolicy.STATIC:
            return False, "static_disabled"
        if policy is MethodPolicy.FIXED_BARRIER:
            boundary = ((candidate.source_round // stride) + 1) * stride
            if current_round < boundary:
                return False, "waiting_fixed_boundary"
        return True, "ready"

    def commit(self, candidate: CandidateUpdate) -> int:
        if candidate is not self.in_flight:
            raise RuntimeError("candidate is not active")
        self.active_version += 1
        self.in_flight = None
        return self.active_version

    def discard(self, candidate: CandidateUpdate) -> None:
        if candidate is self.in_flight:
            self.in_flight = None

    def cancel_request(self, request_id: str) -> None:
        self.signals.discard_request(request_id)
        self.slot_generation += 1
        self.in_flight = None

    def reset(self) -> None:
        self.epoch += 1
        self.active_version = 0
        self.slot_generation += 1
        self.in_flight = None
        self.signals = LatestSignalBatch(self.identity.sha256)
        self.disabled_reason = None

    def disable(self, reason: str) -> None:
        if not reason:
            raise ValueError("disabled cohorts require an evidence reason")
        self.disabled_reason = reason
        self.in_flight = None
