"""Double-buffer publish protocol (spec 4.4, 4.5).

Active storage has a fixed address (CUDA graphs read it in place); side
streams write candidate deltas into staging; publishes happen only at
scheduler / graph-replay boundaries via an atomic staging -> active copy
followed by a version increment. max_in_flight defaults to 1; the value
2 exists only for the L3 parameter-staleness manifest, in which every
pending update carries its source snapshot, source optimizer state and
version ancestry and must be explicitly transported or discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from lightcone_spec.exit_codes import ExactnessViolation
from lightcone_spec.runtime.events import UpdateEventChain, monotonic_us


@dataclass
class ReadyEvent:
    """CPU abstraction of a CUDA done-event; on GPU this maps to
    cudaEventRecord on the side stream."""

    event_id: str
    done: bool = False
    ts_us: Optional[float] = None

    def record(self) -> None:
        self.done = True
        self.ts_us = monotonic_us()


@dataclass
class PendingUpdate:
    update_id: str
    source_round: int
    source_version: int
    candidate_delta: torch.Tensor  # FP32 delta produced by the optimizer
    raw_gradient: torch.Tensor
    events: UpdateEventChain
    ready: ReadyEvent
    # Only populated when max_in_flight = 2 (spec 4.4).
    source_snapshot: Optional[torch.Tensor] = None
    source_optimizer_state: Optional[dict] = None
    version_ancestry: tuple[int, ...] = ()
    logical_delay_left: int = 0


class DoubleBufferStore:
    """Request-local active/staging FP32 parameter storage."""

    def __init__(self, num_params: int, max_in_flight: int = 1):
        if max_in_flight not in (1, 2):
            raise ExactnessViolation("max_in_flight must be 1 or 2")
        self.active = torch.zeros(num_params, dtype=torch.float32)
        self.staging = torch.zeros(num_params, dtype=torch.float32)
        self.active_version = 0
        self.staging_version = -1
        self.max_in_flight = max_in_flight
        self.pending: list[PendingUpdate] = []
        self.in_replay = False
        self.commit_order: list[str] = []
        self.arrival_order: list[str] = []

    # ---- side-stream writes -------------------------------------------

    def can_launch(self) -> bool:
        return len(self.pending) < self.max_in_flight

    def launch(self, update: PendingUpdate) -> None:
        if not self.can_launch():
            raise ExactnessViolation(
                f"launch with {len(self.pending)} updates in flight exceeds "
                f"max_in_flight={self.max_in_flight}"
            )
        if self.max_in_flight == 2 and update.source_snapshot is None:
            raise ExactnessViolation(
                "max_in_flight=2 requires source snapshot + optimizer state "
                "+ version ancestry on every pending update"
            )
        update.events.mark("launch")
        self.pending.append(update)

    def write_staging(self, update: PendingUpdate, new_params: torch.Tensor) -> None:
        if self.in_replay:
            raise ExactnessViolation(
                "update kernel attempted to write during graph replay"
            )
        self.staging.copy_(new_params.to(torch.float32))
        self.staging_version = self.active_version + 1
        update.events.mark("done")
        update.ready.record()

    # ---- graph-boundary polling ----------------------------------------

    def poll_ready(self) -> list[PendingUpdate]:
        ready = [u for u in self.pending if u.ready.done]
        for u in ready:
            if u.update_id not in self.arrival_order:
                self.arrival_order.append(u.update_id)
        return ready

    def publish(self, update: PendingUpdate, new_params: torch.Tensor) -> int:
        """Atomic staging -> active copy at a legal boundary; increments the
        version only after the copy completes. Publishing only affects
        future canvases (in-flight proposals keep their bound version)."""
        if self.in_replay:
            raise ExactnessViolation("publish attempted during graph replay")
        if not update.ready.done:
            raise ExactnessViolation(
                f"publish of {update.update_id} before its done-event fired"
            )
        self.staging.copy_(new_params.to(torch.float32))
        self.active.copy_(self.staging)
        self.active_version += 1
        self.staging_version = -1
        update.events.apply_round = update.events.apply_round
        update.events.mark("commit")
        self.commit_order.append(update.update_id)
        self.pending.remove(update)
        return self.active_version

    def discard(self, update: PendingUpdate) -> None:
        self.staging.zero_()
        self.staging_version = -1
        self.pending.remove(update)

    # ---- replay guard ----------------------------------------------------

    def begin_replay(self) -> None:
        self.in_replay = True

    def end_replay(self) -> None:
        self.in_replay = False

    def read_active(self) -> torch.Tensor:
        return self.active.clone()
