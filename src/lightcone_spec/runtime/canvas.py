"""Canvas version lock (spec 3.2).

A canvas immutably binds: the proposal parameters version, the full
per-position proposal distributions (or a lossless reconstruction), the
acceptance denominator, the rejection residual source, bonus alignment
info, and the RNG substream ids it consumed. Controllers may only decide
when a new parameter version is published; they can never mutate the
proposal a live canvas is bound to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lightcone_spec.exit_codes import ExactnessViolation


@dataclass
class Canvas:
    request_id: str
    round_id: int
    proposal_version: int
    denominator_version: int
    residual_version: int
    draft_tokens: list[int]
    proposal_probs: list[np.ndarray]  # one full distribution per position
    confidence_logits: np.ndarray | None
    rng_substream_ids: list[str] = field(default_factory=list)
    consumed: bool = False
    _alive: bool = True

    def assert_version_locked(self) -> None:
        if not self._alive:
            raise ExactnessViolation(
                f"canvas r{self.round_id}: proposal storage overwritten before "
                "verification consumed it"
            )
        if self.proposal_version != self.denominator_version:
            raise ExactnessViolation(
                f"canvas r{self.round_id}: proposal_version "
                f"{self.proposal_version} != denominator_version "
                f"{self.denominator_version}"
            )
        if self.proposal_version != self.residual_version:
            raise ExactnessViolation(
                f"canvas r{self.round_id}: proposal_version "
                f"{self.proposal_version} != residual_version "
                f"{self.residual_version}"
            )

    def consume(self) -> None:
        self.assert_version_locked()
        self.consumed = True

    def release(self) -> None:
        """Retention lifecycle (spec 8.4): free only after verification and
        the training signal both consumed the canvas."""
        if not self.consumed:
            self._alive = False
        else:
            self.proposal_probs = []
            self._alive = False
