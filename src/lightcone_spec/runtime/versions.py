"""Per-request version state machine (spec 4.2) and the fail-closed
version-mismatch checks (spec 3.3).

Any of the six mismatch conditions terminates the current run unit with
status `failed_exactness`; performance summaries are never generated for
such units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lightcone_spec.exit_codes import ExactnessViolation


@dataclass
class RequestVersionState:
    request_id: str
    tenant_id_hash: str
    stream_id: Optional[str]
    request_epoch: int

    active_version: int = 0
    staging_version: int = -1
    source_version: int = 0
    proposal_version: int = 0
    optimizer_version: int = 0
    controller_version: int = 0
    pending_update_id: Optional[str] = None

    canary_log: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.canary_log.append(reason)
        raise ExactnessViolation(
            f"[request={self.request_id} epoch={self.request_epoch}] {reason}"
        )

    # ---- spec 3.3 checks ----------------------------------------------

    def check_canvas_consistency(
        self,
        proposal_version: int,
        denominator_version: int,
        residual_version: int,
    ) -> None:
        if proposal_version != denominator_version:
            self.fail(
                f"proposal_version {proposal_version} != denominator_version "
                f"{denominator_version}"
            )
        if proposal_version != residual_version:
            self.fail(
                f"proposal_version {proposal_version} != residual_version "
                f"{residual_version}"
            )

    def check_proposal_not_overwritten(self, canvas_alive: bool) -> None:
        if not canvas_alive:
            self.fail("proposal logits overwritten before verification consumed them")

    def check_active_not_written_during_replay(self, wrote_active: bool) -> None:
        if wrote_active:
            self.fail("active storage written by update kernel during graph replay")

    def check_source_matches_active_for_direct_add(self, source_version: int) -> None:
        if source_version != self.active_version:
            self.fail(
                f"candidate delta with source_version {source_version} added "
                f"directly onto active_version {self.active_version} "
                "(requires explicit transport/rebase)"
            )

    def check_slot_ownership(self, slot_tenant_hash: str, slot_epoch: int) -> None:
        if slot_tenant_hash != self.tenant_id_hash:
            self.fail(
                "request read an adapter slot belonging to another tenant "
                f"({slot_tenant_hash} != {self.tenant_id_hash})"
            )
        if slot_epoch != self.request_epoch:
            self.fail(
                f"adapter slot epoch {slot_epoch} != request epoch "
                f"{self.request_epoch} (ABA reuse)"
            )
