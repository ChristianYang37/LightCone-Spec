"""Exactness canaries (spec 3.3, 3.4, 15.4).

The harness must be able to detect a deliberately wrong version
protocol: the injected race canary produces a joint-law TV of at least
0.05 versus the target law, and if the harness fails to flag it the
harness itself fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CanaryCounter:
    version_mismatch: int = 0
    proposal_overwrite: int = 0
    replay_write: int = 0
    source_active_conflict: int = 0
    tenant_mismatch: int = 0
    epoch_aba: int = 0
    details: list[str] = field(default_factory=list)

    def record(self, kind: str, detail: str) -> None:
        if not hasattr(self, kind):
            raise ValueError(f"unknown canary kind {kind}")
        setattr(self, kind, getattr(self, kind) + 1)
        self.details.append(f"{kind}: {detail}")

    def total(self) -> int:
        return (
            self.version_mismatch
            + self.proposal_overwrite
            + self.replay_write
            + self.source_active_conflict
            + self.tenant_mismatch
            + self.epoch_aba
        )

    def to_dict(self) -> dict:
        return {
            "version_mismatch": self.version_mismatch,
            "proposal_overwrite": self.proposal_overwrite,
            "replay_write": self.replay_write,
            "source_active_conflict": self.source_active_conflict,
            "tenant_mismatch": self.tenant_mismatch,
            "epoch_aba": self.epoch_aba,
            "total": self.total(),
            "details": list(self.details),
        }
