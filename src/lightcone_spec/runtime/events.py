"""Update event chain (spec 4.3).

T_snapshot < T_teacher <= T_launch <= T_done <= T_commit < T_exposure.
Events are recorded with a CPU monotonic clock plus (on GPU) CUDA
events; the CPU reference engine uses the same record type with
simulated timestamps.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from lightcone_spec.exit_codes import ExactnessViolation


def monotonic_us() -> float:
    return time.monotonic_ns() / 1000.0


@dataclass
class UpdateEventChain:
    update_id: str
    source_round: int
    source_version: int

    snapshot_ts_us: Optional[float] = None
    teacher_ts_us: Optional[float] = None
    launch_ts_us: Optional[float] = None
    done_ts_us: Optional[float] = None
    commit_ts_us: Optional[float] = None
    exposure_ts_us: Optional[float] = None

    apply_round: Optional[int] = None
    exposure_round: Optional[int] = None

    launch_event_id: str = field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:12]}")
    done_event_id: str = field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:12]}")
    commit_event_id: Optional[str] = None

    def mark(self, name: str, ts_us: Optional[float] = None) -> None:
        ts = monotonic_us() if ts_us is None else ts_us
        setattr(self, f"{name}_ts_us", ts)
        if name == "commit":
            self.commit_event_id = f"ev-{uuid.uuid4().hex[:12]}"

    def validate(self, allow_incomplete: bool = True) -> None:
        chain = [
            ("snapshot", self.snapshot_ts_us, "strict"),
            ("teacher", self.teacher_ts_us, "lte"),
            ("launch", self.launch_ts_us, "lte"),
            ("done", self.done_ts_us, "lte"),
            ("commit", self.commit_ts_us, "strict_next"),
            ("exposure", self.exposure_ts_us, None),
        ]
        prev_name, prev_ts = None, None
        for i, (name, ts, _) in enumerate(chain):
            if ts is None:
                if allow_incomplete:
                    remaining = [c for c in chain[i + 1 :] if c[1] is not None]
                    if remaining:
                        raise ExactnessViolation(
                            f"update {self.update_id}: event {name} missing but a "
                            f"later event was recorded"
                        )
                    return
                raise ExactnessViolation(
                    f"update {self.update_id}: event {name} missing"
                )
            if prev_ts is not None:
                strict = (prev_name == "snapshot") or (name == "exposure")
                if strict and not (prev_ts < ts):
                    raise ExactnessViolation(
                        f"update {self.update_id}: {prev_name} ({prev_ts}) must be "
                        f"strictly before {name} ({ts})"
                    )
                if not strict and not (prev_ts <= ts):
                    raise ExactnessViolation(
                        f"update {self.update_id}: {prev_name} ({prev_ts}) must be "
                        f"<= {name} ({ts})"
                    )
            prev_name, prev_ts = name, ts
