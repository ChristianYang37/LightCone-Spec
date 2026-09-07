"""Atomic resource leases for independent cells and indivisible paired blocks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .protocol import Job, memory_budget_policy


def logical_unit_key(job: Job, *, splittable: bool = False) -> tuple:
    p = job.parameters
    policy = p.get("logical_memory_budget_policy", memory_budget_policy(job))
    stage = str(p.get("source_node", job.node)).removesuffix("-segments")
    if p.get("pairing_key") is not None:
        return ("paired", stage, str(p["pairing_key"]), job.block,
                p.get("topology", "tp1_dp1"), policy)
    if job.block is not None:
        return (
            "block", stage, job.model, p.get("comparison_backend", job.backend),
            p.get("topology", "tp1_dp1"), p.get("panel"),
            policy, job.block,
        )
    if not splittable:
        parent = p.get("original_parent_job_id", p.get("parent_job_id"))
        if parent:
            return ("parent", str(parent))
    return ("cell", str(p.get("replaces_job_id", job.job_id)))


@dataclass(frozen=True)
class WorkUnit:
    key: tuple
    jobs: tuple[Job, ...]
    devices: int
    sessions: tuple[tuple, ...]
    work: float
    pin: tuple[int, ...] = ()
    isolated: bool = False
    execution_policy: str = "automatic_units_v3"
    legacy_cpus: tuple[int, ...] = ()
    execution_pin: tuple[int, ...] = ()


class WorkPool:
    """One owner per unit; a waiting two-device unit drains single-device leases.

    The SQLite start remains the final cell claim. This lock only owns resources
    inside the unique runner, and never modifies the registered job payload.
    """

    def __init__(self, units: tuple[WorkUnit, ...], gpus: tuple[int, ...]):
        self.pending = list(units)
        self.gpus = gpus
        self.active: dict[int, tuple[WorkUnit, tuple[int, ...]]] = {}
        self.condition = threading.Condition()
        self.single_claims = 0
        self.events: list[dict] = []
        self.wait_reasons: dict[int, str] = {}

    def claim(self, worker: int, preferred: tuple | None, stop: threading.Event,
              failed: threading.Event) -> tuple[WorkUnit, tuple[int, ...]] | None:
        with self.condition:
            while self.pending and not stop.is_set() and not failed.is_set():
                occupied = {gpu for _, devices in self.active.values() for gpu in devices}
                # Defer hard isolation, but not genuine TP2/DP2 compute. A bounded
                # two-single quantum prevents an endlessly warm TP1 queue starving TP2.
                ordinary = [unit for unit in self.pending if not unit.isolated]
                ready = ordinary or self.pending
                doubles = [unit for unit in ready if unit.devices == 2]
                drain = min(doubles, key=self._order) if doubles and self.single_claims >= 2 else None
                candidates = []
                for unit in ready:
                    devices = unit.pin or (self.gpus if unit.devices == 2 else (worker,))
                    if worker not in devices or occupied.intersection(devices):
                        continue
                    if drain is not None and unit is not drain:
                        continue
                    if unit.isolated and self.active:
                        continue
                    candidates.append((unit, devices))
                if candidates:
                    unit, devices = min(candidates, key=lambda item: (
                        0 if preferred in item[0].sessions else 1, *self._order(item[0]),
                    ))
                    self.pending.remove(unit)
                    self.wait_reasons.pop(worker, None)
                    self.active[worker] = (unit, devices)
                    self.single_claims = self.single_claims + 1 if unit.devices == 1 else 0
                    self.events.append({"event": "claim", "time": time.time(),
                                        "worker": worker, "unit": list(unit.key),
                                        "reserved_gpu_ids": list(devices),
                                        "co_running_units": [list(u.key) for w, (u, _) in
                                                             self.active.items() if w != worker]})
                    return unit, devices
                # Completion wakes this wait immediately; timeout only observes SIGINT.
                reason = ("dual_device_drain" if drain else "hard_isolation" if not ordinary
                          else "unit_affinity_or_owned_devices")
                if self.wait_reasons.get(worker) != reason:
                    self.wait_reasons[worker] = reason
                    self.events.append({"event": "idle", "time": time.time(),
                                        "worker": worker, "reason": reason})
                self.condition.wait(timeout=0.2)
            self.events.append({"event": "idle", "time": time.time(), "worker": worker,
                                "reason": "interrupted_or_failed" if stop.is_set() or failed.is_set()
                                else "unit_tail"})
            return None

    @staticmethod
    def _order(unit: WorkUnit) -> tuple:
        return (-unit.work, min(job.ordinal for job in unit.jobs), unit.jobs[0].job_id)

    def release(self, worker: int) -> None:
        with self.condition:
            unit, _ = self.active.pop(worker)
            self.events.append({"event": "release", "time": time.time(), "worker": worker,
                                "unit": list(unit.key)})
            self.condition.notify_all()

    def handoff(self, worker: int, session: tuple, stop: threading.Event,
                failed: threading.Event) -> tuple[WorkUnit, tuple[int, ...]] | None:
        """Transfer a warm TP1 lease without exposing its resident server to TP2."""
        with self.condition:
            current, devices = self.active[worker]
            if (stop.is_set() or failed.is_set() or current.devices != 1
                    or current.key[0] != "cell" or current.isolated
                    or (self.single_claims >= 2 and any(
                        u.devices == 2 and not u.isolated for u in self.pending))):
                return None
            candidates = [u for u in self.pending
                          if u.key[0] == "cell" and not u.isolated and u.devices == 1
                          and (not u.pin or u.pin == devices) and session in u.sessions
                          and not any(j.parameters.get("clean_server_per_cell") for j in u.jobs)]
            if not candidates:
                return None
            unit = min(candidates, key=self._order)
            self.pending.remove(unit)
            self.active[worker] = (unit, devices)
            self.single_claims += 1
            self.events.append({"event": "warm_handoff", "time": time.time(),
                                "worker": worker, "completed_unit": list(current.key),
                                "unit": list(unit.key), "reserved_gpu_ids": list(devices)})
            return unit, devices
