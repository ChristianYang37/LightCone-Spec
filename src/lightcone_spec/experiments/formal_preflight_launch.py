"""Source-owned pre-allocation caps for the ten formal preflight rows.

The durable dispatch receipt owns the exact typed :class:`BudgetPlan`, physical
waves, capacity gate, and replay-reserved controls.  This module projects those
authorities into per-process and provider-wave hard caps and commits one whole
wave to an append-only ledger before any child process can be created.  A
caller never supplies a timeout, a GPU, a port, or a wave membership.
"""

from __future__ import annotations

import fcntl
import os
import stat
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

from lightcone_spec.experiments.formal_dispatch import (
    FormalPreflightDispatchReceipt,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "lightcone_formal_preflight_launch_cap_protocol",
        "authority": (
            "durable_dispatch_schema2_typed_budget_plan_capacity_gate_"
            "physical_wave_and_reserved_controls"
        ),
        "process_cap": "per_cell_budget_wall_time_quota_envelope",
        "provider_cap": ("whole_two_gpu_wave_timeout_with_exact_per_cell_attribution"),
        "retry": "exact_experiment_budget_retry_allowance",
        "allocation": "one_atomic_append_only_wave_consumption_before_popen",
        "caller_timeout_gpu_port_wave": "forbidden",
    }
)

_NS_PER_MS = 1_000_000


def _sha256(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lower-case SHA-256")
    return value


def _strict(label: str, value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    return dict(value)


def _private_directory(path: Path, *, create: bool) -> Path:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise ValueError("formal preflight ledger directory is not canonical")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(status.st_mode)
        or path.is_symlink()
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError("formal preflight ledger directory is not private")
    return path


@dataclass(frozen=True)
class FormalPreflightLaunchCellCap:
    materialized_cell_id: str
    registry_cell_id: str
    runner_kind: str
    experiment_budget_sha256: str
    physical_assignment_sha256: str
    topology_mode: str
    interference_mode: Literal["isolated", "concurrent"] | None
    interference_repetition: int | None
    interference_slot: int | None
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    localhost_ports: tuple[int, ...]
    wave_index: int
    wave_cell_ids: tuple[str, ...]
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    provider_reserved_gpu_count: Literal[1, 2]
    maximum_compute_gpu_ns_per_attempt: int
    maximum_provider_reserved_gpu_ns_per_attempt: int
    allowed_attempts: int

    def __post_init__(self) -> None:
        for label, value in (
            ("materialized cell", self.materialized_cell_id),
            ("registry cell", self.registry_cell_id),
            ("budget", self.experiment_budget_sha256),
            ("assignment", self.physical_assignment_sha256),
        ):
            _sha256(f"formal preflight cap {label}", value)
        if self.runner_kind not in {
            "first_party_compile",
            "first_party_exactness",
            "first_party_interference",
        }:
            raise ValueError("formal preflight cap runner differs")
        if self.runner_kind == "first_party_interference":
            if (
                self.interference_mode not in {"isolated", "concurrent"}
                or self.interference_repetition not in {0, 1}
                or self.interference_slot not in {0, 1}
            ):
                raise ValueError("formal preflight interference identity differs")
        elif (
            self.interference_mode is not None
            or self.interference_repetition is not None
            or self.interference_slot is not None
        ):
            raise ValueError("non-interference cap carries interference identity")
        if self.topology_mode not in {"tp1_dp1", "tp2_dp1", "tp1_dp2"}:
            raise ValueError("formal preflight cap topology differs")
        if (
            not self.gpu_uuids
            or len(self.gpu_uuids) != len(set(self.gpu_uuids))
            or tuple(uuid for group in self.rank_groups for uuid in group)
            != self.gpu_uuids
            or not self.localhost_ports
            or len(self.localhost_ports) != len(set(self.localhost_ports))
            or any(
                type(port) is not int or not 1024 <= port <= 65535
                for port in self.localhost_ports
            )
            or type(self.wave_index) is not int
            or self.wave_index < 0
            or self.wave_cell_ids != tuple(sorted(set(self.wave_cell_ids)))
            or self.materialized_cell_id not in self.wave_cell_ids
            or self.provider_reserved_gpu_count not in {1, 2}
            or type(self.allowed_attempts) is not int
            or self.allowed_attempts < 1
        ):
            raise ValueError("formal preflight cap physical wave differs")
        for label, value in (
            ("process timeout", self.process_hard_timeout_ns),
            ("provider timeout", self.provider_wave_hard_timeout_ns),
            ("compute cap", self.maximum_compute_gpu_ns_per_attempt),
            ("provider cap", self.maximum_provider_reserved_gpu_ns_per_attempt),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"formal preflight cap {label} is invalid")
        if (
            self.process_hard_timeout_ns > self.provider_wave_hard_timeout_ns
            or self.maximum_compute_gpu_ns_per_attempt
            != self.process_hard_timeout_ns * len(self.gpu_uuids)
            or self.maximum_provider_reserved_gpu_ns_per_attempt
            != self.provider_wave_hard_timeout_ns * self.provider_reserved_gpu_count
        ):
            raise ValueError("formal preflight cap arithmetic differs")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
            "localhost_ports": list(self.localhost_ports),
            "wave_cell_ids": list(self.wave_cell_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal preflight launch cell cap",
            value,
            set(cls.__dataclass_fields__),
        )
        for name in ("gpu_uuids", "rank_groups", "localhost_ports", "wave_cell_ids"):
            if type(row[name]) is not list:
                raise TypeError(f"formal preflight cap {name} is not an array")
        row["gpu_uuids"] = tuple(row["gpu_uuids"])
        row["rank_groups"] = tuple(tuple(group) for group in row["rank_groups"])
        row["localhost_ports"] = tuple(row["localhost_ports"])
        row["wave_cell_ids"] = tuple(row["wave_cell_ids"])
        return cls(**row)


@dataclass(frozen=True)
class FormalPreflightLaunchCapSchedule:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_preflight_launch_cap_schedule"]
    protocol_sha256: str
    dispatch_receipt: CanonicalJsonProofBinding
    dispatch_receipt_sha256: str
    protocol_lock_sha256: str
    registry_sha256: str
    inventory_sha256: str
    materialization_receipt_sha256: str
    budget_plan_sha256: str
    capacity_schedule_sha256: str
    capacity_gate_sha256: str
    dispatch_reservation_sha256: str
    ledger_root: str
    cell_caps: tuple[FormalPreflightLaunchCellCap, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_preflight_launch_cap_schedule"
            or self.protocol_sha256 != FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256
        ):
            raise ValueError("formal preflight launch cap schedule schema differs")
        if type(self.dispatch_receipt) is not CanonicalJsonProofBinding:
            raise TypeError("formal preflight cap dispatch receipt is not path-bound")
        for label, value in (
            ("dispatch receipt", self.dispatch_receipt_sha256),
            ("protocol lock", self.protocol_lock_sha256),
            ("registry", self.registry_sha256),
            ("inventory", self.inventory_sha256),
            ("materialization", self.materialization_receipt_sha256),
            ("BudgetPlan", self.budget_plan_sha256),
            ("capacity schedule", self.capacity_schedule_sha256),
            ("capacity gate", self.capacity_gate_sha256),
            ("dispatch reservation", self.dispatch_reservation_sha256),
        ):
            _sha256(f"formal preflight cap {label}", value)
        ledger = Path(self.ledger_root)
        if not ledger.is_absolute() or ledger != ledger.resolve(strict=False):
            raise ValueError("formal preflight cap ledger root is not canonical")
        ids = tuple(row.materialized_cell_id for row in self.cell_caps)
        if len(self.cell_caps) != 10 or ids != tuple(sorted(set(ids))):
            raise ValueError("formal preflight cap coverage is not exact ten")
        by_wave: dict[int, list[FormalPreflightLaunchCellCap]] = {}
        for row in self.cell_caps:
            by_wave.setdefault(row.wave_index, []).append(row)
        if tuple(sorted(by_wave)) != tuple(range(len(by_wave))):
            raise ValueError("formal preflight cap wave indexes are not contiguous")
        for rows in by_wave.values():
            ordered = tuple(sorted(rows, key=lambda row: row.materialized_cell_id))
            wave_ids = tuple(row.materialized_cell_id for row in ordered)
            if (
                len(ordered) not in {1, 2}
                or any(row.wave_cell_ids != wave_ids for row in ordered)
                or len({row.provider_wave_hard_timeout_ns for row in ordered}) != 1
                or sum(row.provider_reserved_gpu_count for row in ordered) != 2
            ):
                raise ValueError("formal preflight cap provider wave differs")
            if len(ordered) == 2 and (
                any(row.topology_mode != "tp1_dp1" for row in ordered)
                or any(
                    row.runner_kind != "first_party_interference"
                    or row.interference_mode != "concurrent"
                    for row in ordered
                )
                or len({row.interference_repetition for row in ordered}) != 1
                or {row.interference_slot for row in ordered} != {0, 1}
                or len({row.process_hard_timeout_ns for row in ordered}) != 1
                or len({uuid for row in ordered for uuid in row.gpu_uuids}) != 2
                or len({port for row in ordered for port in row.localhost_ports})
                != sum(len(row.localhost_ports) for row in ordered)
            ):
                raise ValueError("formal preflight paired wave is not atomic TP1")
            if len(ordered) == 1 and ordered[0].interference_mode == "concurrent":
                raise ValueError("formal preflight concurrent row lost its peer")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "dispatch_receipt": self.dispatch_receipt.to_dict(),
            "cell_caps": [row.to_dict() for row in self.cell_caps],
            "schedule_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal preflight launch cap schedule",
            value,
            {*cls.__dataclass_fields__, "schedule_sha256"},
        )
        declared = _sha256("formal preflight cap schedule", row.pop("schedule_sha256"))
        raw_caps = row.pop("cell_caps")
        if type(raw_caps) is not list:
            raise TypeError("formal preflight cap rows are not an array")
        row["dispatch_receipt"] = CanonicalJsonProofBinding.from_dict(
            row["dispatch_receipt"]
        )
        schedule = cls(
            **row,
            cell_caps=tuple(
                FormalPreflightLaunchCellCap.from_dict(item) for item in raw_caps
            ),
        )
        if schedule.sha256 != declared:
            raise ValueError("formal preflight cap schedule digest differs")
        return schedule

    def cap_for_registry_cell(self, cell_id: str) -> FormalPreflightLaunchCellCap:
        rows = tuple(row for row in self.cell_caps if row.registry_cell_id == cell_id)
        if len(rows) != 1:
            raise ValueError("formal preflight cap lacks exact registry cell")
        return rows[0]


def _build_schedule(
    *,
    receipt_binding: CanonicalJsonProofBinding,
    receipt: FormalPreflightDispatchReceipt,
    ledger_root: Path,
    current_ns: int,
) -> FormalPreflightLaunchCapSchedule:
    token = receipt.revalidate(current_ns=current_ns)
    budgets = {row.cell_id: row for row in receipt.budget_plan.require_ready()}
    bindings = {row.registry_cell_id: row for row in token.subject.execution_bindings}
    materialized = {
        row.registry_cell_id: row.materialized_cell_id for row in bindings.values()
    }
    rows: list[FormalPreflightLaunchCellCap] = []

    def interference_identity(
        binding: object,
    ) -> tuple[Literal["isolated", "concurrent"] | None, int | None, int | None]:
        cell = binding.cell  # type: ignore[attr-defined]
        if cell.identity.task != "simultaneous_single_gpu_interference":
            return None, None, None
        variant = str(cell.identity.variant)
        for mode in ("isolated", "concurrent"):
            for slot in range(2):
                if variant == f"{mode}_slot_{slot}":
                    repetition = int(cell.identity.block)
                    if repetition in {0, 1}:
                        return mode, repetition, slot  # type: ignore[return-value]
        raise ValueError("formal preflight interference cell identity differs")

    for wave in token.dispatch_plan.waves:
        registry_ids = tuple(
            sorted(assignment.work_item.item_id for assignment in wave.assignments)
        )
        if len(registry_ids) not in {1, 2}:
            raise ValueError("formal preflight cap supports only one/two-row waves")
        wave_bindings = tuple(bindings[cell_id] for cell_id in registry_ids)
        wave_ids = tuple(sorted(materialized[cell_id] for cell_id in registry_ids))
        process_caps = tuple(
            budgets[cell_id].wall_time.quota_envelope * _NS_PER_MS
            for cell_id in registry_ids
        )
        if len(registry_ids) == 2 and (
            len(set(process_caps)) != 1
            or any(
                binding.cell.identity.topology != "tp1_dp1" for binding in wave_bindings
            )
        ):
            raise ValueError("formal preflight shared wave has unequal/non-TP1 caps")
        wave_interference = tuple(interference_identity(row) for row in wave_bindings)
        if len(registry_ids) == 2 and (
            any(mode != "concurrent" for mode, _repetition, _slot in wave_interference)
            or len({repetition for _mode, repetition, _slot in wave_interference}) != 1
            or {slot for _mode, _repetition, slot in wave_interference} != {0, 1}
        ):
            raise ValueError(
                "formal preflight shared wave is not one concurrent interference pair"
            )
        if len(registry_ids) == 1 and wave_interference[0][0] == "concurrent":
            raise ValueError("formal preflight concurrent row is not paired")
        provider_timeout = max(process_caps)
        provider_count = 1 if len(registry_ids) == 2 else 2
        for binding, process_timeout in zip(wave_bindings, process_caps, strict=True):
            mode, repetition, slot = interference_identity(binding)
            budget = budgets[binding.registry_cell_id]
            if (
                budget.sha256 != binding.experiment_budget_sha256
                or budget.gpu_count != len(binding.gpu_uuids)
                or budget.retry_allowance != 1
                or budget.fixed_instance_billed_gpu_ms.quota_envelope
                < provider_timeout * 2 // _NS_PER_MS
            ):
                raise ValueError("formal preflight cap BudgetPlan row differs")
            rows.append(
                FormalPreflightLaunchCellCap(
                    materialized_cell_id=binding.materialized_cell_id,
                    registry_cell_id=binding.registry_cell_id,
                    runner_kind=binding.runner_kind,
                    experiment_budget_sha256=budget.sha256,
                    physical_assignment_sha256=binding.assignment_sha256,
                    topology_mode=binding.cell.identity.topology,
                    interference_mode=mode,
                    interference_repetition=repetition,
                    interference_slot=slot,
                    gpu_uuids=binding.gpu_uuids,
                    rank_groups=binding.rank_groups,
                    localhost_ports=binding.assignment.ports,
                    wave_index=wave.wave_index,
                    wave_cell_ids=wave_ids,
                    process_hard_timeout_ns=process_timeout,
                    provider_wave_hard_timeout_ns=provider_timeout,
                    provider_reserved_gpu_count=provider_count,
                    maximum_compute_gpu_ns_per_attempt=(
                        process_timeout * len(binding.gpu_uuids)
                    ),
                    maximum_provider_reserved_gpu_ns_per_attempt=(
                        provider_timeout * provider_count
                    ),
                    allowed_attempts=budget.retry_allowance + 1,
                )
            )
    lock = receipt.registry_verification_receipt.signed_protocol_lock.payload
    return FormalPreflightLaunchCapSchedule(
        schema_version=1,
        kind="lightcone_formal_preflight_launch_cap_schedule",
        protocol_sha256=FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256,
        dispatch_receipt=receipt_binding,
        dispatch_receipt_sha256=receipt.sha256,
        protocol_lock_sha256=lock.sha256,
        registry_sha256=lock.registry_sha256,
        inventory_sha256=receipt.inventory.sha256,
        materialization_receipt_sha256=receipt.signed_materialization.payload.sha256,
        budget_plan_sha256=receipt.budget_plan.sha256,
        capacity_schedule_sha256=receipt.capacity_schedule.sha256,
        capacity_gate_sha256=receipt.capacity_gate.sha256,
        dispatch_reservation_sha256=receipt.reservation.reservation_sha256,
        ledger_root=str(ledger_root),
        cell_caps=tuple(sorted(rows, key=lambda row: row.materialized_cell_id)),
    )


def materialize_formal_preflight_launch_cap_schedule(
    *,
    dispatch_receipt_path: str | Path,
    output_path: str | Path,
    current_ns: int,
) -> FormalPreflightLaunchCapSchedule:
    output = Path(output_path)
    if not output.is_absolute() or output != output.resolve(strict=False):
        raise ValueError("formal preflight cap output path is not canonical")
    receipt_binding = CanonicalJsonProofBinding.bind(dispatch_receipt_path)
    receipt = FormalPreflightDispatchReceipt.from_dict(receipt_binding.reopen())
    if receipt.sha256 != receipt_binding.semantic_sha256:
        raise ValueError("formal preflight dispatch receipt identity differs")
    ledger_root = output.parent / "formal-preflight-launch-ledgers"
    schedule = _build_schedule(
        receipt_binding=receipt_binding,
        receipt=receipt,
        ledger_root=ledger_root,
        current_ns=current_ns,
    )
    publish_canonical_json_no_replace(output, schedule.to_dict())
    rebound = FormalPreflightLaunchCapSchedule.from_dict(
        CanonicalJsonProofBinding.bind(output).reopen()
    )
    if rebound != schedule:
        raise RuntimeError("formal preflight cap schedule changed during publication")
    return schedule


def revalidate_formal_preflight_launch_cap_schedule(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalPreflightLaunchCapSchedule:
    binding = CanonicalJsonProofBinding.bind(path)
    schedule = FormalPreflightLaunchCapSchedule.from_dict(binding.reopen())
    receipt_binding = CanonicalJsonProofBinding.bind(
        schedule.dispatch_receipt.absolute_path
    )
    receipt = FormalPreflightDispatchReceipt.from_dict(receipt_binding.reopen())
    expected = _build_schedule(
        receipt_binding=receipt_binding,
        receipt=receipt,
        ledger_root=Path(schedule.ledger_root),
        current_ns=current_ns,
    )
    if (
        schedule.dispatch_receipt != receipt_binding
        or schedule.dispatch_receipt_sha256 != receipt.sha256
        or schedule != expected
        or binding.semantic_sha256 != schedule.sha256
    ):
        raise ValueError("formal preflight cap schedule differs from rebuild")
    return schedule


@dataclass(frozen=True)
class FormalPreflightLaunchWaveConsumption:
    schema_version: Literal[1]
    kind: Literal["lightcone_formal_preflight_launch_wave_consumption"]
    protocol_sha256: str
    schedule: CanonicalJsonProofBinding
    schedule_sha256: str
    dispatch_receipt_sha256: str
    budget_plan_sha256: str
    capacity_gate_sha256: str
    dispatch_reservation_sha256: str
    wave_index: int
    materialized_cell_ids: tuple[str, ...]
    registry_cell_ids: tuple[str, ...]
    attempt_index: int
    process_hard_timeout_ns: int
    provider_wave_hard_timeout_ns: int
    compute_charge_gpu_ns: int
    provider_reserved_charge_gpu_ns: int
    consumed_ns: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.kind != "lightcone_formal_preflight_launch_wave_consumption"
            or self.protocol_sha256 != FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256
        ):
            raise ValueError("formal preflight wave consumption schema differs")
        if type(self.schedule) is not CanonicalJsonProofBinding:
            raise TypeError("formal preflight wave consumption lacks schedule binding")
        for label, value in (
            ("schedule", self.schedule_sha256),
            ("dispatch receipt", self.dispatch_receipt_sha256),
            ("BudgetPlan", self.budget_plan_sha256),
            ("capacity gate", self.capacity_gate_sha256),
            ("dispatch reservation", self.dispatch_reservation_sha256),
        ):
            _sha256(f"formal preflight wave consumption {label}", value)
        if (
            type(self.wave_index) is not int
            or self.wave_index < 0
            or self.materialized_cell_ids
            != tuple(sorted(set(self.materialized_cell_ids)))
            or len(self.materialized_cell_ids) not in {1, 2}
            or self.registry_cell_ids != tuple(sorted(set(self.registry_cell_ids)))
            or len(self.registry_cell_ids) != len(self.materialized_cell_ids)
            or type(self.attempt_index) is not int
            or self.attempt_index not in {0, 1}
        ):
            raise ValueError("formal preflight wave consumption coverage differs")
        for label, value in (
            ("process timeout", self.process_hard_timeout_ns),
            ("provider timeout", self.provider_wave_hard_timeout_ns),
            ("compute charge", self.compute_charge_gpu_ns),
            ("provider charge", self.provider_reserved_charge_gpu_ns),
            ("consumed time", self.consumed_ns),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"formal preflight wave consumption {label} is invalid"
                )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "schedule": self.schedule.to_dict(),
            "materialized_cell_ids": list(self.materialized_cell_ids),
            "registry_cell_ids": list(self.registry_cell_ids),
            "consumption_sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        row = _strict(
            "formal preflight wave consumption",
            value,
            {*cls.__dataclass_fields__, "consumption_sha256"},
        )
        declared = _sha256(
            "formal preflight wave consumption", row.pop("consumption_sha256")
        )
        for name in ("materialized_cell_ids", "registry_cell_ids"):
            if type(row[name]) is not list:
                raise TypeError(
                    f"formal preflight wave consumption {name} is not an array"
                )
            row[name] = tuple(row[name])
        row["schedule"] = CanonicalJsonProofBinding.from_dict(row["schedule"])
        result = cls(**row)
        if result.sha256 != declared:
            raise ValueError("formal preflight wave consumption digest differs")
        return result


def _wave_caps(
    schedule: FormalPreflightLaunchCapSchedule,
    *,
    registry_cell_id: str,
) -> tuple[FormalPreflightLaunchCellCap, ...]:
    cap = schedule.cap_for_registry_cell(registry_cell_id)
    rows = tuple(
        sorted(
            (row for row in schedule.cell_caps if row.wave_index == cap.wave_index),
            key=lambda row: row.materialized_cell_id,
        )
    )
    if tuple(row.materialized_cell_id for row in rows) != cap.wave_cell_ids:
        raise ValueError("formal preflight cap wave membership changed")
    return rows


def _consumption_path(
    schedule: FormalPreflightLaunchCapSchedule,
    *,
    wave_index: int,
    attempt_index: int,
) -> Path:
    return (
        Path(schedule.ledger_root)
        / schedule.sha256
        / f"wave-{wave_index:03d}"
        / f"attempt-{attempt_index}.json"
    )


def consume_formal_preflight_launch_wave(
    schedule_path: str | Path,
    *,
    registry_cell_id: str,
    consumed_ns: int,
    current_ns: int,
) -> CanonicalJsonProofBinding:
    """Atomically reserve an entire source-owned wave before ``Popen``."""

    schedule_binding = CanonicalJsonProofBinding.bind(schedule_path)
    schedule = revalidate_formal_preflight_launch_cap_schedule(
        schedule_path, current_ns=current_ns
    )
    rows = _wave_caps(schedule, registry_cell_id=registry_cell_id)
    if type(consumed_ns) is not int or consumed_ns < 1 or consumed_ns > current_ns:
        raise ValueError("formal preflight wave consumption time is invalid")
    wave_root = _private_directory(
        Path(schedule.ledger_root) / schedule.sha256 / f"wave-{rows[0].wave_index:03d}",
        create=True,
    )
    lock_path = wave_root / "ledger.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        existing = tuple(sorted(wave_root.glob("attempt-*.json")))
        if any(path.is_symlink() for path in existing):
            raise ValueError("formal preflight wave ledger contains a symlink")
        attempt = len(existing)
        allowed = min(row.allowed_attempts for row in rows)
        if attempt >= allowed:
            raise ValueError("formal preflight wave exceeds registered attempts")
        expected_existing = tuple(
            _consumption_path(
                schedule, wave_index=rows[0].wave_index, attempt_index=index
            )
            for index in range(attempt)
        )
        if existing != expected_existing:
            raise ValueError("formal preflight wave ledger is not contiguous")
        process_timeout = max(row.process_hard_timeout_ns for row in rows)
        provider_timeout = rows[0].provider_wave_hard_timeout_ns
        consumption = FormalPreflightLaunchWaveConsumption(
            schema_version=1,
            kind="lightcone_formal_preflight_launch_wave_consumption",
            protocol_sha256=FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256,
            schedule=schedule_binding,
            schedule_sha256=schedule.sha256,
            dispatch_receipt_sha256=schedule.dispatch_receipt_sha256,
            budget_plan_sha256=schedule.budget_plan_sha256,
            capacity_gate_sha256=schedule.capacity_gate_sha256,
            dispatch_reservation_sha256=schedule.dispatch_reservation_sha256,
            wave_index=rows[0].wave_index,
            materialized_cell_ids=tuple(row.materialized_cell_id for row in rows),
            registry_cell_ids=tuple(sorted(row.registry_cell_id for row in rows)),
            attempt_index=attempt,
            process_hard_timeout_ns=process_timeout,
            provider_wave_hard_timeout_ns=provider_timeout,
            compute_charge_gpu_ns=sum(
                row.maximum_compute_gpu_ns_per_attempt for row in rows
            ),
            provider_reserved_charge_gpu_ns=sum(
                row.maximum_provider_reserved_gpu_ns_per_attempt for row in rows
            ),
            consumed_ns=consumed_ns,
        )
        destination = _consumption_path(
            schedule, wave_index=rows[0].wave_index, attempt_index=attempt
        )
        publish_canonical_json_no_replace(destination, consumption.to_dict())
        binding = CanonicalJsonProofBinding.bind(destination)
        if (
            FormalPreflightLaunchWaveConsumption.from_dict(binding.reopen())
            != consumption
        ):
            raise RuntimeError("formal preflight wave consumption changed")
        return binding
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def validate_formal_preflight_launch_wave_consumption(
    path: str | Path,
    *,
    current_ns: int,
) -> FormalPreflightLaunchWaveConsumption:
    binding = CanonicalJsonProofBinding.bind(path)
    consumption = FormalPreflightLaunchWaveConsumption.from_dict(binding.reopen())
    schedule = revalidate_formal_preflight_launch_cap_schedule(
        consumption.schedule.absolute_path,
        current_ns=current_ns,
    )
    rows = _wave_caps(schedule, registry_cell_id=consumption.registry_cell_ids[0])
    expected_path = _consumption_path(
        schedule,
        wave_index=rows[0].wave_index,
        attempt_index=consumption.attempt_index,
    )
    expected = FormalPreflightLaunchWaveConsumption(
        schema_version=1,
        kind="lightcone_formal_preflight_launch_wave_consumption",
        protocol_sha256=FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256,
        schedule=CanonicalJsonProofBinding.bind(consumption.schedule.absolute_path),
        schedule_sha256=schedule.sha256,
        dispatch_receipt_sha256=schedule.dispatch_receipt_sha256,
        budget_plan_sha256=schedule.budget_plan_sha256,
        capacity_gate_sha256=schedule.capacity_gate_sha256,
        dispatch_reservation_sha256=schedule.dispatch_reservation_sha256,
        wave_index=rows[0].wave_index,
        materialized_cell_ids=tuple(row.materialized_cell_id for row in rows),
        registry_cell_ids=tuple(sorted(row.registry_cell_id for row in rows)),
        attempt_index=consumption.attempt_index,
        process_hard_timeout_ns=max(row.process_hard_timeout_ns for row in rows),
        provider_wave_hard_timeout_ns=rows[0].provider_wave_hard_timeout_ns,
        compute_charge_gpu_ns=sum(
            row.maximum_compute_gpu_ns_per_attempt for row in rows
        ),
        provider_reserved_charge_gpu_ns=sum(
            row.maximum_provider_reserved_gpu_ns_per_attempt for row in rows
        ),
        consumed_ns=consumption.consumed_ns,
    )
    if (
        Path(path) != expected_path
        or consumption != expected
        or binding.semantic_sha256 != consumption.sha256
        or consumption.consumed_ns > current_ns
    ):
        raise ValueError("formal preflight wave consumption differs from authority")
    return consumption


__all__ = [
    "FORMAL_PREFLIGHT_LAUNCH_CAP_PROTOCOL_SHA256",
    "FormalPreflightLaunchCapSchedule",
    "FormalPreflightLaunchCellCap",
    "FormalPreflightLaunchWaveConsumption",
    "consume_formal_preflight_launch_wave",
    "materialize_formal_preflight_launch_cap_schedule",
    "revalidate_formal_preflight_launch_cap_schedule",
    "validate_formal_preflight_launch_wave_consumption",
]
