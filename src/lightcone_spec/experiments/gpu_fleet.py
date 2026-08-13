"""Deterministic multi-host composition for same-host GPU dispatch plans.

The fleet layer distributes independent scientific work across hosts.  It does
not introduce a second GPU placement implementation: every host-local plan is
issued by :class:`~lightcone_spec.experiments.gpu_pool.GpuPoolScheduler` and
therefore retains the existing topology, interference, budget, and affinity
invariants.  A gang is always placed wholly on one host.  This release fails
closed when satisfying a gang would require cross-host collectives.

This module is pure.  It does not open SSH connections, inspect devices, or
write evidence.  Remote transports consume the content-bound host plans and
return :class:`FleetWaveReceipt` rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import PurePosixPath
from typing import Any

from lightcone_spec.experiments.gpu_pool import (
    ArtifactSidecar,
    CapabilityRejectionError,
    DispatchWaveExecutionReceipt,
    GpuAssignment,
    GpuDispatchPlan,
    GpuDispatchWave,
    GpuInventory,
    GpuPoolScheduler,
    InterferenceEnvelope,
    PoolWorkItem,
)
from lightcone_spec.experiments.registry import (
    ExperimentRegistry,
    WorkloadClass,
    content_sha256,
)

CROSS_HOST_COLLECTIVES_UNVALIDATED = "cross_host_collectives_unvalidated"
_HOST_EXCLUSIVE_IO_CLASSES = frozenset(
    {WorkloadClass.PROFILE, WorkloadClass.DOWNLOAD, WorkloadClass.COMPILE}
)
_SHA256_LENGTH = 64


def _strict_object(
    name: str,
    value: object,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    actual = set(value)
    missing = fields - actual
    unknown = actual - fields
    if missing or unknown:
        raise ValueError(
            f"{name} fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def _strict_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _strict_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    _require_text(name, value)
    return value


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _strict_text(name, value)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lower-case SHA-256")


def _require_namespace(name: str, value: str) -> None:
    _require_text(name, value)
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{name} must use canonical POSIX path syntax")
    path = PurePosixPath(value)
    if value in {".", "/"} or ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be a canonical non-root namespace")


def _qualify(namespace: str, logical_root: str) -> str:
    _require_namespace("host namespace", namespace)
    _require_namespace("logical resource root", logical_root)
    if logical_root.startswith("/"):
        raise ValueError("logical resource roots must be relative")
    qualified = f"{namespace}/{logical_root}"
    _require_namespace("qualified resource root", qualified)
    return qualified


def _require_exact_json(actual: object, expected: object, *, path: str) -> None:
    """Compare canonical JSON values without Python's bool/int equivalence."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} differs from the issued canonical object")
        for key in expected:
            _require_exact_json(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} differs from the issued canonical array")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_json(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} differs from the issued canonical value")


def _validate_sidecar(
    *,
    artifact_kind: str,
    artifact_sha256: str,
    sidecar: ArtifactSidecar | Mapping[str, Any] | None,
) -> None:
    if sidecar is None:
        return
    parsed = (
        sidecar
        if isinstance(sidecar, ArtifactSidecar)
        else ArtifactSidecar.from_dict(sidecar)
    )
    if parsed.artifact_kind != artifact_kind:
        raise ValueError("artifact sidecar kind mismatch")
    if parsed.artifact_sha256 != artifact_sha256:
        raise ValueError("artifact sidecar SHA-256 mismatch")


@dataclass(frozen=True)
class HostInventoryBinding:
    """One verified single-host inventory and its host-local calibration."""

    schema_version: int
    host_id: str
    inventory: GpuInventory
    interference_envelope: InterferenceEnvelope

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "only host-inventory binding schema version 1 is supported"
            )
        _require_text("host_id", self.host_id)
        if type(self.inventory) is not GpuInventory:
            raise TypeError("host inventory must be an exact GpuInventory")
        if type(self.interference_envelope) is not InterferenceEnvelope:
            raise TypeError(
                "host interference binding must be an exact InterferenceEnvelope"
            )
        if self.inventory.host_ids != (self.host_id,):
            raise ValueError("host inventory binding must contain exactly its host")
        hardware_envelopes = {
            device.hardware_envelope_sha256 for device in self.inventory.devices
        }
        if any(
            rule.hardware_envelope_sha256 not in hardware_envelopes
            for rule in self.interference_envelope.rules
        ):
            raise ValueError(
                "host interference envelope references another hardware envelope"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "host_id": self.host_id,
            "inventory": self.inventory.to_dict(),
            "inventory_sha256": self.inventory.sha256,
            "interference_envelope": self.interference_envelope.to_dict(),
            "interference_envelope_sha256": self.interference_envelope.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> HostInventoryBinding:
        row = _strict_object(
            "host inventory binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "host_id",
                    "inventory",
                    "inventory_sha256",
                    "interference_envelope",
                    "interference_envelope_sha256",
                }
            ),
        )
        inventory = GpuInventory.from_dict(row["inventory"])
        interference = InterferenceEnvelope.from_dict(row["interference_envelope"])
        inventory_sha256 = _strict_text(
            "host inventory SHA-256", row["inventory_sha256"]
        )
        interference_sha256 = _strict_text(
            "host interference SHA-256",
            row["interference_envelope_sha256"],
        )
        if inventory_sha256 != inventory.sha256:
            raise ValueError("host inventory digest mismatch")
        if interference_sha256 != interference.sha256:
            raise ValueError("host interference-envelope digest mismatch")
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            host_id=_strict_text("host_id", row["host_id"]),
            inventory=inventory,
            interference_envelope=interference,
        )


@dataclass(frozen=True)
class GpuFleetInventory:
    """Canonical aggregation of independently verified host inventories."""

    schema_version: int
    hosts: tuple[HostInventoryBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only GPU fleet inventory schema version 1 is supported")
        if not self.hosts:
            raise ValueError("GPU fleet inventory must contain at least one host")
        if any(type(binding) is not HostInventoryBinding for binding in self.hosts):
            raise TypeError("fleet hosts must be exact HostInventoryBinding rows")
        host_ids = tuple(binding.host_id for binding in self.hosts)
        if host_ids != tuple(sorted(host_ids)) or len(host_ids) != len(set(host_ids)):
            raise ValueError("fleet host bindings must be sorted and unique")
        gpu_uuids = tuple(
            device.uuid
            for binding in self.hosts
            for device in binding.inventory.devices
        )
        if len(gpu_uuids) != len(set(gpu_uuids)):
            raise ValueError("fleet inventories contain a duplicate physical GPU UUID")

    @property
    def host_ids(self) -> tuple[str, ...]:
        return tuple(binding.host_id for binding in self.hosts)

    @property
    def gpu_count(self) -> int:
        return sum(len(binding.inventory.devices) for binding in self.hosts)

    def host(self, host_id: str) -> HostInventoryBinding:
        matches = tuple(binding for binding in self.hosts if binding.host_id == host_id)
        if len(matches) != 1:
            raise ValueError(f"unknown fleet host {host_id!r}")
        return matches[0]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hosts": [binding.to_dict() for binding in self.hosts],
            "host_binding_sha256": [binding.sha256 for binding in self.hosts],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> GpuFleetInventory:
        row = _strict_object(
            "GPU fleet inventory",
            value,
            frozenset({"schema_version", "hosts", "host_binding_sha256"}),
        )
        hosts = tuple(
            HostInventoryBinding.from_dict(item)
            for item in _strict_list("fleet hosts", row["hosts"])
        )
        declared = tuple(
            _strict_text("host binding SHA-256", item)
            for item in _strict_list("host_binding_sha256", row["host_binding_sha256"])
        )
        if declared != tuple(binding.sha256 for binding in hosts):
            raise ValueError("fleet host-binding digest list mismatch")
        fleet = cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            hosts=hosts,
        )
        _validate_sidecar(
            artifact_kind="gpu_fleet_inventory.v1",
            artifact_sha256=fleet.sha256,
            sidecar=sidecar,
        )
        return fleet

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "gpu_fleet_inventory.v1", self.sha256)


def assemble_gpu_fleet_inventory(
    hosts: Sequence[HostInventoryBinding],
) -> GpuFleetInventory:
    """Assemble a canonical fleet without weakening per-host verification."""

    rows = tuple(hosts)
    if any(type(binding) is not HostInventoryBinding for binding in rows):
        raise TypeError("fleet assembly requires HostInventoryBinding rows")
    return GpuFleetInventory(1, tuple(sorted(rows, key=lambda row: row.host_id)))


@dataclass(frozen=True)
class HostExecutionBinding:
    """Host-local resource namespaces and dispatch-port authority."""

    schema_version: int
    host_id: str
    inventory_sha256: str
    interference_envelope_sha256: str
    port_start: int
    port_end: int
    cache_namespace: str
    evidence_namespace: str
    contention_domain: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "only host-execution binding schema version 1 is supported"
            )
        _require_text("host_id", self.host_id)
        _require_sha256("inventory_sha256", self.inventory_sha256)
        _require_sha256(
            "interference_envelope_sha256",
            self.interference_envelope_sha256,
        )
        if (
            isinstance(self.port_start, bool)
            or isinstance(self.port_end, bool)
            or not isinstance(self.port_start, int)
            or not isinstance(self.port_end, int)
            or not 1024 <= self.port_start <= self.port_end <= 65_535
        ):
            raise ValueError("host execution port range is invalid")
        _require_namespace("cache_namespace", self.cache_namespace)
        _require_namespace("evidence_namespace", self.evidence_namespace)
        cache_parts = PurePosixPath(self.cache_namespace).parts
        evidence_parts = PurePosixPath(self.evidence_namespace).parts
        if (
            cache_parts[: len(evidence_parts)] == evidence_parts
            or evidence_parts[: len(cache_parts)] == cache_parts
        ):
            raise ValueError("cache and evidence namespaces must not overlap")
        _require_text("contention_domain", self.contention_domain)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def cache_root_for(self, assignment: GpuAssignment) -> str:
        return _qualify(self.cache_namespace, assignment.work_item.claim.cache_root)

    def evidence_root_for(self, assignment: GpuAssignment) -> str:
        return _qualify(
            self.evidence_namespace,
            assignment.work_item.claim.evidence_root,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "host_id": self.host_id,
            "inventory_sha256": self.inventory_sha256,
            "interference_envelope_sha256": self.interference_envelope_sha256,
            "port_start": self.port_start,
            "port_end": self.port_end,
            "cache_namespace": self.cache_namespace,
            "evidence_namespace": self.evidence_namespace,
            "contention_domain": self.contention_domain,
        }

    @classmethod
    def from_dict(cls, value: object) -> HostExecutionBinding:
        row = _strict_object(
            "host execution binding",
            value,
            frozenset(
                {
                    "schema_version",
                    "host_id",
                    "inventory_sha256",
                    "interference_envelope_sha256",
                    "port_start",
                    "port_end",
                    "cache_namespace",
                    "evidence_namespace",
                    "contention_domain",
                }
            ),
        )
        return cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            host_id=_strict_text("host_id", row["host_id"]),
            inventory_sha256=_strict_text("inventory_sha256", row["inventory_sha256"]),
            interference_envelope_sha256=_strict_text(
                "interference_envelope_sha256",
                row["interference_envelope_sha256"],
            ),
            port_start=_strict_int("port_start", row["port_start"]),
            port_end=_strict_int("port_end", row["port_end"]),
            cache_namespace=_strict_text("cache_namespace", row["cache_namespace"]),
            evidence_namespace=_strict_text(
                "evidence_namespace", row["evidence_namespace"]
            ),
            contention_domain=_strict_text(
                "contention_domain", row["contention_domain"]
            ),
        )


@dataclass(frozen=True)
class HostDispatchPlan:
    """One unmodified single-host plan plus its execution binding."""

    host_id: str
    host_inventory_sha256: str
    execution_binding: HostExecutionBinding
    dispatch_plan: GpuDispatchPlan

    def __post_init__(self) -> None:
        _require_text("host_id", self.host_id)
        _require_sha256("host_inventory_sha256", self.host_inventory_sha256)
        if type(self.execution_binding) is not HostExecutionBinding:
            raise TypeError("host dispatch plan needs an exact execution binding")
        if type(self.dispatch_plan) is not GpuDispatchPlan:
            raise TypeError("host dispatch plan needs an exact GpuDispatchPlan")
        if self.execution_binding.host_id != self.host_id:
            raise ValueError("host execution binding identity mismatch")
        if (
            self.execution_binding.inventory_sha256 != self.host_inventory_sha256
            or self.dispatch_plan.inventory_sha256 != self.host_inventory_sha256
        ):
            raise ValueError("host dispatch inventory identity mismatch")
        if (
            self.dispatch_plan.interference_envelope_sha256
            != self.execution_binding.interference_envelope_sha256
        ):
            raise ValueError("host dispatch interference identity mismatch")
        if not self.dispatch_plan.waves:
            raise ValueError("host dispatch wrapper cannot contain an empty plan")
        for wave in self.dispatch_plan.waves:
            if any(
                port < self.execution_binding.port_start
                or port > self.execution_binding.port_end
                for assignment in wave.assignments
                for port in assignment.ports
            ):
                raise ValueError("host assignment uses a port outside its binding")
            cache_roots = tuple(
                self.execution_binding.cache_root_for(assignment)
                for assignment in wave.assignments
            )
            evidence_roots = tuple(
                self.execution_binding.evidence_root_for(assignment)
                for assignment in wave.assignments
            )
            if len(cache_roots) != len(set(cache_roots)):
                raise ValueError("host wave overlaps a qualified cache root")
            if len(evidence_roots) != len(set(evidence_roots)):
                raise ValueError("host wave overlaps a qualified evidence root")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "host_inventory_sha256": self.host_inventory_sha256,
            "execution_binding": self.execution_binding.to_dict(),
            "execution_binding_sha256": self.execution_binding.sha256,
            "dispatch_plan": self.dispatch_plan.to_dict(),
            "dispatch_plan_sha256": self.dispatch_plan.sha256,
        }


@dataclass(frozen=True)
class GpuFleetAssignment:
    """Host-bound projection of one existing GPU assignment."""

    host_id: str
    host_inventory_sha256: str
    host_dispatch_plan_sha256: str
    execution_binding_sha256: str
    local_wave_index: int
    local_wave_sha256: str
    assignment: GpuAssignment
    cache_root: str
    evidence_root: str
    contention_domain: str

    def __post_init__(self) -> None:
        _require_text("host_id", self.host_id)
        for name in (
            "host_inventory_sha256",
            "host_dispatch_plan_sha256",
            "execution_binding_sha256",
            "local_wave_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.local_wave_index, bool)
            or not isinstance(self.local_wave_index, int)
            or self.local_wave_index < 0
        ):
            raise ValueError("local_wave_index must be non-negative")
        if type(self.assignment) is not GpuAssignment:
            raise TypeError("fleet assignment must wrap an exact GpuAssignment")
        _require_namespace("fleet cache root", self.cache_root)
        _require_namespace("fleet evidence root", self.evidence_root)
        _require_text("contention_domain", self.contention_domain)

    @cached_property
    def assignment_id(self) -> str:
        return content_sha256(self.to_dict())

    @cached_property
    def sha256(self) -> str:
        return self.assignment_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "host_inventory_sha256": self.host_inventory_sha256,
            "host_dispatch_plan_sha256": self.host_dispatch_plan_sha256,
            "execution_binding_sha256": self.execution_binding_sha256,
            "local_wave_index": self.local_wave_index,
            "local_wave_sha256": self.local_wave_sha256,
            "assignment": self.assignment.to_dict(),
            "assignment_sha256": self.assignment.sha256,
            "cache_root": self.cache_root,
            "evidence_root": self.evidence_root,
            "contention_domain": self.contention_domain,
        }

    @classmethod
    def from_dict(cls, value: object) -> GpuFleetAssignment:
        row = _strict_object(
            "GPU fleet assignment",
            value,
            frozenset(
                {
                    "host_id",
                    "host_inventory_sha256",
                    "host_dispatch_plan_sha256",
                    "execution_binding_sha256",
                    "local_wave_index",
                    "local_wave_sha256",
                    "assignment",
                    "assignment_sha256",
                    "cache_root",
                    "evidence_root",
                    "contention_domain",
                }
            ),
        )
        assignment = GpuAssignment.from_dict(row["assignment"])
        if _strict_text("assignment_sha256", row["assignment_sha256"]) != (
            assignment.sha256
        ):
            raise ValueError("fleet assignment local-assignment digest mismatch")
        return cls(
            host_id=_strict_text("host_id", row["host_id"]),
            host_inventory_sha256=_strict_text(
                "host_inventory_sha256", row["host_inventory_sha256"]
            ),
            host_dispatch_plan_sha256=_strict_text(
                "host_dispatch_plan_sha256",
                row["host_dispatch_plan_sha256"],
            ),
            execution_binding_sha256=_strict_text(
                "execution_binding_sha256", row["execution_binding_sha256"]
            ),
            local_wave_index=_strict_int("local_wave_index", row["local_wave_index"]),
            local_wave_sha256=_strict_text(
                "local_wave_sha256", row["local_wave_sha256"]
            ),
            assignment=assignment,
            cache_root=_strict_text("cache_root", row["cache_root"]),
            evidence_root=_strict_text("evidence_root", row["evidence_root"]),
            contention_domain=_strict_text(
                "contention_domain", row["contention_domain"]
            ),
        )


@dataclass(frozen=True)
class GpuFleetDispatchWave:
    """Concurrent host-local waves with shared-I/O isolation."""

    wave_index: int
    assignments: tuple[GpuFleetAssignment, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.wave_index, bool)
            or not isinstance(self.wave_index, int)
            or self.wave_index < 0
        ):
            raise ValueError("fleet wave index must be non-negative")
        if not self.assignments:
            raise ValueError("fleet dispatch wave cannot be empty")
        if any(type(row) is not GpuFleetAssignment for row in self.assignments):
            raise TypeError("fleet wave assignments must be exact rows")
        order = tuple((row.host_id, row.assignment.sha256) for row in self.assignments)
        if order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise ValueError("fleet wave assignments must be canonical and unique")
        local_waves: dict[str, tuple[int, str]] = {}
        resources: dict[str, tuple[set[str], set[int], set[str], set[str]]] = {}
        domain_classes: dict[str, set[WorkloadClass]] = {}
        domain_hosts: dict[str, set[str]] = {}
        for row in self.assignments:
            identity = (row.local_wave_index, row.local_wave_sha256)
            if row.host_id in local_waves and local_waves[row.host_id] != identity:
                raise ValueError("fleet wave mixes local waves from one host")
            local_waves[row.host_id] = identity
            gpu, ports, caches, evidence = resources.setdefault(
                row.host_id,
                (set(), set(), set(), set()),
            )
            if gpu & set(row.assignment.gpu_uuids):
                raise ValueError("fleet wave overlaps a host-local GPU")
            if ports & set(row.assignment.ports):
                raise ValueError("fleet wave overlaps a host-local port")
            if row.cache_root in caches or row.evidence_root in evidence:
                raise ValueError("fleet wave overlaps a host-local resource root")
            gpu.update(row.assignment.gpu_uuids)
            ports.update(row.assignment.ports)
            caches.add(row.cache_root)
            evidence.add(row.evidence_root)
            domain_classes.setdefault(row.contention_domain, set()).add(
                row.assignment.work_item.claim.workload_class
            )
            domain_hosts.setdefault(row.contention_domain, set()).add(row.host_id)
        for domain, classes in domain_classes.items():
            if len(domain_hosts[domain]) > 1 and classes.intersection(
                _HOST_EXCLUSIVE_IO_CLASSES
            ):
                raise ValueError(
                    "shared contention domain overlaps download/compile/profile work"
                )

    @property
    def host_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.host_id for row in self.assignments}))

    @property
    def estimated_wall_seconds(self) -> float:
        return max(
            row.assignment.work_item.claim.estimated_duration_seconds
            for row in self.assignments
        )

    @property
    def estimated_gpu_seconds(self) -> float:
        return sum(
            row.assignment.work_item.claim.estimated_gpu_seconds
            for row in self.assignments
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "assignments": [row.to_dict() for row in self.assignments],
            "assignment_sha256": [row.sha256 for row in self.assignments],
            "estimated_wall_seconds": self.estimated_wall_seconds,
            "estimated_gpu_seconds": self.estimated_gpu_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> GpuFleetDispatchWave:
        row = _strict_object(
            "GPU fleet dispatch wave",
            value,
            frozenset(
                {
                    "wave_index",
                    "assignments",
                    "assignment_sha256",
                    "estimated_wall_seconds",
                    "estimated_gpu_seconds",
                }
            ),
        )
        assignments = tuple(
            GpuFleetAssignment.from_dict(item)
            for item in _strict_list("fleet assignments", row["assignments"])
        )
        declared = tuple(
            _strict_text("fleet assignment SHA-256", item)
            for item in _strict_list("assignment_sha256", row["assignment_sha256"])
        )
        if declared != tuple(item.sha256 for item in assignments):
            raise ValueError("fleet wave assignment digest list mismatch")
        wave = cls(
            wave_index=_strict_int("wave_index", row["wave_index"]),
            assignments=assignments,
        )
        for name, expected in (
            ("estimated_wall_seconds", wave.estimated_wall_seconds),
            ("estimated_gpu_seconds", wave.estimated_gpu_seconds),
        ):
            actual = row[name]
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(actual)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"fleet wave {name} mismatch")
        return wave


@dataclass(frozen=True)
class GpuFleetDispatchPlan:
    """Complete immutable multi-host plan composed from single-host plans."""

    schema_version: int
    registry_sha256: str
    fleet_inventory_sha256: str
    receipts_sha256: str
    seed: int
    host_plans: tuple[HostDispatchPlan, ...]
    waves: tuple[GpuFleetDispatchWave, ...]
    completed_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only GPU fleet dispatch schema version 1 is supported")
        for name in (
            "registry_sha256",
            "fleet_inventory_sha256",
            "receipts_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("fleet dispatch seed must be non-negative")
        host_ids = tuple(plan.host_id for plan in self.host_plans)
        if host_ids != tuple(sorted(host_ids)) or len(host_ids) != len(set(host_ids)):
            raise ValueError("fleet host plans must be sorted and unique")
        if any(
            plan.dispatch_plan.registry_sha256 != self.registry_sha256
            for plan in self.host_plans
        ):
            raise ValueError("fleet host plans mix registries")
        if any(
            plan.dispatch_plan.receipts_sha256 != self.receipts_sha256
            for plan in self.host_plans
        ):
            raise ValueError("fleet host plans mix receipt authorities")
        if bool(self.host_plans) != bool(self.waves):
            raise ValueError(
                "fleet host plans and fleet waves must be empty or non-empty together"
            )
        if tuple(wave.wave_index for wave in self.waves) != tuple(
            range(len(self.waves))
        ):
            raise ValueError("fleet wave indexes must be contiguous")
        wrappers = tuple(row for wave in self.waves for row in wave.assignments)
        if len({row.sha256 for row in wrappers}) != len(wrappers):
            raise ValueError("fleet dispatch plan contains duplicate assignments")
        if len({row.assignment.work_item.item_id for row in wrappers}) != len(wrappers):
            raise ValueError("fleet dispatch plan contains duplicate cells")
        host_plan_by_id = {plan.host_id: plan for plan in self.host_plans}
        expected_local_waves: dict[tuple[str, int], GpuDispatchWave] = {
            (plan.host_id, wave.wave_index): wave
            for plan in self.host_plans
            for wave in plan.dispatch_plan.waves
        }
        observed: dict[tuple[str, int], list[GpuFleetAssignment]] = {}
        serial_hosts: dict[str, str] = {}
        for row in wrappers:
            plan = host_plan_by_id.get(row.host_id)
            if plan is None:
                raise ValueError("fleet assignment references an unknown host plan")
            binding = plan.execution_binding
            if (
                row.host_inventory_sha256 != plan.host_inventory_sha256
                or row.host_dispatch_plan_sha256 != plan.dispatch_plan.sha256
                or row.execution_binding_sha256 != binding.sha256
                or row.cache_root != binding.cache_root_for(row.assignment)
                or row.evidence_root != binding.evidence_root_for(row.assignment)
                or row.contention_domain != binding.contention_domain
            ):
                raise ValueError("fleet assignment differs from its host binding")
            local = expected_local_waves.get((row.host_id, row.local_wave_index))
            if local is None or row.local_wave_sha256 != local.sha256:
                raise ValueError("fleet assignment references an unknown local wave")
            if row.assignment.sha256 not in {
                assignment.sha256 for assignment in local.assignments
            }:
                raise ValueError("fleet assignment is absent from its local wave")
            observed.setdefault((row.host_id, row.local_wave_index), []).append(row)
            serial = row.assignment.work_item.serial_group_key
            if serial is not None:
                prior = serial_hosts.setdefault(serial, row.host_id)
                if prior != row.host_id:
                    raise ValueError("serial scientific group spans fleet hosts")
        if set(observed) != set(expected_local_waves):
            raise ValueError("fleet waves do not cover every host-local wave")
        for key, local in expected_local_waves.items():
            expected_ids = {assignment.sha256 for assignment in local.assignments}
            actual_ids = {row.assignment.sha256 for row in observed[key]}
            if actual_ids != expected_ids:
                raise ValueError("fleet wave has incomplete host-local coverage")
        completed = self.completed_cell_ids
        if completed != tuple(sorted(set(completed))) or any(
            not _is_sha256(cell_id) for cell_id in completed
        ):
            raise ValueError("fleet completed cell IDs must be canonical SHA-256s")
        if any(
            plan.dispatch_plan.completed_cell_ids != completed
            for plan in self.host_plans
        ):
            raise ValueError("fleet host plans disagree on completed cells")

    @property
    def assignments(self) -> tuple[GpuFleetAssignment, ...]:
        return tuple(row for wave in self.waves for row in wave.assignments)

    @property
    def estimated_wall_seconds(self) -> float:
        return sum(wave.estimated_wall_seconds for wave in self.waves)

    @property
    def estimated_gpu_seconds(self) -> float:
        return sum(wave.estimated_gpu_seconds for wave in self.waves)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_sha256": self.registry_sha256,
            "fleet_inventory_sha256": self.fleet_inventory_sha256,
            "receipts_sha256": self.receipts_sha256,
            "seed": self.seed,
            "host_plans": [plan.to_dict() for plan in self.host_plans],
            "host_plan_sha256": [plan.sha256 for plan in self.host_plans],
            "waves": [wave.to_dict() for wave in self.waves],
            "wave_sha256": [wave.sha256 for wave in self.waves],
            "completed_cell_ids": list(self.completed_cell_ids),
            "estimated_wall_seconds": self.estimated_wall_seconds,
            "estimated_gpu_seconds": self.estimated_gpu_seconds,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        planning_context: GpuFleetPlanningContext,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> GpuFleetDispatchPlan:
        if not isinstance(planning_context, GpuFleetPlanningContext):
            raise TypeError("planning_context must be a GpuFleetPlanningContext")
        row = _strict_object(
            "GPU fleet dispatch plan",
            value,
            frozenset(
                {
                    "schema_version",
                    "registry_sha256",
                    "fleet_inventory_sha256",
                    "receipts_sha256",
                    "seed",
                    "host_plans",
                    "host_plan_sha256",
                    "waves",
                    "wave_sha256",
                    "completed_cell_ids",
                    "estimated_wall_seconds",
                    "estimated_gpu_seconds",
                }
            ),
        )
        expected = planning_context.issue_plan()
        _require_exact_json(dict(row), expected.to_dict(), path="fleet_plan")
        _validate_sidecar(
            artifact_kind="gpu_fleet_dispatch_plan.v1",
            artifact_sha256=expected.sha256,
            sidecar=sidecar,
        )
        return expected

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "gpu_fleet_dispatch_plan.v1", self.sha256)


class FleetCapabilityRejectionError(CapabilityRejectionError):
    """Fail-closed fleet placement rejection with a stable reason code."""

    def __init__(self, reason_code: str, detail: str) -> None:
        _require_text("fleet rejection reason_code", reason_code)
        _require_text("fleet rejection detail", detail)
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class GpuFleetScheduler:
    """Partition affinity groups, then delegate placement to host schedulers."""

    registry: ExperimentRegistry
    fleet_inventory: GpuFleetInventory
    execution_bindings: tuple[HostExecutionBinding, ...]
    seed: int = 20260811

    def __post_init__(self) -> None:
        if type(self.registry) is not ExperimentRegistry:
            raise TypeError("fleet scheduler registry must be exact")
        if type(self.fleet_inventory) is not GpuFleetInventory:
            raise TypeError("fleet scheduler inventory must be exact")
        if any(
            type(row) is not HostExecutionBinding for row in self.execution_bindings
        ):
            raise TypeError("fleet execution bindings must be exact rows")
        host_ids = tuple(row.host_id for row in self.execution_bindings)
        if host_ids != tuple(sorted(host_ids)) or len(host_ids) != len(set(host_ids)):
            raise ValueError("fleet execution bindings must be sorted and unique")
        if host_ids != self.fleet_inventory.host_ids:
            raise ValueError("fleet execution bindings must cover every host exactly")
        for execution in self.execution_bindings:
            inventory = self.fleet_inventory.host(execution.host_id)
            if (
                execution.inventory_sha256 != inventory.inventory.sha256
                or execution.interference_envelope_sha256
                != inventory.interference_envelope.sha256
            ):
                raise ValueError("fleet execution binding differs from host inventory")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("fleet scheduler seed must be non-negative")

    def schedule_work_items(
        self,
        work_items: Sequence[PoolWorkItem],
        *,
        receipts_sha256: str,
        completed_cell_ids: Sequence[str] = (),
        budget_sha256_by_cell: Mapping[str, str] | None = None,
    ) -> GpuFleetDispatchPlan:
        _require_sha256("receipts_sha256", receipts_sha256)
        items = tuple(work_items)
        if any(type(item) is not PoolWorkItem for item in items):
            raise TypeError(
                "fleet scheduler work items must be exact PoolWorkItem rows"
            )
        item_ids = tuple(item.item_id for item in items)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("fleet work items contain duplicate cells")
        completed = tuple(sorted(completed_cell_ids))
        if completed != tuple(completed_cell_ids) or len(completed) != len(
            set(completed)
        ):
            raise ValueError("completed cell IDs must be sorted and unique")
        budgets = dict(budget_sha256_by_cell or {})
        if budgets and set(budgets) != set(item_ids):
            raise ValueError("fleet budget bindings must cover every work item")
        for cell_id, digest in budgets.items():
            _require_sha256("fleet budget cell ID", cell_id)
            _require_sha256("fleet budget SHA-256", digest)

        grouped: dict[str, list[PoolWorkItem]] = {}
        for item in items:
            # ``serial_group_key`` is the confirmation-block authority and
            # therefore dominates a narrower placement affinity if both are
            # present.  Its rows must share a host even though the host-local
            # scheduler deliberately places them in different waves.
            key = (
                item.serial_group_key
                or item.affinity_key
                or content_sha256(
                    {"kind": "independent_fleet_item", "item_id": item.item_id}
                )
            )
            grouped.setdefault(key, []).append(item)
        groups = tuple(
            (key, tuple(sorted(rows, key=lambda row: row.item_id)))
            for key, rows in sorted(grouped.items())
        )
        by_host: dict[str, list[PoolWorkItem]] = {
            host_id: [] for host_id in self.fleet_inventory.host_ids
        }
        normalized_load = {host_id: 0.0 for host_id in by_host}
        execution_by_host = {row.host_id: row for row in self.execution_bindings}
        for group_key, group in groups:
            feasible = tuple(
                host_id
                for host_id in self.fleet_inventory.host_ids
                if self._host_accepts_group(
                    host_id,
                    group,
                    receipts_sha256=receipts_sha256,
                    completed_cell_ids=completed,
                    budgets=budgets,
                )
            )
            if not feasible:
                reason = (
                    CROSS_HOST_COLLECTIVES_UNVALIDATED
                    if self._looks_cross_host_only(group)
                    else "no_same_host_capability_placement"
                )
                raise FleetCapabilityRejectionError(
                    reason,
                    f"affinity group {group_key} has no complete single-host placement",
                )
            rotation = int(
                content_sha256({"seed": self.seed, "affinity_group": group_key})[:16],
                16,
            ) % len(self.fleet_inventory.host_ids)
            host_order = {
                host_id: index
                for index, host_id in enumerate(self.fleet_inventory.host_ids)
            }
            selected = min(
                feasible,
                key=lambda host_id: (
                    normalized_load[host_id],
                    (host_order[host_id] - rotation)
                    % len(self.fleet_inventory.host_ids),
                    host_id,
                ),
            )
            by_host[selected].extend(group)
            capacity = sum(
                device.ready
                for device in self.fleet_inventory.host(selected).inventory.devices
            )
            normalized_load[selected] += sum(
                item.claim.estimated_gpu_seconds for item in group
            ) / max(capacity, 1)

        host_plans: list[HostDispatchPlan] = []
        for host_id in self.fleet_inventory.host_ids:
            host_items = tuple(by_host[host_id])
            if not host_items:
                continue
            inventory = self.fleet_inventory.host(host_id)
            execution = execution_by_host[host_id]
            local = GpuPoolScheduler(
                registry=self.registry,
                inventory=inventory.inventory,
                interference_envelope=inventory.interference_envelope,
                port_start=execution.port_start,
                port_end=execution.port_end,
                seed=self.seed,
            ).schedule_work_items(
                host_items,
                receipts_sha256=receipts_sha256,
                completed_cell_ids=completed,
                budget_sha256_by_cell=(
                    None
                    if not budgets
                    else {item.item_id: budgets[item.item_id] for item in host_items}
                ),
            )
            host_plans.append(
                HostDispatchPlan(
                    host_id=host_id,
                    host_inventory_sha256=inventory.inventory.sha256,
                    execution_binding=execution,
                    dispatch_plan=local,
                )
            )
        frozen_hosts = tuple(host_plans)
        waves = self._compose_fleet_waves(frozen_hosts)
        if items and (not frozen_hosts or not waves):
            raise RuntimeError("non-empty fleet work produced an empty dispatch plan")
        return GpuFleetDispatchPlan(
            schema_version=1,
            registry_sha256=self.registry.sha256,
            fleet_inventory_sha256=self.fleet_inventory.sha256,
            receipts_sha256=receipts_sha256,
            seed=self.seed,
            host_plans=frozen_hosts,
            waves=waves,
            completed_cell_ids=completed,
        )

    def _host_accepts_group(
        self,
        host_id: str,
        group: tuple[PoolWorkItem, ...],
        *,
        receipts_sha256: str,
        completed_cell_ids: tuple[str, ...],
        budgets: Mapping[str, str],
    ) -> bool:
        inventory = self.fleet_inventory.host(host_id)
        execution = next(
            row for row in self.execution_bindings if row.host_id == host_id
        )
        try:
            GpuPoolScheduler(
                registry=self.registry,
                inventory=inventory.inventory,
                interference_envelope=inventory.interference_envelope,
                port_start=execution.port_start,
                port_end=execution.port_end,
                seed=self.seed,
            ).schedule_work_items(
                group,
                receipts_sha256=receipts_sha256,
                completed_cell_ids=completed_cell_ids,
                budget_sha256_by_cell=(
                    None
                    if not budgets
                    else {item.item_id: budgets[item.item_id] for item in group}
                ),
            )
        except CapabilityRejectionError:
            return False
        return True

    def _looks_cross_host_only(self, group: tuple[PoolWorkItem, ...]) -> bool:
        item = max(group, key=lambda row: row.claim.gang_shape.gpu_count)
        requested = item.claim.gang_shape.gpu_count
        if requested <= 1:
            return False
        exact = set(item.claim.exact_gpu_uuids)
        capable_by_host = tuple(
            sum(
                item.claim.homogeneous.accepts(device)
                and (not exact or device.uuid in exact)
                for device in binding.inventory.devices
            )
            for binding in self.fleet_inventory.hosts
        )
        return sum(capable_by_host) >= requested and max(capable_by_host) < requested

    @staticmethod
    def _wrap_local_wave(
        host_plan: HostDispatchPlan,
        wave: GpuDispatchWave,
    ) -> tuple[GpuFleetAssignment, ...]:
        binding = host_plan.execution_binding
        return tuple(
            GpuFleetAssignment(
                host_id=host_plan.host_id,
                host_inventory_sha256=host_plan.host_inventory_sha256,
                host_dispatch_plan_sha256=host_plan.dispatch_plan.sha256,
                execution_binding_sha256=binding.sha256,
                local_wave_index=wave.wave_index,
                local_wave_sha256=wave.sha256,
                assignment=assignment,
                cache_root=binding.cache_root_for(assignment),
                evidence_root=binding.evidence_root_for(assignment),
                contention_domain=binding.contention_domain,
            )
            for assignment in wave.assignments
        )

    def _compose_fleet_waves(
        self,
        host_plans: tuple[HostDispatchPlan, ...],
    ) -> tuple[GpuFleetDispatchWave, ...]:
        next_index = {plan.host_id: 0 for plan in host_plans}
        by_host = {plan.host_id: plan for plan in host_plans}
        result: list[GpuFleetDispatchWave] = []
        while any(
            next_index[host_id] < len(plan.dispatch_plan.waves)
            for host_id, plan in by_host.items()
        ):
            rotation = int(
                content_sha256({"seed": self.seed, "fleet_wave_index": len(result)})[
                    :16
                ],
                16,
            ) % max(len(host_plans), 1)
            ordered_hosts = tuple(
                sorted(
                    by_host,
                    key=lambda host_id: (
                        (self.fleet_inventory.host_ids.index(host_id) - rotation)
                        % len(self.fleet_inventory.host_ids),
                        host_id,
                    ),
                )
            )
            selected: list[GpuFleetAssignment] = []
            selected_hosts: list[str] = []
            for host_id in ordered_hosts:
                plan = by_host[host_id]
                index = next_index[host_id]
                if index >= len(plan.dispatch_plan.waves):
                    continue
                wrapped = self._wrap_local_wave(
                    plan,
                    plan.dispatch_plan.waves[index],
                )
                candidate = tuple(
                    sorted(
                        (*selected, *wrapped),
                        key=lambda row: (row.host_id, row.assignment.sha256),
                    )
                )
                try:
                    GpuFleetDispatchWave(len(result), candidate)
                except ValueError as error:
                    if "shared contention domain" not in str(error):
                        raise
                    continue
                selected.extend(wrapped)
                selected_hosts.append(host_id)
            if not selected:
                raise RuntimeError("fleet wave composition made no progress")
            frozen = tuple(
                sorted(selected, key=lambda row: (row.host_id, row.assignment.sha256))
            )
            result.append(GpuFleetDispatchWave(len(result), frozen))
            for host_id in selected_hosts:
                next_index[host_id] += 1
        return tuple(result)


@dataclass(frozen=True)
class GpuFleetPlanningContext:
    """Raw low-level inputs used to reissue and verify a fleet plan."""

    registry: ExperimentRegistry
    fleet_inventory: GpuFleetInventory
    execution_bindings: tuple[HostExecutionBinding, ...]
    work_items: tuple[PoolWorkItem, ...]
    receipts_sha256: str
    completed_cell_ids: tuple[str, ...] = ()
    budget_sha256_by_cell: tuple[tuple[str, str], ...] = ()
    seed: int = 20260811

    def __post_init__(self) -> None:
        _require_sha256("receipts_sha256", self.receipts_sha256)
        if self.completed_cell_ids != tuple(sorted(set(self.completed_cell_ids))):
            raise ValueError("fleet planning completed cells must be canonical")
        if self.budget_sha256_by_cell != tuple(sorted(self.budget_sha256_by_cell)):
            raise ValueError("fleet planning budget bindings must be canonical")
        if len({cell_id for cell_id, _ in self.budget_sha256_by_cell}) != len(
            self.budget_sha256_by_cell
        ):
            raise ValueError("fleet planning budget bindings contain duplicate cells")
        GpuFleetScheduler(
            registry=self.registry,
            fleet_inventory=self.fleet_inventory,
            execution_bindings=self.execution_bindings,
            seed=self.seed,
        )

    def issue_plan(self) -> GpuFleetDispatchPlan:
        return GpuFleetScheduler(
            registry=self.registry,
            fleet_inventory=self.fleet_inventory,
            execution_bindings=self.execution_bindings,
            seed=self.seed,
        ).schedule_work_items(
            self.work_items,
            receipts_sha256=self.receipts_sha256,
            completed_cell_ids=self.completed_cell_ids,
            budget_sha256_by_cell=(
                None
                if not self.budget_sha256_by_cell
                else dict(self.budget_sha256_by_cell)
            ),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "gpu_fleet_planning_context",
                "registry_sha256": self.registry.sha256,
                "fleet_inventory_sha256": self.fleet_inventory.sha256,
                "execution_binding_sha256s": [
                    row.sha256 for row in self.execution_bindings
                ],
                "work_item_sha256s": sorted(row.sha256 for row in self.work_items),
                "receipts_sha256": self.receipts_sha256,
                "completed_cell_ids": self.completed_cell_ids,
                "budget_sha256_by_cell": self.budget_sha256_by_cell,
                "seed": self.seed,
            }
        )


class HostWaveStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HostWaveReceipt:
    """One host's content-bound outcome for a fleet wave attempt."""

    host_id: str
    fleet_plan_sha256: str
    fleet_wave_sha256: str
    host_dispatch_plan_sha256: str
    local_wave_index: int
    local_wave_sha256: str
    attempt: int
    status: HostWaveStatus
    dispatch_receipt: DispatchWaveExecutionReceipt | None
    failure_sha256: str | None
    prior_host_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text("host_id", self.host_id)
        for name in (
            "fleet_plan_sha256",
            "fleet_wave_sha256",
            "host_dispatch_plan_sha256",
            "local_wave_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.local_wave_index, bool)
            or not isinstance(self.local_wave_index, int)
            or self.local_wave_index < 0
        ):
            raise ValueError("host receipt local wave index must be non-negative")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("host receipt attempt must be positive")
        if not isinstance(self.status, HostWaveStatus):
            raise TypeError("host receipt status must be HostWaveStatus")
        if (
            self.dispatch_receipt is not None
            and type(self.dispatch_receipt) is not DispatchWaveExecutionReceipt
        ):
            raise TypeError("host dispatch receipt must be exact")
        if self.dispatch_receipt is not None and (
            self.dispatch_receipt.plan_sha256 != self.host_dispatch_plan_sha256
            or self.dispatch_receipt.wave_index != self.local_wave_index
            or self.dispatch_receipt.wave_sha256 != self.local_wave_sha256
        ):
            raise ValueError("host receipt dispatch identity mismatch")
        if self.status is HostWaveStatus.SUCCEEDED:
            if self.dispatch_receipt is None or not self.dispatch_receipt.succeeded:
                raise ValueError(
                    "successful host receipt needs a successful raw receipt"
                )
            if self.failure_sha256 is not None:
                raise ValueError(
                    "successful host receipt cannot carry failure evidence"
                )
        else:
            if not _is_sha256(self.failure_sha256):
                raise ValueError(
                    "failed host receipt needs content-bound failure evidence"
                )
            if self.dispatch_receipt is not None and self.dispatch_receipt.succeeded:
                raise ValueError("failed host receipt cannot wrap a successful wave")
        if self.attempt == 1:
            if self.prior_host_receipt_sha256 is not None:
                raise ValueError("first host attempt cannot bind a prior receipt")
        elif not _is_sha256(self.prior_host_receipt_sha256):
            raise ValueError("retried host receipt must bind its prior receipt")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "fleet_plan_sha256": self.fleet_plan_sha256,
            "fleet_wave_sha256": self.fleet_wave_sha256,
            "host_dispatch_plan_sha256": self.host_dispatch_plan_sha256,
            "local_wave_index": self.local_wave_index,
            "local_wave_sha256": self.local_wave_sha256,
            "attempt": self.attempt,
            "status": self.status.value,
            "dispatch_receipt": (
                None
                if self.dispatch_receipt is None
                else self.dispatch_receipt.to_dict()
            ),
            "dispatch_receipt_sha256": (
                None if self.dispatch_receipt is None else self.dispatch_receipt.sha256
            ),
            "failure_sha256": self.failure_sha256,
            "prior_host_receipt_sha256": self.prior_host_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> HostWaveReceipt:
        row = _strict_object(
            "host wave receipt",
            value,
            frozenset(
                {
                    "host_id",
                    "fleet_plan_sha256",
                    "fleet_wave_sha256",
                    "host_dispatch_plan_sha256",
                    "local_wave_index",
                    "local_wave_sha256",
                    "attempt",
                    "status",
                    "dispatch_receipt",
                    "dispatch_receipt_sha256",
                    "failure_sha256",
                    "prior_host_receipt_sha256",
                }
            ),
        )
        raw_receipt = row["dispatch_receipt"]
        dispatch_receipt = (
            None
            if raw_receipt is None
            else DispatchWaveExecutionReceipt.from_dict(raw_receipt)
        )
        declared = row["dispatch_receipt_sha256"]
        if dispatch_receipt is None:
            if declared is not None:
                raise ValueError("absent host dispatch receipt has a digest")
        elif (
            _strict_text("dispatch_receipt_sha256", declared) != dispatch_receipt.sha256
        ):
            raise ValueError("host dispatch receipt digest mismatch")
        return cls(
            host_id=_strict_text("host_id", row["host_id"]),
            fleet_plan_sha256=_strict_text(
                "fleet_plan_sha256", row["fleet_plan_sha256"]
            ),
            fleet_wave_sha256=_strict_text(
                "fleet_wave_sha256", row["fleet_wave_sha256"]
            ),
            host_dispatch_plan_sha256=_strict_text(
                "host_dispatch_plan_sha256",
                row["host_dispatch_plan_sha256"],
            ),
            local_wave_index=_strict_int("local_wave_index", row["local_wave_index"]),
            local_wave_sha256=_strict_text(
                "local_wave_sha256", row["local_wave_sha256"]
            ),
            attempt=_strict_int("attempt", row["attempt"]),
            status=HostWaveStatus(_strict_text("status", row["status"])),
            dispatch_receipt=dispatch_receipt,
            failure_sha256=_optional_text("failure_sha256", row["failure_sha256"]),
            prior_host_receipt_sha256=_optional_text(
                "prior_host_receipt_sha256", row["prior_host_receipt_sha256"]
            ),
        )


@dataclass(frozen=True)
class FleetWaveReceipt:
    """Failure-isolated aggregation for one immutable fleet dispatch wave."""

    schema_version: int
    fleet_plan_sha256: str
    fleet_wave_index: int
    fleet_wave_sha256: str
    attempt: int
    host_receipts: tuple[HostWaveReceipt, ...]
    prior_fleet_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only fleet-wave receipt schema version 1 is supported")
        _require_sha256("fleet_plan_sha256", self.fleet_plan_sha256)
        _require_sha256("fleet_wave_sha256", self.fleet_wave_sha256)
        if (
            isinstance(self.fleet_wave_index, bool)
            or not isinstance(self.fleet_wave_index, int)
            or self.fleet_wave_index < 0
        ):
            raise ValueError("fleet receipt wave index must be non-negative")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("fleet receipt attempt must be positive")
        if not self.host_receipts:
            raise ValueError("fleet wave receipt needs host outcomes")
        host_ids = tuple(row.host_id for row in self.host_receipts)
        if host_ids != tuple(sorted(host_ids)) or len(host_ids) != len(set(host_ids)):
            raise ValueError("fleet host receipts must be sorted and unique")
        if any(
            row.fleet_plan_sha256 != self.fleet_plan_sha256
            or row.fleet_wave_sha256 != self.fleet_wave_sha256
            for row in self.host_receipts
        ):
            raise ValueError("fleet receipt mixes plan/wave identities")
        if self.attempt == 1:
            if self.prior_fleet_receipt_sha256 is not None:
                raise ValueError("first fleet attempt cannot bind a prior receipt")
        elif not _is_sha256(self.prior_fleet_receipt_sha256):
            raise ValueError("retried fleet receipt must bind its prior receipt")

    @property
    def succeeded(self) -> bool:
        return all(row.status is HostWaveStatus.SUCCEEDED for row in self.host_receipts)

    @property
    def failed_host_ids(self) -> tuple[str, ...]:
        return tuple(
            row.host_id
            for row in self.host_receipts
            if row.status is HostWaveStatus.FAILED
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fleet_plan_sha256": self.fleet_plan_sha256,
            "fleet_wave_index": self.fleet_wave_index,
            "fleet_wave_sha256": self.fleet_wave_sha256,
            "attempt": self.attempt,
            "host_receipts": [row.to_dict() for row in self.host_receipts],
            "host_receipt_sha256": [row.sha256 for row in self.host_receipts],
            "prior_fleet_receipt_sha256": self.prior_fleet_receipt_sha256,
        }

    def validate_for_plan(self, plan: GpuFleetDispatchPlan) -> None:
        if type(plan) is not GpuFleetDispatchPlan:
            raise TypeError("fleet receipt validation needs an exact plan")
        if self.fleet_plan_sha256 != plan.sha256:
            raise ValueError("fleet receipt belongs to another plan")
        if self.fleet_wave_index >= len(plan.waves):
            raise ValueError("fleet receipt references an unknown wave")
        wave = plan.waves[self.fleet_wave_index]
        if self.fleet_wave_sha256 != wave.sha256:
            raise ValueError("fleet receipt wave identity mismatch")
        host_plan_by_id = {row.host_id: row for row in plan.host_plans}
        assignments_by_host: dict[str, list[GpuFleetAssignment]] = {}
        for assignment in wave.assignments:
            assignments_by_host.setdefault(assignment.host_id, []).append(assignment)
        if tuple(row.host_id for row in self.host_receipts) != tuple(
            sorted(assignments_by_host)
        ):
            raise ValueError("fleet receipt must cover every participating host")
        for receipt in self.host_receipts:
            assignments = assignments_by_host[receipt.host_id]
            first = assignments[0]
            if (
                receipt.host_dispatch_plan_sha256
                != host_plan_by_id[receipt.host_id].dispatch_plan.sha256
                or receipt.local_wave_index != first.local_wave_index
                or receipt.local_wave_sha256 != first.local_wave_sha256
            ):
                raise ValueError("host receipt differs from its fleet assignment")
            if receipt.dispatch_receipt is not None:
                actual = {
                    row.assignment_sha256
                    for row in receipt.dispatch_receipt.assignment_receipts
                }
                expected = {row.assignment.sha256 for row in assignments}
                if actual != expected:
                    raise ValueError("host receipt has incomplete assignment coverage")

    @classmethod
    def aggregate(
        cls,
        plan: GpuFleetDispatchPlan,
        *,
        fleet_wave_index: int,
        host_receipts: Sequence[HostWaveReceipt],
        prior_receipt: FleetWaveReceipt | None = None,
    ) -> FleetWaveReceipt:
        if type(plan) is not GpuFleetDispatchPlan:
            raise TypeError("fleet receipt aggregation needs an exact plan")
        rows = tuple(sorted(host_receipts, key=lambda row: row.host_id))
        if prior_receipt is None:
            receipt = cls(
                schema_version=1,
                fleet_plan_sha256=plan.sha256,
                fleet_wave_index=fleet_wave_index,
                fleet_wave_sha256=plan.waves[fleet_wave_index].sha256,
                attempt=1,
                host_receipts=rows,
            )
            if any(row.attempt != 1 for row in rows):
                raise ValueError("initial fleet receipt requires first host attempts")
        else:
            prior_receipt.validate_for_plan(plan)
            if prior_receipt.succeeded:
                raise ValueError("a successful fleet wave cannot be retried")
            if prior_receipt.fleet_wave_index != fleet_wave_index:
                raise ValueError("fleet retry cannot move to another wave")
            prior_by_host = {row.host_id: row for row in prior_receipt.host_receipts}
            if {row.host_id for row in rows} != set(prior_by_host):
                raise ValueError("fleet retry cannot add, remove, or migrate hosts")
            changed = False
            for row in rows:
                prior = prior_by_host[row.host_id]
                if prior.status is HostWaveStatus.SUCCEEDED:
                    if row != prior:
                        raise ValueError("successful host evidence must be preserved")
                    continue
                if row == prior:
                    continue
                changed = True
                if (
                    row.attempt != prior.attempt + 1
                    or row.prior_host_receipt_sha256 != prior.sha256
                    or row.host_dispatch_plan_sha256 != prior.host_dispatch_plan_sha256
                    or row.local_wave_sha256 != prior.local_wave_sha256
                ):
                    raise ValueError("failed host retry is not receipt-bound")
            if not changed:
                raise ValueError("fleet retry must advance at least one failed host")
            receipt = cls(
                schema_version=1,
                fleet_plan_sha256=plan.sha256,
                fleet_wave_index=fleet_wave_index,
                fleet_wave_sha256=plan.waves[fleet_wave_index].sha256,
                attempt=prior_receipt.attempt + 1,
                host_receipts=rows,
                prior_fleet_receipt_sha256=prior_receipt.sha256,
            )
        receipt.validate_for_plan(plan)
        return receipt

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        plan: GpuFleetDispatchPlan,
        prior_receipt: FleetWaveReceipt | None = None,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> FleetWaveReceipt:
        row = _strict_object(
            "fleet wave receipt",
            value,
            frozenset(
                {
                    "schema_version",
                    "fleet_plan_sha256",
                    "fleet_wave_index",
                    "fleet_wave_sha256",
                    "attempt",
                    "host_receipts",
                    "host_receipt_sha256",
                    "prior_fleet_receipt_sha256",
                }
            ),
        )
        host_receipts = tuple(
            HostWaveReceipt.from_dict(item)
            for item in _strict_list("host_receipts", row["host_receipts"])
        )
        declared = tuple(
            _strict_text("host receipt SHA-256", item)
            for item in _strict_list("host_receipt_sha256", row["host_receipt_sha256"])
        )
        if declared != tuple(receipt.sha256 for receipt in host_receipts):
            raise ValueError("fleet host-receipt digest list mismatch")
        receipt = cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            fleet_plan_sha256=_strict_text(
                "fleet_plan_sha256", row["fleet_plan_sha256"]
            ),
            fleet_wave_index=_strict_int("fleet_wave_index", row["fleet_wave_index"]),
            fleet_wave_sha256=_strict_text(
                "fleet_wave_sha256", row["fleet_wave_sha256"]
            ),
            attempt=_strict_int("attempt", row["attempt"]),
            host_receipts=host_receipts,
            prior_fleet_receipt_sha256=_optional_text(
                "prior_fleet_receipt_sha256", row["prior_fleet_receipt_sha256"]
            ),
        )
        receipt.validate_for_plan(plan)
        if receipt.attempt > 1:
            if prior_receipt is None:
                raise ValueError("retried fleet receipt needs its prior raw receipt")
            expected = cls.aggregate(
                plan,
                fleet_wave_index=receipt.fleet_wave_index,
                host_receipts=receipt.host_receipts,
                prior_receipt=prior_receipt,
            )
            if receipt != expected:
                raise ValueError("fleet retry differs from its prior receipt chain")
        elif prior_receipt is not None:
            raise ValueError("first fleet receipt cannot consume prior evidence")
        _validate_sidecar(
            artifact_kind="fleet_wave_receipt.v1",
            artifact_sha256=receipt.sha256,
            sidecar=sidecar,
        )
        return receipt

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "fleet_wave_receipt.v1", self.sha256)


# Compact compatibility spellings for callers that lead with ``Fleet``.
FleetGpuAssignment = GpuFleetAssignment
FleetAssignment = GpuFleetAssignment
FleetDispatchPlan = GpuFleetDispatchPlan
FleetGpuDispatchPlan = GpuFleetDispatchPlan
FleetGpuDispatchWave = GpuFleetDispatchWave
