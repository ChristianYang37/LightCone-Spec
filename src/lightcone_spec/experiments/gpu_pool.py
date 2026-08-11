"""Deterministic, fail-closed scheduling for a same-host GPU pool.

The industrial registry deliberately records scientific cell identities.  A
physical GPU assignment is a separate, content-bound dispatch artifact: it may
change when the same registry is dry-run on a different host without changing
the scientific identity of any cell.

This module has no CUDA, process, filesystem, or network side effects.  The
only side-effecting entry point is :func:`execute_dispatch_plan`, whose caller
must inject an async assignment runner.  Execution is wave-ordered and returns
immutable cost receipts.  Resume remains blocked until a durable raw-terminal
store can authorize every prior attempt.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from itertools import combinations
from typing import Any

from lightcone_spec.experiments.planning import (
    ConfirmationFamilyPowerReductionArtifact,
    ExperimentBudget,
    FamilyActivationArtifact,
    ReducerActivationArtifact,
    materialize_confirmation_prefix,
    verify_confirmation_pilot_activation,
)
from lightcone_spec.experiments.registry import (
    PILOT_BLOCKS,
    CellIdentity,
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    ResourceClaim,
    WorkloadClass,
    content_sha256,
    serving_cell_rejection_reason,
)

_LOWER_SHA256_LENGTH = 64
_EXCLUSIVE_HOST_CLASSES = frozenset(
    {WorkloadClass.PROFILE, WorkloadClass.DOWNLOAD, WorkloadClass.COMPILE}
)
_SUPPORTED_POOL_SIZES = frozenset({1, 2, 4, 8, 16})


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


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _strict_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _strict_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    _require_text(name, value)
    return value


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _strict_text(name, value)


def _optional_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _strict_int(name, value)


def _optional_float(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _strict_float(name, value)


@dataclass(frozen=True)
class ArtifactSidecar:
    """Minimal content sidecar for durable JSON artifacts."""

    schema_version: int
    artifact_kind: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only artifact-sidecar schema version 1 is supported")
        _require_text("artifact_kind", self.artifact_kind)
        _require_sha256("artifact_sha256", self.artifact_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactSidecar:
        row = _strict_object(
            "artifact sidecar",
            value,
            frozenset({"schema_version", "artifact_kind", "artifact_sha256"}),
        )
        return cls(
            schema_version=_strict_int("sidecar schema_version", row["schema_version"]),
            artifact_kind=_strict_text("artifact_kind", row["artifact_kind"]),
            artifact_sha256=_strict_text("artifact_sha256", row["artifact_sha256"]),
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


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


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name} must be non-empty single-line text")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _LOWER_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lower-case SHA-256")


def _require_positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _merge_monotonic_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return the canonical union of non-negative monotonic intervals."""

    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, finish in ordered:
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(finish, bool)
            or not isinstance(finish, int)
            or start < 0
            or finish < start
        ):
            raise ValueError("monotonic intervals require 0 <= start <= finish")
        if not merged or start > merged[-1][1]:
            merged.append((start, finish))
        else:
            prior_start, prior_finish = merged[-1]
            merged[-1] = (prior_start, max(prior_finish, finish))
    return tuple(merged)


def _interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(
        finish - start for start, finish in _merge_monotonic_intervals(intervals)
    )


class GpuAvailability(str, Enum):
    """Inventory readiness state.  Only ``READY`` devices can be allocated."""

    READY = "READY"
    DRAINING = "DRAINING"
    RESERVED = "RESERVED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class GpuDevice:
    """One content-bound physical-device declaration."""

    uuid: str
    host_id: str
    model: str
    memory_bytes: int
    compute_capability: tuple[int, int]
    pci_bus_id: str
    pci_root: str
    numa_node: int
    interconnects: tuple[str, ...]
    peer_access_class: str
    clock_policy: str
    power_limit_watts: float
    thermal_limit_celsius: float
    availability: GpuAvailability
    reserved_processes: tuple[str, ...]
    allowed_topology_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "uuid",
            "host_id",
            "model",
            "pci_bus_id",
            "pci_root",
            "peer_access_class",
            "clock_policy",
        ):
            _require_text(name, getattr(self, name))
        if (
            isinstance(self.memory_bytes, bool)
            or not isinstance(self.memory_bytes, int)
            or self.memory_bytes <= 0
        ):
            raise ValueError("memory_bytes must be a positive integer")
        if len(self.compute_capability) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.compute_capability
        ):
            raise ValueError("compute_capability must be two non-negative integers")
        if (
            isinstance(self.numa_node, bool)
            or not isinstance(self.numa_node, int)
            or self.numa_node < 0
        ):
            raise ValueError("numa_node must be a non-negative integer")
        if not self.interconnects or len(set(self.interconnects)) != len(
            self.interconnects
        ):
            raise ValueError("interconnects must be non-empty and unique")
        if tuple(sorted(self.interconnects)) != self.interconnects:
            raise ValueError("interconnects must be canonically sorted")
        for interconnect in self.interconnects:
            _require_text("interconnect", interconnect)
        _require_positive_finite("power_limit_watts", self.power_limit_watts)
        _require_positive_finite("thermal_limit_celsius", self.thermal_limit_celsius)
        if not isinstance(self.availability, GpuAvailability):
            raise TypeError("availability must be a GpuAvailability")
        if len(set(self.reserved_processes)) != len(self.reserved_processes):
            raise ValueError("reserved_processes must be unique")
        if tuple(sorted(self.reserved_processes)) != self.reserved_processes:
            raise ValueError("reserved_processes must be canonically sorted")
        for process in self.reserved_processes:
            _require_text("reserved process", process)
        if len(set(self.allowed_topology_groups)) != len(self.allowed_topology_groups):
            raise ValueError("allowed_topology_groups must be unique")
        if tuple(sorted(self.allowed_topology_groups)) != self.allowed_topology_groups:
            raise ValueError("allowed_topology_groups must be canonically sorted")
        for group_id in self.allowed_topology_groups:
            _require_text("allowed topology group", group_id)

    @property
    def ready(self) -> bool:
        return (
            self.availability is GpuAvailability.READY and not self.reserved_processes
        )

    @cached_property
    def hardware_envelope_sha256(self) -> str:
        """Bind every hardware/load attribute relevant to co-tenancy.

        UUID, PCI address, NUMA placement, and topology-group membership are
        intentionally excluded so homogeneous devices on the same attested host
        share an envelope.  The host identity remains included: calibration on
        one host cannot authorize another host.
        """

        return content_sha256(
            {
                "host_id": self.host_id,
                "model": self.model,
                "memory_bytes": self.memory_bytes,
                "compute_capability": self.compute_capability,
                "interconnects": self.interconnects,
                "peer_access_class": self.peer_access_class,
                "clock_policy": self.clock_policy,
                "power_limit_watts": self.power_limit_watts,
                "thermal_limit_celsius": self.thermal_limit_celsius,
            }
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "host_id": self.host_id,
            "model": self.model,
            "memory_bytes": self.memory_bytes,
            "compute_capability": list(self.compute_capability),
            "pci_bus_id": self.pci_bus_id,
            "pci_root": self.pci_root,
            "numa_node": self.numa_node,
            "interconnects": list(self.interconnects),
            "peer_access_class": self.peer_access_class,
            "clock_policy": self.clock_policy,
            "power_limit_watts": self.power_limit_watts,
            "thermal_limit_celsius": self.thermal_limit_celsius,
            "availability": self.availability.value,
            "reserved_processes": list(self.reserved_processes),
            "allowed_topology_groups": list(self.allowed_topology_groups),
        }

    @classmethod
    def from_dict(cls, value: object) -> GpuDevice:
        fields = frozenset(
            {
                "uuid",
                "host_id",
                "model",
                "memory_bytes",
                "compute_capability",
                "pci_bus_id",
                "pci_root",
                "numa_node",
                "interconnects",
                "peer_access_class",
                "clock_policy",
                "power_limit_watts",
                "thermal_limit_celsius",
                "availability",
                "reserved_processes",
                "allowed_topology_groups",
            }
        )
        row = _strict_object("GPU device", value, fields)
        compute = _strict_list("compute_capability", row["compute_capability"])
        return cls(
            uuid=_strict_text("GPU UUID", row["uuid"]),
            host_id=_strict_text("host_id", row["host_id"]),
            model=_strict_text("GPU model", row["model"]),
            memory_bytes=_strict_int("memory_bytes", row["memory_bytes"]),
            compute_capability=tuple(
                _strict_int("compute capability component", component)
                for component in compute
            ),
            pci_bus_id=_strict_text("pci_bus_id", row["pci_bus_id"]),
            pci_root=_strict_text("pci_root", row["pci_root"]),
            numa_node=_strict_int("numa_node", row["numa_node"]),
            interconnects=tuple(
                _strict_text("interconnect", item)
                for item in _strict_list("interconnects", row["interconnects"])
            ),
            peer_access_class=_strict_text(
                "peer_access_class", row["peer_access_class"]
            ),
            clock_policy=_strict_text("clock_policy", row["clock_policy"]),
            power_limit_watts=_strict_float(
                "power_limit_watts", row["power_limit_watts"]
            ),
            thermal_limit_celsius=_strict_float(
                "thermal_limit_celsius", row["thermal_limit_celsius"]
            ),
            availability=GpuAvailability(
                _strict_text("availability", row["availability"])
            ),
            reserved_processes=tuple(
                _strict_text("reserved process", item)
                for item in _strict_list(
                    "reserved_processes", row["reserved_processes"]
                )
            ),
            allowed_topology_groups=tuple(
                _strict_text("allowed topology group", item)
                for item in _strict_list(
                    "allowed_topology_groups", row["allowed_topology_groups"]
                )
            ),
        )


@dataclass(frozen=True)
class GpuTopologyGroup:
    """An allowed, attested peer-access group on one host."""

    group_id: str
    host_id: str
    gpu_uuids: tuple[str, ...]
    fabric: str
    bandwidth_class: str

    def __post_init__(self) -> None:
        for name in ("group_id", "host_id", "fabric", "bandwidth_class"):
            _require_text(name, getattr(self, name))
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("topology-group GPU UUIDs must be non-empty and unique")
        if tuple(sorted(self.gpu_uuids)) != self.gpu_uuids:
            raise ValueError("topology-group GPU UUIDs must be canonically sorted")
        for uuid in self.gpu_uuids:
            _require_text("topology-group GPU UUID", uuid)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "host_id": self.host_id,
            "gpu_uuids": list(self.gpu_uuids),
            "fabric": self.fabric,
            "bandwidth_class": self.bandwidth_class,
        }

    @classmethod
    def from_dict(cls, value: object) -> GpuTopologyGroup:
        row = _strict_object(
            "GPU topology group",
            value,
            frozenset(
                {"group_id", "host_id", "gpu_uuids", "fabric", "bandwidth_class"}
            ),
        )
        return cls(
            group_id=_strict_text("group_id", row["group_id"]),
            host_id=_strict_text("host_id", row["host_id"]),
            gpu_uuids=tuple(
                _strict_text("topology GPU UUID", item)
                for item in _strict_list("gpu_uuids", row["gpu_uuids"])
            ),
            fabric=_strict_text("fabric", row["fabric"]),
            bandwidth_class=_strict_text("bandwidth_class", row["bandwidth_class"]),
        )


@dataclass(frozen=True)
class GpuInventory:
    """Immutable signed-or-content-bound inventory input.

    ``source_receipt_sha256`` binds the external discovery/attestation receipt.
    Signature verification belongs at the inventory ingestion boundary; this
    scheduler consumes only the already verified, immutable declaration.
    """

    schema_version: int
    devices: tuple[GpuDevice, ...]
    topology_groups: tuple[GpuTopologyGroup, ...]
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only GPU inventory schema version 1 is supported")
        if not self.devices:
            raise ValueError("GPU inventory must contain at least one device")
        if any(not isinstance(device, GpuDevice) for device in self.devices):
            raise TypeError("inventory devices must be GpuDevice declarations")
        if any(
            not isinstance(group, GpuTopologyGroup) for group in self.topology_groups
        ):
            raise TypeError("inventory topology groups must be declarations")
        uuids = tuple(device.uuid for device in self.devices)
        if len(uuids) != len(set(uuids)):
            raise ValueError("GPU inventory contains duplicate UUIDs")
        if tuple(sorted(uuids)) != uuids:
            raise ValueError("GPU inventory devices must be sorted by UUID")
        group_ids = tuple(group.group_id for group in self.topology_groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("GPU inventory contains duplicate topology groups")
        if tuple(sorted(group_ids)) != group_ids:
            raise ValueError("topology groups must be sorted by group_id")
        device_by_uuid = {device.uuid: device for device in self.devices}
        for group in self.topology_groups:
            if set(group.gpu_uuids) - set(device_by_uuid):
                raise ValueError("topology group references a GPU outside inventory")
            if any(
                device_by_uuid[uuid].host_id != group.host_id
                for uuid in group.gpu_uuids
            ):
                raise ValueError("topology group cannot span hosts")
            if any(
                group.group_id not in device_by_uuid[uuid].allowed_topology_groups
                for uuid in group.gpu_uuids
            ):
                raise ValueError("device does not allow its declared topology group")
        known_groups = set(group_ids)
        if any(
            set(device.allowed_topology_groups) - known_groups
            for device in self.devices
        ):
            raise ValueError("device references an undeclared topology group")
        _require_sha256("source_receipt_sha256", self.source_receipt_sha256)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "devices": [device.to_dict() for device in self.devices],
            "topology_groups": [group.to_dict() for group in self.topology_groups],
            "source_receipt_sha256": self.source_receipt_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> GpuInventory:
        row = _strict_object(
            "GPU inventory",
            value,
            frozenset(
                {
                    "schema_version",
                    "devices",
                    "topology_groups",
                    "source_receipt_sha256",
                }
            ),
        )
        inventory = cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            devices=tuple(
                GpuDevice.from_dict(item)
                for item in _strict_list("devices", row["devices"])
            ),
            topology_groups=tuple(
                GpuTopologyGroup.from_dict(item)
                for item in _strict_list("topology_groups", row["topology_groups"])
            ),
            source_receipt_sha256=_strict_text(
                "source_receipt_sha256", row["source_receipt_sha256"]
            ),
        )
        _validate_sidecar(
            artifact_kind="gpu_inventory.v1",
            artifact_sha256=inventory.sha256,
            sidecar=sidecar,
        )
        return inventory

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "gpu_inventory.v1", self.sha256)

    @property
    def host_ids(self) -> tuple[str, ...]:
        return tuple(sorted({device.host_id for device in self.devices}))

    def device(self, uuid: str) -> GpuDevice:
        for device in self.devices:
            if device.uuid == uuid:
                return device
        raise ValueError(f"unknown GPU UUID {uuid!r}")


@dataclass(frozen=True)
class GangShape:
    """Tensor/data-parallel rank shape allocated atomically."""

    tensor_parallel_size: int = 1
    data_parallel_size: int = 1

    def __post_init__(self) -> None:
        for name in ("tensor_parallel_size", "data_parallel_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def gpu_count(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def signature(self) -> str:
        return f"tp{self.tensor_parallel_size}_dp{self.data_parallel_size}"

    def to_dict(self) -> dict[str, int]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }

    @classmethod
    def from_dict(cls, value: object) -> GangShape:
        row = _strict_object(
            "gang shape",
            value,
            frozenset({"tensor_parallel_size", "data_parallel_size"}),
        )
        return cls(
            tensor_parallel_size=_strict_int(
                "tensor_parallel_size", row["tensor_parallel_size"]
            ),
            data_parallel_size=_strict_int(
                "data_parallel_size", row["data_parallel_size"]
            ),
        )


@dataclass(frozen=True)
class HomogeneousDeviceConstraint:
    """Minimum capability and exact allowed hardware classes."""

    model: str | None = None
    minimum_memory_bytes: int = 1
    minimum_compute_capability: tuple[int, int] = (0, 0)
    allowed_peer_access_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model is not None:
            _require_text("constraint model", self.model)
        if (
            isinstance(self.minimum_memory_bytes, bool)
            or not isinstance(self.minimum_memory_bytes, int)
            or self.minimum_memory_bytes < 1
        ):
            raise ValueError("minimum_memory_bytes must be a positive integer")
        if len(self.minimum_compute_capability) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.minimum_compute_capability
        ):
            raise ValueError(
                "minimum_compute_capability must be two non-negative integers"
            )
        if len(set(self.allowed_peer_access_classes)) != len(
            self.allowed_peer_access_classes
        ):
            raise ValueError("allowed peer-access classes must be unique")
        if tuple(sorted(self.allowed_peer_access_classes)) != (
            self.allowed_peer_access_classes
        ):
            raise ValueError("allowed peer-access classes must be canonically sorted")
        for value in self.allowed_peer_access_classes:
            _require_text("allowed peer-access class", value)

    def accepts(self, device: GpuDevice) -> bool:
        return (
            device.ready
            and (self.model is None or device.model == self.model)
            and device.memory_bytes >= self.minimum_memory_bytes
            and device.compute_capability >= self.minimum_compute_capability
            and (
                not self.allowed_peer_access_classes
                or device.peer_access_class in self.allowed_peer_access_classes
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "minimum_memory_bytes": self.minimum_memory_bytes,
            "minimum_compute_capability": list(self.minimum_compute_capability),
            "allowed_peer_access_classes": list(self.allowed_peer_access_classes),
        }

    @classmethod
    def from_dict(cls, value: object) -> HomogeneousDeviceConstraint:
        row = _strict_object(
            "homogeneous device constraint",
            value,
            frozenset(
                {
                    "model",
                    "minimum_memory_bytes",
                    "minimum_compute_capability",
                    "allowed_peer_access_classes",
                }
            ),
        )
        compute = _strict_list(
            "minimum_compute_capability", row["minimum_compute_capability"]
        )
        return cls(
            model=_optional_text("constraint model", row["model"]),
            minimum_memory_bytes=_strict_int(
                "minimum_memory_bytes", row["minimum_memory_bytes"]
            ),
            minimum_compute_capability=tuple(
                _strict_int("minimum compute capability component", item)
                for item in compute
            ),
            allowed_peer_access_classes=tuple(
                _strict_text("allowed peer-access class", item)
                for item in _strict_list(
                    "allowed_peer_access_classes",
                    row["allowed_peer_access_classes"],
                )
            ),
        )


@dataclass(frozen=True)
class PoolResourceClaim:
    """Generic resource request independent of a physical GPU assignment."""

    gang_shape: GangShape
    exact_gpu_uuids: tuple[str, ...]
    homogeneous: HomogeneousDeviceConstraint
    allowed_topology_groups: tuple[str, ...]
    allowed_fabrics: tuple[str, ...]
    same_host: bool
    exclusive_gpu: bool
    exclusive_host: bool
    cpu_cores: int
    numa_nodes: tuple[int, ...]
    ram_bytes: int
    disk_io_class: str
    network_class: str
    port_count: int
    exact_ports: tuple[int, ...]
    cache_root: str
    evidence_root: str
    workload_class: WorkloadClass
    interference_class: str
    load_thermal_power_envelope: str
    estimated_duration_seconds: float
    estimated_gpu_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.gang_shape, GangShape):
            raise TypeError("gang_shape must be a GangShape")
        if self.exact_gpu_uuids:
            if len(self.exact_gpu_uuids) != self.gang_shape.gpu_count or len(
                set(self.exact_gpu_uuids)
            ) != len(self.exact_gpu_uuids):
                raise ValueError("exact GPU UUIDs must cover the complete gang")
            for uuid in self.exact_gpu_uuids:
                _require_text("exact GPU UUID", uuid)
        if not isinstance(self.homogeneous, HomogeneousDeviceConstraint):
            raise TypeError("homogeneous must be a HomogeneousDeviceConstraint")
        if len(set(self.allowed_topology_groups)) != len(self.allowed_topology_groups):
            raise ValueError("allowed topology groups must be unique")
        if len(set(self.allowed_fabrics)) != len(self.allowed_fabrics):
            raise ValueError("allowed fabrics must be unique")
        if tuple(sorted(self.allowed_topology_groups)) != self.allowed_topology_groups:
            raise ValueError("allowed topology groups must be canonically sorted")
        if tuple(sorted(self.allowed_fabrics)) != self.allowed_fabrics:
            raise ValueError("allowed fabrics must be canonically sorted")
        for value in self.allowed_topology_groups:
            _require_text("allowed topology group", value)
        for value in self.allowed_fabrics:
            _require_text("allowed fabric", value)
        if self.same_host is not True:
            raise ValueError("this release supports same-host resource claims only")
        if not isinstance(self.exclusive_gpu, bool) or not isinstance(
            self.exclusive_host, bool
        ):
            raise TypeError("exclusive flags must be booleans")
        for name in ("cpu_cores", "ram_bytes", "port_count"):
            value = getattr(self, name)
            minimum = 1 if name in {"cpu_cores", "port_count"} else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} is outside its supported range")
        if len(set(self.numa_nodes)) != len(self.numa_nodes) or any(
            isinstance(node, bool) or not isinstance(node, int) or node < 0
            for node in self.numa_nodes
        ):
            raise ValueError("NUMA nodes must be unique non-negative integers")
        if tuple(sorted(self.numa_nodes)) != self.numa_nodes:
            raise ValueError("NUMA nodes must be canonically sorted")
        if self.exact_ports:
            if len(self.exact_ports) != self.port_count:
                raise ValueError("exact ports must cover port_count")
            if len(set(self.exact_ports)) != len(self.exact_ports):
                raise ValueError("exact ports must be unique")
            if any(port < 1024 or port > 65_535 for port in self.exact_ports):
                raise ValueError("exact ports must be in [1024, 65535]")
        for name in (
            "disk_io_class",
            "network_class",
            "cache_root",
            "evidence_root",
            "interference_class",
            "load_thermal_power_envelope",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.workload_class, WorkloadClass):
            raise TypeError("workload_class must be a WorkloadClass")
        _require_positive_finite(
            "estimated_duration_seconds", self.estimated_duration_seconds
        )
        _require_positive_finite("estimated_gpu_seconds", self.estimated_gpu_seconds)
        expected_gpu_seconds = (
            self.estimated_duration_seconds * self.gang_shape.gpu_count
        )
        if not math.isclose(
            self.estimated_gpu_seconds,
            expected_gpu_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "estimated_gpu_seconds must equal duration times requested GPUs"
            )
        if self.workload_class in _EXCLUSIVE_HOST_CLASSES and not self.exclusive_host:
            raise ValueError(
                "profile/download/compile claims must request exclusive-host access"
            )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    @property
    def contention_class(self) -> str:
        return content_sha256(
            {
                "cpu_cores": self.cpu_cores,
                "numa_nodes": self.numa_nodes,
                "disk_io_class": self.disk_io_class,
                "network_class": self.network_class,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gang_shape": self.gang_shape.to_dict(),
            "exact_gpu_uuids": list(self.exact_gpu_uuids),
            "homogeneous": self.homogeneous.to_dict(),
            "allowed_topology_groups": list(self.allowed_topology_groups),
            "allowed_fabrics": list(self.allowed_fabrics),
            "same_host": self.same_host,
            "exclusive_gpu": self.exclusive_gpu,
            "exclusive_host": self.exclusive_host,
            "cpu_cores": self.cpu_cores,
            "numa_nodes": list(self.numa_nodes),
            "ram_bytes": self.ram_bytes,
            "disk_io_class": self.disk_io_class,
            "network_class": self.network_class,
            "port_count": self.port_count,
            "exact_ports": list(self.exact_ports),
            "cache_root": self.cache_root,
            "evidence_root": self.evidence_root,
            "workload_class": self.workload_class.value,
            "interference_class": self.interference_class,
            "load_thermal_power_envelope": self.load_thermal_power_envelope,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "estimated_gpu_seconds": self.estimated_gpu_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> PoolResourceClaim:
        fields = frozenset(
            {
                "gang_shape",
                "exact_gpu_uuids",
                "homogeneous",
                "allowed_topology_groups",
                "allowed_fabrics",
                "same_host",
                "exclusive_gpu",
                "exclusive_host",
                "cpu_cores",
                "numa_nodes",
                "ram_bytes",
                "disk_io_class",
                "network_class",
                "port_count",
                "exact_ports",
                "cache_root",
                "evidence_root",
                "workload_class",
                "interference_class",
                "load_thermal_power_envelope",
                "estimated_duration_seconds",
                "estimated_gpu_seconds",
            }
        )
        row = _strict_object("pool resource claim", value, fields)
        return cls(
            gang_shape=GangShape.from_dict(row["gang_shape"]),
            exact_gpu_uuids=tuple(
                _strict_text("exact GPU UUID", item)
                for item in _strict_list("exact_gpu_uuids", row["exact_gpu_uuids"])
            ),
            homogeneous=HomogeneousDeviceConstraint.from_dict(row["homogeneous"]),
            allowed_topology_groups=tuple(
                _strict_text("allowed topology group", item)
                for item in _strict_list(
                    "allowed_topology_groups", row["allowed_topology_groups"]
                )
            ),
            allowed_fabrics=tuple(
                _strict_text("allowed fabric", item)
                for item in _strict_list("allowed_fabrics", row["allowed_fabrics"])
            ),
            same_host=_strict_bool("same_host", row["same_host"]),
            exclusive_gpu=_strict_bool("exclusive_gpu", row["exclusive_gpu"]),
            exclusive_host=_strict_bool("exclusive_host", row["exclusive_host"]),
            cpu_cores=_strict_int("cpu_cores", row["cpu_cores"]),
            numa_nodes=tuple(
                _strict_int("NUMA node", item)
                for item in _strict_list("numa_nodes", row["numa_nodes"])
            ),
            ram_bytes=_strict_int("ram_bytes", row["ram_bytes"]),
            disk_io_class=_strict_text("disk_io_class", row["disk_io_class"]),
            network_class=_strict_text("network_class", row["network_class"]),
            port_count=_strict_int("port_count", row["port_count"]),
            exact_ports=tuple(
                _strict_int("exact port", item)
                for item in _strict_list("exact_ports", row["exact_ports"])
            ),
            cache_root=_strict_text("cache_root", row["cache_root"]),
            evidence_root=_strict_text("evidence_root", row["evidence_root"]),
            workload_class=WorkloadClass(
                _strict_text("workload_class", row["workload_class"])
            ),
            interference_class=_strict_text(
                "interference_class", row["interference_class"]
            ),
            load_thermal_power_envelope=_strict_text(
                "load_thermal_power_envelope",
                row["load_thermal_power_envelope"],
            ),
            estimated_duration_seconds=_strict_float(
                "estimated_duration_seconds", row["estimated_duration_seconds"]
            ),
            estimated_gpu_seconds=_strict_float(
                "estimated_gpu_seconds", row["estimated_gpu_seconds"]
            ),
        )


_CELL_IDENTITY_FIELDS = frozenset(
    {
        "experiment",
        "model",
        "backend",
        "task",
        "method",
        "scope",
        "rank",
        "alpha_over_rank",
        "optimizer",
        "learning_rate",
        "schedule",
        "context",
        "regime",
        "width",
        "arrival",
        "slo",
        "cohort",
        "topology",
        "seed",
        "block",
        "gpu_uuids",
        "parameterization",
        "variant",
        "concurrency",
        "load_factor",
        "cohort_count",
    }
)


def _cell_identity_to_dict(identity: CellIdentity) -> dict[str, Any]:
    return {
        "experiment": identity.experiment,
        "model": identity.model,
        "backend": identity.backend,
        "task": identity.task,
        "method": identity.method,
        "scope": identity.scope,
        "rank": identity.rank,
        "alpha_over_rank": identity.alpha_over_rank,
        "optimizer": identity.optimizer,
        "learning_rate": identity.learning_rate,
        "schedule": identity.schedule,
        "context": identity.context,
        "regime": identity.regime,
        "width": identity.width,
        "arrival": identity.arrival,
        "slo": identity.slo,
        "cohort": identity.cohort,
        "topology": identity.topology,
        "seed": identity.seed,
        "block": identity.block,
        "gpu_uuids": list(identity.gpu_uuids),
        "parameterization": identity.parameterization,
        "variant": identity.variant,
        "concurrency": identity.concurrency,
        "load_factor": identity.load_factor,
        "cohort_count": identity.cohort_count,
    }


def _cell_identity_from_dict(value: object) -> CellIdentity:
    row = _strict_object("cell identity", value, _CELL_IDENTITY_FIELDS)
    return CellIdentity(
        experiment=_strict_text("experiment", row["experiment"]),
        model=_strict_text("model", row["model"]),
        backend=_strict_text("backend", row["backend"]),
        task=_strict_text("task", row["task"]),
        method=_strict_text("method", row["method"]),
        scope=_optional_text("scope", row["scope"]),
        rank=_optional_int("rank", row["rank"]),
        alpha_over_rank=_optional_float("alpha_over_rank", row["alpha_over_rank"]),
        optimizer=_optional_text("optimizer", row["optimizer"]),
        learning_rate=_optional_float("learning_rate", row["learning_rate"]),
        schedule=_optional_text("schedule", row["schedule"]),
        context=_optional_int("context", row["context"]),
        regime=_strict_text("regime", row["regime"]),
        width=_optional_int("width", row["width"]),
        arrival=_strict_text("arrival", row["arrival"]),
        slo=_strict_text("slo", row["slo"]),
        cohort=_strict_text("cohort", row["cohort"]),
        topology=_strict_text("topology", row["topology"]),
        seed=_strict_int("seed", row["seed"]),
        block=_strict_int("block", row["block"]),
        gpu_uuids=tuple(
            _strict_text("cell GPU UUID", item)
            for item in _strict_list("gpu_uuids", row["gpu_uuids"])
        ),
        parameterization=_strict_text("parameterization", row["parameterization"]),
        variant=_strict_text("variant", row["variant"]),
        concurrency=_optional_int("concurrency", row["concurrency"]),
        load_factor=_optional_float("load_factor", row["load_factor"]),
        cohort_count=_strict_int("cohort_count", row["cohort_count"]),
    )


def _resource_claim_to_dict(claim: ResourceClaim) -> dict[str, Any]:
    return {
        "gpu_uuids": list(claim.gpu_uuids),
        "ports": list(claim.ports),
        "cache_root": claim.cache_root,
        "evidence_root": claim.evidence_root,
        "workload_class": claim.workload_class.value,
    }


def _resource_claim_from_dict(value: object) -> ResourceClaim:
    row = _strict_object(
        "registered resource claim",
        value,
        frozenset(
            {"gpu_uuids", "ports", "cache_root", "evidence_root", "workload_class"}
        ),
    )
    return ResourceClaim(
        gpu_uuids=tuple(
            _strict_text("registered GPU UUID", item)
            for item in _strict_list("gpu_uuids", row["gpu_uuids"])
        ),
        ports=tuple(
            _strict_int("registered port", item)
            for item in _strict_list("ports", row["ports"])
        ),
        cache_root=_strict_text("cache_root", row["cache_root"]),
        evidence_root=_strict_text("evidence_root", row["evidence_root"]),
        workload_class=WorkloadClass(
            _strict_text("workload_class", row["workload_class"])
        ),
    )


def _experiment_cell_to_dict(cell: ExperimentCell) -> dict[str, Any]:
    return {
        "identity": _cell_identity_to_dict(cell.identity),
        "resources": _resource_claim_to_dict(cell.resources),
        "status": cell.status.value,
        "reason_code": cell.reason_code,
        "reason": cell.reason,
    }


def _experiment_cell_from_dict(value: object) -> ExperimentCell:
    row = _strict_object(
        "experiment cell",
        value,
        frozenset({"identity", "resources", "status", "reason_code", "reason"}),
    )
    return ExperimentCell(
        identity=_cell_identity_from_dict(row["identity"]),
        resources=_resource_claim_from_dict(row["resources"]),
        status=CellStatus(_strict_text("cell status", row["status"])),
        reason_code=_strict_text("reason_code", row["reason_code"]),
        reason=_strict_text("reason", row["reason"]),
    )


@dataclass(frozen=True)
class PoolWorkItem:
    """One registry cell plus its physical resource request."""

    cell: ExperimentCell
    claim: PoolResourceClaim
    affinity_key: str | None = None
    serial_group_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cell, ExperimentCell):
            raise TypeError("cell must be an ExperimentCell")
        if not isinstance(self.claim, PoolResourceClaim):
            raise TypeError("claim must be a PoolResourceClaim")
        if self.affinity_key is not None:
            _require_sha256("affinity_key", self.affinity_key)
        if self.serial_group_key is not None:
            _require_sha256("serial_group_key", self.serial_group_key)

    @property
    def item_id(self) -> str:
        return self.cell.cell_id

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": _experiment_cell_to_dict(self.cell),
            "cell_sha256": self.cell.sha256,
            "claim": self.claim.to_dict(),
            "affinity_key": self.affinity_key,
            "serial_group_key": self.serial_group_key,
        }

    @classmethod
    def from_dict(cls, value: object) -> PoolWorkItem:
        row = _strict_object(
            "pool work item",
            value,
            frozenset(
                {
                    "cell",
                    "cell_sha256",
                    "claim",
                    "affinity_key",
                    "serial_group_key",
                }
            ),
        )
        cell = _experiment_cell_from_dict(row["cell"])
        declared_cell_sha256 = _strict_text("cell_sha256", row["cell_sha256"])
        _require_sha256("cell_sha256", declared_cell_sha256)
        if declared_cell_sha256 != cell.sha256:
            raise ValueError("work-item cell SHA-256 mismatch")
        return cls(
            cell=cell,
            claim=PoolResourceClaim.from_dict(row["claim"]),
            affinity_key=_optional_text("affinity_key", row["affinity_key"]),
            serial_group_key=_optional_text(
                "serial_group_key", row["serial_group_key"]
            ),
        )


@dataclass(frozen=True)
class InterferenceRule:
    """Exact calibrated permission for one co-run class and cardinality."""

    hardware_envelope_sha256: str
    workload_class: WorkloadClass
    co_run_signature: str
    simultaneous_jobs: int
    gang_shape: str
    load_thermal_power_envelope: str
    contention_class: str
    evidence_sha256: str
    status: str = "PASS"

    def __post_init__(self) -> None:
        _require_sha256("hardware_envelope_sha256", self.hardware_envelope_sha256)
        if not isinstance(self.workload_class, WorkloadClass):
            raise TypeError("workload_class must be a WorkloadClass")
        for name in (
            "co_run_signature",
            "gang_shape",
            "load_thermal_power_envelope",
            "contention_class",
        ):
            _require_text(name, getattr(self, name))
        if (
            isinstance(self.simultaneous_jobs, bool)
            or not isinstance(self.simultaneous_jobs, int)
            or self.simultaneous_jobs < 2
        ):
            raise ValueError("interference rules require at least two jobs")
        _require_sha256("interference evidence_sha256", self.evidence_sha256)
        if self.status not in {"PASS", "FAIL"}:
            raise ValueError("interference rule status must be PASS or FAIL")

    @property
    def key(self) -> tuple[str, str, str, int, str, str, str]:
        return (
            self.hardware_envelope_sha256,
            self.workload_class.value,
            self.co_run_signature,
            self.simultaneous_jobs,
            self.gang_shape,
            self.load_thermal_power_envelope,
            self.contention_class,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "workload_class": self.workload_class.value,
            "co_run_signature": self.co_run_signature,
            "simultaneous_jobs": self.simultaneous_jobs,
            "gang_shape": self.gang_shape,
            "load_thermal_power_envelope": self.load_thermal_power_envelope,
            "contention_class": self.contention_class,
            "evidence_sha256": self.evidence_sha256,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: object) -> InterferenceRule:
        row = _strict_object(
            "interference rule",
            value,
            frozenset(
                {
                    "hardware_envelope_sha256",
                    "workload_class",
                    "co_run_signature",
                    "simultaneous_jobs",
                    "gang_shape",
                    "load_thermal_power_envelope",
                    "contention_class",
                    "evidence_sha256",
                    "status",
                }
            ),
        )
        return cls(
            hardware_envelope_sha256=_strict_text(
                "hardware_envelope_sha256", row["hardware_envelope_sha256"]
            ),
            workload_class=WorkloadClass(
                _strict_text("workload_class", row["workload_class"])
            ),
            co_run_signature=_strict_text("co_run_signature", row["co_run_signature"]),
            simultaneous_jobs=_strict_int(
                "simultaneous_jobs", row["simultaneous_jobs"]
            ),
            gang_shape=_strict_text("gang_shape", row["gang_shape"]),
            load_thermal_power_envelope=_strict_text(
                "load_thermal_power_envelope",
                row["load_thermal_power_envelope"],
            ),
            contention_class=_strict_text("contention_class", row["contention_class"]),
            evidence_sha256=_strict_text("evidence_sha256", row["evidence_sha256"]),
            status=_strict_text("interference status", row["status"]),
        )

    @classmethod
    def for_claim(
        cls,
        *,
        device: GpuDevice,
        claim: PoolResourceClaim,
        simultaneous_jobs: int,
        evidence_sha256: str,
        status: str = "PASS",
    ) -> InterferenceRule:
        """Build an exact homogeneous co-run rule from one registered claim."""

        return cls(
            hardware_envelope_sha256=device.hardware_envelope_sha256,
            workload_class=claim.workload_class,
            co_run_signature=claim.interference_class,
            simultaneous_jobs=simultaneous_jobs,
            gang_shape=claim.gang_shape.signature,
            load_thermal_power_envelope=claim.load_thermal_power_envelope,
            contention_class=claim.contention_class,
            evidence_sha256=evidence_sha256,
            status=status,
        )


@dataclass(frozen=True)
class InterferenceEnvelope:
    """Sealed exact co-tenancy permissions; absent rules deny concurrency."""

    schema_version: int
    rules: tuple[InterferenceRule, ...]
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only interference-envelope schema version 1 is supported")
        _require_sha256("source_receipt_sha256", self.source_receipt_sha256)
        keys = tuple(rule.key for rule in self.rules)
        if len(keys) != len(set(keys)):
            raise ValueError("interference envelope contains duplicate rule keys")
        if tuple(sorted(keys)) != keys:
            raise ValueError("interference rules must be in canonical key order")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rules": [rule.to_dict() for rule in self.rules],
            "source_receipt_sha256": self.source_receipt_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> InterferenceEnvelope:
        row = _strict_object(
            "interference envelope",
            value,
            frozenset({"schema_version", "rules", "source_receipt_sha256"}),
        )
        envelope = cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            rules=tuple(
                InterferenceRule.from_dict(item)
                for item in _strict_list("rules", row["rules"])
            ),
            source_receipt_sha256=_strict_text(
                "source_receipt_sha256", row["source_receipt_sha256"]
            ),
        )
        _validate_sidecar(
            artifact_kind="interference_envelope.v1",
            artifact_sha256=envelope.sha256,
            sidecar=sidecar,
        )
        return envelope

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "interference_envelope.v1", self.sha256)

    @classmethod
    def serial(cls, *, source_receipt_sha256: str) -> InterferenceEnvelope:
        return cls(
            schema_version=1,
            rules=(),
            source_receipt_sha256=source_receipt_sha256,
        )

    def permits(
        self,
        assignments: Sequence[GpuAssignment],
        *,
        inventory: GpuInventory,
    ) -> bool:
        if len(assignments) <= 1:
            return True
        claims = tuple(assignment.work_item.claim for assignment in assignments)
        if len({claim.workload_class for claim in claims}) != 1:
            return False
        if any(claim.exclusive_host for claim in claims):
            return False
        workload_class = claims[0].workload_class
        if workload_class in _EXCLUSIVE_HOST_CLASSES:
            return False
        if workload_class is WorkloadClass.HEADLINE and any(
            claim.workload_class is not WorkloadClass.HEADLINE for claim in claims
        ):
            return False
        co_run_classes = {claim.interference_class for claim in claims}
        gang_shapes = {claim.gang_shape.signature for claim in claims}
        load_envelopes = {claim.load_thermal_power_envelope for claim in claims}
        contention_classes = {claim.contention_class for claim in claims}
        device_envelopes = {
            inventory.device(uuid).hardware_envelope_sha256
            for assignment in assignments
            for uuid in assignment.gpu_uuids
        }
        if any(
            len(values) != 1
            for values in (
                co_run_classes,
                gang_shapes,
                load_envelopes,
                contention_classes,
                device_envelopes,
            )
        ):
            return False
        key = (
            next(iter(device_envelopes)),
            workload_class.value,
            next(iter(co_run_classes)),
            len(assignments),
            next(iter(gang_shapes)),
            next(iter(load_envelopes)),
            next(iter(contention_classes)),
        )
        return any(rule.key == key and rule.status == "PASS" for rule in self.rules)


@dataclass(frozen=True)
class GpuAssignment:
    """A complete atomic placement of one work item."""

    work_item: PoolWorkItem
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    ports: tuple[int, ...]

    def __post_init__(self) -> None:
        claim = self.work_item.claim
        if len(self.gpu_uuids) != claim.gang_shape.gpu_count or len(
            set(self.gpu_uuids)
        ) != len(self.gpu_uuids):
            raise ValueError("assignment must bind every gang rank to one unique GPU")
        flattened = tuple(uuid for group in self.rank_groups for uuid in group)
        if flattened != self.gpu_uuids:
            raise ValueError("rank groups must exactly partition ordered GPU UUIDs")
        if len(self.rank_groups) != claim.gang_shape.data_parallel_size or any(
            len(group) != claim.gang_shape.tensor_parallel_size
            for group in self.rank_groups
        ):
            raise ValueError("rank groups do not match the TP/DP gang shape")
        if len(self.ports) != claim.port_count or len(set(self.ports)) != len(
            self.ports
        ):
            raise ValueError("assignment ports must exactly satisfy port_count")
        if any(port < 1024 or port > 65_535 for port in self.ports):
            raise ValueError("assignment ports must be in [1024, 65535]")
        if claim.exact_gpu_uuids and self.gpu_uuids != claim.exact_gpu_uuids:
            raise ValueError("assignment differs from exact requested GPU UUIDs")
        if claim.exact_ports and self.ports != claim.exact_ports:
            raise ValueError("assignment differs from exact requested ports")

    @cached_property
    def assignment_id(self) -> str:
        return content_sha256(self.to_dict())

    @cached_property
    def sha256(self) -> str:
        return self.assignment_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item": self.work_item.to_dict(),
            "work_item_sha256": self.work_item.sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
            "ports": list(self.ports),
        }

    @classmethod
    def from_dict(cls, value: object) -> GpuAssignment:
        row = _strict_object(
            "GPU assignment",
            value,
            frozenset(
                {
                    "work_item",
                    "work_item_sha256",
                    "gpu_uuids",
                    "rank_groups",
                    "ports",
                }
            ),
        )
        work_item = PoolWorkItem.from_dict(row["work_item"])
        work_item_sha256 = _strict_text("work_item_sha256", row["work_item_sha256"])
        _require_sha256("work_item_sha256", work_item_sha256)
        if work_item_sha256 != work_item.sha256:
            raise ValueError("assignment work-item SHA-256 mismatch")
        rank_groups = tuple(
            tuple(
                _strict_text("rank GPU UUID", uuid)
                for uuid in _strict_list("rank group", group)
            )
            for group in _strict_list("rank_groups", row["rank_groups"])
        )
        return cls(
            work_item=work_item,
            gpu_uuids=tuple(
                _strict_text("assigned GPU UUID", item)
                for item in _strict_list("gpu_uuids", row["gpu_uuids"])
            ),
            rank_groups=rank_groups,
            ports=tuple(
                _strict_int("assigned port", item)
                for item in _strict_list("ports", row["ports"])
            ),
        )


@dataclass(frozen=True)
class GpuDispatchWave:
    """One pre-frozen wave; runtime completion cannot change later co-tenancy."""

    wave_index: int
    assignments: tuple[GpuAssignment, ...]
    interference_envelope_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.wave_index, bool)
            or not isinstance(self.wave_index, int)
            or self.wave_index < 0
        ):
            raise ValueError("wave_index must be a non-negative integer")
        if not self.assignments:
            raise ValueError("a dispatch wave cannot be empty")
        _require_sha256(
            "interference_envelope_sha256", self.interference_envelope_sha256
        )
        assignment_ids = tuple(row.assignment_id for row in self.assignments)
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("dispatch wave contains duplicate assignments")
        used_gpus: set[str] = set()
        used_ports: set[int] = set()
        cache_roots: set[str] = set()
        evidence_roots: set[str] = set()
        serial_groups: set[str] = set()
        for assignment in self.assignments:
            claim = assignment.work_item.claim
            if used_gpus & set(assignment.gpu_uuids):
                raise ValueError("dispatch wave overlaps a GPU")
            if used_ports & set(assignment.ports):
                raise ValueError("dispatch wave overlaps a port")
            if claim.cache_root in cache_roots:
                raise ValueError("dispatch wave overlaps a cache writer")
            if claim.evidence_root in evidence_roots:
                raise ValueError("dispatch wave overlaps an evidence root")
            if (
                assignment.work_item.serial_group_key is not None
                and assignment.work_item.serial_group_key in serial_groups
            ):
                raise ValueError("serial scientific group cannot share a wave")
            used_gpus.update(assignment.gpu_uuids)
            used_ports.update(assignment.ports)
            cache_roots.add(claim.cache_root)
            evidence_roots.add(claim.evidence_root)
            if assignment.work_item.serial_group_key is not None:
                serial_groups.add(assignment.work_item.serial_group_key)
        if len(self.assignments) > 1 and any(
            assignment.work_item.claim.exclusive_host for assignment in self.assignments
        ):
            raise ValueError("exclusive-host work cannot share a dispatch wave")

    @property
    def estimated_wall_seconds(self) -> float:
        return max(
            assignment.work_item.claim.estimated_duration_seconds
            for assignment in self.assignments
        )

    @property
    def estimated_gpu_seconds(self) -> float:
        return sum(
            assignment.work_item.claim.estimated_gpu_seconds
            for assignment in self.assignments
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_index": self.wave_index,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "assignment_sha256": [assignment.sha256 for assignment in self.assignments],
            "interference_envelope_sha256": self.interference_envelope_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> GpuDispatchWave:
        row = _strict_object(
            "GPU dispatch wave",
            value,
            frozenset(
                {
                    "wave_index",
                    "assignments",
                    "assignment_sha256",
                    "interference_envelope_sha256",
                }
            ),
        )
        assignments = tuple(
            GpuAssignment.from_dict(item)
            for item in _strict_list("assignments", row["assignments"])
        )
        declared = tuple(
            _strict_text("assignment_sha256", item)
            for item in _strict_list("assignment_sha256", row["assignment_sha256"])
        )
        if declared != tuple(assignment.sha256 for assignment in assignments):
            raise ValueError("dispatch-wave assignment SHA-256 list mismatch")
        return cls(
            wave_index=_strict_int("wave_index", row["wave_index"]),
            assignments=assignments,
            interference_envelope_sha256=_strict_text(
                "interference_envelope_sha256",
                row["interference_envelope_sha256"],
            ),
        )


@dataclass(frozen=True)
class GpuDispatchPlan:
    """Complete immutable dispatch schedule and cost estimate."""

    schema_version: int
    registry_sha256: str
    inventory_sha256: str
    receipts_sha256: str
    interference_envelope_sha256: str
    budget_sha256_by_cell: tuple[tuple[str, str], ...]
    seed: int
    waves: tuple[GpuDispatchWave, ...]
    completed_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only GPU dispatch-plan schema version 1 is supported")
        for name in (
            "registry_sha256",
            "inventory_sha256",
            "receipts_sha256",
            "interference_envelope_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if tuple(wave.wave_index for wave in self.waves) != tuple(
            range(len(self.waves))
        ):
            raise ValueError("dispatch wave indexes must be contiguous")
        assignment_ids = tuple(
            assignment.assignment_id
            for wave in self.waves
            for assignment in wave.assignments
        )
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("dispatch plan contains duplicate assignments")
        cell_ids = tuple(
            assignment.work_item.item_id
            for wave in self.waves
            for assignment in wave.assignments
        )
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("dispatch plan contains duplicate cells")
        budget_cells = tuple(cell_id for cell_id, _ in self.budget_sha256_by_cell)
        if self.budget_sha256_by_cell != tuple(
            sorted(self.budget_sha256_by_cell)
        ) or len(budget_cells) != len(set(budget_cells)):
            raise ValueError("dispatch budget bindings must be canonical and unique")
        if self.budget_sha256_by_cell and set(budget_cells) != set(cell_ids):
            raise ValueError(
                "dispatch budget bindings must cover every scheduled cell exactly"
            )
        for cell_id, budget_sha256 in self.budget_sha256_by_cell:
            _require_sha256("dispatch budget cell ID", cell_id)
            _require_sha256("dispatch ExperimentBudget SHA-256", budget_sha256)
        if len(self.completed_cell_ids) != len(set(self.completed_cell_ids)):
            raise ValueError("completed cell IDs must be unique")
        if any(not _is_sha256(cell_id) for cell_id in self.completed_cell_ids):
            raise ValueError("completed cell IDs must be lower-case SHA-256")
        if tuple(sorted(self.completed_cell_ids)) != self.completed_cell_ids:
            raise ValueError("completed cell IDs must be canonically sorted")
        if set(self.completed_cell_ids) & set(cell_ids):
            raise ValueError("completed cells cannot remain in dispatch waves")

    @property
    def estimated_wall_seconds(self) -> float:
        return sum(wave.estimated_wall_seconds for wave in self.waves)

    @property
    def estimated_gpu_seconds(self) -> float:
        return sum(wave.estimated_gpu_seconds for wave in self.waves)

    @property
    def estimated_gpu_hours(self) -> float:
        return self.estimated_gpu_seconds / 3600.0

    @property
    def scientific_budget_bound(self) -> bool:
        """Whether every assignment is bound to its full ExperimentBudget."""

        return bool(self.budget_sha256_by_cell) or not self.waves

    def budget_sha256_for(self, cell_id: str) -> str:
        matches = tuple(
            digest
            for bound_cell_id, digest in self.budget_sha256_by_cell
            if bound_cell_id == cell_id
        )
        if len(matches) != 1:
            raise ValueError("dispatch cell lacks one exact ExperimentBudget binding")
        return matches[0]

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the full executable plan, not a digest-only summary."""

        return {
            "schema_version": self.schema_version,
            "registry_sha256": self.registry_sha256,
            "inventory_sha256": self.inventory_sha256,
            "receipts_sha256": self.receipts_sha256,
            "interference_envelope_sha256": self.interference_envelope_sha256,
            "budget_sha256_by_cell": [
                {
                    "cell_id": cell_id,
                    "experiment_budget_sha256": budget_sha256,
                }
                for cell_id, budget_sha256 in self.budget_sha256_by_cell
            ],
            "scientific_budget_bound": self.scientific_budget_bound,
            "seed": self.seed,
            "waves": [wave.to_dict() for wave in self.waves],
            "wave_sha256": [wave.sha256 for wave in self.waves],
            "completed_cell_ids": list(self.completed_cell_ids),
            "estimated_wall_seconds": self.estimated_wall_seconds,
            "estimated_gpu_seconds": self.estimated_gpu_seconds,
            "estimated_gpu_hours": self.estimated_gpu_hours,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        planning_context: GpuDispatchPlanningContext,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> GpuDispatchPlan:
        row = _strict_object(
            "GPU dispatch plan",
            value,
            frozenset(
                {
                    "schema_version",
                    "registry_sha256",
                    "inventory_sha256",
                    "receipts_sha256",
                    "interference_envelope_sha256",
                    "budget_sha256_by_cell",
                    "scientific_budget_bound",
                    "seed",
                    "waves",
                    "wave_sha256",
                    "completed_cell_ids",
                    "estimated_wall_seconds",
                    "estimated_gpu_seconds",
                    "estimated_gpu_hours",
                }
            ),
        )
        waves = tuple(
            GpuDispatchWave.from_dict(item)
            for item in _strict_list("waves", row["waves"])
        )
        declared_waves = tuple(
            _strict_text("wave_sha256", item)
            for item in _strict_list("wave_sha256", row["wave_sha256"])
        )
        if declared_waves != tuple(wave.sha256 for wave in waves):
            raise ValueError("dispatch-plan wave SHA-256 list mismatch")
        budget_rows = _strict_list(
            "budget_sha256_by_cell", row["budget_sha256_by_cell"]
        )
        budget_bindings = tuple(
            (
                _strict_text("budget cell_id", budget_row["cell_id"]),
                _strict_text(
                    "experiment_budget_sha256",
                    budget_row["experiment_budget_sha256"],
                ),
            )
            for budget_row in (
                _strict_object(
                    "dispatch budget binding",
                    value,
                    frozenset({"cell_id", "experiment_budget_sha256"}),
                )
                for value in budget_rows
            )
        )
        plan = cls(
            schema_version=_strict_int("schema_version", row["schema_version"]),
            registry_sha256=_strict_text("registry_sha256", row["registry_sha256"]),
            inventory_sha256=_strict_text("inventory_sha256", row["inventory_sha256"]),
            receipts_sha256=_strict_text("receipts_sha256", row["receipts_sha256"]),
            interference_envelope_sha256=_strict_text(
                "interference_envelope_sha256",
                row["interference_envelope_sha256"],
            ),
            budget_sha256_by_cell=budget_bindings,
            seed=_strict_int("seed", row["seed"]),
            waves=waves,
            completed_cell_ids=tuple(
                _strict_text("completed cell ID", item)
                for item in _strict_list(
                    "completed_cell_ids", row["completed_cell_ids"]
                )
            ),
        )
        if (
            _strict_bool("scientific_budget_bound", row["scientific_budget_bound"])
            is not plan.scientific_budget_bound
        ):
            raise ValueError("dispatch-plan budget-bound status mismatch")
        estimates = (
            (
                "estimated_wall_seconds",
                plan.estimated_wall_seconds,
                _strict_float("estimated_wall_seconds", row["estimated_wall_seconds"]),
            ),
            (
                "estimated_gpu_seconds",
                plan.estimated_gpu_seconds,
                _strict_float("estimated_gpu_seconds", row["estimated_gpu_seconds"]),
            ),
            (
                "estimated_gpu_hours",
                plan.estimated_gpu_hours,
                _strict_float("estimated_gpu_hours", row["estimated_gpu_hours"]),
            ),
        )
        if any(
            not math.isclose(expected, declared, rel_tol=0.0, abs_tol=1e-12)
            for _, expected, declared in estimates
        ):
            raise ValueError("dispatch-plan cost summary mismatch")
        validate_dispatch_plan_for_planning(plan, planning_context=planning_context)
        _validate_sidecar(
            artifact_kind="gpu_dispatch_plan.v1",
            artifact_sha256=plan.sha256,
            sidecar=sidecar,
        )
        return plan

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "gpu_dispatch_plan.v1", self.sha256)


class CapabilityRejectionError(RuntimeError):
    """Raised before dispatch when an exact claim has no valid gang placement."""


def _identity_without_method_or_gpu(cell: ExperimentCell) -> Mapping[str, Any]:
    identity = cell.identity
    return {
        "experiment": identity.experiment,
        "model": identity.model,
        "backend": identity.backend,
        "task": identity.task,
        "scope": identity.scope,
        "rank": identity.rank,
        "alpha_over_rank": identity.alpha_over_rank,
        "optimizer": identity.optimizer,
        "learning_rate": identity.learning_rate,
        "schedule": identity.schedule,
        "context": identity.context,
        "regime": identity.regime,
        "width": identity.width,
        "arrival": identity.arrival,
        "slo": identity.slo,
        "cohort": identity.cohort,
        "topology": identity.topology,
        "seed": identity.seed,
        "block": identity.block,
        "parameterization": identity.parameterization,
        "variant": identity.variant,
        "concurrency": identity.concurrency,
        "load_factor": identity.load_factor,
        "cohort_count": identity.cohort_count,
    }


def _confirmation_serial_key(cell: ExperimentCell) -> str:
    """Bind one paired confirmation family/block across method variants."""

    identity = cell.identity
    backend = identity.backend
    if identity.experiment == "E3b" and backend == "NONE":
        backend = "DFLASH"
    return content_sha256(
        {
            "experiment": identity.experiment,
            "model": identity.model,
            "backend_family": backend,
            "task": identity.task,
            "context": identity.context,
            "regime": identity.regime,
            "arrival": identity.arrival,
            "slo": identity.slo,
            "cohort": identity.cohort,
            "topology": identity.topology,
            "seed": identity.seed,
            "block": identity.block,
            "variant": identity.variant,
            "concurrency": identity.concurrency,
            "load_factor": identity.load_factor,
            "cohort_count": identity.cohort_count,
        }
    )


def _registry_gang_shape(cell: ExperimentCell) -> GangShape:
    topology = cell.identity.topology.lower()
    gpu_count = cell.resources.gpu_count
    if topology == "two_replica_tp1_dp2" or "two_independent_tp1" in topology:
        shape = GangShape(tensor_parallel_size=1, data_parallel_size=gpu_count)
    elif "tp2" in topology:
        shape = GangShape(tensor_parallel_size=2, data_parallel_size=gpu_count // 2)
    else:
        shape = GangShape(tensor_parallel_size=gpu_count, data_parallel_size=1)
    if shape.gpu_count != gpu_count:
        raise ValueError("registry topology and resource GPU count disagree")
    return shape


def registry_pool_work_item(
    cell: ExperimentCell,
    *,
    estimated_duration_seconds: float,
) -> PoolWorkItem:
    """Adapt a registry cell to a fungible physical-pool claim.

    The original two UUIDs remain part of ``cell.cell_id``.  They are a legacy
    registry sharding declaration, not authority to choose physical devices on
    a different inventory.  The returned dispatch assignment binds the actual
    UUIDs without mutating the cell.
    """

    _require_positive_finite("estimated_duration_seconds", estimated_duration_seconds)
    shape = _registry_gang_shape(cell)
    affinity_key: str | None = None
    serial_group_key: str | None = None
    if cell.identity.method in {"tts", "l0"} and cell.identity.experiment in {
        "E1",
        "E2",
        "E1a",
    }:
        affinity_key = content_sha256(_identity_without_method_or_gpu(cell))
    if cell.identity.experiment in {"E3b", "E5"}:
        serial_group_key = _confirmation_serial_key(cell)
        affinity_key = serial_group_key
    contention_class = content_sha256(
        {
            "cpu_cores": 1,
            "numa_nodes": (),
            "disk_io_class": "registered",
            "network_class": "registered",
        }
    )
    claim = PoolResourceClaim(
        gang_shape=shape,
        exact_gpu_uuids=(),
        homogeneous=HomogeneousDeviceConstraint(),
        allowed_topology_groups=(),
        allowed_fabrics=(),
        same_host=True,
        exclusive_gpu=True,
        exclusive_host=cell.resources.workload_class in _EXCLUSIVE_HOST_CLASSES,
        cpu_cores=1,
        numa_nodes=(),
        ram_bytes=0,
        disk_io_class="registered",
        network_class="registered",
        port_count=len(cell.resources.ports),
        exact_ports=(),
        cache_root=cell.resources.cache_root,
        evidence_root=cell.resources.evidence_root,
        workload_class=cell.resources.workload_class,
        interference_class=content_sha256(
            {
                "workload_class": cell.resources.workload_class,
                "experiment": cell.identity.experiment,
                "gang_shape": shape.signature,
                "contention_class": contention_class,
            }
        ),
        load_thermal_power_envelope="registered",
        estimated_duration_seconds=float(estimated_duration_seconds),
        estimated_gpu_seconds=float(estimated_duration_seconds * shape.gpu_count),
    )
    return PoolWorkItem(
        cell=cell,
        claim=claim,
        affinity_key=affinity_key,
        serial_group_key=serial_group_key,
    )


def _adaptive_pair_key(cell: ExperimentCell) -> str:
    return content_sha256(_identity_without_method_or_gpu(cell))


def _dispatch_order_key(
    item: PoolWorkItem, *, registry_seed: int
) -> tuple[int, int, str, str, str]:
    identity = item.cell.identity
    phase = 0 if identity.block in PILOT_BLOCKS else 1
    paired_group = item.serial_group_key or content_sha256(
        {
            **_identity_without_method_or_gpu(item.cell),
            "registry_seed": registry_seed,
        }
    )
    method_order = content_sha256(
        {
            "paired_group": paired_group,
            "method": identity.method,
            "seed": identity.seed,
        }
    )
    return phase, identity.block, paired_group, method_order, item.item_id


def _gang_load_key(
    gang: tuple[str, ...],
    *,
    device_load_seconds: Mapping[str, float],
    device_order: Mapping[str, int],
    rotation: int,
    pool_size: int,
) -> tuple[float, tuple[int, ...], tuple[str, ...]]:
    peak_load = max(device_load_seconds[uuid] for uuid in gang)
    rotated = tuple((device_order[uuid] - rotation) % pool_size for uuid in gang)
    return peak_load, rotated, gang


@dataclass(frozen=True)
class GpuPoolScheduler:
    """Pure deterministic scheduler for any same-host inventory with N >= 1."""

    registry: ExperimentRegistry
    inventory: GpuInventory
    interference_envelope: InterferenceEnvelope
    port_start: int = 24_000
    port_end: int = 65_535
    seed: int = 20260811

    def __post_init__(self) -> None:
        if not isinstance(self.registry, ExperimentRegistry):
            raise TypeError("registry must be an ExperimentRegistry")
        if not isinstance(self.inventory, GpuInventory):
            raise TypeError("inventory must be a GpuInventory")
        if not isinstance(self.interference_envelope, InterferenceEnvelope):
            raise TypeError("interference_envelope must be an InterferenceEnvelope")
        if len(self.inventory.host_ids) != 1:
            raise ValueError(
                "multi-host inventories are unsupported; use one same-host inventory"
            )
        if (
            isinstance(self.port_start, bool)
            or isinstance(self.port_end, bool)
            or not 1024 <= self.port_start <= self.port_end <= 65_535
        ):
            raise ValueError("scheduler port range is invalid")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("scheduler seed must be a non-negative integer")

    def schedule(
        self,
        *,
        budgets_by_cell_id: Mapping[str, ExperimentBudget],
        receipts: Sequence[ExperimentReceipt] = (),
        completed_cell_ids: Sequence[str] = (),
        activation_artifact: ReducerActivationArtifact | None = None,
        family_activations: Sequence[FamilyActivationArtifact] = (),
        family_power_reductions: Sequence[
            ConfirmationFamilyPowerReductionArtifact
        ] = (),
    ) -> GpuDispatchPlan:
        """Select the ready stage and freeze all physical waves before execution."""

        completed = tuple(completed_cell_ids)
        if len(completed) != len(set(completed)):
            raise ValueError("completed cell identities must be unique")
        known_ids = {cell.cell_id for cell in self.registry.cells}
        if set(completed) - known_ids:
            raise ValueError("completed cells include an identity outside the registry")
        experiment = self.registry.ready_experiment(receipts)
        if experiment is None:
            if family_activations or family_power_reductions:
                raise ValueError(
                    "confirmation family artifacts do not match a ready stage"
                )
            pending: tuple[ExperimentCell, ...] = ()
        else:
            activated = self._activated_cell_ids(
                experiment=experiment,
                receipts=receipts,
                artifact=activation_artifact,
            )
            confirmation_activated = self._activated_confirmation_cell_ids(
                experiment=experiment,
                completed=completed,
                artifacts=family_activations,
                power_reductions=family_power_reductions,
            )
            if confirmation_activated is not None:
                if activated is not None:
                    raise ValueError("stage cannot mix activation protocols")
                activated = confirmation_activated
            pending = tuple(
                cell
                for cell in self.registry.cells_for(experiment)
                if self._dispatchable(cell)
                and cell.cell_id not in completed
                and (activated is None or cell.cell_id in activated)
            )
        budgets = dict(budgets_by_cell_id)
        if set(budgets) != {cell.cell_id for cell in pending}:
            raise ValueError(
                "dispatch requires exact ExperimentBudget coverage for pending cells"
            )
        for cell in pending:
            budget = budgets[cell.cell_id]
            if (
                not isinstance(budget, ExperimentBudget)
                or budget.cell_id != cell.cell_id
                or budget.experiment != cell.identity.experiment
                or budget.method != cell.identity.method
                or budget.workload_class is not cell.resources.workload_class
                or budget.gpu_count != cell.resources.gpu_count
                or budget.topology != cell.identity.topology
                or budget.measured_gpu_ms is not None
            ):
                raise ValueError("dispatch budget differs from its registry cell")
            if budget.fixed_instance_billed_gpu_ms != budget.wall_time.scale(
                len(self.inventory.devices)
            ):
                raise ValueError(
                    "dispatch budget must bill the exact fixed GPU inventory"
                )
        work_items = tuple(
            registry_pool_work_item(
                cell,
                estimated_duration_seconds=(
                    budgets[cell.cell_id].wall_time.registered / 1000.0
                ),
            )
            for cell in pending
        )
        return self.schedule_work_items(
            work_items,
            receipts_sha256=content_sha256(
                tuple(sorted(receipt.sha256 for receipt in receipts))
            ),
            completed_cell_ids=tuple(sorted(completed)),
            budget_sha256_by_cell={
                cell_id: budget.sha256 for cell_id, budget in budgets.items()
            },
        )

    def schedule_work_items(
        self,
        work_items: Sequence[PoolWorkItem],
        *,
        receipts_sha256: str,
        completed_cell_ids: Sequence[str] = (),
        budget_sha256_by_cell: Mapping[str, str] | None = None,
    ) -> GpuDispatchPlan:
        """Freeze deterministic placements and waves for selected work.

        The low-level generic API permits an unbound diagnostic plan when
        ``budget_sha256_by_cell`` is omitted.  Such a plan can be inspected and
        round-tripped, but :func:`execute_dispatch_plan` refuses to dispatch it.
        The industrial :meth:`schedule` entry point always supplies complete
        ExperimentBudget bindings.
        """

        _require_sha256("receipts_sha256", receipts_sha256)
        items = tuple(work_items)
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("work items contain duplicate cell identities")
        budget_bindings = dict(budget_sha256_by_cell or {})
        if budget_bindings and set(budget_bindings) != {item.item_id for item in items}:
            raise ValueError(
                "dispatch budget bindings must cover every work item exactly"
            )
        for cell_id, budget_sha256 in budget_bindings.items():
            _require_sha256("dispatch budget cell ID", cell_id)
            _require_sha256("dispatch ExperimentBudget SHA-256", budget_sha256)
        if any(not self._dispatchable(item.cell) for item in items):
            raise ValueError(
                "work items contain a blocked or non-executable method/capability"
            )
        registry_ids = {cell.cell_id for cell in self.registry.cells}
        if any(item.item_id not in registry_ids for item in items):
            raise ValueError("work item is not bound to this registry")
        registry_by_id = {cell.cell_id: cell for cell in self.registry.cells}
        completed = tuple(completed_cell_ids)
        if len(completed) != len(set(completed)):
            raise ValueError("completed cell identities must be unique")
        if any(not _is_sha256(cell_id) for cell_id in completed):
            raise ValueError("completed cell identities must be lower-case SHA-256")
        if set(completed) - registry_ids:
            raise ValueError("completed cells include an identity outside the registry")
        if set(completed) & {item.item_id for item in items}:
            raise ValueError("completed cells cannot remain in the work-item set")
        for item in items:
            registered = registry_by_id[item.item_id]
            if item.cell.sha256 != registered.sha256:
                raise ValueError("work item changes its registered cell declaration")
            claim = item.claim
            if (
                claim.gang_shape != _registry_gang_shape(registered)
                or claim.port_count != len(registered.resources.ports)
                or claim.cache_root != registered.resources.cache_root
                or claim.evidence_root != registered.resources.evidence_root
                or claim.workload_class is not registered.resources.workload_class
            ):
                raise ValueError("pool claim changes a registered resource invariant")
            canonical = registry_pool_work_item(
                registered,
                estimated_duration_seconds=claim.estimated_duration_seconds,
            ).claim
            if (
                claim.exclusive_gpu != canonical.exclusive_gpu
                or claim.exclusive_host != canonical.exclusive_host
                or claim.cpu_cores != canonical.cpu_cores
                or claim.numa_nodes != canonical.numa_nodes
                or claim.ram_bytes != canonical.ram_bytes
                or claim.disk_io_class != canonical.disk_io_class
                or claim.network_class != canonical.network_class
                or claim.interference_class != canonical.interference_class
                or claim.load_thermal_power_envelope
                != canonical.load_thermal_power_envelope
            ):
                raise ValueError("pool claim changes its registered isolation class")
        ordered = tuple(
            sorted(
                items,
                key=lambda item: _dispatch_order_key(item, registry_seed=self.seed),
            )
        )
        candidates = {item.item_id: self._candidate_gangs(item) for item in ordered}
        affinity_assignments: dict[str, tuple[str, ...]] = {}
        device_load_seconds = {device.uuid: 0.0 for device in self.inventory.devices}
        placements: list[GpuAssignment] = []
        for item in ordered:
            gangs = candidates[item.item_id]
            if (
                item.affinity_key is not None
                and item.affinity_key in affinity_assignments
            ):
                bound = affinity_assignments[item.affinity_key]
                gangs = tuple(gang for gang in gangs if gang == bound)
                if not gangs:
                    raise CapabilityRejectionError(
                        "affinity group has no common capability/topology placement"
                    )
            if not gangs:
                raise CapabilityRejectionError(
                    f"cell {item.item_id} has no ready capability/topology placement"
                )
            rotation = int(
                content_sha256(
                    {
                        "seed": self.seed,
                        "affinity_key": item.affinity_key,
                        "item_id": item.item_id,
                    }
                )[:16],
                16,
            )
            device_order = {
                device.uuid: index
                for index, device in enumerate(self.inventory.devices)
            }

            selected = min(
                gangs,
                key=lambda gang: _gang_load_key(
                    gang,
                    device_load_seconds=device_load_seconds,
                    device_order=device_order,
                    rotation=rotation,
                    pool_size=len(self.inventory.devices),
                ),
            )
            if item.affinity_key is not None:
                affinity_assignments.setdefault(item.affinity_key, selected)
            for uuid in selected:
                device_load_seconds[uuid] += item.claim.estimated_duration_seconds
            rank_groups = tuple(
                tuple(
                    selected[
                        replica * item.claim.gang_shape.tensor_parallel_size + rank
                    ]
                    for rank in range(item.claim.gang_shape.tensor_parallel_size)
                )
                for replica in range(item.claim.gang_shape.data_parallel_size)
            )
            placements.append(
                GpuAssignment(
                    work_item=item,
                    gpu_uuids=selected,
                    rank_groups=rank_groups,
                    ports=self._placeholder_ports(item.claim),
                )
            )
        waves: list[list[GpuAssignment]] = []
        for placement in placements:
            placed = False
            for wave in waves:
                candidate = self._bind_wave_ports((*wave, placement))
                if candidate is None:
                    continue
                if self._wave_compatible(candidate):
                    wave[:] = candidate
                    placed = True
                    break
            if not placed:
                bound = self._bind_wave_ports((placement,))
                if bound is None:
                    raise CapabilityRejectionError(
                        "scheduler port range cannot satisfy one atomic assignment"
                    )
                wave = list(bound)
                if not self._wave_compatible(wave):
                    raise CapabilityRejectionError(
                        "single assignment violates its resource declaration"
                    )
                waves.append(wave)
        frozen_waves = tuple(
            GpuDispatchWave(
                wave_index=index,
                assignments=tuple(wave),
                interference_envelope_sha256=self.interference_envelope.sha256,
            )
            for index, wave in enumerate(waves)
        )
        return GpuDispatchPlan(
            schema_version=1,
            registry_sha256=self.registry.sha256,
            inventory_sha256=self.inventory.sha256,
            receipts_sha256=receipts_sha256,
            interference_envelope_sha256=self.interference_envelope.sha256,
            budget_sha256_by_cell=tuple(sorted(budget_bindings.items())),
            seed=self.seed,
            waves=frozen_waves,
            completed_cell_ids=tuple(sorted(completed)),
        )

    def _candidate_gangs(self, item: PoolWorkItem) -> tuple[tuple[str, ...], ...]:
        claim = item.claim
        device_by_uuid = {device.uuid: device for device in self.inventory.devices}
        if claim.exact_gpu_uuids:
            if set(claim.exact_gpu_uuids) - set(device_by_uuid):
                return ()
            raw_candidates = (claim.exact_gpu_uuids,)
        else:
            capable = tuple(
                device.uuid
                for device in self.inventory.devices
                if claim.homogeneous.accepts(device)
            )
            raw_candidates = tuple(combinations(capable, claim.gang_shape.gpu_count))
        accepted: list[tuple[str, ...]] = []
        for candidate in raw_candidates:
            devices = tuple(device_by_uuid[uuid] for uuid in candidate)
            if any(not claim.homogeneous.accepts(device) for device in devices):
                continue
            if len({device.host_id for device in devices}) != 1:
                continue
            if len({device.hardware_envelope_sha256 for device in devices}) != 1:
                continue
            if not self._topology_accepts(candidate, claim):
                continue
            accepted.append(tuple(candidate))
        return tuple(sorted(accepted))

    def _topology_accepts(
        self, candidate: tuple[str, ...], claim: PoolResourceClaim
    ) -> bool:
        tp = claim.gang_shape.tensor_parallel_size
        if tp == 1:
            return True
        groups = tuple(
            group
            for group in self.inventory.topology_groups
            if (
                not claim.allowed_topology_groups
                or group.group_id in claim.allowed_topology_groups
            )
            and (not claim.allowed_fabrics or group.fabric in claim.allowed_fabrics)
        )
        if not groups:
            return False
        for replica in range(claim.gang_shape.data_parallel_size):
            start = replica * tp
            rank_group = set(candidate[start : start + tp])
            if not any(rank_group <= set(group.gpu_uuids) for group in groups):
                return False
        return True

    def _placeholder_ports(self, claim: PoolResourceClaim) -> tuple[int, ...]:
        if claim.exact_ports:
            return claim.exact_ports
        return tuple(range(self.port_start, self.port_start + claim.port_count))

    def _bind_wave_ports(
        self, assignments: Sequence[GpuAssignment]
    ) -> list[GpuAssignment] | None:
        used: set[int] = set()
        bound: list[GpuAssignment] = []
        for assignment in assignments:
            claim = assignment.work_item.claim
            if claim.exact_ports:
                ports = claim.exact_ports
                if (
                    used & set(ports)
                    or min(ports) < self.port_start
                    or max(ports) > self.port_end
                ):
                    return None
            else:
                ports_list: list[int] = []
                for port in range(self.port_start, self.port_end + 1):
                    if port not in used:
                        ports_list.append(port)
                    if len(ports_list) == claim.port_count:
                        break
                if len(ports_list) != claim.port_count:
                    return None
                ports = tuple(ports_list)
            used.update(ports)
            bound.append(replace(assignment, ports=ports))
        return bound

    def _wave_compatible(self, assignments: Sequence[GpuAssignment]) -> bool:
        used_gpus: set[str] = set()
        used_ports: set[int] = set()
        caches: set[str] = set()
        evidence: set[str] = set()
        serial_groups: set[str] = set()
        for assignment in assignments:
            claim = assignment.work_item.claim
            if used_gpus & set(assignment.gpu_uuids):
                return False
            if used_ports & set(assignment.ports):
                return False
            if claim.cache_root in caches or claim.evidence_root in evidence:
                return False
            serial = assignment.work_item.serial_group_key
            if serial is not None and serial in serial_groups:
                return False
            used_gpus.update(assignment.gpu_uuids)
            used_ports.update(assignment.ports)
            caches.add(claim.cache_root)
            evidence.add(claim.evidence_root)
            if serial is not None:
                serial_groups.add(serial)
        return self.interference_envelope.permits(assignments, inventory=self.inventory)

    @staticmethod
    def _dispatchable(cell: ExperimentCell) -> bool:
        if cell.status is not CellStatus.UNMEASURED:
            return False
        if cell.identity.experiment == "preflight":
            return cell.identity.method == "target_only"
        if cell.resources.workload_class in {
            WorkloadClass.DOWNLOAD,
            WorkloadClass.COMPILE,
        }:
            return cell.identity.method == "target_only"
        if cell.identity.method != "target_only":
            return False
        return serving_cell_rejection_reason(cell) is None

    def _activated_cell_ids(
        self,
        *,
        experiment: str,
        receipts: Sequence[ExperimentReceipt],
        artifact: ReducerActivationArtifact | None,
    ) -> set[str] | None:
        plan = None if artifact is None else artifact.plan
        if experiment not in {"E1", "E2"}:
            if artifact is not None:
                raise ValueError("activation plan does not match the ready stage")
            return None
        if artifact is None or plan is None:
            raise ValueError(f"{experiment} requires a sealed stage activation plan")
        if artifact.schema_version != 1:
            raise ValueError("activation artifact schema is unsupported")
        if (
            plan.registry_sha256 != self.registry.sha256
            or plan.experiment != experiment
        ):
            raise ValueError("stage activation plan identity mismatch")
        receipt_by_name = self.registry.validate_receipts(receipts)
        dependency = self.registry.definition(experiment).dependencies[0]
        if plan.dependency_receipt_sha256 != receipt_by_name[dependency].sha256:
            raise ValueError(
                "activation plan does not bind the direct dependency receipt"
            )
        stage_cells = {
            cell.cell_id: cell for cell in self.registry.cells_for(experiment)
        }
        dispositions = (
            plan.activated_cell_ids
            + plan.not_applicable_cell_ids
            + plan.blocked_cell_ids
            + plan.deferred_cell_ids
        )
        if set(dispositions) != set(stage_cells):
            raise ValueError("activation plan must disposition every stage cell")
        if any(
            not stage_cells[cell_id].runnable for cell_id in plan.activated_cell_ids
        ):
            raise ValueError("activation plan cannot enable a registry-blocked cell")
        base_blocked = {
            cell_id for cell_id, cell in stage_cells.items() if not cell.runnable
        }
        if not base_blocked <= set(plan.blocked_cell_ids):
            raise ValueError("activation plan must preserve registry-blocked cells")
        if experiment == "E1":
            if plan.activation_round != "e3a_locked_reference":
                raise ValueError("E1 activation must bind the E3a reference slice")
            activated = [stage_cells[cell_id] for cell_id in plan.activated_cell_ids]
            widths = {
                cell.identity.width
                for cell in activated
                if cell.identity.method != "target_only"
            }
            concurrencies = {cell.identity.concurrency for cell in activated}
            if len(widths) != 1 or len(concurrencies) != 1:
                raise ValueError("E1 activation must select one width/load slice")
            if {cell.identity.method for cell in activated} != {
                "target_only",
                "static",
                "tts",
                "l0",
            }:
                raise ValueError("E1 activation must retain the four core methods")
        else:
            if plan.activation_round not in {
                "halving_0",
                "halving_1",
                "halving_2",
                "halving_3",
            }:
                raise ValueError(
                    "E2 activation round must name halving_0 through halving_3"
                )
            marker = f"halving_stage={plan.activation_round[-1]}:"
            activated = [stage_cells[cell_id] for cell_id in plan.activated_cell_ids]
            if any(marker not in cell.identity.variant for cell in activated):
                raise ValueError("E2 activation mixes successive-halving stages")
            adaptive: dict[str, set[str]] = {}
            for cell in activated:
                if cell.identity.method in {"tts", "l0"}:
                    adaptive.setdefault(_adaptive_pair_key(cell), set()).add(
                        cell.identity.method
                    )
            if not adaptive or any(
                methods != {"tts", "l0"} for methods in adaptive.values()
            ):
                raise ValueError("E2 activation must retain matched TTS/L0 pairs")
        if any(
            not self._dispatchable(stage_cells[cell_id])
            for cell_id in plan.activated_cell_ids
        ):
            raise ValueError(
                "activation contains methods not executable by the validated release"
            )
        return set(plan.activated_cell_ids)

    def _activated_confirmation_cell_ids(
        self,
        *,
        experiment: str,
        completed: tuple[str, ...],
        artifacts: Sequence[FamilyActivationArtifact],
        power_reductions: Sequence[ConfirmationFamilyPowerReductionArtifact],
    ) -> set[str] | None:
        artifact_rows = tuple(artifacts)
        power_rows = tuple(power_reductions)
        if experiment not in {"E3b", "E5"}:
            if artifact_rows or power_rows:
                raise ValueError(
                    "confirmation family artifacts do not match the ready stage"
                )
            return None
        if not artifact_rows:
            raise ValueError(
                "confirmation stages require reducer-generated family activations"
            )
        if any(not isinstance(row, FamilyActivationArtifact) for row in artifact_rows):
            raise TypeError("family activations must be FamilyActivationArtifact rows")
        if any(
            not isinstance(row, ConfirmationFamilyPowerReductionArtifact)
            for row in power_rows
        ):
            raise TypeError("family power reductions must be raw reducer artifacts")

        by_round: dict[tuple[str, str], FamilyActivationArtifact] = {}
        for artifact in artifact_rows:
            family = artifact.family
            if (
                family.registry_sha256 != self.registry.sha256
                or family.experiment != experiment
            ):
                raise ValueError(
                    "confirmation activation belongs to another stage or registry"
                )
            key = (family.sha256, artifact.activation_round)
            if key in by_round:
                raise ValueError("duplicate confirmation family activation")
            by_round[key] = artifact

        reduction_by_family: dict[str, ConfirmationFamilyPowerReductionArtifact] = {}
        for reduction in power_rows:
            family = reduction.family
            if (
                family.registry_sha256 != self.registry.sha256
                or family.experiment != experiment
            ):
                raise ValueError(
                    "confirmation power plan belongs to another stage or registry"
                )
            if family.sha256 in reduction_by_family:
                raise ValueError("duplicate confirmation family power reduction")
            reduction_by_family[family.sha256] = reduction

        pilots = {
            family_sha256: artifact
            for (family_sha256, round_name), artifact in by_round.items()
            if round_name == "excluded_pilots"
        }
        finals = {
            family_sha256: artifact
            for (family_sha256, round_name), artifact in by_round.items()
            if round_name == "final_prefix"
        }
        for pilot in pilots.values():
            verify_confirmation_pilot_activation(
                self.registry,
                family=pilot.family,
                artifact=pilot,
            )
        missing_pilots = set(finals) - pilots.keys()
        if missing_pilots:
            raise ValueError(
                "final-prefix activation requires its matching pilot artifact"
            )
        missing_power = set(finals) - reduction_by_family.keys()
        if missing_power:
            raise ValueError(
                "final-prefix activation requires its exact family power reduction"
            )
        unused_power = set(reduction_by_family) - finals.keys()
        if unused_power:
            raise ValueError(
                "family power reduction has no matching final-prefix activation"
            )

        completed_set = set(completed)
        activated: set[str] = set()
        for family_sha256, pilot in pilots.items():
            activated.update(pilot.activated_cell_ids)
            final = finals.get(family_sha256)
            if final is None:
                continue
            if not set(pilot.activated_cell_ids) <= completed_set:
                raise ValueError(
                    "final-prefix activation requires prior completion of its family pilots"
                )
            reduction = reduction_by_family[family_sha256]
            if reduction.family != pilot.family or final.family != pilot.family:
                raise ValueError("confirmation artifacts cross family boundaries")
            expected = materialize_confirmation_prefix(
                self.registry,
                family=pilot.family,
                reduction=reduction,
                pilot_activation=pilot,
            )
            if final != expected:
                raise ValueError(
                    "family final activation is not reducer-generated from its "
                    "power reduction"
                )
            activated.update(final.activated_cell_ids)
        return activated


@dataclass(frozen=True)
class GpuDispatchPlanningContext:
    """Inputs that deterministically issue one plan for inspection or execution.

    Keeping the raw reducer artifacts here prevents a plan from authorizing
    itself with copied digests.  Consumers re-run the sole scheduler and
    require object-for-object equality.  Bare completed cell IDs are admitted
    only for planning; :class:`GpuDispatchExecutionContext` rejects them until
    a durable terminal-artifact store can validate their authority.
    """

    registry: ExperimentRegistry
    inventory: GpuInventory
    interference_envelope: InterferenceEnvelope
    budgets: tuple[ExperimentBudget, ...]
    receipts: tuple[ExperimentReceipt, ...] = ()
    completed_cell_ids: tuple[str, ...] = ()
    activation_artifact: ReducerActivationArtifact | None = None
    family_activations: tuple[FamilyActivationArtifact, ...] = ()
    family_power_reductions: tuple[ConfirmationFamilyPowerReductionArtifact, ...] = ()
    port_start: int = 24_000
    port_end: int = 65_535
    seed: int = 20260811

    def __post_init__(self) -> None:
        if not isinstance(self.registry, ExperimentRegistry):
            raise TypeError("execution-context registry must be an ExperimentRegistry")
        if not isinstance(self.inventory, GpuInventory):
            raise TypeError("execution-context inventory must be a GpuInventory")
        if not isinstance(self.interference_envelope, InterferenceEnvelope):
            raise TypeError(
                "execution-context interference envelope must be an "
                "InterferenceEnvelope"
            )
        if any(not isinstance(budget, ExperimentBudget) for budget in self.budgets):
            raise TypeError("execution-context budgets must be ExperimentBudget rows")
        budget_ids = tuple(budget.cell_id for budget in self.budgets)
        if len(budget_ids) != len(set(budget_ids)):
            raise ValueError("execution-context budgets contain duplicate cells")
        if budget_ids != tuple(sorted(budget_ids)):
            raise ValueError("execution-context budgets must be sorted by cell ID")
        if any(not isinstance(receipt, ExperimentReceipt) for receipt in self.receipts):
            raise TypeError("execution-context receipts must be ExperimentReceipt rows")
        if any(
            not isinstance(artifact, FamilyActivationArtifact)
            for artifact in self.family_activations
        ):
            raise TypeError(
                "execution-context family activations must be reducer artifacts"
            )
        if any(
            not isinstance(reduction, ConfirmationFamilyPowerReductionArtifact)
            for reduction in self.family_power_reductions
        ):
            raise TypeError(
                "dispatch-context family power reductions must be raw reducer artifacts"
            )
        if self.activation_artifact is not None and not isinstance(
            self.activation_artifact, ReducerActivationArtifact
        ):
            raise TypeError(
                "execution-context activation must be a reducer activation artifact"
            )
        if len(self.completed_cell_ids) != len(set(self.completed_cell_ids)):
            raise ValueError("execution-context completed cells must be unique")
        if self.completed_cell_ids != tuple(sorted(self.completed_cell_ids)):
            raise ValueError("execution-context completed cells must be sorted")
        GpuPoolScheduler(
            registry=self.registry,
            inventory=self.inventory,
            interference_envelope=self.interference_envelope,
            port_start=self.port_start,
            port_end=self.port_end,
            seed=self.seed,
        )

    @property
    def budgets_by_cell_id(self) -> dict[str, ExperimentBudget]:
        return {budget.cell_id: budget for budget in self.budgets}

    def issue_plan(self) -> GpuDispatchPlan:
        """Run the sole scheduler from raw trusted artifacts."""

        scheduler = GpuPoolScheduler(
            registry=self.registry,
            inventory=self.inventory,
            interference_envelope=self.interference_envelope,
            port_start=self.port_start,
            port_end=self.port_end,
            seed=self.seed,
        )
        return scheduler.schedule(
            budgets_by_cell_id=self.budgets_by_cell_id,
            receipts=self.receipts,
            completed_cell_ids=self.completed_cell_ids,
            activation_artifact=self.activation_artifact,
            family_activations=self.family_activations,
            family_power_reductions=self.family_power_reductions,
        )

    def authority_dict(self) -> dict[str, Any]:
        """Return the digest-only identity of every scheduler authority input."""

        return {
            "schema_version": 1,
            "kind": "gpu_dispatch_planning_context",
            "registry_sha256": self.registry.sha256,
            "inventory_sha256": self.inventory.sha256,
            "interference_envelope_sha256": self.interference_envelope.sha256,
            "budget_sha256s": [budget.sha256 for budget in self.budgets],
            "receipt_sha256s": [receipt.sha256 for receipt in self.receipts],
            "completed_cell_ids": list(self.completed_cell_ids),
            "activation_artifact_sha256": (
                None
                if self.activation_artifact is None
                else self.activation_artifact.sha256
            ),
            "family_activation_sha256s": [
                artifact.sha256 for artifact in self.family_activations
            ],
            "family_power_reduction_sha256s": [
                reduction.sha256 for reduction in self.family_power_reductions
            ],
            "port_start": self.port_start,
            "port_end": self.port_end,
            "seed": self.seed,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.authority_dict())


@dataclass(frozen=True)
class GpuDispatchExecutionContext(GpuDispatchPlanningContext):
    """Scheduler authority that is safe to consume at an execution boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.completed_cell_ids:
            raise ValueError(
                "execution context cannot trust bare completed cell IDs; use a "
                "planning context until durable terminal bindings are available"
            )


def validate_dispatch_plan_for_planning(
    plan: GpuDispatchPlan,
    *,
    planning_context: GpuDispatchPlanningContext,
) -> None:
    """Require the exact plan issued from the supplied scheduler inputs."""

    if not isinstance(plan, GpuDispatchPlan):
        raise TypeError("dispatch plan must be a GpuDispatchPlan")
    if not isinstance(planning_context, GpuDispatchPlanningContext):
        raise TypeError("planning_context must be a GpuDispatchPlanningContext")
    expected = planning_context.issue_plan()
    if plan != expected:
        raise ValueError(
            "dispatch plan is not the exact plan issued from its planning context"
        )


def validate_dispatch_plan_for_execution(
    plan: GpuDispatchPlan,
    *,
    execution_context: GpuDispatchExecutionContext,
) -> None:
    """Require the exact plan issued from trusted scheduler inputs."""

    if not isinstance(plan, GpuDispatchPlan):
        raise TypeError("dispatch plan must be a GpuDispatchPlan")
    if not isinstance(execution_context, GpuDispatchExecutionContext):
        raise TypeError("execution_context must be a GpuDispatchExecutionContext")
    validate_dispatch_plan_for_planning(
        plan,
        planning_context=execution_context,
    )


class AssignmentExecutionStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AssignmentExecutionReceipt:
    """Terminal result plus cumulative monotonic cost for one assignment."""

    plan_sha256: str
    wave_sha256: str
    assignment_sha256: str
    budget_sha256: str
    attempt: int
    status: AssignmentExecutionStatus
    terminal_receipt_sha256: str | None
    failure_sha256: str | None
    prior_attempt_receipt_sha256: str | None
    gpu_count: int
    fixed_instance_gpu_count: int
    attempt_intervals_monotonic_ns: tuple[tuple[int, int], ...]
    attributed_gpu_ns: int
    attributed_fixed_instance_gpu_ns: int
    accounting_semantics: str = "per_assignment_attribution_not_schedule_ledger_v1"

    def __post_init__(self) -> None:
        for name in (
            "plan_sha256",
            "wave_sha256",
            "assignment_sha256",
            "budget_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.status, AssignmentExecutionStatus):
            raise TypeError("status must be an AssignmentExecutionStatus")
        if self.attempt != len(self.attempt_intervals_monotonic_ns):
            raise ValueError("attempt intervals must cover every cumulative attempt")
        canonical_intervals = _merge_monotonic_intervals(
            self.attempt_intervals_monotonic_ns
        )
        if canonical_intervals != self.attempt_intervals_monotonic_ns:
            raise ValueError(
                "assignment attempt intervals must be sorted and non-overlapping"
            )
        if self.attempt == 1:
            if self.prior_attempt_receipt_sha256 is not None:
                raise ValueError("first attempt cannot bind a prior attempt receipt")
        elif not _is_sha256(self.prior_attempt_receipt_sha256):
            raise ValueError("retried assignment must bind its prior attempt receipt")
        for name in ("gpu_count", "fixed_instance_gpu_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.fixed_instance_gpu_count < self.gpu_count:
            raise ValueError("fixed-instance GPU count cannot be below gang size")
        cumulative_elapsed_ns = _interval_union_ns(self.attempt_intervals_monotonic_ns)
        if self.attributed_gpu_ns != cumulative_elapsed_ns * self.gpu_count:
            raise ValueError("assignment GPU attribution differs from monotonic time")
        if (
            self.attributed_fixed_instance_gpu_ns
            != cumulative_elapsed_ns * self.fixed_instance_gpu_count
        ):
            raise ValueError(
                "assignment fixed-instance attribution differs from monotonic time"
            )
        if self.accounting_semantics != (
            "per_assignment_attribution_not_schedule_ledger_v1"
        ):
            raise ValueError("assignment accounting semantics are unsupported")
        if self.status is AssignmentExecutionStatus.SUCCEEDED:
            if not _is_sha256(self.terminal_receipt_sha256):
                raise ValueError("successful assignment needs a terminal receipt")
            if self.failure_sha256 is not None:
                raise ValueError("successful assignment cannot carry failure evidence")
        else:
            if not _is_sha256(self.failure_sha256):
                raise ValueError("failed assignment needs bound failure evidence")
            if self.terminal_receipt_sha256 is not None:
                raise ValueError("failed assignment cannot carry a terminal receipt")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "wave_sha256": self.wave_sha256,
            "assignment_sha256": self.assignment_sha256,
            "budget_sha256": self.budget_sha256,
            "attempt": self.attempt,
            "status": self.status.value,
            "terminal_receipt_sha256": self.terminal_receipt_sha256,
            "failure_sha256": self.failure_sha256,
            "prior_attempt_receipt_sha256": self.prior_attempt_receipt_sha256,
            "gpu_count": self.gpu_count,
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "attempt_intervals_monotonic_ns": [
                [start, finish] for start, finish in self.attempt_intervals_monotonic_ns
            ],
            "cumulative_elapsed_ns": _interval_union_ns(
                self.attempt_intervals_monotonic_ns
            ),
            "attributed_gpu_ns": self.attributed_gpu_ns,
            "attributed_fixed_instance_gpu_ns": (self.attributed_fixed_instance_gpu_ns),
            "accounting_semantics": self.accounting_semantics,
        }

    @classmethod
    def from_dict(cls, value: object) -> AssignmentExecutionReceipt:
        row = _strict_object(
            "assignment execution receipt",
            value,
            frozenset(
                {
                    "plan_sha256",
                    "wave_sha256",
                    "assignment_sha256",
                    "budget_sha256",
                    "attempt",
                    "status",
                    "terminal_receipt_sha256",
                    "failure_sha256",
                    "prior_attempt_receipt_sha256",
                    "gpu_count",
                    "fixed_instance_gpu_count",
                    "attempt_intervals_monotonic_ns",
                    "cumulative_elapsed_ns",
                    "attributed_gpu_ns",
                    "attributed_fixed_instance_gpu_ns",
                    "accounting_semantics",
                }
            ),
        )
        intervals = tuple(
            tuple(
                _strict_int("attempt monotonic endpoint", endpoint)
                for endpoint in _strict_list("attempt monotonic interval", interval)
            )
            for interval in _strict_list(
                "attempt_intervals_monotonic_ns",
                row["attempt_intervals_monotonic_ns"],
            )
        )
        receipt = cls(
            plan_sha256=_strict_text("plan_sha256", row["plan_sha256"]),
            wave_sha256=_strict_text("wave_sha256", row["wave_sha256"]),
            assignment_sha256=_strict_text(
                "assignment_sha256", row["assignment_sha256"]
            ),
            budget_sha256=_strict_text("budget_sha256", row["budget_sha256"]),
            attempt=_strict_int("attempt", row["attempt"]),
            status=AssignmentExecutionStatus(
                _strict_text("assignment status", row["status"])
            ),
            terminal_receipt_sha256=_optional_text(
                "terminal_receipt_sha256", row["terminal_receipt_sha256"]
            ),
            failure_sha256=_optional_text("failure_sha256", row["failure_sha256"]),
            prior_attempt_receipt_sha256=_optional_text(
                "prior_attempt_receipt_sha256",
                row["prior_attempt_receipt_sha256"],
            ),
            gpu_count=_strict_int("gpu_count", row["gpu_count"]),
            fixed_instance_gpu_count=_strict_int(
                "fixed_instance_gpu_count", row["fixed_instance_gpu_count"]
            ),
            attempt_intervals_monotonic_ns=intervals,
            attributed_gpu_ns=_strict_int(
                "attributed_gpu_ns", row["attributed_gpu_ns"]
            ),
            attributed_fixed_instance_gpu_ns=_strict_int(
                "attributed_fixed_instance_gpu_ns",
                row["attributed_fixed_instance_gpu_ns"],
            ),
            accounting_semantics=_strict_text(
                "accounting_semantics", row["accounting_semantics"]
            ),
        )
        if _strict_int("cumulative_elapsed_ns", row["cumulative_elapsed_ns"]) != (
            _interval_union_ns(receipt.attempt_intervals_monotonic_ns)
        ):
            raise ValueError("assignment cumulative elapsed time mismatch")
        return receipt


@dataclass(frozen=True)
class DispatchWaveExecutionReceipt:
    """All sibling outcomes for one frozen wave, including partial failures."""

    plan_sha256: str
    wave_index: int
    wave_sha256: str
    assignment_receipts: tuple[AssignmentExecutionReceipt, ...]
    inventory_sha256: str
    fixed_instance_gpu_count: int
    active_intervals_monotonic_ns: tuple[tuple[int, int], ...]
    fixed_instance_actual_billed_gpu_ns: int
    per_assignment_attributed_gpu_ns: int
    per_assignment_attributed_fixed_instance_gpu_ns: int
    accounting_semantics: str = "interval_union_schedule_ledger_v1"

    def __post_init__(self) -> None:
        _require_sha256("plan_sha256", self.plan_sha256)
        _require_sha256("wave_sha256", self.wave_sha256)
        _require_sha256("inventory_sha256", self.inventory_sha256)
        if (
            isinstance(self.wave_index, bool)
            or not isinstance(self.wave_index, int)
            or self.wave_index < 0
        ):
            raise ValueError("wave_index must be a non-negative integer")
        if not self.assignment_receipts:
            raise ValueError("wave receipt requires assignment receipts")
        ids = tuple(row.assignment_sha256 for row in self.assignment_receipts)
        if len(ids) != len(set(ids)):
            raise ValueError("wave receipt contains duplicate assignment receipts")
        if (
            isinstance(self.fixed_instance_gpu_count, bool)
            or not isinstance(self.fixed_instance_gpu_count, int)
            or self.fixed_instance_gpu_count < 1
        ):
            raise ValueError("wave fixed-instance GPU count must be positive")
        expected_intervals = _merge_monotonic_intervals(
            tuple(
                interval
                for receipt in self.assignment_receipts
                for interval in receipt.attempt_intervals_monotonic_ns
            )
        )
        if self.active_intervals_monotonic_ns != expected_intervals:
            raise ValueError("wave active intervals differ from assignment attempts")
        union_ns = _interval_union_ns(self.active_intervals_monotonic_ns)
        if (
            self.fixed_instance_actual_billed_gpu_ns
            != union_ns * self.fixed_instance_gpu_count
        ):
            raise ValueError("wave fixed-instance ledger differs from interval union")
        if self.per_assignment_attributed_gpu_ns != sum(
            receipt.attributed_gpu_ns for receipt in self.assignment_receipts
        ):
            raise ValueError("wave per-assignment GPU attribution mismatch")
        if self.per_assignment_attributed_fixed_instance_gpu_ns != sum(
            receipt.attributed_fixed_instance_gpu_ns
            for receipt in self.assignment_receipts
        ):
            raise ValueError("wave per-assignment fixed attribution mismatch")
        if any(
            receipt.fixed_instance_gpu_count != self.fixed_instance_gpu_count
            for receipt in self.assignment_receipts
        ):
            raise ValueError("wave assignments disagree on fixed-instance GPU count")
        if self.accounting_semantics != "interval_union_schedule_ledger_v1":
            raise ValueError("wave accounting semantics are unsupported")

    @property
    def succeeded(self) -> bool:
        return all(
            receipt.status is AssignmentExecutionStatus.SUCCEEDED
            for receipt in self.assignment_receipts
        )

    @property
    def partial_sibling_failure(self) -> bool:
        statuses = {receipt.status for receipt in self.assignment_receipts}
        return statuses == {
            AssignmentExecutionStatus.SUCCEEDED,
            AssignmentExecutionStatus.FAILED,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "wave_index": self.wave_index,
            "wave_sha256": self.wave_sha256,
            "assignment_receipts": [
                receipt.to_dict() for receipt in self.assignment_receipts
            ],
            "assignment_receipt_sha256": [
                receipt.sha256 for receipt in self.assignment_receipts
            ],
            "inventory_sha256": self.inventory_sha256,
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "active_intervals_monotonic_ns": [
                [start, finish] for start, finish in self.active_intervals_monotonic_ns
            ],
            "active_interval_union_ns": _interval_union_ns(
                self.active_intervals_monotonic_ns
            ),
            "fixed_instance_actual_billed_gpu_ns": (
                self.fixed_instance_actual_billed_gpu_ns
            ),
            "per_assignment_attributed_gpu_ns": (self.per_assignment_attributed_gpu_ns),
            "per_assignment_attributed_fixed_instance_gpu_ns": (
                self.per_assignment_attributed_fixed_instance_gpu_ns
            ),
            "accounting_semantics": self.accounting_semantics,
        }

    @classmethod
    def from_dict(cls, value: object) -> DispatchWaveExecutionReceipt:
        row = _strict_object(
            "dispatch-wave execution receipt",
            value,
            frozenset(
                {
                    "plan_sha256",
                    "wave_index",
                    "wave_sha256",
                    "assignment_receipts",
                    "assignment_receipt_sha256",
                    "inventory_sha256",
                    "fixed_instance_gpu_count",
                    "active_intervals_monotonic_ns",
                    "active_interval_union_ns",
                    "fixed_instance_actual_billed_gpu_ns",
                    "per_assignment_attributed_gpu_ns",
                    "per_assignment_attributed_fixed_instance_gpu_ns",
                    "accounting_semantics",
                }
            ),
        )
        receipts = tuple(
            AssignmentExecutionReceipt.from_dict(item)
            for item in _strict_list("assignment_receipts", row["assignment_receipts"])
        )
        declared = tuple(
            _strict_text("assignment_receipt_sha256", item)
            for item in _strict_list(
                "assignment_receipt_sha256",
                row["assignment_receipt_sha256"],
            )
        )
        if declared != tuple(receipt.sha256 for receipt in receipts):
            raise ValueError("wave receipt assignment SHA-256 list mismatch")
        intervals = tuple(
            tuple(
                _strict_int("wave monotonic endpoint", endpoint)
                for endpoint in _strict_list("wave monotonic interval", interval)
            )
            for interval in _strict_list(
                "active_intervals_monotonic_ns",
                row["active_intervals_monotonic_ns"],
            )
        )
        receipt = cls(
            plan_sha256=_strict_text("plan_sha256", row["plan_sha256"]),
            wave_index=_strict_int("wave_index", row["wave_index"]),
            wave_sha256=_strict_text("wave_sha256", row["wave_sha256"]),
            assignment_receipts=receipts,
            inventory_sha256=_strict_text("inventory_sha256", row["inventory_sha256"]),
            fixed_instance_gpu_count=_strict_int(
                "fixed_instance_gpu_count", row["fixed_instance_gpu_count"]
            ),
            active_intervals_monotonic_ns=intervals,
            fixed_instance_actual_billed_gpu_ns=_strict_int(
                "fixed_instance_actual_billed_gpu_ns",
                row["fixed_instance_actual_billed_gpu_ns"],
            ),
            per_assignment_attributed_gpu_ns=_strict_int(
                "per_assignment_attributed_gpu_ns",
                row["per_assignment_attributed_gpu_ns"],
            ),
            per_assignment_attributed_fixed_instance_gpu_ns=_strict_int(
                "per_assignment_attributed_fixed_instance_gpu_ns",
                row["per_assignment_attributed_fixed_instance_gpu_ns"],
            ),
            accounting_semantics=_strict_text(
                "accounting_semantics", row["accounting_semantics"]
            ),
        )
        if _strict_int(
            "active_interval_union_ns", row["active_interval_union_ns"]
        ) != _interval_union_ns(receipt.active_intervals_monotonic_ns):
            raise ValueError("wave active interval union mismatch")
        return receipt


class DispatchExecutionPhase(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class DispatchExecutionState:
    """Pure wave-ordered execution state machine."""

    plan_sha256: str
    phase: DispatchExecutionPhase
    next_wave_index: int
    wave_receipt_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("plan_sha256", self.plan_sha256)
        if not isinstance(self.phase, DispatchExecutionPhase):
            raise TypeError("phase must be a DispatchExecutionPhase")
        if (
            isinstance(self.next_wave_index, bool)
            or not isinstance(self.next_wave_index, int)
            or self.next_wave_index < 0
        ):
            raise ValueError("next_wave_index must be non-negative")
        if len(self.wave_receipt_sha256) != self.next_wave_index:
            raise ValueError("state receipt chain must cover every attempted wave")
        for digest in self.wave_receipt_sha256:
            _require_sha256("wave receipt digest", digest)

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self)

    def begin(self, *, total_waves: int) -> DispatchExecutionState:
        if self.phase not in {
            DispatchExecutionPhase.PLANNED,
            DispatchExecutionPhase.FAILED,
        }:
            raise ValueError("only planned or failed execution can begin")
        if self.next_wave_index > total_waves:
            raise ValueError("execution state exceeds dispatch-plan wave count")
        return replace(self, phase=DispatchExecutionPhase.RUNNING)

    def accept(
        self,
        receipt: DispatchWaveExecutionReceipt,
        *,
        total_waves: int,
    ) -> DispatchExecutionState:
        if self.phase is not DispatchExecutionPhase.RUNNING:
            raise ValueError("wave receipt requires RUNNING execution state")
        if receipt.plan_sha256 != self.plan_sha256:
            raise ValueError("wave receipt belongs to another dispatch plan")
        if receipt.wave_index != self.next_wave_index:
            raise ValueError("wave receipt is not the next contiguous wave")
        chain = self.wave_receipt_sha256 + (receipt.sha256,)
        next_index = self.next_wave_index + 1
        if not receipt.succeeded:
            phase = DispatchExecutionPhase.FAILED
        elif next_index == total_waves:
            phase = DispatchExecutionPhase.COMPLETE
        else:
            phase = DispatchExecutionPhase.RUNNING
        return DispatchExecutionState(
            plan_sha256=self.plan_sha256,
            phase=phase,
            next_wave_index=next_index,
            wave_receipt_sha256=chain,
        )


@dataclass(frozen=True)
class DispatchScheduleReceipt:
    """Whole-schedule receipt and the sole accepted resume input."""

    plan_sha256: str
    phase: DispatchExecutionPhase
    wave_receipts: tuple[DispatchWaveExecutionReceipt, ...]
    inventory_sha256: str
    fixed_instance_gpu_count: int
    active_intervals_monotonic_ns: tuple[tuple[int, int], ...]
    fixed_instance_actual_billed_gpu_ns: int
    per_assignment_attributed_gpu_ns: int
    per_assignment_attributed_fixed_instance_gpu_ns: int
    prior_schedule_receipt_sha256: str | None = None
    accounting_semantics: str = "interval_union_schedule_ledger_v1"

    def __post_init__(self) -> None:
        _require_sha256("plan_sha256", self.plan_sha256)
        _require_sha256("inventory_sha256", self.inventory_sha256)
        if not isinstance(self.phase, DispatchExecutionPhase):
            raise TypeError("phase must be a DispatchExecutionPhase")
        indexes = tuple(receipt.wave_index for receipt in self.wave_receipts)
        if indexes != tuple(range(len(self.wave_receipts))):
            raise ValueError("schedule receipt must be a contiguous wave prefix")
        if any(
            receipt.plan_sha256 != self.plan_sha256 for receipt in self.wave_receipts
        ):
            raise ValueError("schedule receipt mixes dispatch plans")
        if (
            isinstance(self.fixed_instance_gpu_count, bool)
            or not isinstance(self.fixed_instance_gpu_count, int)
            or self.fixed_instance_gpu_count < 1
        ):
            raise ValueError("schedule fixed-instance GPU count must be positive")
        if any(
            receipt.inventory_sha256 != self.inventory_sha256
            or receipt.fixed_instance_gpu_count != self.fixed_instance_gpu_count
            for receipt in self.wave_receipts
        ):
            raise ValueError("schedule receipt mixes inventory billing identities")
        expected_intervals = _merge_monotonic_intervals(
            tuple(
                interval
                for receipt in self.wave_receipts
                for interval in receipt.active_intervals_monotonic_ns
            )
        )
        if self.active_intervals_monotonic_ns != expected_intervals:
            raise ValueError("schedule intervals differ from its wave receipts")
        union_ns = _interval_union_ns(self.active_intervals_monotonic_ns)
        if (
            self.fixed_instance_actual_billed_gpu_ns
            != union_ns * self.fixed_instance_gpu_count
        ):
            raise ValueError(
                "schedule fixed-instance ledger differs from interval union"
            )
        if self.per_assignment_attributed_gpu_ns != sum(
            receipt.per_assignment_attributed_gpu_ns for receipt in self.wave_receipts
        ):
            raise ValueError("schedule per-assignment GPU attribution mismatch")
        if self.per_assignment_attributed_fixed_instance_gpu_ns != sum(
            receipt.per_assignment_attributed_fixed_instance_gpu_ns
            for receipt in self.wave_receipts
        ):
            raise ValueError("schedule per-assignment fixed attribution mismatch")
        if self.accounting_semantics != "interval_union_schedule_ledger_v1":
            raise ValueError("schedule accounting semantics are unsupported")
        if self.prior_schedule_receipt_sha256 is not None:
            _require_sha256(
                "prior_schedule_receipt_sha256",
                self.prior_schedule_receipt_sha256,
            )
        if self.phase is DispatchExecutionPhase.COMPLETE:
            if not all(receipt.succeeded for receipt in self.wave_receipts):
                raise ValueError("COMPLETE schedule receipt contains failed work")
        elif self.phase is DispatchExecutionPhase.FAILED:
            if not self.wave_receipts or self.wave_receipts[-1].succeeded:
                raise ValueError("FAILED schedule receipt lacks a failed terminal wave")
        else:
            raise ValueError("durable schedule receipts must be COMPLETE or FAILED")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "phase": self.phase.value,
            "wave_receipts": [receipt.to_dict() for receipt in self.wave_receipts],
            "wave_receipt_sha256": [receipt.sha256 for receipt in self.wave_receipts],
            "inventory_sha256": self.inventory_sha256,
            "fixed_instance_gpu_count": self.fixed_instance_gpu_count,
            "active_intervals_monotonic_ns": [
                [start, finish] for start, finish in self.active_intervals_monotonic_ns
            ],
            "active_interval_union_ns": _interval_union_ns(
                self.active_intervals_monotonic_ns
            ),
            "fixed_instance_actual_billed_gpu_ns": (
                self.fixed_instance_actual_billed_gpu_ns
            ),
            "per_assignment_attributed_gpu_ns": (self.per_assignment_attributed_gpu_ns),
            "per_assignment_attributed_fixed_instance_gpu_ns": (
                self.per_assignment_attributed_fixed_instance_gpu_ns
            ),
            "prior_schedule_receipt_sha256": self.prior_schedule_receipt_sha256,
            "accounting_semantics": self.accounting_semantics,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        plan: GpuDispatchPlan,
        execution_context: GpuDispatchExecutionContext,
        sidecar: ArtifactSidecar | Mapping[str, Any] | None = None,
    ) -> DispatchScheduleReceipt:
        row = _strict_object(
            "dispatch schedule receipt",
            value,
            frozenset(
                {
                    "plan_sha256",
                    "phase",
                    "wave_receipts",
                    "wave_receipt_sha256",
                    "inventory_sha256",
                    "fixed_instance_gpu_count",
                    "active_intervals_monotonic_ns",
                    "active_interval_union_ns",
                    "fixed_instance_actual_billed_gpu_ns",
                    "per_assignment_attributed_gpu_ns",
                    "per_assignment_attributed_fixed_instance_gpu_ns",
                    "prior_schedule_receipt_sha256",
                    "accounting_semantics",
                }
            ),
        )
        wave_receipts = tuple(
            DispatchWaveExecutionReceipt.from_dict(item)
            for item in _strict_list("wave_receipts", row["wave_receipts"])
        )
        declared = tuple(
            _strict_text("wave_receipt_sha256", item)
            for item in _strict_list("wave_receipt_sha256", row["wave_receipt_sha256"])
        )
        if declared != tuple(receipt.sha256 for receipt in wave_receipts):
            raise ValueError("schedule receipt wave SHA-256 list mismatch")
        intervals = tuple(
            tuple(
                _strict_int("schedule monotonic endpoint", endpoint)
                for endpoint in _strict_list("schedule monotonic interval", interval)
            )
            for interval in _strict_list(
                "active_intervals_monotonic_ns",
                row["active_intervals_monotonic_ns"],
            )
        )
        receipt = cls(
            plan_sha256=_strict_text("plan_sha256", row["plan_sha256"]),
            phase=DispatchExecutionPhase(_strict_text("dispatch phase", row["phase"])),
            wave_receipts=wave_receipts,
            inventory_sha256=_strict_text("inventory_sha256", row["inventory_sha256"]),
            fixed_instance_gpu_count=_strict_int(
                "fixed_instance_gpu_count", row["fixed_instance_gpu_count"]
            ),
            active_intervals_monotonic_ns=intervals,
            fixed_instance_actual_billed_gpu_ns=_strict_int(
                "fixed_instance_actual_billed_gpu_ns",
                row["fixed_instance_actual_billed_gpu_ns"],
            ),
            per_assignment_attributed_gpu_ns=_strict_int(
                "per_assignment_attributed_gpu_ns",
                row["per_assignment_attributed_gpu_ns"],
            ),
            per_assignment_attributed_fixed_instance_gpu_ns=_strict_int(
                "per_assignment_attributed_fixed_instance_gpu_ns",
                row["per_assignment_attributed_fixed_instance_gpu_ns"],
            ),
            prior_schedule_receipt_sha256=_optional_text(
                "prior_schedule_receipt_sha256",
                row["prior_schedule_receipt_sha256"],
            ),
            accounting_semantics=_strict_text(
                "accounting_semantics", row["accounting_semantics"]
            ),
        )
        if _strict_int(
            "active_interval_union_ns", row["active_interval_union_ns"]
        ) != _interval_union_ns(receipt.active_intervals_monotonic_ns):
            raise ValueError("schedule active interval union mismatch")
        _validate_dispatch_resume_structure(
            plan,
            receipt,
            execution_context=execution_context,
        )
        _validate_sidecar(
            artifact_kind="dispatch_schedule_receipt.v1",
            artifact_sha256=receipt.sha256,
            sidecar=sidecar,
        )
        return receipt

    def sidecar(self) -> ArtifactSidecar:
        return ArtifactSidecar(1, "dispatch_schedule_receipt.v1", self.sha256)


AssignmentRunner = Callable[[GpuAssignment], Awaitable[str]]


def _validate_wave_receipt(
    plan: GpuDispatchPlan,
    wave: GpuDispatchWave,
    receipt: DispatchWaveExecutionReceipt,
    *,
    execution_context: GpuDispatchExecutionContext,
) -> None:
    if receipt.plan_sha256 != plan.sha256 or receipt.wave_sha256 != wave.sha256:
        raise ValueError("resume wave receipt identity mismatch")
    expected = {assignment.assignment_id for assignment in wave.assignments}
    actual = {row.assignment_sha256 for row in receipt.assignment_receipts}
    if expected != actual:
        raise ValueError("resume wave receipt has incomplete or extra assignments")
    if (
        receipt.inventory_sha256 != execution_context.inventory.sha256
        or receipt.fixed_instance_gpu_count != len(execution_context.inventory.devices)
    ):
        raise ValueError("resume wave receipt belongs to another inventory ledger")
    assignment_by_id = {
        assignment.assignment_id: assignment for assignment in wave.assignments
    }
    budgets = execution_context.budgets_by_cell_id
    for row in receipt.assignment_receipts:
        assignment = assignment_by_id[row.assignment_sha256]
        budget = budgets[assignment.work_item.item_id]
        if row.plan_sha256 != plan.sha256 or row.wave_sha256 != wave.sha256:
            raise ValueError("resume assignment receipt plan/wave mismatch")
        if row.budget_sha256 != budget.sha256:
            raise ValueError("resume assignment receipt budget mismatch")
        if row.gpu_count != len(assignment.gpu_uuids):
            raise ValueError("resume assignment receipt gang-size mismatch")
        if row.attempt > budget.retry_allowance + 1:
            raise ValueError("resume assignment receipt exceeds retry allowance")


def _validate_dispatch_resume_structure(
    plan: GpuDispatchPlan,
    receipt: DispatchScheduleReceipt,
    *,
    execution_context: GpuDispatchExecutionContext,
) -> None:
    """Validate receipt structure without granting resume authority."""

    validate_dispatch_plan_for_execution(plan, execution_context=execution_context)
    if receipt.plan_sha256 != plan.sha256:
        raise ValueError("resume receipt belongs to another dispatch plan")
    if (
        receipt.inventory_sha256 != execution_context.inventory.sha256
        or receipt.fixed_instance_gpu_count != len(execution_context.inventory.devices)
    ):
        raise ValueError("resume receipt belongs to another inventory ledger")
    if len(receipt.wave_receipts) > len(plan.waves):
        raise ValueError("resume receipt exceeds dispatch-plan wave count")
    for wave_receipt in receipt.wave_receipts:
        _validate_wave_receipt(
            plan,
            plan.waves[wave_receipt.wave_index],
            wave_receipt,
            execution_context=execution_context,
        )
    if receipt.phase is DispatchExecutionPhase.COMPLETE and len(
        receipt.wave_receipts
    ) != len(plan.waves):
        raise ValueError("complete resume receipt does not cover the whole plan")


def validate_dispatch_resume(
    plan: GpuDispatchPlan,
    receipt: DispatchScheduleReceipt,
    *,
    execution_context: GpuDispatchExecutionContext,
) -> None:
    """Fail closed until raw terminal artifacts can authorize resume."""

    _validate_dispatch_resume_structure(
        plan,
        receipt,
        execution_context=execution_context,
    )
    raise ValueError(
        "dispatch resume is blocked until a durable terminal artifact store "
        "can revalidate raw terminal identity"
    )


def _make_wave_execution_receipt(
    *,
    plan: GpuDispatchPlan,
    wave: GpuDispatchWave,
    assignment_receipts: tuple[AssignmentExecutionReceipt, ...],
    execution_context: GpuDispatchExecutionContext,
) -> DispatchWaveExecutionReceipt:
    intervals = _merge_monotonic_intervals(
        tuple(
            interval
            for receipt in assignment_receipts
            for interval in receipt.attempt_intervals_monotonic_ns
        )
    )
    fixed_count = len(execution_context.inventory.devices)
    return DispatchWaveExecutionReceipt(
        plan_sha256=plan.sha256,
        wave_index=wave.wave_index,
        wave_sha256=wave.sha256,
        assignment_receipts=assignment_receipts,
        inventory_sha256=execution_context.inventory.sha256,
        fixed_instance_gpu_count=fixed_count,
        active_intervals_monotonic_ns=intervals,
        fixed_instance_actual_billed_gpu_ns=(
            _interval_union_ns(intervals) * fixed_count
        ),
        per_assignment_attributed_gpu_ns=sum(
            receipt.attributed_gpu_ns for receipt in assignment_receipts
        ),
        per_assignment_attributed_fixed_instance_gpu_ns=sum(
            receipt.attributed_fixed_instance_gpu_ns for receipt in assignment_receipts
        ),
    )


def _make_schedule_receipt(
    *,
    plan: GpuDispatchPlan,
    phase: DispatchExecutionPhase,
    wave_receipts: tuple[DispatchWaveExecutionReceipt, ...],
    execution_context: GpuDispatchExecutionContext,
    prior_schedule_receipt_sha256: str | None,
) -> DispatchScheduleReceipt:
    intervals = _merge_monotonic_intervals(
        tuple(
            interval
            for receipt in wave_receipts
            for interval in receipt.active_intervals_monotonic_ns
        )
    )
    fixed_count = len(execution_context.inventory.devices)
    return DispatchScheduleReceipt(
        plan_sha256=plan.sha256,
        phase=phase,
        wave_receipts=wave_receipts,
        inventory_sha256=execution_context.inventory.sha256,
        fixed_instance_gpu_count=fixed_count,
        active_intervals_monotonic_ns=intervals,
        fixed_instance_actual_billed_gpu_ns=(
            _interval_union_ns(intervals) * fixed_count
        ),
        per_assignment_attributed_gpu_ns=sum(
            receipt.per_assignment_attributed_gpu_ns for receipt in wave_receipts
        ),
        per_assignment_attributed_fixed_instance_gpu_ns=sum(
            receipt.per_assignment_attributed_fixed_instance_gpu_ns
            for receipt in wave_receipts
        ),
        prior_schedule_receipt_sha256=prior_schedule_receipt_sha256,
    )


async def _run_assignment(
    *,
    plan: GpuDispatchPlan,
    wave: GpuDispatchWave,
    assignment: GpuAssignment,
    attempt: int,
    budget: ExperimentBudget,
    fixed_instance_gpu_count: int,
    prior_attempt_receipt: AssignmentExecutionReceipt | None,
    runner: AssignmentRunner,
) -> AssignmentExecutionReceipt:
    started_ns = time.monotonic_ns()
    try:
        terminal_sha256 = await runner(assignment)
        finished_ns = time.monotonic_ns()
        _require_sha256("runner terminal receipt", terminal_sha256)
        intervals = (
            ()
            if prior_attempt_receipt is None
            else prior_attempt_receipt.attempt_intervals_monotonic_ns
        ) + ((started_ns, finished_ns),)
        elapsed_ns = _interval_union_ns(intervals)
        return AssignmentExecutionReceipt(
            plan_sha256=plan.sha256,
            wave_sha256=wave.sha256,
            assignment_sha256=assignment.assignment_id,
            budget_sha256=budget.sha256,
            attempt=attempt,
            status=AssignmentExecutionStatus.SUCCEEDED,
            terminal_receipt_sha256=terminal_sha256,
            failure_sha256=None,
            prior_attempt_receipt_sha256=(
                None if prior_attempt_receipt is None else prior_attempt_receipt.sha256
            ),
            gpu_count=len(assignment.gpu_uuids),
            fixed_instance_gpu_count=fixed_instance_gpu_count,
            attempt_intervals_monotonic_ns=intervals,
            attributed_gpu_ns=elapsed_ns * len(assignment.gpu_uuids),
            attributed_fixed_instance_gpu_ns=(elapsed_ns * fixed_instance_gpu_count),
        )
    except Exception as error:  # noqa: BLE001 - receipt-bound runner boundary
        finished_ns = time.monotonic_ns()
        intervals = (
            ()
            if prior_attempt_receipt is None
            else prior_attempt_receipt.attempt_intervals_monotonic_ns
        ) + ((started_ns, finished_ns),)
        elapsed_ns = _interval_union_ns(intervals)
        failure_sha256 = content_sha256(
            {
                "exception_type": type(error).__qualname__,
                "message": str(error),
                "assignment_sha256": assignment.assignment_id,
                "budget_sha256": budget.sha256,
                "attempt": attempt,
                "started_monotonic_ns": started_ns,
                "finished_monotonic_ns": finished_ns,
            }
        )
        return AssignmentExecutionReceipt(
            plan_sha256=plan.sha256,
            wave_sha256=wave.sha256,
            assignment_sha256=assignment.assignment_id,
            budget_sha256=budget.sha256,
            attempt=attempt,
            status=AssignmentExecutionStatus.FAILED,
            terminal_receipt_sha256=None,
            failure_sha256=failure_sha256,
            prior_attempt_receipt_sha256=(
                None if prior_attempt_receipt is None else prior_attempt_receipt.sha256
            ),
            gpu_count=len(assignment.gpu_uuids),
            fixed_instance_gpu_count=fixed_instance_gpu_count,
            attempt_intervals_monotonic_ns=intervals,
            attributed_gpu_ns=elapsed_ns * len(assignment.gpu_uuids),
            attributed_fixed_instance_gpu_ns=(elapsed_ns * fixed_instance_gpu_count),
        )


async def execute_dispatch_plan(
    plan: GpuDispatchPlan,
    *,
    execution_context: GpuDispatchExecutionContext,
    runner: AssignmentRunner,
    resume_receipt: DispatchScheduleReceipt | None = None,
) -> DispatchScheduleReceipt:
    """Execute frozen waves with structured sibling completion.

    A failed sibling does not erase successful sibling receipts.  Execution
    stops before the next wave.  Durable resume inputs are structurally checked
    but rejected until a raw terminal-artifact store can authorize every prior
    attempt.  No bare ``completed_cell_ids`` resume path exists.
    """

    if not plan.scientific_budget_bound:
        raise ValueError("dispatch plan lacks exact per-cell ExperimentBudget bindings")
    validate_dispatch_plan_for_execution(plan, execution_context=execution_context)

    if resume_receipt is not None:
        validate_dispatch_resume(
            plan,
            resume_receipt,
            execution_context=execution_context,
        )
    else:
        prior_receipts = []
        failed_prior = None
        start_wave_index = 0
    if not plan.waves:
        return _make_schedule_receipt(
            plan=plan,
            phase=DispatchExecutionPhase.COMPLETE,
            wave_receipts=(),
            execution_context=execution_context,
            prior_schedule_receipt_sha256=(
                None if resume_receipt is None else resume_receipt.sha256
            ),
        )
    state = DispatchExecutionState(
        plan_sha256=plan.sha256,
        phase=(
            DispatchExecutionPhase.FAILED
            if failed_prior is not None
            else DispatchExecutionPhase.PLANNED
        ),
        next_wave_index=start_wave_index,
        wave_receipt_sha256=tuple(row.sha256 for row in prior_receipts),
    ).begin(total_waves=len(plan.waves))
    for wave in plan.waves[start_wave_index:]:
        successful_prior: dict[str, AssignmentExecutionReceipt] = {}
        prior_attempts: dict[str, AssignmentExecutionReceipt] = {}
        if failed_prior is not None and wave.wave_index == failed_prior.wave_index:
            successful_prior = {
                receipt.assignment_sha256: receipt
                for receipt in failed_prior.assignment_receipts
                if receipt.status is AssignmentExecutionStatus.SUCCEEDED
            }
            prior_attempts = {
                receipt.assignment_sha256: receipt
                for receipt in failed_prior.assignment_receipts
                if receipt.status is AssignmentExecutionStatus.FAILED
            }
        pending: list[
            tuple[
                GpuAssignment,
                int,
                ExperimentBudget,
                AssignmentExecutionReceipt | None,
            ]
        ] = []
        for assignment in wave.assignments:
            if assignment.assignment_id in successful_prior:
                continue
            prior_attempt = prior_attempts.get(assignment.assignment_id)
            attempt = 1 if prior_attempt is None else prior_attempt.attempt + 1
            budget = execution_context.budgets_by_cell_id[assignment.work_item.item_id]
            if attempt > budget.retry_allowance + 1:
                raise ValueError(
                    "dispatch retry would exceed the ExperimentBudget allowance"
                )
            pending.append((assignment, attempt, budget, prior_attempt))
        tasks: dict[str, asyncio.Task[AssignmentExecutionReceipt]] = {}
        async with asyncio.TaskGroup() as group:
            for assignment, attempt, budget, prior_attempt in pending:
                tasks[assignment.assignment_id] = group.create_task(
                    _run_assignment(
                        plan=plan,
                        wave=wave,
                        assignment=assignment,
                        attempt=attempt,
                        budget=budget,
                        fixed_instance_gpu_count=len(
                            execution_context.inventory.devices
                        ),
                        prior_attempt_receipt=prior_attempt,
                        runner=runner,
                    )
                )
        assignment_receipts = tuple(
            successful_prior.get(assignment.assignment_id)
            or tasks[assignment.assignment_id].result()
            for assignment in wave.assignments
        )
        wave_receipt = _make_wave_execution_receipt(
            plan=plan,
            wave=wave,
            assignment_receipts=assignment_receipts,
            execution_context=execution_context,
        )
        if failed_prior is not None and wave.wave_index == failed_prior.wave_index:
            prior_receipts.append(wave_receipt)
            failed_prior = None
        else:
            prior_receipts.append(wave_receipt)
        state = state.accept(wave_receipt, total_waves=len(plan.waves))
        if state.phase is DispatchExecutionPhase.FAILED:
            return _make_schedule_receipt(
                plan=plan,
                phase=DispatchExecutionPhase.FAILED,
                wave_receipts=tuple(prior_receipts),
                execution_context=execution_context,
                prior_schedule_receipt_sha256=(
                    None if resume_receipt is None else resume_receipt.sha256
                ),
            )
    return _make_schedule_receipt(
        plan=plan,
        phase=DispatchExecutionPhase.COMPLETE,
        wave_receipts=tuple(prior_receipts),
        execution_context=execution_context,
        prior_schedule_receipt_sha256=(
            None if resume_receipt is None else resume_receipt.sha256
        ),
    )


def supported_pool_size(size: int) -> bool:
    """Return whether ``size`` is one of the required regression inventories.

    The scheduler itself accepts every positive N; this helper names the exact
    1/2/4/8/16 release-gate matrix without accidentally restricting N=3, N=6,
    or larger hosts.
    """

    return (
        not isinstance(size, bool)
        and isinstance(size, int)
        and size in _SUPPORTED_POOL_SIZES
    )
