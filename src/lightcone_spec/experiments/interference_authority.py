"""Path-bound raw authority for same-host interference calibration.

An :class:`~lightcone_spec.experiments.gpu_pool.InterferenceEnvelope` is a
compact scheduling input, not scientific authority.  This module defines the
raw authority that a future release must replay before a calibrated ``PASS``
rule can enter that envelope.  It binds the complete inventory and probe
receipt, the derived physical hardware envelope, a calibration-only manifest,
and every first-party terminal authority used for the isolated/concurrent
comparison.

The current protocol intentionally has no registered performance-equivalence
threshold.  Raw evidence can therefore establish a hard ``FAIL`` (for example
wrong topology, missing coverage, unsafe counters, or changed token
trajectories), but it cannot establish ``PASS``.  Formal calibrated execution
is blocked with
``interference_calibration_acceptance_protocol_unregistered``.  The serial
deny-all envelope remains the only evidence-free scheduling policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyarrow.parquet as pq

from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    GpuInventory,
    InterferenceEnvelope,
    PoolResourceClaim,
)
from lightcone_spec.experiments.registry import WorkloadClass, content_sha256
from lightcone_spec.runtime.attestation import (
    RELEASE_TRUSTED_ATTESTER_POLICY,
    require_release_trusted_attester_policy,
)
from lightcone_spec.telemetry.writer import load_completed_evidence

if TYPE_CHECKING:
    from lightcone_spec.experiments.completion_authority import (
        AssignmentTerminalAuthority,
    )

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SOURCE_ROLES = frozenset(
    {
        "gpu_inventory",
        "gpu_inventory_source_receipt",
        "interference_hardware_envelope",
        "interference_calibration_protocol",
        "interference_calibration_manifest",
    }
)
_SAFETY_COUNTERS = (
    "exactness_violations",
    "version_mismatches",
    "fallbacks",
    "nonfinite_updates",
    "oom_events",
    "retractions",
    "communicator_failures",
)

TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON = (
    "trusted_hardware_attester_unavailable"
)
INTERFERENCE_ACCEPTANCE_PROTOCOL_UNREGISTERED_REASON = (
    "interference_calibration_acceptance_protocol_unregistered"
)
CALIBRATED_INTERFERENCE_RAW_AUTHORITY_REQUIRED_REASON = (
    "calibrated_interference_raw_authority_required"
)

INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256 = content_sha256(
    {
        "schema_version": 1,
        "kind": "interference_calibration_raw_reducer_protocol",
        "source_authority": (
            "path-bound inventory+probe+hardware+manifest+terminal authorities"
        ),
        "comparison": "paired isolated versus exact-cardinality concurrent jobs",
        "hard_failures": [
            "missing_or_duplicate_request_coverage",
            "terminal_token_trajectory_change",
            "nonzero_or_missing_safety_counter",
            "invalid_hardware_evidence",
            "isolated_interval_overlap",
            "concurrent_interval_nonoverlap",
            "wrong_claim_load_topology_or_contention_identity",
        ],
        "performance_metrics": [
            "completed_request_goodput_tps_from_raw_timestamps_and_tokens",
            "request_latency_p99_ms_from_raw_timestamps",
        ],
        "cardinality": "exact_only_no_upward_inference",
        "confirmation_data": "forbidden",
        "acceptance_threshold": "UNREGISTERED",
    }
)


class InterferenceCalibrationBlockedError(RuntimeError):
    """Raw inputs are honest but cannot authorize calibrated concurrency."""

    def __init__(self, reason_code: str) -> None:
        if type(reason_code) is not str or not reason_code:
            raise ValueError("interference BLOCKED reason must be non-empty text")
        self.reason_code = reason_code
        super().__init__(f"interference calibration is BLOCKED: {reason_code}")


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _require_sha256(label: str, value: object) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lower-case SHA-256")
    return str(value)


def _strict_text(label: str, value: object) -> str:
    if type(value) is not str or not value or "\n" in value:
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def _strict_int(label: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_float(label: str, value: object, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _strict_object(label: str, value: object, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{label} must be one JSON object")
    if set(value) != fields:
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - set(value))}, "
            f"unknown={sorted(set(value) - fields)}"
        )
    return value


def _strict_list(label: str, value: object) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a JSON array")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_finite_json(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("non-finite JSON numbers are forbidden")
    if type(value) is list:
        for item in value:
            _validate_finite_json(item)
    elif type(value) is dict:
        for item in value.values():
            _validate_finite_json(item)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _decode_json(body: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    _validate_finite_json(value)
    if type(value) is not dict:
        raise TypeError(f"{label} must contain one JSON object")
    return value


def _exact_path(path: str | Path, *, label: str) -> Path:
    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} source is missing") from error
    if resolved != source:
        raise ValueError(f"{label} path must be resolved and non-symlink")
    return source


def _regular_file_bytes(path: str | Path, *, label: str) -> bytes:
    source = _exact_path(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is not a readable regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read()
        after = os.fstat(descriptor)
        current = source.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
            or current.st_size != after.st_size
            or current.st_mtime_ns != after.st_mtime_ns
            or after.st_size != len(body)
        ):
            raise RuntimeError(f"{label} changed while it was read")
        return body
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class RawInterferenceJsonBinding:
    """Stable path, bytes, sidecar, and semantic identity for one raw source."""

    schema_version: int
    role: str
    path: str
    sidecar_path: str
    semantic_sha256: str
    file_sha256: str
    sidecar_file_sha256: str
    size: int
    sidecar_size: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only raw interference JSON binding schema 1 is supported")
        if self.role not in _SOURCE_ROLES:
            raise ValueError("raw interference JSON role is unsupported")
        source = Path(self.path)
        sidecar = Path(self.sidecar_path)
        if not source.is_absolute() or source.resolve() != source:
            raise ValueError("raw interference path must be absolute and resolved")
        if sidecar != Path(f"{source}.sha256"):
            raise ValueError("raw interference sidecar path is not exact")
        if not sidecar.is_absolute() or sidecar.resolve() != sidecar:
            raise ValueError("raw interference sidecar must be absolute and resolved")
        for name in (
            "semantic_sha256",
            "file_sha256",
            "sidecar_file_sha256",
        ):
            _require_sha256(f"raw interference {name}", getattr(self, name))
        if type(self.size) is not int or self.size < 1:
            raise ValueError("raw interference JSON size must be positive")
        if type(self.sidecar_size) is not int or self.sidecar_size != 65:
            raise ValueError("raw interference sidecar must be one SHA-256 line")

    @classmethod
    def from_path(cls, role: str, path: str | Path) -> RawInterferenceJsonBinding:
        if role not in _SOURCE_ROLES:
            raise ValueError("raw interference JSON role is unsupported")
        source = _exact_path(path, label=role)
        sidecar = _exact_path(f"{source}.sha256", label=f"{role} sidecar")
        body = _regular_file_bytes(source, label=role)
        sidecar_body = _regular_file_bytes(sidecar, label=f"{role} sidecar")
        value = _decode_json(body, label=role)
        semantic_sha256 = content_sha256(value)
        if sidecar_body != f"{semantic_sha256}\n".encode("ascii"):
            raise ValueError(f"{role} sidecar differs from canonical JSON identity")
        binding = cls(
            schema_version=1,
            role=role,
            path=str(source),
            sidecar_path=str(sidecar),
            semantic_sha256=semantic_sha256,
            file_sha256=hashlib.sha256(body).hexdigest(),
            sidecar_file_sha256=hashlib.sha256(sidecar_body).hexdigest(),
            size=len(body),
            sidecar_size=len(sidecar_body),
        )
        binding.load()
        return binding

    def load(self) -> dict[str, Any]:
        body = _regular_file_bytes(self.path, label=f"bound {self.role}")
        sidecar_body = _regular_file_bytes(
            self.sidecar_path, label=f"bound {self.role} sidecar"
        )
        if (
            len(body) != self.size
            or len(sidecar_body) != self.sidecar_size
            or hashlib.sha256(body).hexdigest() != self.file_sha256
            or hashlib.sha256(sidecar_body).hexdigest() != self.sidecar_file_sha256
            or sidecar_body != f"{self.semantic_sha256}\n".encode("ascii")
        ):
            raise RuntimeError(f"bound {self.role} bytes or sidecar changed")
        value = _decode_json(body, label=f"bound {self.role}")
        if content_sha256(value) != self.semantic_sha256:
            raise RuntimeError(f"bound {self.role} semantic identity changed")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "path": self.path,
            "sidecar_path": self.sidecar_path,
            "semantic_sha256": self.semantic_sha256,
            "file_sha256": self.file_sha256,
            "sidecar_file_sha256": self.sidecar_file_sha256,
            "size": self.size,
            "sidecar_size": self.sidecar_size,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class InterferenceHardwareDevice:
    uuid: str
    hardware_envelope_sha256: str
    pci_root: str
    numa_node: int
    allowed_topology_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        _strict_text("hardware GPU UUID", self.uuid)
        _require_sha256("device hardware envelope", self.hardware_envelope_sha256)
        _strict_text("device PCI root", self.pci_root)
        _strict_int("device NUMA node", self.numa_node)
        if self.allowed_topology_groups != tuple(
            sorted(set(self.allowed_topology_groups))
        ):
            raise ValueError("hardware topology groups must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "uuid": self.uuid,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "pci_root": self.pci_root,
            "numa_node": self.numa_node,
            "allowed_topology_groups": list(self.allowed_topology_groups),
        }

    @classmethod
    def from_dict(cls, value: object) -> InterferenceHardwareDevice:
        row = _strict_object(
            "interference hardware device",
            value,
            frozenset(
                {
                    "uuid",
                    "hardware_envelope_sha256",
                    "pci_root",
                    "numa_node",
                    "allowed_topology_groups",
                }
            ),
        )
        return cls(
            uuid=_strict_text("hardware GPU UUID", row["uuid"]),
            hardware_envelope_sha256=_require_sha256(
                "device hardware envelope", row["hardware_envelope_sha256"]
            ),
            pci_root=_strict_text("device PCI root", row["pci_root"]),
            numa_node=_strict_int("device NUMA node", row["numa_node"]),
            allowed_topology_groups=tuple(
                _strict_text("allowed topology group", item)
                for item in _strict_list(
                    "allowed topology groups", row["allowed_topology_groups"]
                )
            ),
        )


@dataclass(frozen=True)
class InterferenceHardwareTopology:
    group_id: str
    gpu_uuids: tuple[str, ...]
    fabric: str
    bandwidth_class: str

    def __post_init__(self) -> None:
        for label, value in (
            ("topology group", self.group_id),
            ("topology fabric", self.fabric),
            ("topology bandwidth", self.bandwidth_class),
        ):
            _strict_text(label, value)
        if not self.gpu_uuids or self.gpu_uuids != tuple(sorted(set(self.gpu_uuids))):
            raise ValueError("hardware topology GPUs must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "gpu_uuids": list(self.gpu_uuids),
            "fabric": self.fabric,
            "bandwidth_class": self.bandwidth_class,
        }

    @classmethod
    def from_dict(cls, value: object) -> InterferenceHardwareTopology:
        row = _strict_object(
            "interference hardware topology",
            value,
            frozenset({"group_id", "gpu_uuids", "fabric", "bandwidth_class"}),
        )
        return cls(
            group_id=_strict_text("topology group", row["group_id"]),
            gpu_uuids=tuple(
                _strict_text("topology GPU", item)
                for item in _strict_list("topology GPUs", row["gpu_uuids"])
            ),
            fabric=_strict_text("topology fabric", row["fabric"]),
            bandwidth_class=_strict_text("topology bandwidth", row["bandwidth_class"]),
        )


@dataclass(frozen=True)
class InterferenceHardwareEnvelope:
    """Exact physical locality/fabric view used by calibration claims."""

    schema_version: int
    kind: Literal["interference_hardware_envelope"]
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    host_id: str
    devices: tuple[InterferenceHardwareDevice, ...]
    topology_groups: tuple[InterferenceHardwareTopology, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "interference_hardware_envelope":
            raise ValueError("interference hardware envelope schema is unsupported")
        _require_sha256("hardware inventory", self.inventory_sha256)
        _require_sha256(
            "hardware inventory receipt", self.inventory_source_receipt_sha256
        )
        _strict_text("hardware host", self.host_id)
        device_ids = tuple(device.uuid for device in self.devices)
        group_ids = tuple(group.group_id for group in self.topology_groups)
        if not device_ids or device_ids != tuple(sorted(set(device_ids))):
            raise ValueError("hardware devices must be UUID-sorted and unique")
        if group_ids != tuple(sorted(set(group_ids))):
            raise ValueError("hardware topology groups must be sorted and unique")
        if any(
            set(group.gpu_uuids) - set(device_ids) for group in self.topology_groups
        ):
            raise ValueError("hardware topology references an unknown GPU")

    @classmethod
    def from_inventory(cls, inventory: GpuInventory) -> InterferenceHardwareEnvelope:
        if type(inventory) is not GpuInventory:
            raise TypeError("hardware envelope requires an exact GpuInventory")
        if len(inventory.host_ids) != 1:
            raise ValueError("interference calibration is same-host only")
        return cls(
            schema_version=1,
            kind="interference_hardware_envelope",
            inventory_sha256=inventory.sha256,
            inventory_source_receipt_sha256=inventory.source_receipt_sha256,
            host_id=inventory.host_ids[0],
            devices=tuple(
                InterferenceHardwareDevice(
                    uuid=device.uuid,
                    hardware_envelope_sha256=device.hardware_envelope_sha256,
                    pci_root=device.pci_root,
                    numa_node=device.numa_node,
                    allowed_topology_groups=device.allowed_topology_groups,
                )
                for device in inventory.devices
            ),
            topology_groups=tuple(
                InterferenceHardwareTopology(
                    group_id=group.group_id,
                    gpu_uuids=group.gpu_uuids,
                    fabric=group.fabric,
                    bandwidth_class=group.bandwidth_class,
                )
                for group in inventory.topology_groups
            ),
        )

    @classmethod
    def from_dict(cls, value: object) -> InterferenceHardwareEnvelope:
        row = _strict_object(
            "interference hardware envelope",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "inventory_sha256",
                    "inventory_source_receipt_sha256",
                    "host_id",
                    "devices",
                    "topology_groups",
                }
            ),
        )
        return cls(
            schema_version=_strict_int(
                "hardware envelope schema", row["schema_version"], minimum=1
            ),
            kind=_strict_text("hardware envelope kind", row["kind"]),
            inventory_sha256=_require_sha256(
                "hardware inventory", row["inventory_sha256"]
            ),
            inventory_source_receipt_sha256=_require_sha256(
                "hardware inventory receipt", row["inventory_source_receipt_sha256"]
            ),
            host_id=_strict_text("hardware host", row["host_id"]),
            devices=tuple(
                InterferenceHardwareDevice.from_dict(item)
                for item in _strict_list("hardware devices", row["devices"])
            ),
            topology_groups=tuple(
                InterferenceHardwareTopology.from_dict(item)
                for item in _strict_list(
                    "hardware topology groups", row["topology_groups"]
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "inventory_sha256": self.inventory_sha256,
            "inventory_source_receipt_sha256": (self.inventory_source_receipt_sha256),
            "host_id": self.host_id,
            "devices": [device.to_dict() for device in self.devices],
            "topology_groups": [group.to_dict() for group in self.topology_groups],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class InterferenceCalibrationProtocol:
    """Pre-confirmation protocol input; schema 1 deliberately cannot PASS."""

    schema_version: int
    kind: Literal["interference_calibration_protocol"]
    reducer_protocol_sha256: str
    inventory_sha256: str
    hardware_envelope_sha256: str
    data_partition: Literal["interference_calibration_only"]
    confirmation_data_visible: Literal[False]
    acceptance_status: Literal["UNREGISTERED"]
    minimum_isolated_repetitions: int
    minimum_concurrent_repetitions: int
    goodput_ratio_floor: None
    p99_latency_ratio_ceiling: None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "interference_calibration_protocol":
            raise ValueError("interference calibration protocol schema is unsupported")
        if (
            self.reducer_protocol_sha256
            != INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256
        ):
            raise ValueError("interference calibration reducer protocol changed")
        _require_sha256("protocol inventory", self.inventory_sha256)
        _require_sha256("protocol hardware envelope", self.hardware_envelope_sha256)
        if (
            self.data_partition != "interference_calibration_only"
            or self.confirmation_data_visible is not False
        ):
            raise ValueError("confirmation data cannot enter interference calibration")
        if (
            self.acceptance_status != "UNREGISTERED"
            or self.goodput_ratio_floor is not None
            or self.p99_latency_ratio_ceiling is not None
        ):
            raise ValueError("schema-1 interference acceptance thresholds are absent")
        _strict_int(
            "minimum isolated repetitions",
            self.minimum_isolated_repetitions,
            minimum=1,
        )
        _strict_int(
            "minimum concurrent repetitions",
            self.minimum_concurrent_repetitions,
            minimum=1,
        )

    @classmethod
    def from_dict(cls, value: object) -> InterferenceCalibrationProtocol:
        row = _strict_object(
            "interference calibration protocol",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "reducer_protocol_sha256",
                    "inventory_sha256",
                    "hardware_envelope_sha256",
                    "data_partition",
                    "confirmation_data_visible",
                    "acceptance_status",
                    "minimum_isolated_repetitions",
                    "minimum_concurrent_repetitions",
                    "goodput_ratio_floor",
                    "p99_latency_ratio_ceiling",
                }
            ),
        )
        if type(row["confirmation_data_visible"]) is not bool:
            raise TypeError("confirmation-data visibility must be boolean")
        return cls(
            schema_version=_strict_int("protocol schema", row["schema_version"]),
            kind=_strict_text("protocol kind", row["kind"]),
            reducer_protocol_sha256=_require_sha256(
                "protocol reducer", row["reducer_protocol_sha256"]
            ),
            inventory_sha256=_require_sha256(
                "protocol inventory", row["inventory_sha256"]
            ),
            hardware_envelope_sha256=_require_sha256(
                "protocol hardware", row["hardware_envelope_sha256"]
            ),
            data_partition=_strict_text(
                "protocol data partition", row["data_partition"]
            ),
            confirmation_data_visible=row["confirmation_data_visible"],
            acceptance_status=_strict_text(
                "protocol acceptance status", row["acceptance_status"]
            ),
            minimum_isolated_repetitions=_strict_int(
                "minimum isolated repetitions",
                row["minimum_isolated_repetitions"],
                minimum=1,
            ),
            minimum_concurrent_repetitions=_strict_int(
                "minimum concurrent repetitions",
                row["minimum_concurrent_repetitions"],
                minimum=1,
            ),
            goodput_ratio_floor=row["goodput_ratio_floor"],
            p99_latency_ratio_ceiling=row["p99_latency_ratio_ceiling"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "reducer_protocol_sha256": self.reducer_protocol_sha256,
            "inventory_sha256": self.inventory_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "data_partition": self.data_partition,
            "confirmation_data_visible": self.confirmation_data_visible,
            "acceptance_status": self.acceptance_status,
            "minimum_isolated_repetitions": self.minimum_isolated_repetitions,
            "minimum_concurrent_repetitions": self.minimum_concurrent_repetitions,
            "goodput_ratio_floor": self.goodput_ratio_floor,
            "p99_latency_ratio_ceiling": self.p99_latency_ratio_ceiling,
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


@dataclass(frozen=True)
class InterferenceCalibrationRun:
    """One exact isolated or concurrent terminal observation declaration."""

    observation_id: str
    mode: Literal["isolated", "concurrent"]
    repetition: int
    slot: int
    terminal_authority_sha256: str
    assignment_sha256: str
    cell_id: str
    execution_plan_sha256: str
    budget_sha256: str
    load_plan_sha256: str
    run_nonce_sha256: str
    gpu_uuids: tuple[str, ...]
    rank_groups: tuple[tuple[str, ...], ...]
    topology_sha256: str
    hardware_envelope_sha256: str
    workload_class: WorkloadClass
    co_run_signature: str
    gang_shape: str
    load_thermal_power_envelope: str
    cpu_cores: int
    numa_nodes: tuple[int, ...]
    ram_bytes: int
    disk_io_class: str
    network_class: str
    contention_class: str
    data_partition: Literal["interference_calibration_only"]

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.observation_id) is None:
            raise ValueError("interference observation ID is unsafe")
        if self.mode not in {"isolated", "concurrent"}:
            raise ValueError("interference observation mode is unsupported")
        _strict_int("interference repetition", self.repetition)
        _strict_int("interference slot", self.slot)
        for label, value in (
            ("terminal authority", self.terminal_authority_sha256),
            ("assignment", self.assignment_sha256),
            ("cell", self.cell_id),
            ("execution plan", self.execution_plan_sha256),
            ("budget", self.budget_sha256),
            ("load plan", self.load_plan_sha256),
            ("run nonce", self.run_nonce_sha256),
            ("topology", self.topology_sha256),
            ("hardware envelope", self.hardware_envelope_sha256),
            ("contention class", self.contention_class),
        ):
            _require_sha256(label, value)
        if not self.gpu_uuids or len(set(self.gpu_uuids)) != len(self.gpu_uuids):
            raise ValueError("calibration run requires unique physical GPUs")
        if (
            tuple(uuid for group in self.rank_groups for uuid in group)
            != self.gpu_uuids
        ):
            raise ValueError("calibration rank groups must partition physical GPUs")
        if not isinstance(self.workload_class, WorkloadClass):
            raise TypeError("calibration workload class must be exact")
        for label, value in (
            ("co-run signature", self.co_run_signature),
            ("gang shape", self.gang_shape),
            ("load/thermal/power envelope", self.load_thermal_power_envelope),
            ("disk I/O class", self.disk_io_class),
            ("network class", self.network_class),
        ):
            _strict_text(label, value)
        _strict_int("calibration CPU cores", self.cpu_cores, minimum=1)
        _strict_int("calibration RAM bytes", self.ram_bytes)
        if self.numa_nodes != tuple(sorted(set(self.numa_nodes))) or any(
            type(node) is not int or node < 0 for node in self.numa_nodes
        ):
            raise ValueError("calibration NUMA nodes must be sorted and unique")
        if self.data_partition != "interference_calibration_only":
            raise ValueError("confirmation observations cannot enter calibration")

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.repetition, self.slot, self.observation_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "mode": self.mode,
            "repetition": self.repetition,
            "slot": self.slot,
            "terminal_authority_sha256": self.terminal_authority_sha256,
            "assignment_sha256": self.assignment_sha256,
            "cell_id": self.cell_id,
            "execution_plan_sha256": self.execution_plan_sha256,
            "budget_sha256": self.budget_sha256,
            "load_plan_sha256": self.load_plan_sha256,
            "run_nonce_sha256": self.run_nonce_sha256,
            "gpu_uuids": list(self.gpu_uuids),
            "rank_groups": [list(group) for group in self.rank_groups],
            "topology_sha256": self.topology_sha256,
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "workload_class": self.workload_class.value,
            "co_run_signature": self.co_run_signature,
            "gang_shape": self.gang_shape,
            "load_thermal_power_envelope": self.load_thermal_power_envelope,
            "cpu_cores": self.cpu_cores,
            "numa_nodes": list(self.numa_nodes),
            "ram_bytes": self.ram_bytes,
            "disk_io_class": self.disk_io_class,
            "network_class": self.network_class,
            "contention_class": self.contention_class,
            "data_partition": self.data_partition,
        }

    @classmethod
    def from_dict(cls, value: object) -> InterferenceCalibrationRun:
        fields = frozenset(
            {
                "observation_id",
                "mode",
                "repetition",
                "slot",
                "terminal_authority_sha256",
                "assignment_sha256",
                "cell_id",
                "execution_plan_sha256",
                "budget_sha256",
                "load_plan_sha256",
                "run_nonce_sha256",
                "gpu_uuids",
                "rank_groups",
                "topology_sha256",
                "hardware_envelope_sha256",
                "workload_class",
                "co_run_signature",
                "gang_shape",
                "load_thermal_power_envelope",
                "cpu_cores",
                "numa_nodes",
                "ram_bytes",
                "disk_io_class",
                "network_class",
                "contention_class",
                "data_partition",
            }
        )
        row = _strict_object("interference calibration run", value, fields)
        return cls(
            observation_id=_strict_text("observation ID", row["observation_id"]),
            mode=_strict_text("observation mode", row["mode"]),
            repetition=_strict_int("observation repetition", row["repetition"]),
            slot=_strict_int("observation slot", row["slot"]),
            terminal_authority_sha256=_require_sha256(
                "terminal authority", row["terminal_authority_sha256"]
            ),
            assignment_sha256=_require_sha256("assignment", row["assignment_sha256"]),
            cell_id=_require_sha256("cell", row["cell_id"]),
            execution_plan_sha256=_require_sha256(
                "execution plan", row["execution_plan_sha256"]
            ),
            budget_sha256=_require_sha256("budget", row["budget_sha256"]),
            load_plan_sha256=_require_sha256("load plan", row["load_plan_sha256"]),
            run_nonce_sha256=_require_sha256("run nonce", row["run_nonce_sha256"]),
            gpu_uuids=tuple(
                _strict_text("calibration GPU", item)
                for item in _strict_list("calibration GPUs", row["gpu_uuids"])
            ),
            rank_groups=tuple(
                tuple(
                    _strict_text("rank GPU", item)
                    for item in _strict_list("rank group", group)
                )
                for group in _strict_list("rank groups", row["rank_groups"])
            ),
            topology_sha256=_require_sha256("topology", row["topology_sha256"]),
            hardware_envelope_sha256=_require_sha256(
                "hardware envelope", row["hardware_envelope_sha256"]
            ),
            workload_class=WorkloadClass(
                _strict_text("workload class", row["workload_class"])
            ),
            co_run_signature=_strict_text("co-run signature", row["co_run_signature"]),
            gang_shape=_strict_text("gang shape", row["gang_shape"]),
            load_thermal_power_envelope=_strict_text(
                "load/thermal/power envelope",
                row["load_thermal_power_envelope"],
            ),
            cpu_cores=_strict_int("CPU cores", row["cpu_cores"], minimum=1),
            numa_nodes=tuple(
                _strict_int("NUMA node", item)
                for item in _strict_list("NUMA nodes", row["numa_nodes"])
            ),
            ram_bytes=_strict_int("RAM bytes", row["ram_bytes"]),
            disk_io_class=_strict_text("disk I/O class", row["disk_io_class"]),
            network_class=_strict_text("network class", row["network_class"]),
            contention_class=_require_sha256(
                "contention class", row["contention_class"]
            ),
            data_partition=_strict_text("data partition", row["data_partition"]),
        )


@dataclass(frozen=True)
class InterferenceCalibrationGroup:
    group_id: str
    simultaneous_jobs: int
    isolated: tuple[InterferenceCalibrationRun, ...]
    concurrent: tuple[InterferenceCalibrationRun, ...]

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.group_id) is None:
            raise ValueError("interference calibration group ID is unsafe")
        _strict_int("simultaneous jobs", self.simultaneous_jobs, minimum=2)
        if not self.isolated or not self.concurrent:
            raise ValueError("calibration group requires isolated and concurrent runs")
        if any(row.mode != "isolated" for row in self.isolated) or any(
            row.mode != "concurrent" for row in self.concurrent
        ):
            raise ValueError("calibration runs are in the wrong group mode")
        if tuple(row.key for row in self.isolated) != tuple(
            sorted(row.key for row in self.isolated)
        ) or tuple(row.key for row in self.concurrent) != tuple(
            sorted(row.key for row in self.concurrent)
        ):
            raise ValueError("calibration group runs must be canonically ordered")
        observations = tuple(
            row.observation_id for row in (*self.isolated, *self.concurrent)
        )
        authorities = tuple(
            row.terminal_authority_sha256 for row in (*self.isolated, *self.concurrent)
        )
        nonces = tuple(
            row.run_nonce_sha256 for row in (*self.isolated, *self.concurrent)
        )
        if len(observations) != len(set(observations)):
            raise ValueError("calibration observation IDs must be unique")
        if len(authorities) != len(set(authorities)):
            raise ValueError("terminal authorities cannot be reused as observations")
        if len(nonces) != len(set(nonces)):
            raise ValueError("calibration run nonces must be unique")
        isolated_keys = {(row.repetition, row.slot) for row in self.isolated}
        concurrent_keys = {(row.repetition, row.slot) for row in self.concurrent}
        if isolated_keys != concurrent_keys:
            raise ValueError("isolated/concurrent calibration pairs differ")
        repetitions = {row.repetition for row in self.concurrent}
        expected_slots = set(range(self.simultaneous_jobs))
        for repetition in repetitions:
            if {
                row.slot for row in self.concurrent if row.repetition == repetition
            } != expected_slots or {
                row.slot for row in self.isolated if row.repetition == repetition
            } != expected_slots:
                raise ValueError("calibration cardinality lacks exact slot coverage")
        common = {
            (
                row.workload_class,
                row.co_run_signature,
                row.gang_shape,
                row.load_plan_sha256,
                row.load_thermal_power_envelope,
                row.cpu_cores,
                row.numa_nodes,
                row.ram_bytes,
                row.disk_io_class,
                row.network_class,
                row.contention_class,
                row.hardware_envelope_sha256,
            )
            for row in (*self.isolated, *self.concurrent)
        }
        if len(common) != 1:
            raise ValueError(
                "calibration group mixes claim/load/topology contention classes"
            )
        for repetition in repetitions:
            concurrent_rows = tuple(
                row for row in self.concurrent if row.repetition == repetition
            )
            used = [uuid for row in concurrent_rows for uuid in row.gpu_uuids]
            if len(used) != len(set(used)):
                raise ValueError("concurrent calibration assignments overlap a GPU")

    @property
    def repetitions(self) -> tuple[int, ...]:
        return tuple(sorted({row.repetition for row in self.concurrent}))

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "simultaneous_jobs": self.simultaneous_jobs,
            "isolated": [row.to_dict() for row in self.isolated],
            "concurrent": [row.to_dict() for row in self.concurrent],
        }

    @classmethod
    def from_dict(cls, value: object) -> InterferenceCalibrationGroup:
        row = _strict_object(
            "interference calibration group",
            value,
            frozenset({"group_id", "simultaneous_jobs", "isolated", "concurrent"}),
        )
        return cls(
            group_id=_strict_text("calibration group ID", row["group_id"]),
            simultaneous_jobs=_strict_int(
                "simultaneous jobs", row["simultaneous_jobs"], minimum=2
            ),
            isolated=tuple(
                InterferenceCalibrationRun.from_dict(item)
                for item in _strict_list("isolated calibration runs", row["isolated"])
            ),
            concurrent=tuple(
                InterferenceCalibrationRun.from_dict(item)
                for item in _strict_list(
                    "concurrent calibration runs", row["concurrent"]
                )
            ),
        )


@dataclass(frozen=True)
class InterferenceCalibrationManifest:
    schema_version: int
    kind: Literal["interference_calibration_manifest"]
    inventory_sha256: str
    inventory_source_receipt_sha256: str
    hardware_envelope_sha256: str
    protocol_sha256: str
    data_partition: Literal["interference_calibration_only"]
    confirmation_data_visible: Literal[False]
    groups: tuple[InterferenceCalibrationGroup, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.kind != "interference_calibration_manifest":
            raise ValueError("interference calibration manifest schema is unsupported")
        for label, value in (
            ("manifest inventory", self.inventory_sha256),
            ("manifest inventory receipt", self.inventory_source_receipt_sha256),
            ("manifest hardware envelope", self.hardware_envelope_sha256),
            ("manifest protocol", self.protocol_sha256),
        ):
            _require_sha256(label, value)
        if (
            self.data_partition != "interference_calibration_only"
            or self.confirmation_data_visible is not False
        ):
            raise ValueError("confirmation data cannot enter calibration manifest")
        group_ids = tuple(group.group_id for group in self.groups)
        if not group_ids or group_ids != tuple(sorted(set(group_ids))):
            raise ValueError("calibration groups must be ID-sorted and unique")
        all_runs = tuple(
            row for group in self.groups for row in (*group.isolated, *group.concurrent)
        )
        authority_ids = tuple(row.terminal_authority_sha256 for row in all_runs)
        observation_ids = tuple(row.observation_id for row in all_runs)
        if len(authority_ids) != len(set(authority_ids)) or len(observation_ids) != len(
            set(observation_ids)
        ):
            raise ValueError("calibration runs cannot be shared across groups")

    @classmethod
    def from_dict(cls, value: object) -> InterferenceCalibrationManifest:
        row = _strict_object(
            "interference calibration manifest",
            value,
            frozenset(
                {
                    "schema_version",
                    "kind",
                    "inventory_sha256",
                    "inventory_source_receipt_sha256",
                    "hardware_envelope_sha256",
                    "protocol_sha256",
                    "data_partition",
                    "confirmation_data_visible",
                    "groups",
                }
            ),
        )
        if type(row["confirmation_data_visible"]) is not bool:
            raise TypeError("confirmation-data visibility must be boolean")
        return cls(
            schema_version=_strict_int("manifest schema", row["schema_version"]),
            kind=_strict_text("manifest kind", row["kind"]),
            inventory_sha256=_require_sha256(
                "manifest inventory", row["inventory_sha256"]
            ),
            inventory_source_receipt_sha256=_require_sha256(
                "manifest inventory receipt",
                row["inventory_source_receipt_sha256"],
            ),
            hardware_envelope_sha256=_require_sha256(
                "manifest hardware envelope", row["hardware_envelope_sha256"]
            ),
            protocol_sha256=_require_sha256(
                "manifest protocol", row["protocol_sha256"]
            ),
            data_partition=_strict_text("data partition", row["data_partition"]),
            confirmation_data_visible=row["confirmation_data_visible"],
            groups=tuple(
                InterferenceCalibrationGroup.from_dict(item)
                for item in _strict_list("calibration groups", row["groups"])
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "inventory_sha256": self.inventory_sha256,
            "inventory_source_receipt_sha256": (self.inventory_source_receipt_sha256),
            "hardware_envelope_sha256": self.hardware_envelope_sha256,
            "protocol_sha256": self.protocol_sha256,
            "data_partition": self.data_partition,
            "confirmation_data_visible": self.confirmation_data_visible,
            "groups": [group.to_dict() for group in self.groups],
        }

    @cached_property
    def sha256(self) -> str:
        return content_sha256(self.to_dict())


def _physical_topology_sha256(
    assignment: GpuAssignment, inventory: GpuInventory
) -> str:
    groups = tuple(
        group.to_dict()
        for group in inventory.topology_groups
        if set(assignment.gpu_uuids) <= set(group.gpu_uuids)
    )
    return content_sha256(
        {
            "schema_version": 1,
            "host_ids": sorted(
                {inventory.device(uuid).host_id for uuid in assignment.gpu_uuids}
            ),
            "gpu_uuids": list(assignment.gpu_uuids),
            "rank_groups": [list(group) for group in assignment.rank_groups],
            "devices": [
                {
                    "uuid": uuid,
                    "hardware_envelope_sha256": (
                        inventory.device(uuid).hardware_envelope_sha256
                    ),
                    "pci_root": inventory.device(uuid).pci_root,
                    "numa_node": inventory.device(uuid).numa_node,
                    "peer_access_class": inventory.device(uuid).peer_access_class,
                }
                for uuid in assignment.gpu_uuids
            ],
            "covering_topology_groups": list(groups),
        }
    )


def _claim_hardware_envelope_sha256(
    assignment: GpuAssignment, inventory: GpuInventory
) -> str:
    values = {
        inventory.device(uuid).hardware_envelope_sha256 for uuid in assignment.gpu_uuids
    }
    if len(values) != 1:
        raise ValueError("calibration gang crosses hardware envelopes")
    return next(iter(values))


def _validate_run_against_assignment(
    run: InterferenceCalibrationRun,
    *,
    assignment: GpuAssignment,
    claim: PoolResourceClaim,
    inventory: GpuInventory,
) -> None:
    expected = {
        "assignment_sha256": assignment.assignment_id,
        "cell_id": assignment.work_item.item_id,
        "gpu_uuids": assignment.gpu_uuids,
        "rank_groups": assignment.rank_groups,
        "topology_sha256": _physical_topology_sha256(assignment, inventory),
        "hardware_envelope_sha256": _claim_hardware_envelope_sha256(
            assignment, inventory
        ),
        "workload_class": claim.workload_class,
        "co_run_signature": claim.interference_class,
        "gang_shape": claim.gang_shape.signature,
        "load_thermal_power_envelope": claim.load_thermal_power_envelope,
        "cpu_cores": claim.cpu_cores,
        "numa_nodes": claim.numa_nodes,
        "ram_bytes": claim.ram_bytes,
        "disk_io_class": claim.disk_io_class,
        "network_class": claim.network_class,
        "contention_class": claim.contention_class,
    }
    if any(getattr(run, name) != value for name, value in expected.items()):
        raise ValueError(
            "calibration manifest differs from exact claim/load/topology/contention"
        )


def _validate_inventory_receipt(value: object, *, inventory: GpuInventory) -> None:
    if type(value) is not dict:
        raise TypeError("GPU inventory source receipt must be one JSON object")
    required = {
        "schema_version",
        "kind",
        "challenge_nonce_sha256",
        "host_id",
        "hostname",
        "machine_id_sha256",
        "commands",
        "parsed_topology",
        "pci_locality",
        "receipt_sha256",
    }
    row = _strict_object("GPU inventory source receipt", value, frozenset(required))
    declared = _require_sha256("inventory receipt", row["receipt_sha256"])
    content = {key: item for key, item in row.items() if key != "receipt_sha256"}
    if (
        row["schema_version"] != 1
        or row["kind"] != "gpu_inventory_probe_receipt"
        or declared != content_sha256(content)
        or declared != inventory.source_receipt_sha256
        or row["host_id"] not in inventory.host_ids
    ):
        raise ValueError("GPU inventory source receipt identity mismatch")
    locality = _strict_list("inventory PCI locality", row["pci_locality"])
    expected = {
        (
            device.uuid,
            device.pci_bus_id,
            device.pci_root,
            device.numa_node,
        )
        for device in inventory.devices
    }
    observed: set[tuple[str, str, str, int]] = set()
    for item in locality:
        location = _strict_object(
            "inventory PCI locality row",
            item,
            frozenset({"index", "uuid", "pci_bus_id", "pci_root", "numa_node"}),
        )
        observed.add(
            (
                _strict_text("inventory locality UUID", location["uuid"]),
                _strict_text("inventory locality PCI bus", location["pci_bus_id"]),
                _strict_text("inventory locality PCI root", location["pci_root"]),
                _strict_int("inventory locality NUMA", location["numa_node"]),
            )
        )
    if observed != expected:
        raise ValueError("GPU inventory receipt PCI/NUMA coverage differs")


@dataclass(frozen=True)
class InterferenceCalibrationSourceAudit:
    inventory: GpuInventory
    hardware_envelope: InterferenceHardwareEnvelope
    protocol: InterferenceCalibrationProtocol
    manifest: InterferenceCalibrationManifest

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "interference_calibration_source_audit",
                "inventory_sha256": self.inventory.sha256,
                "hardware_envelope_sha256": self.hardware_envelope.sha256,
                "protocol_sha256": self.protocol.sha256,
                "manifest_sha256": self.manifest.sha256,
            }
        )


@dataclass(frozen=True)
class InterferenceCalibrationSourceAuthority:
    """Path-bound non-claiming authority over all calibration JSON inputs."""

    inventory: RawInterferenceJsonBinding
    inventory_source_receipt: RawInterferenceJsonBinding
    hardware_envelope: RawInterferenceJsonBinding
    protocol: RawInterferenceJsonBinding
    manifest: RawInterferenceJsonBinding

    def __post_init__(self) -> None:
        expected = (
            (self.inventory, "gpu_inventory"),
            (self.inventory_source_receipt, "gpu_inventory_source_receipt"),
            (self.hardware_envelope, "interference_hardware_envelope"),
            (self.protocol, "interference_calibration_protocol"),
            (self.manifest, "interference_calibration_manifest"),
        )
        if any(
            type(binding) is not RawInterferenceJsonBinding or binding.role != role
            for binding, role in expected
        ):
            raise TypeError("interference source authority has a wrong binding role")
        paths = tuple(binding.path for binding, _ in expected)
        if len(paths) != len(set(paths)):
            raise ValueError("interference source paths must be distinct")

    @classmethod
    def from_paths(
        cls,
        *,
        inventory: str | Path,
        inventory_source_receipt: str | Path,
        hardware_envelope: str | Path,
        protocol: str | Path,
        manifest: str | Path,
    ) -> InterferenceCalibrationSourceAuthority:
        return cls(
            inventory=RawInterferenceJsonBinding.from_path("gpu_inventory", inventory),
            inventory_source_receipt=RawInterferenceJsonBinding.from_path(
                "gpu_inventory_source_receipt", inventory_source_receipt
            ),
            hardware_envelope=RawInterferenceJsonBinding.from_path(
                "interference_hardware_envelope", hardware_envelope
            ),
            protocol=RawInterferenceJsonBinding.from_path(
                "interference_calibration_protocol", protocol
            ),
            manifest=RawInterferenceJsonBinding.from_path(
                "interference_calibration_manifest", manifest
            ),
        )

    def audit(self) -> InterferenceCalibrationSourceAudit:
        inventory = GpuInventory.from_dict(self.inventory.load())
        if inventory.sha256 != self.inventory.semantic_sha256:
            raise ValueError("bound GPU inventory semantic identity mismatch")
        _validate_inventory_receipt(
            self.inventory_source_receipt.load(), inventory=inventory
        )
        hardware = InterferenceHardwareEnvelope.from_dict(self.hardware_envelope.load())
        expected_hardware = InterferenceHardwareEnvelope.from_inventory(inventory)
        if hardware != expected_hardware or hardware.sha256 != (
            self.hardware_envelope.semantic_sha256
        ):
            raise ValueError("hardware envelope differs from full GPU inventory")
        protocol = InterferenceCalibrationProtocol.from_dict(self.protocol.load())
        if protocol.sha256 != self.protocol.semantic_sha256:
            raise ValueError("calibration protocol semantic identity mismatch")
        manifest = InterferenceCalibrationManifest.from_dict(self.manifest.load())
        if manifest.sha256 != self.manifest.semantic_sha256:
            raise ValueError("calibration manifest semantic identity mismatch")
        if (
            protocol.inventory_sha256 != inventory.sha256
            or protocol.hardware_envelope_sha256 != hardware.sha256
            or manifest.inventory_sha256 != inventory.sha256
            or manifest.inventory_source_receipt_sha256
            != inventory.source_receipt_sha256
            or manifest.hardware_envelope_sha256 != hardware.sha256
            or manifest.protocol_sha256 != protocol.sha256
        ):
            raise ValueError("calibration source lineage differs")
        ready_uuids = {device.uuid for device in inventory.devices if device.ready}
        for group in manifest.groups:
            if group.simultaneous_jobs > len(inventory.devices):
                raise ValueError("calibration cardinality exceeds inventory")
            if any(
                not set(run.gpu_uuids) <= ready_uuids
                for run in (*group.isolated, *group.concurrent)
            ):
                raise ValueError("calibration manifest uses a non-ready GPU")
            if len(group.repetitions) < max(
                protocol.minimum_isolated_repetitions,
                protocol.minimum_concurrent_repetitions,
            ):
                raise ValueError("calibration repetition coverage is incomplete")
            for run in (*group.isolated, *group.concurrent):
                device_envelopes = {
                    inventory.device(uuid).hardware_envelope_sha256
                    for uuid in run.gpu_uuids
                }
                if device_envelopes != {run.hardware_envelope_sha256}:
                    raise ValueError("calibration run uses a wrong hardware envelope")
                expected_topology = content_sha256(
                    {
                        "schema_version": 1,
                        "host_ids": sorted(
                            {inventory.device(uuid).host_id for uuid in run.gpu_uuids}
                        ),
                        "gpu_uuids": list(run.gpu_uuids),
                        "rank_groups": [list(row) for row in run.rank_groups],
                        "devices": [
                            {
                                "uuid": uuid,
                                "hardware_envelope_sha256": (
                                    inventory.device(uuid).hardware_envelope_sha256
                                ),
                                "pci_root": inventory.device(uuid).pci_root,
                                "numa_node": inventory.device(uuid).numa_node,
                                "peer_access_class": (
                                    inventory.device(uuid).peer_access_class
                                ),
                            }
                            for uuid in run.gpu_uuids
                        ],
                        "covering_topology_groups": [
                            group_row.to_dict()
                            for group_row in inventory.topology_groups
                            if set(run.gpu_uuids) <= set(group_row.gpu_uuids)
                        ],
                    }
                )
                if run.topology_sha256 != expected_topology:
                    raise ValueError("calibration manifest topology identity is wrong")
                if run.contention_class != content_sha256(
                    {
                        "cpu_cores": run.cpu_cores,
                        "numa_nodes": run.numa_nodes,
                        "disk_io_class": run.disk_io_class,
                        "network_class": run.network_class,
                    }
                ):
                    raise ValueError("calibration contention class is wrong")
        return InterferenceCalibrationSourceAudit(
            inventory=inventory,
            hardware_envelope=hardware,
            protocol=protocol,
            manifest=manifest,
        )

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "interference_calibration_source_authority",
                "inventory_binding_sha256": self.inventory.sha256,
                "inventory_source_receipt_binding_sha256": (
                    self.inventory_source_receipt.sha256
                ),
                "hardware_envelope_binding_sha256": self.hardware_envelope.sha256,
                "protocol_binding_sha256": self.protocol.sha256,
                "manifest_binding_sha256": self.manifest.sha256,
                "reducer_protocol_sha256": (
                    INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256
                ),
            }
        )


@dataclass(frozen=True)
class InterferenceRawObservation:
    """Metrics freshly derived from one trusted terminal evidence chain."""

    observation_id: str
    terminal_authority_sha256: str
    mode: Literal["isolated", "concurrent"]
    repetition: int
    slot: int
    started_ns: int
    finished_ns: int
    request_ids: tuple[str, ...]
    token_trajectory_sha256: str
    completed_requests: int
    output_tokens: int
    goodput_tps: float
    latency_p99_ms: float
    safety_counters: tuple[tuple[str, int], ...]
    hardware_valid: bool

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.observation_id) is None:
            raise ValueError("raw observation ID is unsafe")
        _require_sha256("raw terminal authority", self.terminal_authority_sha256)
        if self.mode not in {"isolated", "concurrent"}:
            raise ValueError("raw observation mode is unsupported")
        _strict_int("raw repetition", self.repetition)
        _strict_int("raw slot", self.slot)
        if (
            type(self.started_ns) is not int
            or type(self.finished_ns) is not int
            or self.started_ns < 0
            or self.finished_ns <= self.started_ns
        ):
            raise ValueError("raw observation interval is invalid")
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("raw observation request coverage is invalid")
        _require_sha256("raw token trajectory", self.token_trajectory_sha256)
        if (
            type(self.completed_requests) is not int
            or not 0 <= self.completed_requests <= len(self.request_ids)
            or type(self.output_tokens) is not int
            or self.output_tokens < 0
        ):
            raise ValueError("raw observation request/token counts are invalid")
        _strict_float("raw goodput", self.goodput_tps, positive=True)
        _strict_float("raw p99 latency", self.latency_p99_ms, positive=True)
        if tuple(name for name, _ in self.safety_counters) != _SAFETY_COUNTERS:
            raise ValueError("raw observation safety coverage is incomplete")
        if any(
            type(value) is not int or value < 0 for _, value in self.safety_counters
        ):
            raise ValueError("raw observation safety counter is invalid")
        if type(self.hardware_valid) is not bool:
            raise TypeError("raw hardware validity must be boolean")


@dataclass(frozen=True)
class InterferenceCalibrationGroupDiagnostic:
    group_id: str
    simultaneous_jobs: int
    status: Literal["FAIL", "UNRESOLVED"]
    reason_codes: tuple[str, ...]
    raw_observation_sha256s: tuple[str, ...]
    goodput_ratios: tuple[tuple[int, int, float], ...]
    p99_latency_ratios: tuple[tuple[int, int, float], ...]

    def __post_init__(self) -> None:
        if self.status not in {"FAIL", "UNRESOLVED"}:
            raise ValueError("interference diagnostic status is unsupported")
        if not self.reason_codes or self.reason_codes != tuple(
            sorted(set(self.reason_codes))
        ):
            raise ValueError("interference diagnostic reasons must be canonical")
        if any(not _is_sha256(value) for value in self.raw_observation_sha256s):
            raise ValueError("raw observation diagnostic digest is invalid")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "group_id": self.group_id,
                "simultaneous_jobs": self.simultaneous_jobs,
                "status": self.status,
                "reason_codes": list(self.reason_codes),
                "raw_observation_sha256s": list(self.raw_observation_sha256s),
                "goodput_ratios": [list(row) for row in self.goodput_ratios],
                "p99_latency_ratios": [list(row) for row in self.p99_latency_ratios],
            }
        )


def _raw_observation_sha256(value: InterferenceRawObservation) -> str:
    return content_sha256(
        {
            "observation_id": value.observation_id,
            "terminal_authority_sha256": value.terminal_authority_sha256,
            "mode": value.mode,
            "repetition": value.repetition,
            "slot": value.slot,
            "started_ns": value.started_ns,
            "finished_ns": value.finished_ns,
            "request_ids": list(value.request_ids),
            "token_trajectory_sha256": value.token_trajectory_sha256,
            "completed_requests": value.completed_requests,
            "output_tokens": value.output_tokens,
            "goodput_tps": value.goodput_tps,
            "latency_p99_ms": value.latency_p99_ms,
            "safety_counters": [list(row) for row in value.safety_counters],
            "hardware_valid": value.hardware_valid,
        }
    )


def diagnose_interference_calibration(
    group: InterferenceCalibrationGroup,
    observations: Sequence[InterferenceRawObservation],
) -> InterferenceCalibrationGroupDiagnostic:
    """Reduce raw metrics without minting a scheduling permission.

    This diagnostic function is intentionally incapable of returning ``PASS``.
    Only the path-bound formal authority may eventually produce an envelope,
    after a new registered acceptance protocol is added.
    """

    if type(group) is not InterferenceCalibrationGroup:
        raise TypeError("interference diagnostic requires one exact group")
    rows = tuple(observations)
    declared = {row.observation_id: row for row in (*group.isolated, *group.concurrent)}
    by_id = {row.observation_id: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(declared):
        raise ValueError("raw observations differ from manifest coverage")
    if any(
        row.terminal_authority_sha256
        != declared[row.observation_id].terminal_authority_sha256
        or row.mode != declared[row.observation_id].mode
        or row.repetition != declared[row.observation_id].repetition
        or row.slot != declared[row.observation_id].slot
        for row in rows
    ):
        raise ValueError("raw observation identity differs from manifest")

    reasons: set[str] = set()
    for row in rows:
        if row.completed_requests != len(row.request_ids):
            reasons.add("incomplete_request_coverage")
        if any(value != 0 for _, value in row.safety_counters):
            reasons.add("nonzero_safety_counter")
        if not row.hardware_valid:
            reasons.add("invalid_hardware_evidence")
    isolated = {
        (row.repetition, row.slot): row for row in rows if row.mode == "isolated"
    }
    concurrent = {
        (row.repetition, row.slot): row for row in rows if row.mode == "concurrent"
    }
    goodput_ratios: list[tuple[int, int, float]] = []
    latency_ratios: list[tuple[int, int, float]] = []
    for key in sorted(isolated):
        left = isolated[key]
        right = concurrent[key]
        if (
            left.request_ids != right.request_ids
            or left.token_trajectory_sha256 != right.token_trajectory_sha256
        ):
            reasons.add("terminal_token_trajectory_change")
        goodput_ratios.append((key[0], key[1], right.goodput_tps / left.goodput_tps))
        latency_ratios.append(
            (key[0], key[1], right.latency_p99_ms / left.latency_p99_ms)
        )
    isolated_intervals = sorted(
        (row.started_ns, row.finished_ns) for row in isolated.values()
    )
    if any(
        right_start < left_finish
        for (_, left_finish), (right_start, _) in pairwise(isolated_intervals)
    ):
        reasons.add("isolated_interval_overlap")
    for repetition in group.repetitions:
        concurrent_rows = tuple(
            row for row in concurrent.values() if row.repetition == repetition
        )
        if max(row.started_ns for row in concurrent_rows) >= min(
            row.finished_ns for row in concurrent_rows
        ):
            reasons.add("concurrent_interval_nonoverlap")
    status: Literal["FAIL", "UNRESOLVED"]
    if reasons:
        status = "FAIL"
    else:
        status = "UNRESOLVED"
        reasons.add(INTERFERENCE_ACCEPTANCE_PROTOCOL_UNREGISTERED_REASON)
    return InterferenceCalibrationGroupDiagnostic(
        group_id=group.group_id,
        simultaneous_jobs=group.simultaneous_jobs,
        status=status,
        reason_codes=tuple(sorted(reasons)),
        raw_observation_sha256s=tuple(
            _raw_observation_sha256(row)
            for row in sorted(rows, key=lambda item: item.observation_id)
        ),
        goodput_ratios=tuple(goodput_ratios),
        p99_latency_ratios=tuple(latency_ratios),
    )


def require_release_interference_attester() -> None:
    """Require the source-owned release trust root, never a caller policy."""

    policy = require_release_trusted_attester_policy(RELEASE_TRUSTED_ATTESTER_POLICY)
    if not policy.release_ready:
        raise InterferenceCalibrationBlockedError(
            TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON
        )


def _percentile_99(values: Sequence[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("p99 latency requires completed requests")
    index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return ordered[index]


def _output_token_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    raw = row.get("output_token_ids")
    if type(raw) is not str:
        raise ValueError("calibration request lacks ordered output token IDs")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("calibration output token IDs are invalid JSON") from error
    if type(value) is not list or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise ValueError("calibration output token IDs are malformed")
    return tuple(value)


def _load_raw_observation(
    declared: InterferenceCalibrationRun,
    authority: AssignmentTerminalAuthority,
    *,
    inventory: GpuInventory,
) -> InterferenceRawObservation:
    plan = authority.plan
    assignment = next(
        (
            row
            for wave in plan.dispatch_plan.waves
            for row in wave.assignments
            if row.assignment_id == declared.assignment_sha256
        ),
        None,
    )
    if assignment is None:
        raise ValueError("calibration terminal plan lacks its assignment")
    binding = authority.revalidate(
        registry=plan.dispatch_context.registry,
        inventory=inventory,
        assignment_sha256=assignment.assignment_id,
        budget_sha256=plan.budget.sha256,
        physical_gpu_uuids=assignment.gpu_uuids,
    )
    root = Path(plan.runtime_plan.cell.resources.evidence_root)
    request_rows: tuple[dict[str, Any], ...] | None = None
    performance_rows: list[dict[str, Any]] = []
    for rank in range(len(assignment.gpu_uuids)):
        completed = load_completed_evidence(root, run_id=binding.run_id, rank=rank)
        if completed is None:
            raise ValueError("calibration terminal evidence is incomplete")
        rank_runs = pq.read_table(completed["run"]).to_pylist()
        rank_requests = tuple(pq.read_table(completed["request"]).to_pylist())
        rank_performance = pq.read_table(completed["performance"]).to_pylist()
        if len(rank_runs) != 1 or len(rank_performance) != 1:
            raise ValueError("calibration terminal rank coverage is malformed")
        started = rank_runs[0].get("started_ns")
        finished = rank_runs[0].get("completed_ns")
        if type(started) is not int or type(finished) is not int:
            raise ValueError("calibration run lacks terminal timestamps")
        if request_rows is None:
            request_rows = rank_requests
        elif request_rows != rank_requests:
            raise ValueError("calibration gang ranks disagree on request evidence")
        performance_rows.extend(rank_performance)
    if request_rows is None or not request_rows:
        raise ValueError("calibration observation has no requests")
    request_ids: list[str] = []
    token_rows: list[tuple[str, tuple[int, ...]]] = []
    arrivals: list[int] = []
    completions: list[int] = []
    latencies_ms: list[float] = []
    output_tokens = 0
    for row in request_rows:
        request_id = row.get("request_id")
        arrival = row.get("arrival_ns")
        completed_ns = row.get("completed_ns")
        if (
            type(request_id) is not str
            or not request_id
            or type(arrival) is not int
            or type(completed_ns) is not int
            or completed_ns <= arrival
            or row.get("outcome_status") != "completed"
            or row.get("error_code") is not None
        ):
            raise ValueError("calibration request terminal row is incomplete")
        tokens = _output_token_ids(row)
        request_ids.append(request_id)
        token_rows.append((request_id, tokens))
        arrivals.append(arrival)
        completions.append(completed_ns)
        latencies_ms.append((completed_ns - arrival) / 1_000_000.0)
        output_tokens += len(tokens)
    if len(request_ids) != len(set(request_ids)) or output_tokens < 1:
        raise ValueError("calibration request coverage is duplicated or empty")
    elapsed_ns = max(completions) - min(arrivals)
    if elapsed_ns <= 0:
        raise ValueError("calibration scored interval is invalid")
    safety = []
    for counter in _SAFETY_COUNTERS:
        values = tuple(row.get(counter) for row in performance_rows)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("calibration safety evidence is missing")
        safety.append((counter, sum(values)))
    return InterferenceRawObservation(
        observation_id=declared.observation_id,
        terminal_authority_sha256=authority.sha256,
        mode=declared.mode,
        repetition=declared.repetition,
        slot=declared.slot,
        started_ns=min(arrivals),
        finished_ns=max(completions),
        request_ids=tuple(request_ids),
        token_trajectory_sha256=content_sha256(token_rows),
        completed_requests=len(request_ids),
        output_tokens=output_tokens,
        goodput_tps=output_tokens / (elapsed_ns / 1_000_000_000.0),
        latency_p99_ms=_percentile_99(latencies_ms),
        safety_counters=tuple(safety),
        hardware_valid=True,
    )


@dataclass(frozen=True)
class InterferenceCalibrationReduction:
    authority_sha256: str
    source_audit_sha256: str
    diagnostics: tuple[InterferenceCalibrationGroupDiagnostic, ...]

    def __post_init__(self) -> None:
        _require_sha256("interference authority", self.authority_sha256)
        _require_sha256("interference source audit", self.source_audit_sha256)
        group_ids = tuple(row.group_id for row in self.diagnostics)
        if not group_ids or group_ids != tuple(sorted(set(group_ids))):
            raise ValueError("interference diagnostics must be group-sorted and unique")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": 1,
                "kind": "interference_calibration_raw_reduction",
                "authority_sha256": self.authority_sha256,
                "source_audit_sha256": self.source_audit_sha256,
                "diagnostic_sha256s": [row.sha256 for row in self.diagnostics],
                "acceptance_status": "UNREGISTERED",
            }
        )


@dataclass(frozen=True)
class InterferenceCalibrationAuthority:
    """Raw terminal authority; incapable of silently accepting bare rule SHA values."""

    schema_version: int
    source: InterferenceCalibrationSourceAuthority
    terminal_authorities: tuple[AssignmentTerminalAuthority, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("only interference calibration authority schema 1 works")
        if type(self.source) is not InterferenceCalibrationSourceAuthority:
            raise TypeError("calibration authority requires exact path-bound sources")
        from lightcone_spec.experiments.completion_authority import (
            AssignmentTerminalAuthority,
        )

        if any(
            type(authority) is not AssignmentTerminalAuthority
            for authority in self.terminal_authorities
        ):
            raise TypeError("calibration requires exact terminal authorities")
        identities = tuple(authority.sha256 for authority in self.terminal_authorities)
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("calibration terminal authorities must be unique")

    @cached_property
    def sha256(self) -> str:
        return content_sha256(
            {
                "schema_version": self.schema_version,
                "kind": "interference_calibration_authority",
                "source_authority_sha256": self.source.sha256,
                "terminal_authority_sha256s": [
                    authority.sha256 for authority in self.terminal_authorities
                ],
                "trusted_attester_policy_sha256": (
                    RELEASE_TRUSTED_ATTESTER_POLICY.sha256
                ),
                "reducer_protocol_sha256": (
                    INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256
                ),
            }
        )

    def audit_inputs(self) -> InterferenceCalibrationSourceAudit:
        audit = self.source.audit()
        declared = {
            run.terminal_authority_sha256: run
            for group in audit.manifest.groups
            for run in (*group.isolated, *group.concurrent)
        }
        actual = {
            authority.sha256: authority for authority in self.terminal_authorities
        }
        if set(actual) != set(declared):
            raise ValueError("terminal authority coverage differs from raw manifest")
        for authority_sha256, run in declared.items():
            authority = actual[authority_sha256]
            plan = authority.plan
            if (
                plan.dispatch_context.inventory != audit.inventory
                or plan.sha256 != run.execution_plan_sha256
                or plan.budget.sha256 != run.budget_sha256
                or plan.load_plan.paired_replay_sha256 != run.load_plan_sha256
                or authority.run_nonce_sha256 != run.run_nonce_sha256
            ):
                raise ValueError("calibration terminal authority has foreign lineage")
            identity = plan.runtime_plan.cell.identity
            if (
                identity.experiment != "preflight"
                or identity.task != "simultaneous_single_gpu_interference"
            ):
                raise ValueError(
                    "confirmation or scientific-result data cannot enter "
                    "interference calibration"
                )
            assignment = next(
                (
                    row
                    for wave in plan.dispatch_plan.waves
                    for row in wave.assignments
                    if row.assignment_id == run.assignment_sha256
                ),
                None,
            )
            if assignment is None:
                raise ValueError("calibration plan lacks declared assignment")
            _validate_run_against_assignment(
                run,
                assignment=assignment,
                claim=assignment.work_item.claim,
                inventory=audit.inventory,
            )
        return audit

    def revalidate(self) -> InterferenceCalibrationReduction:
        """Replay full raw evidence and return only a non-authorizing diagnostic."""

        audit = self.audit_inputs()
        require_release_interference_attester()
        authorities = {
            authority.sha256: authority for authority in self.terminal_authorities
        }
        diagnostics = []
        for group in audit.manifest.groups:
            observations = tuple(
                _load_raw_observation(
                    run,
                    authorities[run.terminal_authority_sha256],
                    inventory=audit.inventory,
                )
                for run in (*group.isolated, *group.concurrent)
            )
            diagnostics.append(diagnose_interference_calibration(group, observations))
        return InterferenceCalibrationReduction(
            authority_sha256=self.sha256,
            source_audit_sha256=audit.sha256,
            diagnostics=tuple(diagnostics),
        )

    def require_envelope(self) -> InterferenceEnvelope:
        """Refuse calibrated scheduling until an acceptance protocol is registered."""

        self.revalidate()
        raise InterferenceCalibrationBlockedError(
            INTERFERENCE_ACCEPTANCE_PROTOCOL_UNREGISTERED_REASON
        )


def require_calibrated_interference_execution_authority(
    envelope: InterferenceEnvelope,
    *,
    authority: InterferenceCalibrationAuthority | None,
) -> None:
    """Formal-consumer hook for a calibrated (non-serial) envelope.

    Serial deny-all envelopes need no calibration authority.  Any rule-bearing
    envelope requires the exact raw authority and can currently only end in a
    named BLOCKED state; a caller-provided rule/evidence digest is never enough.
    """

    if type(envelope) is not InterferenceEnvelope:
        raise TypeError("interference execution requires an exact envelope")
    if not envelope.rules:
        if authority is not None:
            raise ValueError("serial interference envelope cannot carry calibration")
        return
    if type(authority) is not InterferenceCalibrationAuthority:
        raise InterferenceCalibrationBlockedError(
            CALIBRATED_INTERFERENCE_RAW_AUTHORITY_REQUIRED_REASON
        )
    reduced = authority.require_envelope()
    if reduced != envelope:  # pragma: no cover - current protocol always blocks
        raise ValueError("calibrated envelope differs from its raw authority")


__all__ = [
    "CALIBRATED_INTERFERENCE_RAW_AUTHORITY_REQUIRED_REASON",
    "INTERFERENCE_ACCEPTANCE_PROTOCOL_UNREGISTERED_REASON",
    "INTERFERENCE_CALIBRATION_REDUCER_PROTOCOL_SHA256",
    "TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON",
    "InterferenceCalibrationAuthority",
    "InterferenceCalibrationBlockedError",
    "InterferenceCalibrationGroup",
    "InterferenceCalibrationGroupDiagnostic",
    "InterferenceCalibrationManifest",
    "InterferenceCalibrationProtocol",
    "InterferenceCalibrationReduction",
    "InterferenceCalibrationRun",
    "InterferenceCalibrationSourceAudit",
    "InterferenceCalibrationSourceAuthority",
    "InterferenceHardwareDevice",
    "InterferenceHardwareEnvelope",
    "InterferenceHardwareTopology",
    "InterferenceRawObservation",
    "RawInterferenceJsonBinding",
    "diagnose_interference_calibration",
    "require_calibrated_interference_execution_authority",
    "require_release_interference_attester",
]
