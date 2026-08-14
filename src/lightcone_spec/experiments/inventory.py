"""First-party, content-bound GPU inventory and safe interference inputs."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lightcone_spec.doctor import _parse_topology
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
    InterferenceEnvelope,
    _canonical_pci_bus_id,
    content_sha256,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PCI_SYSFS_BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
_GPU_QUERY = (
    "index,uuid,name,memory.total,driver_version,compute_cap,pci.bus_id,"
    "power.limit,temperature.gpu.tlimit,clocks.max.sm,persistence_mode"
)


def _canonical_sha256(value: object) -> str:
    return content_sha256(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_command(argv: Sequence[str]) -> str:
    completed = subprocess.run(
        tuple(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"inventory probe failed for {argv[0]!r}: exit {completed.returncode}"
        )
    return completed.stdout


def _strict_csv_rows(
    value: str, *, columns: int, label: str
) -> tuple[tuple[str, ...], ...]:
    rows = tuple(
        tuple(field.strip() for field in row)
        for row in csv.reader(io.StringIO(value))
        if any(field.strip() for field in row)
    )
    if any(len(row) != columns or any(not field for field in row) for row in rows):
        raise ValueError(f"{label} has an ambiguous CSV schema")
    return rows


def _pci_locality(pci_bus_id: str, *, sysfs_root: Path) -> tuple[str, int]:
    canonical_bus_id = _canonical_pci_bus_id(pci_bus_id)
    device_path = sysfs_root / "bus" / "pci" / "devices" / canonical_bus_id
    try:
        resolved = device_path.resolve(strict=True)
        numa_node = int((device_path / "numa_node").read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise ValueError("GPU PCI/NUMA locality is unavailable") from error
    if numa_node < 0:
        raise ValueError("GPU NUMA locality is unresolved")
    roots = tuple(part for part in resolved.parts if _PCI_SYSFS_BDF.fullmatch(part))
    if not roots:
        raise ValueError("GPU PCI root cannot be derived from sysfs")
    return roots[0].lower(), numa_node


def _machine_identity(*, machine_id_path: Path, hostname: str) -> tuple[str, str]:
    if not hostname.strip():
        raise ValueError("host name is unavailable")
    if (
        not machine_id_path.is_file()
        or machine_id_path.is_symlink()
        or not machine_id_path.read_text(encoding="utf-8").strip()
    ):
        raise ValueError("stable machine identity is unavailable")
    machine_id_sha256 = _file_sha256(machine_id_path)
    return (
        content_sha256(
            {
                "schema_version": 1,
                "hostname": hostname,
                "machine_id_sha256": machine_id_sha256,
            }
        ),
        machine_id_sha256,
    )


def collect_gpu_inventory(
    *,
    challenge_nonce_sha256: str,
    command_runner: Callable[[Sequence[str]], str] = _run_command,
    sysfs_root: str | Path = "/sys",
    machine_id_path: str | Path = "/etc/machine-id",
    hostname: str | None = None,
) -> tuple[GpuInventory, dict[str, Any]]:
    """Collect one exact same-host inventory or fail without an artifact.

    The returned receipt intentionally contains probe output and is therefore a
    runtime artifact, not source material.  Its digest is the inventory's
    ``source_receipt_sha256``.
    """

    if (
        not isinstance(challenge_nonce_sha256, str)
        or _SHA256.fullmatch(challenge_nonce_sha256) is None
    ):
        raise ValueError("inventory challenge nonce must be lowercase SHA-256")
    host_name = socket.gethostname() if hostname is None else hostname
    host_id, machine_id_sha256 = _machine_identity(
        machine_id_path=Path(machine_id_path), hostname=host_name
    )
    gpu_argv = (
        "nvidia-smi",
        f"--query-gpu={_GPU_QUERY}",
        "--format=csv,noheader,nounits",
    )
    process_argv = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    )
    topology_argv = ("nvidia-smi", "topo", "-m")
    gpu_raw = command_runner(gpu_argv)
    process_raw = command_runner(process_argv)
    topology_raw = command_runner(topology_argv)
    gpu_rows = _strict_csv_rows(gpu_raw, columns=11, label="GPU inventory probe")
    if not gpu_rows:
        raise ValueError("GPU inventory probe returned no devices")
    try:
        indexed = tuple(sorted(gpu_rows, key=lambda row: int(row[0])))
    except ValueError as error:
        raise ValueError("GPU inventory index is invalid") from error
    if tuple(int(row[0]) for row in indexed) != tuple(range(len(indexed))):
        raise ValueError("GPU inventory indices must be contiguous from zero")
    if len({row[1] for row in indexed}) != len(indexed):
        raise ValueError("GPU inventory contains duplicate UUIDs")

    process_rows = _strict_csv_rows(process_raw, columns=3, label="GPU process probe")
    known_uuids = {row[1] for row in indexed}
    if any(row[0] not in known_uuids for row in process_rows):
        raise ValueError("GPU process probe references an unknown device")
    processes_by_uuid: dict[str, list[str]] = {uuid: [] for uuid in known_uuids}
    for uuid, pid, process_name in process_rows:
        if not pid.isdigit() or int(pid) <= 0:
            raise ValueError("GPU process probe contains an invalid PID")
        processes_by_uuid[uuid].append(f"{pid}:{process_name}")

    topology = _parse_topology(topology_raw)
    expected_rows = [f"GPU{index}" for index in range(len(indexed))]
    if (
        topology.get("parse_error") is not None
        or topology.get("gpu_rows") != expected_rows
        or not isinstance(topology.get("pairs"), list)
    ):
        raise ValueError("GPU topology probe is incomplete")
    pairs = tuple(topology["pairs"])
    if len(pairs) != len(indexed) * (len(indexed) - 1) // 2 or any(
        not isinstance(pair, dict)
        or pair.get("link") != pair.get("reciprocal_link")
        or not isinstance(pair.get("link"), str)
        or not pair["link"]
        for pair in pairs
    ):
        raise ValueError("GPU topology probe is not symmetric and complete")

    locality_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    for row in indexed:
        (
            index,
            uuid,
            model,
            memory_mib,
            driver_version,
            compute_capability,
            pci_bus_id,
            power_limit,
            thermal_limit,
            max_sm_clock,
            persistence_mode,
        ) = row
        try:
            memory_bytes = int(memory_mib) * 1024 * 1024
            compute = tuple(
                int(component) for component in compute_capability.split(".")
            )
            power_watts = float(power_limit)
            thermal_celsius = float(thermal_limit)
            max_sm_mhz = int(max_sm_clock)
        except ValueError as error:
            raise ValueError("GPU numeric inventory fields are invalid") from error
        if len(compute) != 2 or max_sm_mhz <= 0:
            raise ValueError("GPU compute capability or clock is invalid")
        canonical_pci_bus_id = _canonical_pci_bus_id(pci_bus_id)
        pci_root, numa_node = _pci_locality(
            canonical_pci_bus_id, sysfs_root=Path(sysfs_root)
        )
        locality_rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "pci_bus_id": canonical_pci_bus_id,
                "pci_root": pci_root,
                "numa_node": numa_node,
            }
        )
        link_types = sorted(
            {
                str(pair["link"])
                for pair in pairs
                if pair["left"] == f"GPU{index}" or pair["right"] == f"GPU{index}"
            }
        )
        parsed_rows.append(
            {
                "uuid": uuid,
                "model": model,
                "memory_bytes": memory_bytes,
                "driver_version": driver_version,
                "compute_capability": compute,
                "pci_bus_id": canonical_pci_bus_id,
                "pci_root": pci_root,
                "numa_node": numa_node,
                "interconnects": tuple(link_types or ("NONE_SINGLE_GPU",)),
                "peer_access_class": "topology:" + ",".join(link_types or ("none",)),
                "clock_policy": (
                    f"persistence={persistence_mode};max_sm_mhz={max_sm_mhz}"
                ),
                "power_limit_watts": power_watts,
                "thermal_limit_celsius": thermal_celsius,
                "reserved_processes": tuple(sorted(processes_by_uuid[uuid])),
            }
        )

    topology_group_id = (
        "same-host-" + content_sha256({"host_id": host_id, "pairs": pairs})[:16]
    )
    receipt_content: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gpu_inventory_probe_receipt",
        "challenge_nonce_sha256": challenge_nonce_sha256,
        "host_id": host_id,
        "hostname": host_name,
        "machine_id_sha256": machine_id_sha256,
        "commands": {
            "gpu": {"argv": list(gpu_argv), "stdout": gpu_raw},
            "processes": {"argv": list(process_argv), "stdout": process_raw},
            "topology": {"argv": list(topology_argv), "stdout": topology_raw},
        },
        "parsed_topology": topology,
        "pci_locality": locality_rows,
    }
    receipt_sha256 = _canonical_sha256(receipt_content)
    receipt = {**receipt_content, "receipt_sha256": receipt_sha256}
    devices = tuple(
        sorted(
            (
                GpuDevice(
                    uuid=row["uuid"],
                    host_id=host_id,
                    model=row["model"],
                    memory_bytes=row["memory_bytes"],
                    compute_capability=row["compute_capability"],
                    pci_bus_id=row["pci_bus_id"],
                    pci_root=row["pci_root"],
                    numa_node=row["numa_node"],
                    interconnects=row["interconnects"],
                    peer_access_class=row["peer_access_class"],
                    clock_policy=row["clock_policy"],
                    power_limit_watts=row["power_limit_watts"],
                    thermal_limit_celsius=row["thermal_limit_celsius"],
                    availability=(
                        GpuAvailability.READY
                        if not row["reserved_processes"]
                        else GpuAvailability.RESERVED
                    ),
                    reserved_processes=row["reserved_processes"],
                    allowed_topology_groups=(topology_group_id,),
                )
                for row in parsed_rows
            ),
            key=lambda device: device.uuid,
        )
    )
    group = GpuTopologyGroup(
        group_id=topology_group_id,
        host_id=host_id,
        gpu_uuids=tuple(device.uuid for device in devices),
        fabric="mixed:"
        + ",".join(sorted({str(pair["link"]) for pair in pairs}) or ["none"]),
        bandwidth_class=content_sha256({"pairs": pairs}),
    )
    inventory = GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(group,),
        source_receipt_sha256=receipt_sha256,
    )
    return inventory, receipt


def build_serial_interference_envelope(
    inventory: GpuInventory,
) -> tuple[InterferenceEnvelope, dict[str, Any]]:
    """Build the only evidence-free safe envelope: no simultaneous jobs."""

    if not isinstance(inventory, GpuInventory):
        raise TypeError("serial interference policy requires a GPU inventory")
    receipt_content: Mapping[str, Any] = {
        "schema_version": 1,
        "kind": "serial_interference_policy_receipt",
        "inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "policy": "deny_all_simultaneous_execution",
        "calibration_receipt_sha256s": [],
    }
    receipt_sha256 = content_sha256(receipt_content)
    receipt = {**receipt_content, "receipt_sha256": receipt_sha256}
    return (
        InterferenceEnvelope.serial(source_receipt_sha256=receipt_sha256),
        receipt,
    )


__all__ = ["build_serial_interference_envelope", "collect_gpu_inventory"]
