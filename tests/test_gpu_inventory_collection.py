from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments.inventory import (
    build_serial_interference_envelope,
    collect_gpu_inventory,
)


def _probe_environment(tmp_path: Path, *, gpu_count: int) -> tuple[Path, Path]:
    sysfs = tmp_path / "sys"
    for index in range(gpu_count):
        device = sysfs / "bus" / "pci" / "devices" / f"0000:{index + 1:02x}:00.0"
        device.mkdir(parents=True)
        (device / "numa_node").write_text(str(index % 2), encoding="utf-8")
    machine_id = tmp_path / "machine-id"
    machine_id.write_text("industrial-host-identity\n", encoding="utf-8")
    return sysfs, machine_id


def _outputs(*, gpu_count: int, processes: str = "") -> dict[str, str]:
    gpu_rows = "\n".join(
        (
            f"{index}, GPU-{index}, NVIDIA RTX PRO 6000 Blackwell Server Edition, "
            f"97887, 580.65.06, 12.0, 0000:{index + 1:02x}:00.0, "
            "600.0, 90.0, 2400, Enabled"
        )
        for index in range(gpu_count)
    )
    names = [f"GPU{index}" for index in range(gpu_count)]
    topology = [" ".join((*names, "CPU", "Affinity"))]
    for left_index, left in enumerate(names):
        links = [
            "X" if left_index == right_index else "PHB"
            for right_index in range(gpu_count)
        ]
        topology.append(" ".join((left, *links, "0-31")))
    return {
        "--query-gpu": gpu_rows + "\n",
        "--query-compute-apps": processes,
        "topo": "\n".join(topology) + "\n",
    }


def _runner(outputs: dict[str, str]):
    def run(argv) -> str:
        if any(str(value).startswith("--query-gpu=") for value in argv):
            return outputs["--query-gpu"]
        if any(str(value).startswith("--query-compute-apps=") for value in argv):
            return outputs["--query-compute-apps"]
        if tuple(argv[-2:]) == ("topo", "-m"):
            return outputs["topo"]
        raise AssertionError(f"unexpected command: {argv!r}")

    return run


@pytest.mark.parametrize("gpu_count", (1, 2, 4, 8, 16))
def test_first_party_inventory_collection_is_deterministic(
    tmp_path: Path, gpu_count: int
) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=gpu_count)
    kwargs = {
        "challenge_nonce_sha256": "a" * 64,
        "command_runner": _runner(_outputs(gpu_count=gpu_count)),
        "sysfs_root": sysfs,
        "machine_id_path": machine_id,
        "hostname": "same-host",
    }
    first, first_receipt = collect_gpu_inventory(**kwargs)
    second, second_receipt = collect_gpu_inventory(**kwargs)
    assert first == second
    assert first_receipt == second_receipt
    assert len(first.devices) == gpu_count
    assert first.source_receipt_sha256 == first_receipt["receipt_sha256"]
    assert first.host_ids == (first.devices[0].host_id,)
    assert all(device.ready for device in first.devices)
    assert len(first.topology_groups) == 1
    assert set(first.topology_groups[0].gpu_uuids) == {
        device.uuid for device in first.devices
    }


def test_inventory_marks_reserved_process_without_hiding_it(tmp_path: Path) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=2)
    outputs = _outputs(
        gpu_count=2,
        processes="GPU-1, 31415, python-worker\n",
    )
    inventory, receipt = collect_gpu_inventory(
        challenge_nonce_sha256="b" * 64,
        command_runner=_runner(outputs),
        sysfs_root=sysfs,
        machine_id_path=machine_id,
        hostname="same-host",
    )
    by_uuid = {device.uuid: device for device in inventory.devices}
    assert by_uuid["GPU-0"].ready
    assert not by_uuid["GPU-1"].ready
    assert by_uuid["GPU-1"].reserved_processes == ("31415:python-worker",)
    assert "GPU-1, 31415, python-worker" in receipt["commands"]["processes"]["stdout"]


def test_inventory_binds_extended_pci_domain_to_canonical_sysfs_bdf(
    tmp_path: Path,
) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=1)
    devices = sysfs / "bus" / "pci" / "devices"
    old_device = devices / "0000:01:00.0"
    (old_device / "numa_node").unlink()
    old_device.rmdir()
    resolved_device = sysfs / "devices" / "pci0000:a7" / "0000:a7:01.0" / "0000:a8:00.0"
    resolved_device.mkdir(parents=True)
    (resolved_device / "numa_node").write_text("1", encoding="utf-8")
    (devices / "0000:a8:00.0").symlink_to(resolved_device, target_is_directory=True)
    outputs = _outputs(gpu_count=1)
    outputs["--query-gpu"] = outputs["--query-gpu"].replace(
        "0000:01:00.0", "00000000:A8:00.0"
    )

    inventory, receipt = collect_gpu_inventory(
        challenge_nonce_sha256="c" * 64,
        command_runner=_runner(outputs),
        sysfs_root=sysfs,
        machine_id_path=machine_id,
        hostname="same-host",
    )

    assert inventory.devices[0].pci_bus_id == "0000:a8:00.0"
    assert inventory.devices[0].pci_root == "0000:a7:01.0"
    assert "00000000:A8:00.0" in receipt["commands"]["gpu"]["stdout"]
    assert receipt["pci_locality"] == [
        {
            "index": 0,
            "uuid": "GPU-0",
            "pci_bus_id": "0000:a8:00.0",
            "pci_root": "0000:a7:01.0",
            "numa_node": 1,
        }
    ]


@pytest.mark.parametrize(
    "pci_bus_id",
    (
        "0000000:01:00.0",
        "000000000:01:00.0",
        "00000000:1:00.0",
        "00000000:01:000.0",
        "00000000:01:20.0",
        "00000000:01:00.8",
        "00000000:01:00.0/..",
        "00010000:01:00.0",
    ),
)
def test_inventory_rejects_malformed_extended_pci_bus_ids(
    tmp_path: Path,
    pci_bus_id: str,
) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=1)
    outputs = _outputs(gpu_count=1)
    outputs["--query-gpu"] = outputs["--query-gpu"].replace("0000:01:00.0", pci_bus_id)

    with pytest.raises(ValueError, match="PCI bus identity is invalid"):
        collect_gpu_inventory(
            challenge_nonce_sha256="d" * 64,
            command_runner=_runner(outputs),
            sysfs_root=sysfs,
            machine_id_path=machine_id,
            hostname="same-host",
        )


def test_inventory_rejects_mixed_width_aliases_for_one_physical_bdf(
    tmp_path: Path,
) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=2)
    outputs = _outputs(gpu_count=2)
    outputs["--query-gpu"] = outputs["--query-gpu"].replace(
        "0000:01:00.0", "00000000:01:00.0"
    )
    outputs["--query-gpu"] = outputs["--query-gpu"].replace(
        "0000:02:00.0", "0000:01:00.0"
    )

    with pytest.raises(ValueError, match="duplicate PCI bus identities"):
        collect_gpu_inventory(
            challenge_nonce_sha256="e" * 64,
            command_runner=_runner(outputs),
            sysfs_root=sysfs,
            machine_id_path=machine_id,
            hostname="same-host",
        )


def test_inventory_rejects_incomplete_or_asymmetric_topology(tmp_path: Path) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=2)
    outputs = _outputs(gpu_count=2)
    outputs["topo"] = "GPU0 GPU1\nGPU0 X PHB\nGPU1 SYS X\n"
    with pytest.raises(ValueError, match="symmetric"):
        collect_gpu_inventory(
            challenge_nonce_sha256="c" * 64,
            command_runner=_runner(outputs),
            sysfs_root=sysfs,
            machine_id_path=machine_id,
            hostname="same-host",
        )


def test_inventory_accepts_nvidia_smi_sgr_topology_header(tmp_path: Path) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=2)
    outputs = _outputs(gpu_count=2)
    topology_lines = outputs["topo"].splitlines()
    topology_lines[0] = f"\x1b[4m{topology_lines[0]}\x1b[0m"
    outputs["topo"] = "\n".join(topology_lines) + "\n"

    inventory, receipt = collect_gpu_inventory(
        challenge_nonce_sha256="f" * 64,
        command_runner=_runner(outputs),
        sysfs_root=sysfs,
        machine_id_path=machine_id,
        hostname="same-host",
    )

    assert len(inventory.devices) == 2
    assert all(device.ready for device in inventory.devices)
    assert receipt["parsed_topology"] == {
        "gpu_rows": ["GPU0", "GPU1"],
        "pairs": [
            {
                "left": "GPU0",
                "right": "GPU1",
                "link": "PHB",
                "reciprocal_link": "PHB",
            }
        ],
        "parse_error": None,
    }


def test_evidence_free_interference_policy_is_strictly_serial(tmp_path: Path) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=8)
    inventory, _ = collect_gpu_inventory(
        challenge_nonce_sha256="d" * 64,
        command_runner=_runner(_outputs(gpu_count=8)),
        sysfs_root=sysfs,
        machine_id_path=machine_id,
        hostname="same-host",
    )
    envelope, receipt = build_serial_interference_envelope(inventory)
    assert envelope.rules == ()
    assert envelope.source_receipt_sha256 == receipt["receipt_sha256"]
    assert receipt["policy"] == "deny_all_simultaneous_execution"


def test_inventory_and_serial_envelope_cli_write_bound_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sysfs, machine_id = _probe_environment(tmp_path, gpu_count=4)
    inventory, receipt = collect_gpu_inventory(
        challenge_nonce_sha256="e" * 64,
        command_runner=_runner(_outputs(gpu_count=4)),
        sysfs_root=sysfs,
        machine_id_path=machine_id,
        hostname="same-host",
    )
    cli = import_module("lightcone_spec.cli.main")
    monkeypatch.setattr(
        cli,
        "collect_gpu_inventory",
        lambda **_kwargs: (inventory, receipt),
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_receipt_path = tmp_path / "inventory-receipt.json"
    assert (
        cli._collect_gpu_inventory(
            SimpleNamespace(
                challenge_nonce_sha256="e" * 64,
                receipt_output=str(inventory_receipt_path),
                output=str(inventory_path),
            )
        )
        == 0
    )
    assert json.loads(inventory_path.read_text(encoding="utf-8")) == inventory.to_dict()
    assert Path(f"{inventory_path}.sha256").is_file()

    envelope_path = tmp_path / "interference.json"
    envelope_receipt_path = tmp_path / "interference-receipt.json"
    assert (
        cli._build_interference_envelope(
            SimpleNamespace(
                inventory=str(inventory_path),
                receipt_output=str(envelope_receipt_path),
                output=str(envelope_path),
            )
        )
        == 0
    )
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["rules"] == []
    assert Path(f"{envelope_path}.sha256").is_file()
