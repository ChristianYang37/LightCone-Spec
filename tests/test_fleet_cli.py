from __future__ import annotations

import io
from pathlib import Path

import pytest

import lightcone_spec.cli.main as cli_module
from lightcone_spec.cli.main import (
    _load_gpu_fleet_inventory,
    _write_json,
)
from lightcone_spec.cli.main import (
    main as cli_main,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.registry import content_sha256


def _inventory(host_id: str) -> GpuInventory:
    return GpuInventory(
        schema_version=1,
        devices=(
            GpuDevice(
                uuid=f"GPU-{host_id}",
                host_id=host_id,
                model="H100-SXM",
                memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                pci_bus_id="0000:01:00.0",
                pci_root="root-0",
                numa_node=0,
                interconnects=("PCIe5",),
                peer_access_class="PCIe",
                clock_policy="locked-1980MHz",
                power_limit_watts=700.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=(),
            ),
        ),
        topology_groups=(),
        source_receipt_sha256=content_sha256({"inventory": host_id}),
    )


def _write_host_pair(tmp_path: Path, host_id: str) -> tuple[str, str]:
    inventory = _inventory(host_id)
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256({"envelope": host_id})
    )
    inventory_path = tmp_path / f"{host_id}-inventory.json"
    envelope_path = tmp_path / f"{host_id}-envelope.json"
    _write_json(inventory_path, inventory.to_dict())
    _write_json(envelope_path, envelope.to_dict())
    return str(inventory_path), str(envelope_path)


def test_assemble_gpu_fleet_inventory_cli_round_trips(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "fleet.json"
    host_b_inventory, host_b_envelope = _write_host_pair(tmp_path, "host-b")
    host_a_inventory, host_a_envelope = _write_host_pair(tmp_path, "host-a")
    result = cli_main(
        [
            "assemble-gpu-fleet-inventory",
            "--inventory",
            host_b_inventory,
            "--inventory",
            host_a_inventory,
            "--interference-envelope",
            host_b_envelope,
            "--interference-envelope",
            host_a_envelope,
            "--output",
            str(output),
        ]
    )

    fleet = _load_gpu_fleet_inventory(output)
    assert result == 0
    assert fleet.host_ids == ("host-a", "host-b")
    assert fleet.gpu_count == 2
    assert capsys.readouterr().out == f"{fleet.sha256}\n"
    assert Path(f"{output}.sha256").is_file()


class _BinaryInput:
    def __init__(self, value: bytes) -> None:
        self.buffer = io.BytesIO(value)


class _BinaryOutput:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


def test_execute_dispatch_wave_host_stdin_writes_only_worker_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_request = b'{"canonical":"request"}\n'
    expected_response = b'{"canonical":"response"}\n'

    async def execute(request: bytes) -> tuple[int, bytes]:
        assert request == expected_request
        return 0, expected_response

    stdin = _BinaryInput(expected_request)
    stdout = _BinaryOutput()
    monkeypatch.setattr(cli_module, "execute_host_local_wave_request", execute)
    monkeypatch.setattr(cli_module.sys, "stdin", stdin)
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)

    result = cli_main(["execute-dispatch-wave", "--host-request-stdin"])

    assert result == 0
    assert stdout.buffer.getvalue() == expected_response


def test_execute_dispatch_wave_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli_main(
            [
                "execute-dispatch-wave",
                "--host-request-stdin",
                "--materialization-manifest",
                str(tmp_path / "manifest.json"),
            ]
        )


@pytest.mark.parametrize(
    "request_body",
    (b"", b"x" * (cli_module.MAX_REQUEST_BYTES + 1)),
    ids=("malformed", "oversize"),
)
def test_execute_dispatch_wave_host_stdin_fails_closed_without_extra_stdout(
    request_body: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdin = _BinaryInput(request_body)
    stdout = _BinaryOutput()
    monkeypatch.setattr(cli_module.sys, "stdin", stdin)
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)

    result = cli_main(["execute-dispatch-wave", "--host-request-stdin"])

    assert result == 42
    assert stdout.buffer.getvalue() == b""
