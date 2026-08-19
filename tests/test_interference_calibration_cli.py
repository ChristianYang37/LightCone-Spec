from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lightcone_spec.cli.main import main
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.interference_authority import (
    TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON,
    materialize_interference_calibration_bootstrap_authority,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)
from lightcone_spec.experiments.registry import (
    content_sha256,
)
from lightcone_spec.experiments.stage_activation import (
    materialize_registry_stage_activation,
)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _write_bound(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(
        _canonical_sha256(value) + "\n",
        encoding="ascii",
    )
    return path


def _registry(tmp_path: Path) -> tuple[Path, object]:
    path = tmp_path / "registry.json"
    cache_root = str(tmp_path / "cache")
    evidence_root = str(tmp_path / "evidence")
    assert (
        main(
            [
                "build-industrial-registry",
                "--legacy-diagnostic",
                "--logical-gpu-slot",
                "logical-rank-slot-a",
                "logical-rank-slot-b",
                "--cache-root",
                cache_root,
                "--evidence-root",
                evidence_root,
                "--output",
                str(path),
            ]
        )
        == 0
    )
    return path, build_industrial_registry(
        gpu_uuids=("logical-rank-slot-a", "logical-rank-slot-b"),
        cache_root=cache_root,
        evidence_root=evidence_root,
    )


def _inventory() -> GpuInventory:
    group_id = "same-host-calibration"
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-calibration-{index}",
            host_id="calibration-host",
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="0000:00:00.0",
            numa_node=index,
            interconnects=("NVLink4",),
            peer_access_class="NVSwitch",
            clock_policy="locked",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=(group_id,),
        )
        for index in range(2)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id=group_id,
                host_id="calibration-host",
                gpu_uuids=tuple(device.uuid for device in devices),
                fabric="NVSwitch",
                bandwidth_class="full",
            ),
        ),
        source_receipt_sha256=content_sha256({"inventory": "calibration"}),
    )


def test_cli_materializes_only_the_raw_reducer_bootstrap(tmp_path: Path) -> None:
    registry_path, registry = _registry(tmp_path)
    inventory = _inventory()
    inventory_path = _write_bound(tmp_path / "inventory.json", inventory.to_dict())
    runtime_path = _write_bound(tmp_path / "runtime.json", {"runtime": "preflight"})
    split_path = _write_bound(tmp_path / "split.json", {"split": "tuning"})
    manifest = {
        "schema_version": 1,
        "kind": "industrial_registry_stage_activation_manifest",
        "registry_artifact": str(registry_path),
        "experiment": "preflight",
        "runtime_artifact": str(runtime_path),
        "split_artifact": str(split_path),
        "dependency_receipts": [],
    }
    manifest_path = _write_bound(tmp_path / "activation.json", manifest)
    receipt_path = tmp_path / "bootstrap-receipt.json"
    envelope_path = tmp_path / "bootstrap-envelope.json"

    assert (
        main(
            [
                "materialize-interference-calibration-bootstrap",
                "--registry",
                str(registry_path),
                "--activation-manifest",
                str(manifest_path),
                "--inventory",
                str(inventory_path),
                "--receipt-output",
                str(receipt_path),
                "--output",
                str(envelope_path),
            ]
        )
        == 0
    )

    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=content_sha256({"runtime": "preflight"}),
        split_sha256=content_sha256({"split": "tuning"}),
    )
    authority = materialize_interference_calibration_bootstrap_authority(
        registry,
        inventory,
        activation,
    )
    assert (
        json.loads(receipt_path.read_text(encoding="utf-8")) == authority.source_receipt
    )
    assert (
        InterferenceEnvelope.from_dict(
            json.loads(envelope_path.read_text(encoding="utf-8"))
        )
        == authority.bootstrap_envelope
    )
    assert {rule.simultaneous_jobs for rule in authority.bootstrap_envelope.rules} == {
        2
    }


def test_cli_reducer_blocks_before_opening_untrusted_paths(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    envelope = tmp_path / "must-not-exist.json"

    assert (
        main(
            [
                "reduce-interference-calibration",
                "--authority",
                str(tmp_path / "nonexistent-caller-authority.json"),
                "--envelope-output",
                str(envelope),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "kind": "interference_calibration_reduction_decision",
        "status": "BLOCKED",
        "reason_code": TRUSTED_INTERFERENCE_ATTESTER_UNAVAILABLE_REASON,
        "trusted_attester_id": None,
    }
    assert not envelope.exists()
