from __future__ import annotations

from types import SimpleNamespace

import pytest

import lightcone_spec.cli.main as main_module
from lightcone_spec.experiments import stage_activation
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.preflight_authority import (
    PreflightSealControlBinding,
)
from lightcone_spec.experiments.registry import (
    build_industrial_registry,
    build_legacy_industrial_registry,
    content_sha256,
)


def _sha(label: str) -> str:
    return content_sha256({"preflight-cli-validation": label})


def _inventory() -> GpuInventory:
    uuids = ("GPU-validation-0", "GPU-validation-1")
    return GpuInventory(
        schema_version=1,
        devices=tuple(
            GpuDevice(
                uuid=uuid,
                host_id="validation-host",
                model="RTX-PRO-6000",
                memory_bytes=96 * 1024**3,
                compute_capability=(12, 0),
                pci_bus_id=f"0000:0{index + 1}:00.0",
                pci_root="root-0",
                numa_node=0,
                interconnects=("PCIe5",),
                peer_access_class="pcie-peer",
                clock_policy="locked",
                power_limit_watts=600.0,
                thermal_limit_celsius=83.0,
                availability=GpuAvailability.READY,
                reserved_processes=(),
                allowed_topology_groups=("pair",),
            )
            for index, uuid in enumerate(uuids)
        ),
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair",
                host_id="validation-host",
                gpu_uuids=uuids,
                fabric="PCIe",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=_sha("inventory-source"),
    )


def test_preflight_sealer_consumes_control_binding_in_final_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_legacy_industrial_registry()
    runtime_sha256 = _sha("runtime")
    split_value = {"split": "formal-preflight"}
    split_sha256 = main_module._canonical_sha256(split_value)
    monkeypatch.setattr(
        stage_activation,
        "release_dispatch_rejection_reason",
        lambda _cell: None,
    )
    activation = stage_activation.materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    activated = tuple(
        sorted(
            row.cell_id
            for row in activation.dispositions
            if row.status.value == "ACTIVATED"
        )
    )
    inventory = _inventory()
    raw_completed_sha256 = _sha("raw-completed")
    binding = PreflightSealControlBinding(
        schema_version=1,
        kind="formal_preflight_seal_control_binding",
        status="SEALED",
        registry_sha256=registry.sha256,
        activation_sha256=activation.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=inventory.devices[0].hardware_envelope_sha256,
        raw_completed_cells_sha256=raw_completed_sha256,
        coverage_receipt_sha256=_sha("coverage"),
        coverage_attestation_sha256=_sha("coverage-attestation"),
        capacity_gate_sha256=_sha("capacity"),
        capacity_attestation_sha256=_sha("capacity-attestation"),
        deployment_policy_authorization_sha256=_sha("deployment"),
        trust_bundle_sha256=_sha("bundle"),
        trusted_attester_policy_sha256=_sha("policy"),
        replay_reservation_sha256=_sha("reservation"),
    )
    binding_artifact_sha256 = main_module._canonical_sha256(binding.to_dict())
    written: list[tuple[str, object]] = []
    control_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        main_module, "_load_industrial_registry", lambda _path: registry
    )
    monkeypatch.setattr(main_module, "_load_industrial_receipts", lambda _paths: ())
    monkeypatch.setattr(main_module, "_load_gpu_inventory", lambda _path: inventory)
    monkeypatch.setattr(
        main_module, "_load_stage_activation_plan", lambda _path: activation
    )
    monkeypatch.setattr(main_module, "_load_family_activations", lambda _paths: ())
    monkeypatch.setattr(main_module, "_load_family_power_reductions", lambda _paths: ())
    monkeypatch.setattr(
        main_module,
        "_load_bound_json",
        lambda path: split_value if path == "split.json" else {},
    )
    monkeypatch.setattr(
        main_module,
        "_completed_industrial_cells",
        lambda *_args, **_kwargs: (activated, raw_completed_sha256),
    )
    monkeypatch.setattr(
        main_module,
        "_parse_locked_output_paths",
        lambda _values: {"runtime_envelope": "runtime-envelope.json"},
    )
    monkeypatch.setattr(
        main_module,
        "_validate_preflight_interference_seal_authority",
        lambda **_kwargs: None,
    )

    def control_validator(**kwargs):
        control_calls.append(kwargs)
        return binding

    monkeypatch.setattr(
        main_module,
        "_validate_preflight_control_seal_authorities",
        control_validator,
    )

    def artifact_sha256(path):
        if path == "runtime.json":
            return runtime_sha256
        if path == "control-binding.json":
            return binding_artifact_sha256
        if path == "runtime-envelope.json":
            return _sha("runtime-envelope")
        raise AssertionError(f"unexpected artifact path {path}")

    monkeypatch.setattr(main_module, "_artifact_sha256", artifact_sha256)
    monkeypatch.setattr(
        main_module,
        "_write_json",
        lambda path, value: written.append((str(path), value)),
    )
    args = SimpleNamespace(
        registry="registry.json",
        experiment="preflight",
        runtime_artifact="runtime.json",
        split_artifact="split.json",
        completed_cells="completed.json",
        inventory="inventory.json",
        e2_final_stage_manifest=None,
        interference_calibration_authority="interference.json",
        preflight_coverage_receipt="coverage.json",
        preflight_coverage_attestation="coverage-attestation.json",
        stage_capacity_gate="capacity.json",
        stage_capacity_attestation="capacity-attestation.json",
        control_replay_store="/validation/replay",
        preflight_control_binding_output="control-binding.json",
        activation_plan="activation.json",
        family_activation=[],
        family_power_plan=[],
        dependency_receipt=[],
        locked_output=["runtime_envelope=runtime-envelope.json"],
        output="receipt.json",
    )

    assert main_module._seal_industrial_stage(args) == 0
    assert len(control_calls) == 1
    assert written[0] == ("control-binding.json", binding.to_dict())
    receipt = written[1][1]
    assert receipt["completed_cells_sha256"] == binding_artifact_sha256
    assert receipt["experiment"] == "preflight"


def test_legacy_sealer_rejects_signed_staged_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = build_industrial_registry()
    monkeypatch.setattr(
        main_module, "_load_industrial_registry", lambda _path: registry
    )
    monkeypatch.setattr(main_module, "_load_industrial_receipts", lambda _paths: ())
    monkeypatch.setattr(main_module, "_load_gpu_inventory", lambda _path: _inventory())
    monkeypatch.setattr(main_module, "_load_stage_activation_plan", lambda _path: None)
    monkeypatch.setattr(main_module, "_load_family_activations", lambda _paths: ())
    monkeypatch.setattr(main_module, "_load_family_power_reductions", lambda _paths: ())
    monkeypatch.setattr(main_module, "_artifact_sha256", lambda _path: _sha("runtime"))
    monkeypatch.setattr(main_module, "_load_bound_json", lambda _path: {"split": "x"})
    monkeypatch.setattr(
        main_module,
        "_completed_industrial_cells",
        lambda *_args, **_kwargs: ((_sha("cell"),), _sha("completed")),
    )
    monkeypatch.setattr(
        main_module,
        "_parse_locked_output_paths",
        lambda _values: {"runtime_envelope": "runtime-envelope.json"},
    )
    args = SimpleNamespace(
        registry="registry.json",
        experiment="preflight",
        runtime_artifact="runtime.json",
        split_artifact="split.json",
        completed_cells="completed.json",
        inventory="inventory.json",
        e2_final_stage_manifest=None,
        interference_calibration_authority="interference.json",
        preflight_coverage_receipt=None,
        preflight_coverage_attestation=None,
        stage_capacity_gate=None,
        stage_capacity_attestation=None,
        control_replay_store=None,
        preflight_control_binding_output=None,
        activation_plan=None,
        family_activation=[],
        family_power_plan=[],
        dependency_receipt=[],
        locked_output=["runtime_envelope=runtime-envelope.json"],
        output="receipt.json",
    )
    with pytest.raises(ValueError, match="legacy diagnostic only"):
        main_module._seal_industrial_stage(args)


def test_local_interference_cli_requires_exact_distinct_eight_cell_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cell_ids = tuple(_sha(f"interference-cell-{index}") for index in range(8))
    native_result = [
        f"{cell}=result-{index}.json" for index, cell in enumerate(cell_ids)
    ]
    native_itl = [f"{cell}=itl-{index}.json" for index, cell in enumerate(cell_ids)]
    assert main_module._parse_preflight_cell_proof_paths(native_result) == {
        cell: f"result-{index}.json" for index, cell in enumerate(cell_ids)
    }
    with pytest.raises(ValueError, match="exactly eight"):
        main_module._parse_preflight_cell_proof_paths(native_result[:-1])
    with pytest.raises(ValueError, match="unique"):
        main_module._parse_preflight_cell_proof_paths(
            [*native_result[:-1], native_result[0]]
        )

    args = main_module._parser().parse_args(
        [
            "qualify-formal-preflight-interference",
            "--dispatch-receipt",
            "dispatch.json",
            "--remote-raw-receipt",
            "raw.json",
            *(
                argument
                for value in native_result
                for argument in ("--native-result-proof", value)
            ),
            *(
                argument
                for value in native_itl
                for argument in ("--native-itl-proof", value)
            ),
            "--aggregate-control",
            "control.json",
            "--control-replay-store",
            str(tmp_path),
            "--now-ns",
            "123",
            "--output",
            "proof.json",
        ]
    )
    token = object()
    control = object()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_dispatch."
        "load_formal_preflight_dispatch_receipt",
        lambda path, current_ns: token,
    )
    monkeypatch.setattr(
        main_module.ControlArtifactAttestation,
        "from_dict",
        classmethod(lambda cls, value: control),
    )
    monkeypatch.setattr(main_module, "_load_bound_json", lambda path: {})

    def qualify(dispatch, **kwargs):
        calls.append({"dispatch": dispatch, **kwargs})
        return SimpleNamespace(status="PASSED", artifact_sha256=_sha("proof"))

    monkeypatch.setattr(
        "lightcone_spec.experiments.formal_preflight_execution."
        "qualify_formal_preflight_interference_locally",
        qualify,
    )

    assert main_module._qualify_formal_preflight_interference_cli(args) == 42
    assert len(calls) == 1
    assert calls[0]["dispatch"] is token
    assert calls[0]["remote_raw_receipt_path"] == "raw.json"
    assert calls[0]["native_result_proof_paths"] == {
        cell: f"result-{index}.json" for index, cell in enumerate(cell_ids)
    }
    assert calls[0]["native_itl_proof_paths"] == {
        cell: f"itl-{index}.json" for index, cell in enumerate(cell_ids)
    }
    assert calls[0]["aggregate_control_attestation"] is control
    assert '"formal_coverage_complete": false' in capsys.readouterr().out


def test_generic_coverage_cli_cannot_forge_preflight_dispositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "stage_materialization_receipt_from_dict",
        lambda _value: SimpleNamespace(stage="preflight"),
    )
    monkeypatch.setattr(main_module, "_load_bound_json", lambda _path: {})
    with pytest.raises(ValueError, match="reducer-owned"):
        main_module._create_stage_coverage(
            SimpleNamespace(
                materialization="materialization.json",
                dispositions="dispositions.json",
            )
        )
