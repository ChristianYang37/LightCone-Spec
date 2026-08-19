from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightcone_spec.experiments import (
    formal_dispatch,
    formal_preflight_execution,
    gpu_hour_authority,
    preflight_authority,
)
from lightcone_spec.experiments.formal_protocol import content_sha256
from lightcone_spec.experiments.formal_registry import (
    stage_gpu_hour_envelope_from_dict,
)
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    StageGpuHourEnvelope,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)
from lightcone_spec.runtime.scientific_signing import scientific_payload_sha256

cli = importlib.import_module("lightcone_spec.cli.main")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _publish(path: Path, label: str) -> str:
    value = {
        "schema_version": 1,
        "kind": "gpu_hour_operator_test",
        "label": label,
    }
    publish_canonical_json_no_replace(
        path,
        value,
    )
    Path(f"{path}.sha256").write_text(
        f"{content_sha256(value)}\n",
        encoding="ascii",
    )
    return str(path)


def _available_estimate(
    *, source_sha256: str, schedule_sha256: str, materialization_sha256: str
) -> GpuHourEstimate:
    values = {
        "source_pilot_receipt_sha256": source_sha256,
        "source_schedule_sha256": schedule_sha256,
        "source_materialization_receipt_sha256": materialization_sha256,
        "source_inventory_gpu_count": 2,
        "compute_gpu_hours": 1.0,
        "reserved_gpu_hours": 2.4,
        "estimated_wall_hours": 1.0,
        "retry_reserve_gpu_hours": 0.1,
        "profile_reserve_gpu_hours": 0.1,
        "evidence_reserve_gpu_hours": 0.1,
    }
    return GpuHourEstimate(
        status="AVAILABLE",
        derivation_sha256=content_sha256(values),
        **values,
    )


def _install_operator_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, str]]:
    protocol = SimpleNamespace(sha256=_sha("protocol-lock"))
    inventory = SimpleNamespace(sha256=_sha("inventory"))
    materialization = SimpleNamespace(sha256=_sha("materialization"))
    token = SimpleNamespace(
        sha256=_sha("dispatch-token"),
        protocol_lock=protocol,
        dispatch_context=SimpleNamespace(inventory=inventory),
    )
    durable = SimpleNamespace(
        registry_verification_receipt=SimpleNamespace(
            signed_protocol_lock=SimpleNamespace(payload=protocol)
        ),
        signed_materialization=SimpleNamespace(payload=materialization),
        activation=None,
        revalidate=lambda current_ns: token,
    )
    source = SimpleNamespace(inventory_sha256=inventory.sha256)
    runtime_manifest = SimpleNamespace(sha256=_sha("runtime-manifest"))

    monkeypatch.setattr(
        formal_dispatch.FormalPreflightDispatchReceipt,
        "from_dict",
        classmethod(lambda cls, value: durable),
    )
    monkeypatch.setattr(
        preflight_authority.PreflightExecutionSourceAuthority,
        "from_dict",
        classmethod(lambda cls, value: source),
    )
    pointer_coverage = object()
    monkeypatch.setattr(
        preflight_authority.PreflightCoverageReceipt,
        "from_dict",
        classmethod(lambda cls, value: pointer_coverage),
    )
    activation = object()
    durable.activation = activation
    stage_coverage = object()
    monkeypatch.setattr(
        cli,
        "registry_stage_activation_from_dict",
        lambda value: activation,
    )
    monkeypatch.setattr(
        cli,
        "stage_coverage_receipt_from_dict",
        lambda value: stage_coverage,
    )
    monkeypatch.setattr(
        cli,
        "formal_runtime_authority_manifest_from_dict",
        lambda value: runtime_manifest,
    )

    class _FinalEvidence:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        formal_preflight_execution,
        "FormalPreflightFinalEvidence",
        _FinalEvidence,
    )
    monkeypatch.setattr(
        formal_preflight_execution.FormalPreflightRemoteRawEvidenceReceipt,
        "from_dict",
        classmethod(
            lambda cls, value: SimpleNamespace(
                sha256=content_sha256(value),
                revalidate=lambda dispatch: None,
            )
        ),
    )

    inputs = {}
    for name in (
        "dispatch",
        "remote",
        "source",
        "activation",
        "coverage",
        "stage-coverage",
        "runtime",
    ):
        path = (tmp_path / f"{name}.json").resolve()
        inputs[name] = _publish(path, name)
    expected_proofs: dict[str, str] = {}
    proof_args: list[str] = []
    for index in range(8):
        cell_id = _sha(f"interference-cell-{index}")
        path = (tmp_path / f"lifecycle-{index}.json").resolve()
        _publish(path, f"lifecycle-{index}")
        expected_proofs[str(path)] = CanonicalJsonProofBinding.bind(
            path
        ).semantic_sha256
        proof_args.extend(("--interference-lifecycle-proof", f"{cell_id}={path}"))

    def materialize(**kwargs: object) -> StageGpuHourEnvelope:
        assert kwargs["protocol_lock"] is protocol
        assert kwargs["formal_runtime_authority_manifest"] is runtime_manifest
        assert kwargs["inventory"] is inventory
        evidence = kwargs["final_evidence"]
        assert evidence.source_authority is source
        assert evidence.activation is activation
        assert evidence.coverage is pointer_coverage
        assert evidence.materialization is materialization
        assert evidence.stage_coverage is stage_coverage
        proofs = kwargs["interference_lifecycle_proof_inputs"]
        assert type(proofs) is tuple and len(proofs) == 8
        for proof in proofs:
            binding = CanonicalJsonProofBinding.bind(
                proof.lifecycle_proof_artifact_path
            )
            if (
                expected_proofs.get(proof.lifecycle_proof_artifact_path)
                != binding.semantic_sha256
            ):
                raise ValueError("preflight lifecycle proof changed")
        source_sha256 = _sha("preflight-source-manifest")
        schedule_sha256 = _sha("preflight-schedule")
        publish_canonical_json_no_replace(
            kwargs["source_manifest_output_path"],
            {
                "schema_version": 1,
                "kind": "preflight_gpu_hour_source_manifest_test_double",
                "source_sha256": source_sha256,
            },
        )
        return StageGpuHourEnvelope(
            schema_version=2,
            protocol_lock_sha256=protocol.sha256,
            materialization_receipt_sha256=materialization.sha256,
            signed_pilot_receipt_sha256=source_sha256,
            schedule_sha256=schedule_sha256,
            estimate=_available_estimate(
                source_sha256=source_sha256,
                schedule_sha256=schedule_sha256,
                materialization_sha256=materialization.sha256,
            ),
        )

    monkeypatch.setattr(
        gpu_hour_authority,
        "materialize_preflight_gpu_hour_envelope",
        materialize,
    )
    source_output = (tmp_path / "gpu-hour-source.json").resolve()
    output = (tmp_path / "gpu-hour-envelope.json").resolve()
    argv = [
        "materialize-preflight-gpu-hour-envelope",
        "--dispatch-receipt",
        inputs["dispatch"],
        "--remote-raw-receipt",
        inputs["remote"],
        "--source-authority",
        inputs["source"],
        "--activation",
        inputs["activation"],
        "--coverage",
        inputs["coverage"],
        "--stage-coverage",
        inputs["stage-coverage"],
        *proof_args,
        "--formal-runtime-authority-manifest",
        inputs["runtime"],
        "--source-output",
        str(source_output),
        "--now-ns",
        "2000000000",
        "--output",
        str(output),
    ]
    return argv, expected_proofs


def test_preflight_gpu_hour_operator_cli_publishes_schema2_but_requires_source_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, _proofs = _install_operator_sources(tmp_path, monkeypatch)
    assert cli.main(argv) == 0
    output = Path(argv[-1])
    envelope_json = CanonicalJsonProofBinding.bind(output).reopen()
    envelope = stage_gpu_hour_envelope_from_dict(envelope_json)
    assert envelope.schema_version == 2
    with pytest.raises(ValueError, match="validation time is required"):
        scientific_payload_sha256(
            artifact_type="stage-gpu-hour-envelope",
            payload_json=envelope_json,
        )
    with pytest.raises(RuntimeError, match="target already exists"):
        cli.main(argv)


def test_preflight_gpu_hour_operator_rejects_missing_duplicate_and_tampered_proofs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, expected = _install_operator_sources(tmp_path, monkeypatch)
    proof_flags = [
        index
        for index, value in enumerate(argv)
        if value == "--interference-lifecycle-proof"
    ]
    missing = argv[: proof_flags[-1]] + argv[proof_flags[-1] + 2 :]
    with pytest.raises(ValueError, match="exact eight"):
        cli.main(missing)

    duplicate = list(argv)
    duplicate[proof_flags[-1] + 1] = duplicate[proof_flags[0] + 1]
    with pytest.raises(ValueError, match="unique CELL_ID=PATH"):
        cli.main(duplicate)

    tampered_path = Path(next(iter(expected)))
    tampered_path.write_text(
        '{"kind":"gpu_hour_operator_test","label":"tampered","schema_version":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="proof changed"):
        cli.main(argv)

    with pytest.raises(SystemExit):
        cli.main([*argv, "--duration-ms", "1"])


def test_staged_prospective_operator_emits_explicit_blocked_then_ready_without_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = SimpleNamespace(sha256=_sha("staged-protocol-lock"))
    runtime_manifest = SimpleNamespace(sha256=_sha("staged-runtime-manifest"))
    inventory = SimpleNamespace(sha256=_sha("staged-inventory"))
    materialization = SimpleNamespace(
        stage="E1",
        sha256=_sha("staged-e1-materialization"),
    )
    receipt = SimpleNamespace(
        signed_protocol_lock=SimpleNamespace(payload=protocol),
        cumulative_signed_materializations=(SimpleNamespace(payload=materialization),),
        revalidate=lambda current_ns: SimpleNamespace(
            inventory_sha256=inventory.sha256,
            verified_ns=current_ns,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_formal_registry_receipt_path",
        lambda _path, *, now_ns: receipt,
    )
    monkeypatch.setattr(
        cli,
        "formal_runtime_authority_manifest_from_dict",
        lambda _value: runtime_manifest,
    )
    monkeypatch.setattr(cli, "_load_gpu_inventory", lambda _path: inventory)

    registry_path = _publish(
        (tmp_path / "staged-registry.json").resolve(), "staged-registry"
    )
    runtime_path = _publish(
        (tmp_path / "staged-runtime.json").resolve(), "staged-runtime"
    )
    inventory_path = _publish(
        (tmp_path / "staged-inventory.json").resolve(), "staged-inventory"
    )
    completed_path = _publish(
        (tmp_path / "staged-completed-source.json").resolve(),
        "staged-completed-source",
    )

    class _Manifest:
        def __init__(self, *, status: str, source_path: str) -> None:
            self.status = status
            self._value = {
                "schema_version": 1,
                "kind": "staged_prospective_gpu_hour_source_manifest_test_double",
                "status": status,
                "source_path": source_path,
                "minimum_pilot_cell_ids": (
                    [_sha("minimum-e1-pilot")] if status == "BLOCKED" else []
                ),
            }
            self.sha256 = content_sha256(self._value)

        def to_dict(self) -> dict[str, object]:
            return dict(self._value)

    calls: list[dict[str, object]] = []

    def materialize(**kwargs: object):
        calls.append(dict(kwargs))
        assert kwargs["protocol_lock"] is protocol
        assert kwargs["formal_runtime_authority_manifest"] is runtime_manifest
        assert kwargs["materialization"] is materialization
        assert kwargs["inventory"] is inventory
        source_path = str(kwargs["source_manifest_output_path"])
        completed = kwargs["completed_source_manifest_path"]
        status = "BLOCKED" if completed is None else "READY"
        manifest = _Manifest(status=status, source_path=source_path)
        publish_canonical_json_no_replace(source_path, manifest.to_dict())
        if status == "BLOCKED":
            return manifest, None
        source_sha256 = manifest.sha256
        schedule_sha256 = _sha("staged-schedule")
        envelope = StageGpuHourEnvelope(
            schema_version=2,
            protocol_lock_sha256=protocol.sha256,
            materialization_receipt_sha256=materialization.sha256,
            signed_pilot_receipt_sha256=source_sha256,
            schedule_sha256=schedule_sha256,
            estimate=_available_estimate(
                source_sha256=source_sha256,
                schedule_sha256=schedule_sha256,
                materialization_sha256=materialization.sha256,
            ),
        )
        return manifest, envelope

    monkeypatch.setattr(
        cli,
        "materialize_staged_prospective_gpu_hour_envelope",
        materialize,
    )

    def argv(*, suffix: str, completed: bool) -> list[str]:
        values = [
            "materialize-staged-prospective-gpu-hours",
            "--stage",
            "E1",
            "--materialization-sha256",
            materialization.sha256,
            "--registry-verification-receipt",
            registry_path,
            "--formal-runtime-authority-manifest",
            runtime_path,
            "--inventory",
            inventory_path,
            "--source-output",
            str((tmp_path / f"staged-source-{suffix}.json").resolve()),
            "--now-ns",
            "2000000000",
            "--output",
            str((tmp_path / f"staged-result-{suffix}.json").resolve()),
        ]
        if completed:
            values[9:9] = ["--completed-source-manifest", completed_path]
        return values

    blocked_argv = argv(suffix="blocked", completed=False)
    assert cli.main(blocked_argv) == 42
    blocked = cli._load_bound_json(blocked_argv[-1])
    assert blocked["status"] == "BLOCKED"
    assert blocked["minimum_pilot_cell_ids"] == [_sha("minimum-e1-pilot")]

    ready_argv = argv(suffix="ready", completed=True)
    assert cli.main(ready_argv) == 0
    ready = stage_gpu_hour_envelope_from_dict(cli._load_bound_json(ready_argv[-1]))
    assert ready.materialization_receipt_sha256 == materialization.sha256
    assert calls[0]["completed_source_manifest_path"] is None
    assert calls[1]["completed_source_manifest_path"] == completed_path
    with pytest.raises(SystemExit):
        cli.main([*blocked_argv, "--duration-ms", "1"])


def test_gpu_hour_envelope_proof_publisher_cli_maps_only_bound_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        name: str((tmp_path / f"{name}.json").resolve())
        for name in (
            "protocol",
            "runtime",
            "registry",
            "inventory",
            "final",
            "pilot",
            "source",
            "envelope",
            "coverage",
            "output",
        )
    }
    artifact = SimpleNamespace(sha256=_sha("gpu-hour-proof"))
    calls: list[tuple[str, object]] = []

    def bind(**kwargs: object) -> object:
        calls.append(("bind", dict(kwargs)))
        return artifact

    def publish(value: object, output_path: str) -> None:
        calls.append(("publish", (value, output_path)))

    monkeypatch.setattr(
        cli,
        "bind_formal_stage_gpu_hour_envelope_proof_artifact",
        bind,
    )
    monkeypatch.setattr(
        cli,
        "publish_formal_stage_gpu_hour_envelope_proof_artifact",
        publish,
    )
    argv = [
        "publish-formal-stage-gpu-hour-envelope-proof",
        "--protocol-lock",
        paths["protocol"],
        "--formal-runtime-authority-manifest",
        paths["runtime"],
        "--registry-layer",
        paths["registry"],
        "--inventory",
        paths["inventory"],
        "--final-materialization",
        paths["final"],
        "--pilot-materialization",
        paths["pilot"],
        "--gpu-hour-source-manifest",
        paths["source"],
        "--envelope",
        paths["envelope"],
        "--preflight-coverage-proof",
        paths["coverage"],
        "--now-ns",
        "2000000000",
        "--output",
        paths["output"],
    ]
    assert cli.main(argv) == 0
    assert calls == [
        (
            "bind",
            {
                "protocol_lock_path": paths["protocol"],
                "runtime_authority_path": paths["runtime"],
                "registry_layer_path": paths["registry"],
                "inventory_path": paths["inventory"],
                "final_materialization_path": paths["final"],
                "pilot_materialization_path": paths["pilot"],
                "gpu_hour_source_manifest_path": paths["source"],
                "envelope_path": paths["envelope"],
                "preflight_coverage_proof_path": paths["coverage"],
                "now_ns": 2_000_000_000,
            },
        ),
        ("publish", (artifact, paths["output"])),
    ]
    with pytest.raises(SystemExit):
        cli.main([*argv, "--compute-gpu-hours", "1.0"])
