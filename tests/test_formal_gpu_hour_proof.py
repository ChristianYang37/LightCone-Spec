from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from test_control_attestation import _root_binding
from test_formal_dispatch import _protocol_lock
from test_formal_gpu_hour_registry import (
    HARDWARE_SHA256,
    NOW_NS,
    _bundle,
    _control,
    _deployment,
    _extend_registry_with_e3a,
    _registry_receipt,
    _reservation_for_challenges,
)
from test_gpu_hour_authority import (
    _case,
    _inventory,
    _materialization,
    _phase_values,
    _runtime_manifest,
    _subject_and_binding,
)

from lightcone_spec.experiments import gpu_hour_authority
from lightcone_spec.experiments.formal_gpu_hour_proof import (
    bind_formal_stage_gpu_hour_envelope_proof_artifact,
    publish_formal_stage_gpu_hour_envelope_proof_artifact,
    revalidate_formal_stage_gpu_hour_envelope_proof_artifact,
)
from lightcone_spec.experiments.formal_registry import (
    formal_runtime_authority_manifest_to_dict,
    protocol_lock_to_dict,
    stage_gpu_hour_envelope_to_dict,
    stage_materialization_receipt_to_dict,
)
from lightcone_spec.runtime import release_trust_root as root_module
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)


def _publish(path: Path, value: object) -> Path:
    publish_canonical_json_no_replace(path, value)
    return path


def test_distributed_lifecycle_union_keeps_exact_topology_and_control_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lightcone_spec.orchestration import formal_serving_lift
    from lightcone_spec.orchestration.formal_serving_lift import (
        FormalDistributedLifecycleTimingProofArtifact,
    )
    from lightcone_spec.orchestration.formal_terminal_result import (
        FormalDistributedTerminalResultProofArtifact,
    )
    from lightcone_spec.orchestration.live_sglang import (
        PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256,
        UnsignedPinnedSglangLifecycleTimingReceipt,
    )

    runtime = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime.sha256,
    )
    inventory = _inventory()
    materialization = _materialization(lock.sha256, 1)
    _serving, native, _unused = _subject_and_binding(
        tmp_path=tmp_path,
        lock=lock,
        runtime_manifest=runtime,
        materialization=materialization,
        inventory=inventory,
        cell=materialization.cells[0],
        gpu_uuids=("GPU-0", "GPU-1"),
        suffix="distributed",
    )
    bindings = {}
    for label in (
        "plan",
        "live",
        "launch-admission",
        "launch-consumption",
        "budget-consumption",
        "request-terminal",
        "gang-terminal",
        "pointer-bundle",
        "run-receipt",
    ):
        path = _publish(
            (tmp_path / f"{label}.json").resolve(),
            {"schema_version": 1, "kind": f"typed_test_{label}"},
        )
        bindings[label] = CanonicalJsonProofBinding.bind(path)
    edges, durations = _phase_values(10_000_000_000, 3_600_000_000_000)
    timing = UnsignedPinnedSglangLifecycleTimingReceipt(
        schema_version=1,
        kind="unsigned_pinned_sglang_lifecycle_timing_receipt",
        protocol_sha256=PINNED_SGLANG_LIFECYCLE_TIMING_PROTOCOL_SHA256,
        formal_execution_authorized=False,
        live_run_receipt=bindings["live"],
        formal_launch_admission=bindings["launch-admission"],
        formal_launch_consumption=bindings["launch-consumption"],
        budget_consumption=bindings["budget-consumption"],
        run_binding_sha256=gpu_hour_authority.content_sha256(
            gpu_hour_authority._run_binding_to_dict(native)
        ),
        run_id=native.run_id,
        run_nonce_sha256=native.run_nonce_sha256,
        execution_plan_sha256=native.execution_plan_sha256,
        rank_config_sha256=native.rank_config_sha256,
        attempt_id=native.attempt_id,
        method=native.method,
        inventory_sha256=inventory.sha256,
        gpu_uuids=("GPU-0", "GPU-1"),
        telemetry_detail="headline",
        phase_edges_ns=dict(edges),
        phase_durations_ns=dict(durations),
    )
    timing_path = _publish(
        (tmp_path / "distributed-raw-timing.json").resolve(),
        timing.to_dict(),
    )
    timing_binding = CanonicalJsonProofBinding.bind(timing_path)
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    bundle = _bundle(controller_private)
    authorization = _deployment(
        root_private,
        root_binding=root_binding,
        bundle=bundle,
        inventory_sha256=inventory.sha256,
        nonce=80,
    )
    control = _control(
        controller_private,
        root_binding=root_binding,
        bundle=bundle,
        authorization=authorization,
        artifact_type="rank_aggregate",
        artifact_sha256=gpu_hour_authority.content_sha256(
            {"kind": "distributed-lifecycle-test"}
        ),
        protocol_sha256=gpu_hour_authority.FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256,
        lineage_sha256=gpu_hour_authority.content_sha256(
            {"kind": "distributed-lifecycle-lineage"}
        ),
        nonce=81,
    )
    reservation = _reservation_for_challenges(
        tmp_path,
        label="distributed-terminal-replay",
        challenges=(
            control.challenge.sha256,
            control.deployment_policy_authorization.challenge.sha256,
        ),
    )
    terminal = FormalDistributedTerminalResultProofArtifact(
        schema_version=1,
        kind="formal_distributed_terminal_result_proof_artifact",
        plan=bindings["plan"],
        run_receipt=bindings["run-receipt"],
        request_terminal=bindings["request-terminal"],
        gang_terminal=bindings["gang-terminal"],
        pointer_bundle=bindings["pointer-bundle"],
        lifecycle_timing=timing_binding,
        launch_admission=bindings["launch-admission"],
        launch_consumption=bindings["launch-consumption"],
        budget_consumption=bindings["budget-consumption"],
        control_attestation=control,
        replay_reservation=reservation,
        expected_inventory_sha256=inventory.sha256,
        expected_registry_sha256=lock.registry_sha256,
        expected_root_manifest_sha256=lock.offline_release_trust_root_sha256,
        result={"kind": "formal_distributed_terminal_result_projection"},
    )
    terminal_path = _publish(
        (tmp_path / "distributed-terminal-proof.json").resolve(),
        terminal.to_dict(),
    )
    terminal_binding = CanonicalJsonProofBinding.bind(terminal_path)
    artifact = FormalDistributedLifecycleTimingProofArtifact(
        schema_version=1,
        kind="formal_distributed_lifecycle_timing_proof_artifact",
        plan=bindings["plan"],
        terminal_result_proof=terminal_binding,
        raw_lifecycle_timing=timing_binding,
        launch_admission=bindings["launch-admission"],
        launch_consumption=bindings["launch-consumption"],
        budget_consumption=bindings["budget-consumption"],
        expected_inventory_sha256=inventory.sha256,
        expected_registry_sha256=lock.registry_sha256,
        expected_root_manifest_sha256=lock.offline_release_trust_root_sha256,
        topology_mode="tp2_dp1",
        terminal_projection_sha256=gpu_hour_authority.content_sha256(
            {"kind": "distributed-terminal-projection"}
        ),
    )
    proof_path = _publish(
        (tmp_path / "distributed-lifecycle-proof.json").resolve(),
        artifact.to_dict(),
    )

    def validate_distributed(path, **kwargs):
        assert Path(path) == proof_path
        assert kwargs == {
            "expected_inventory_sha256": inventory.sha256,
            "expected_registry_sha256": lock.registry_sha256,
            "expected_root_manifest_sha256": (lock.offline_release_trust_root_sha256),
            "now_ns": NOW_NS,
        }
        return artifact

    monkeypatch.setattr(
        formal_serving_lift,
        "validate_formal_distributed_lifecycle_timing_proof_artifact",
        validate_distributed,
    )
    _proof, replay, projection = gpu_hour_authority._validate_serving_lifecycle_timing(
        proof_path=str(proof_path),
        native=native,
        topology_mode="tp2_dp1",
        gpu_uuids=("GPU-0", "GPU-1"),
        hardware_envelope_sha256=HARDWARE_SHA256,
        telemetry_detail="headline",
        protocol_lock=lock,
        inventory=inventory,
        now_ns=NOW_NS,
    )
    assert projection.proof_kind == "distributed"
    assert projection.execution_plan_sha256 == native.execution_plan_sha256
    assert projection.control_envelope_sha256 == control.sha256
    assert replay == reservation

    with pytest.raises(ValueError, match="distributed GPU-hour lifecycle identity"):
        monkeypatch.setattr(
            formal_serving_lift,
            "validate_formal_distributed_lifecycle_timing_proof_artifact",
            lambda *_args, **_kwargs: replace(
                artifact,
                topology_mode="tp1_dp2",
            ),
        )
        gpu_hour_authority._validate_serving_lifecycle_timing(
            proof_path=str(proof_path),
            native=native,
            topology_mode="tp2_dp1",
            gpu_uuids=("GPU-0", "GPU-1"),
            hardware_envelope_sha256=HARDWARE_SHA256,
            telemetry_detail="headline",
            protocol_lock=lock,
            inventory=inventory,
            now_ns=NOW_NS,
        )


def _actual_e3a_proof_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root_private = Ed25519PrivateKey.generate()
    controller_private = Ed25519PrivateKey.generate()
    root_binding = _root_binding(root_private)
    controller_bundle = _bundle(controller_private)
    monkeypatch.setattr(
        root_module,
        "load_source_release_ed25519_root",
        lambda: root_binding,
    )
    runtime = _runtime_manifest()
    inventory = _inventory()
    lock, initial, preflight, initial_layer = _registry_receipt(
        tmp_path,
        monkeypatch,
        inventory=inventory,
        runtime_manifest=runtime,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=controller_bundle,
    )
    registry, materialization, registry_layer = _extend_registry_with_e3a(
        tmp_path,
        monkeypatch,
        lock=lock,
        prior_receipt=initial,
        prior_layer_path=initial_layer,
        preflight_materialization=preflight,
        inventory=inventory,
        root_private=root_private,
        controller_private=controller_private,
        root_binding=root_binding,
        bundle=controller_bundle,
    )
    gangs = tuple(
        ("GPU-0",) if index % 2 == 0 else ("GPU-1",)
        for index in range(len(materialization.cells))
    )
    starts = tuple(
        10_000_000_000 + (index // 2) * 4_000_000_000_000
        for index in range(len(materialization.cells))
    )
    *_, source_path, envelope = _case(
        tmp_path / "actual-source",
        gangs=gangs,
        starts=starts,
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime,
        inventory_override=inventory,
        materialization_override=materialization,
    )
    protocol_path = _publish(
        (tmp_path / "protocol-lock.json").resolve(),
        protocol_lock_to_dict(lock),
    )
    runtime_path = _publish(
        (tmp_path / "runtime-authority.json").resolve(),
        formal_runtime_authority_manifest_to_dict(runtime),
    )
    inventory_path = _publish(
        (tmp_path / "inventory.json").resolve(),
        inventory.to_dict(),
    )
    materialization_path = _publish(
        (tmp_path / "e3a-materialization.json").resolve(),
        stage_materialization_receipt_to_dict(materialization),
    )
    envelope_path = _publish(
        (tmp_path / "e3a-gpu-hour-envelope.json").resolve(),
        stage_gpu_hour_envelope_to_dict(envelope),
    )
    artifact = bind_formal_stage_gpu_hour_envelope_proof_artifact(
        protocol_lock_path=protocol_path,
        runtime_authority_path=runtime_path,
        registry_layer_path=registry_layer,
        inventory_path=inventory_path,
        final_materialization_path=materialization_path,
        gpu_hour_source_manifest_path=source_path,
        envelope_path=envelope_path,
        now_ns=NOW_NS,
    )
    return artifact, envelope, registry, source_path


def test_actual_gpu_hour_proof_deep_replays_schema5_layer_and_nested_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, envelope, registry, source_path = _actual_e3a_proof_case(
        tmp_path, monkeypatch
    )
    proof_path = (tmp_path / "formal-gpu-hour-proof.json").resolve()
    binding = publish_formal_stage_gpu_hour_envelope_proof_artifact(
        artifact, proof_path
    )
    assert binding.size < 2 * 1024 * 1024
    assert (
        revalidate_formal_stage_gpu_hour_envelope_proof_artifact(
            proof_path,
            now_ns=NOW_NS,
        )
        == envelope
    )
    with pytest.raises(ValueError, match="aliases direct sources"):
        replace(
            artifact,
            runtime_authority_source=artifact.protocol_lock_source,
        )

    direct_registry_path = (tmp_path / "direct-registry.json").resolve()
    from lightcone_spec.experiments.formal_registry import (
        formal_registry_verification_receipt_to_dict,
    )

    _publish(
        direct_registry_path,
        formal_registry_verification_receipt_to_dict(registry),
    )
    with pytest.raises(ValueError, match="proof-replay layer"):
        bind_formal_stage_gpu_hour_envelope_proof_artifact(
            protocol_lock_path=artifact.protocol_lock_source.absolute_path,
            runtime_authority_path=artifact.runtime_authority_source.absolute_path,
            registry_layer_path=direct_registry_path,
            inventory_path=artifact.inventory_source.absolute_path,
            final_materialization_path=(
                artifact.final_materialization_source.absolute_path
            ),
            gpu_hour_source_manifest_path=(
                artifact.gpu_hour_source_manifest.absolute_path
            ),
            envelope_path=artifact.envelope_source.absolute_path,
            now_ns=NOW_NS,
        )

    raw_source = CanonicalJsonProofBinding.bind(source_path).reopen()
    assert type(raw_source) is dict
    source_path.write_text(
        json.dumps(
            {**raw_source, "schedule_sha256": "0" * 64},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    proof_path = (tmp_path / "tampered-gpu-hour-proof.json").resolve()
    publish_canonical_json_no_replace(proof_path, artifact.to_dict())
    with pytest.raises(ValueError, match="binding changed"):
        revalidate_formal_stage_gpu_hour_envelope_proof_artifact(
            proof_path,
            now_ns=NOW_NS,
        )
