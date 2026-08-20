from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_formal_dispatch import _protocol_lock

from lightcone_spec.config import ModelPair, RunConfig, RuntimeConfig
from lightcone_spec.experiments import formal_stage_execution, gpu_hour_authority
from lightcone_spec.experiments.formal_protocol import (
    FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS,
    FormalRuntimeAuthorityManifest,
    FormalRuntimeAuthorityMember,
)
from lightcone_spec.experiments.formal_stage_execution import (
    FormalServingExecutionSubject,
    VerifiedFormalServingExecutionBinding,
)
from lightcone_spec.experiments.gpu_hour_authority import (
    FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256,
    FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256,
    FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256,
    FormalGpuHourLifecycleBlocked,
    LifecycleGpuHourProofInput,
    LifecycleGpuHourSourceManifest,
    materialize_stage_gpu_hour_envelope_from_lifecycle_proofs,
    revalidate_persisted_stage_gpu_hour_source_manifest,
    revalidate_stage_gpu_hour_source_manifest,
)
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.itl_authority import StageItlExecutionIdentity
from lightcone_spec.experiments.stage_materialization import (
    GpuHourEstimate,
    MaterializedCell,
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
    _materialize_e4_profiler_diagnostic,
)
from lightcone_spec.orchestration import live_sglang
from lightcone_spec.orchestration.formal_failure_physical import (
    FormalE5FailureLifecycleCostProjection,
)
from lightcone_spec.orchestration.live_sglang import (
    VerifiedPinnedSglangLifecycleTimingProof,
)
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.runtime import readiness
from lightcone_spec.runtime.control_attestation import (
    ChallengeReplayReservationBinding,
)
from lightcone_spec.runtime.proof_artifact import (
    CanonicalJsonProofBinding,
    publish_canonical_json_no_replace,
)

_NS_PER_HOUR = 3_600_000_000_000
_EDGE_NAMES = (
    "execution_started_ns",
    "server_ready_ns",
    "begin_started_ns",
    "begin_finished_ns",
    "warmup_started_ns",
    "warmup_finished_ns",
    "reset_started_ns",
    "reset_finished_ns",
    "scored_executor_started_ns",
    "scored_request_started_ns",
    "scored_request_finished_ns",
    "scored_executor_finished_ns",
    "finalize_started_ns",
    "finalize_finished_ns",
    "terminal_published_ns",
    "itl_pointer_published_ns",
    "native_terminal_finished_ns",
    "process_exited_ns",
    "process_group_empty_checked_ns",
    "evidence_flush_started_ns",
    "evidence_flush_finished_ns",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _native_gpu_token(*, inventory_sha256: str, gpu_uuid: str, suffix: str):
    capability = readiness.NATIVE_RUNTIME_RELEASE_CAPABILITY
    tests = readiness.NATIVE_RUNTIME_QUALIFICATION_TESTS["native_hot_path_tp1"]
    receipt = readiness.NativeRuntimeGpuProofReceipt(
        schema_version=1,
        kind="lightcone_native_runtime_gpu_proof",
        suite_id="native_hot_path_tp1",
        topology_mode="tp1_dp1",
        topology_sha256=_sha(f"topology:{suffix}"),
        runner_protocol_sha256=(
            readiness.NATIVE_RUNTIME_SUITE_RUNNER_PROTOCOL_SHA256S[
                "native_hot_path_tp1"
            ]
        ),
        assignment_sha256=_sha(f"assignment:{suffix}"),
        qualification_observation_sha256=_sha(f"observation:{suffix}"),
        source_capability_sha256=capability.sha256,
        pinned_sglang_commit=capability.pinned_sglang_commit,
        patched_sglang_tree=capability.patched_sglang_tree,
        semantic_patch_sha256=capability.semantic_patch_sha256,
        run_nonce_sha256=_sha(f"qualification-nonce:{suffix}"),
        qualification_authority_sha256=(
            readiness.NATIVE_RUNTIME_QUALIFICATION_AUTHORITY_SHA256
        ),
        source_identity_sha256=_sha("source-identity"),
        inventory_sha256=inventory_sha256,
        gpu_uuids=(gpu_uuid,),
        hardware_envelope_sha256=_sha("hardware"),
        junit_xml_sha256=_sha(f"junit:{suffix}"),
        test_names=tests,
        tests_collected=len(tests),
        tests_passed=len(tests),
        tests_failed=0,
        tests_errored=0,
        tests_skipped=0,
    )
    return readiness.VerifiedNativeRuntimeGpuProof(
        receipt=receipt,
        receipt_raw_sha256=_sha(f"receipt-raw:{suffix}"),
        trusted_policy_sha256=_sha("policy"),
        challenge_sha256=_sha(f"challenge:{suffix}"),
        control_envelope_sha256=_sha(f"control:{suffix}"),
        challenge_reservation_sha256=_sha(f"reservation:{suffix}"),
        _verification_tag=readiness._VERIFIED_NATIVE_GPU_PROOF_SENTINEL,
    )


def _reservation(
    tmp_path: Path,
    *,
    label: str,
    reserved_ns: int = 2_000_000_000,
) -> ChallengeReplayReservationBinding:
    challenge = _sha(f"challenge:{label}")
    canonical = {
        "schema_version": 2,
        "kind": "lightcone_control_challenge_reservation",
        "reserved_ns": reserved_ns,
        "challenge_sha256s": [challenge],
    }
    identity = gpu_hour_authority.content_sha256(canonical)
    body = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path = (tmp_path / f"reservation-{identity}.json").resolve()
    path.write_bytes(body)
    path.chmod(0o600)
    return ChallengeReplayReservationBinding(
        schema_version=1,
        kind="lightcone_challenge_replay_reservation_binding",
        path=str(path),
        reservation_sha256=identity,
        raw_sha256=hashlib.sha256(body).hexdigest(),
        size=len(body),
        reserved_ns=reserved_ns,
        challenge_sha256s=(challenge,),
    )


def _runtime_manifest() -> FormalRuntimeAuthorityManifest:
    return FormalRuntimeAuthorityManifest(
        schema_version=2,
        authority_id="gpu-hour-authority-test-v1",
        members=tuple(
            FormalRuntimeAuthorityMember(
                member_id=member_id,
                protocol_sha256=(
                    FORMAL_GPU_HOUR_BUDGET_PROTOCOL_SHA256
                    if member_id == "gpu_hour_budget_reducer"
                    else _sha(f"{member_id}:protocol")
                ),
                runner_sha256=(
                    FORMAL_GPU_HOUR_BUDGET_RUNNER_SHA256
                    if member_id == "gpu_hour_budget_reducer"
                    else _sha(f"{member_id}:runner")
                ),
                test_set_sha256=(
                    FORMAL_GPU_HOUR_BUDGET_TEST_SET_SHA256
                    if member_id == "gpu_hour_budget_reducer"
                    else _sha(f"{member_id}:tests")
                ),
                source_sha256=_sha(f"{member_id}:source"),
            )
            for member_id in FORMAL_RUNTIME_AUTHORITY_MEMBER_IDS
        ),
    )


def _inventory() -> GpuInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{index}",
            host_id="gpu-hour-host",
            model="RTX PRO 6000 Blackwell Server Edition",
            memory_bytes=96 * 1024**3,
            compute_capability=(12, 0),
            pci_bus_id=f"0000:0{index + 1}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("PCIe",),
            peer_access_class="P2P",
            clock_policy="locked",
            power_limit_watts=600.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("pair",),
        )
        for index in range(2)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="pair",
                host_id="gpu-hour-host",
                gpu_uuids=tuple(row.uuid for row in devices),
                fabric="PCIe",
                bandwidth_class="local",
            ),
        ),
        source_receipt_sha256=_sha("inventory-receipt"),
    )


def _materialization(lock_sha256: str, count: int) -> StageMaterializationReceipt:
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E3a",
                    method_role="Static",
                    model="Qwen/Qwen3-8B",
                    backend="DFlash",
                    task="LiveCodeBench",
                    publication_policy="none",
                    recipe_sha256=None,
                    dimensions=(("case", index),),
                )
                for index in range(count)
            ),
            key=lambda row: row.cell_id,
        )
    )
    return StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=lock_sha256,
        upstream_receipt_sha256s=(_sha("preflight-coverage"),),
        source_decision_sha256=_sha("e3a-source"),
        materialization_rule="gpu_hour_test_exact_cells",
        expected_cell_count=len(cells),
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )


def _subject_and_binding(
    *,
    tmp_path: Path,
    lock,
    runtime_manifest: FormalRuntimeAuthorityManifest,
    materialization: StageMaterializationReceipt,
    inventory: GpuInventory,
    cell: MaterializedCell,
    gpu_uuids: tuple[str, ...],
    suffix: str,
    telemetry_detail: str = "headline",
):
    method = {
        "Target-only": "target_only",
        "Static": "static",
        "TTS": "tts",
        "L0-naive": "l0",
        "LightCone-candidate": "l0",
        "LightCone": "l0",
        "TTS-calibration-candidate": "tts",
    }[cell.method_role]
    runtime_path = (tmp_path / f"runtime-{suffix}.json").resolve()
    publish_canonical_json_no_replace(
        runtime_path, {"kind": "test-runtime-proof-binding", "suffix": suffix}
    )
    runtime_binding = CanonicalJsonProofBinding.bind(runtime_path)
    execution_plan_sha256 = _sha(f"execution-plan:{suffix}")
    rank_config_sha256 = _sha(f"rank-config:{suffix}")
    run_id = f"gpu-hour-{suffix}"
    run_nonce_sha256 = _sha(f"run-nonce:{suffix}")
    attempt_id = f"attempt-{suffix}"
    identity = StageItlExecutionIdentity(
        schema_version=1,
        kind="stage_itl_execution_identity",
        materialized_cell_id=cell.cell_id,
        inventory_sha256=inventory.sha256,
        registry_sha256=lock.registry_sha256,
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=rank_config_sha256,
        run_id=run_id,
        run_nonce_sha256=run_nonce_sha256,
        attempt_id=attempt_id,
        method=method,
        runtime_trust_mode=None,
        formal_measurement=None,
    )
    subject = FormalServingExecutionSubject(
        schema_version=4,
        protocol_lock_sha256=lock.sha256,
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
        execution_mapper_authority_sha256=runtime_manifest.member(
            "all_stage_execution_mapper"
        ).sha256,
        materialization_receipt_sha256=materialization.sha256,
        materialized_cell_id=cell.cell_id,
        stage=cell.stage,
        method=method,
        stage_source_binding_sha256=(
            _sha(f"stage-source:{suffix}")
            if cell.stage in {"E3b", "E5", "E6", "E0"}
            else None
        ),
        run_config_sha256=_sha(f"run-config:{suffix}"),
        recipe_authority_sha256s=(_sha("static-recipe"),),
        workload_authority_sha256=_sha("workload"),
        content_verification_receipt_sha256=None,
        prepared_model_member_sha256s=(),
        workload_member_sha256s=(),
        inventory_sha256=inventory.sha256,
        topology_mode="tp1_dp1" if len(gpu_uuids) == 1 else "tp2_dp1",
        gpu_uuids=gpu_uuids,
        runtime_gpu_proof_artifacts=(runtime_binding,),
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=rank_config_sha256,
        execution_identity=identity,
    )
    config = RunConfig(
        method=method,
        model=ModelPair(
            target_revision="1" * 40,
            drafter_revision="2" * 40,
        ),
        runtime=RuntimeConfig(
            sampling_profile_sha256=_sha("sampling-profile"),
            telemetry_detail=telemetry_detail,
            speculation_enabled=method != "target_only",
        ),
    )
    native_gpu_proof = _native_gpu_token(
        inventory_sha256=inventory.sha256,
        gpu_uuid=gpu_uuids[0],
        suffix=suffix,
    )
    binding = VerifiedFormalServingExecutionBinding(
        subject=subject,
        run_config=config,
        runtime_gpu_proof_sha256s=(native_gpu_proof.sha256,),
        verified_native_gpu_proofs=(native_gpu_proof,),
        verified_distributed_gpu_proofs=(),
        verified_nextn_tp2_authority=None,
        hardware_envelope_sha256=native_gpu_proof.hardware_envelope_sha256,
        _construction_seal=(
            formal_stage_execution._VERIFIED_FORMAL_SERVING_EXECUTION_SEAL
        ),
    )
    native = NativeTerminalRunBinding(
        run_id=run_id,
        run_nonce_sha256=run_nonce_sha256,
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=rank_config_sha256,
        attempt_id=attempt_id,
        session_id=f"session-{suffix}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=_sha(f"challenge:{suffix}"),
        method=method,
        reset_scope=(
            "request"
            if cell.method_role in {"TTS", "L0-naive", "TTS-calibration-candidate"}
            else "cohort"
            if method not in {"target_only", "static"}
            else None
        ),
        request_admission_policy=(
            "serialized_native_scheduler_v1"
            if cell.method_role in {"TTS", "L0-naive", "TTS-calibration-candidate"}
            else "cohort_batching_v1"
            if method not in {"target_only", "static"}
            else None
        ),
        runtime_trust_mode=None,
        formal_measurement=None,
        warmup_request_ids=(f"warmup-{suffix}",),
        scored_request_ids=(f"scored-{suffix}",),
    )
    proof_path = (tmp_path / f"lifecycle-proof-{suffix}.json").resolve()
    publish_canonical_json_no_replace(
        proof_path, {"kind": "test-lifecycle-proof", "suffix": suffix}
    )
    return binding, native, proof_path


def _phase_values(
    start_ns: int,
    wall_ns: int,
    evidence_tail_ns: int = 0,
    *,
    telemetry_detail: str = "headline",
):
    # GPU/process work ends at ``wall_ns``. Evidence flushing is a disjoint
    # provider-reservation tail and must not be charged as compute.
    values = tuple(
        start_ns + (wall_ns * min(index, 17)) // 17 for index in range(20)
    ) + (start_ns + wall_ns + evidence_tail_ns,)
    edges = tuple(zip(_EDGE_NAMES, values, strict=True))
    edge = dict(edges)
    durations = (
        ("startup_ns", edge["server_ready_ns"] - edge["execution_started_ns"]),
        ("warmup_ns", edge["warmup_finished_ns"] - edge["warmup_started_ns"]),
        ("adaptation_reset_ns", edge["reset_finished_ns"] - edge["reset_started_ns"]),
        (
            "scored_request_window_ns",
            edge["scored_request_finished_ns"] - edge["scored_request_started_ns"],
        ),
        (
            "drain_ns",
            edge["native_terminal_finished_ns"] - edge["scored_request_finished_ns"],
        ),
        (
            "process_cleanup_ns",
            edge["process_group_empty_checked_ns"]
            - edge["native_terminal_finished_ns"],
        ),
        (
            "evidence_flush_ns",
            edge["evidence_flush_finished_ns"] - edge["evidence_flush_started_ns"],
        ),
        ("reserved_wall_ns", wall_ns + evidence_tail_ns),
        (
            "profile_reserved_ns",
            wall_ns + evidence_tail_ns if telemetry_detail == "profile" else 0,
        ),
    )
    return edges, durations


def _verified_lifecycle(
    *,
    binding: VerifiedFormalServingExecutionBinding,
    native: NativeTerminalRunBinding,
    start_ns: int,
    wall_ns: int,
    suffix: str,
    evidence_tail_ns: int = 0,
    telemetry_detail: str = "headline",
) -> VerifiedPinnedSglangLifecycleTimingProof:
    value = object.__new__(VerifiedPinnedSglangLifecycleTimingProof)
    edges, durations = _phase_values(
        start_ns,
        wall_ns,
        evidence_tail_ns,
        telemetry_detail=telemetry_detail,
    )
    fields = {
        "raw_timing_sha256": _sha(f"raw:{suffix}"),
        "live_run_receipt_sha256": _sha(f"live:{suffix}"),
        "native_result_proof_sha256": _sha(f"result:{suffix}"),
        "run_binding_sha256": _sha(f"binding:{suffix}"),
        "run_id": native.run_id,
        "run_nonce_sha256": native.run_nonce_sha256,
        "execution_plan_sha256": native.execution_plan_sha256,
        "rank_config_sha256": native.rank_config_sha256,
        "attempt_id": native.attempt_id,
        "method": native.method,
        "inventory_sha256": binding.subject.inventory_sha256,
        "registry_sha256": binding.subject.execution_identity.registry_sha256,
        "root_manifest_sha256": _protocol_lock().offline_release_trust_root_sha256,
        "hardware_envelope_sha256": binding.hardware_envelope_sha256,
        "gpu_uuids": binding.subject.gpu_uuids,
        "telemetry_detail": telemetry_detail,
        "phase_edges_ns": edges,
        "phase_durations_ns": durations,
        "control_envelope_sha256": _sha(f"control:{suffix}"),
        "replay_reservation_sha256": _sha(f"reservation:{suffix}"),
    }
    for name, field_value in fields.items():
        object.__setattr__(value, name, field_value)
    return value


def _case(
    tmp_path: Path,
    *,
    gangs: tuple[tuple[str, ...], ...],
    starts: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
    evidence_tail_ns: int = 0,
    telemetry_detail: str = "headline",
    lock_override=None,
    runtime_manifest_override: FormalRuntimeAuthorityManifest | None = None,
    inventory_override: GpuInventory | None = None,
    materialization_override: StageMaterializationReceipt | None = None,
):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_manifest = runtime_manifest_override or _runtime_manifest()
    lock = lock_override or replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    inventory = inventory_override or _inventory()
    materialization = materialization_override or _materialization(
        lock.sha256, len(gangs)
    )
    if len(materialization.cells) != len(gangs):
        raise ValueError("GPU-hour test materialization/gang count differs")
    proof_inputs = []
    verified_by_path = {}
    execution_payloads = {}
    execution_artifacts = {}
    lifecycle_artifacts = {}
    for index, (cell, gpu_uuids, start_ns) in enumerate(
        zip(materialization.cells, gangs, starts, strict=True)
    ):
        suffix = str(index)
        binding, native, proof_path = _subject_and_binding(
            tmp_path=tmp_path,
            lock=lock,
            runtime_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            cell=cell,
            gpu_uuids=gpu_uuids,
            suffix=suffix,
            telemetry_detail=telemetry_detail,
        )
        execution_path = (tmp_path / f"execution-proof-{suffix}.json").resolve()
        publish_canonical_json_no_replace(
            execution_path,
            {"kind": "test-execution-proof", "cell_id": cell.cell_id},
        )
        execution_payloads[str(execution_path)] = (
            gpu_hour_authority._execution_proof_payload(binding)
        )
        execution_artifacts[str(execution_path)] = SimpleNamespace(
            control_attestation=SimpleNamespace(
                sha256=_sha(f"execution-control:{suffix}")
            ),
            replay_reservation=_reservation(
                tmp_path,
                label=f"execution:{suffix}",
            ),
        )
        lifecycle_artifacts[str(proof_path)] = SimpleNamespace(
            replay_reservation=_reservation(
                tmp_path,
                label=f"lifecycle:{suffix}",
            )
        )
        proof_inputs.append(
            LifecycleGpuHourProofInput(
                execution_binding=binding,
                native_run_binding=native,
                lifecycle_proof_artifact_path=str(proof_path),
                execution_proof_artifact_path=str(execution_path),
            )
        )
        verified_by_path[str(proof_path)] = _verified_lifecycle(
            binding=binding,
            native=native,
            start_ns=start_ns,
            wall_ns=_NS_PER_HOUR,
            suffix=suffix,
            evidence_tail_ns=evidence_tail_ns,
            telemetry_detail=telemetry_detail,
        )
        object.__setattr__(
            verified_by_path[str(proof_path)],
            "replay_reservation_sha256",
            lifecycle_artifacts[str(proof_path)].replay_reservation.reservation_sha256,
        )
        object.__setattr__(
            verified_by_path[str(proof_path)],
            "root_manifest_sha256",
            lock.offline_release_trust_root_sha256,
        )

    def validate(path, **_kwargs):
        return verified_by_path[path]

    monkeypatch.setattr(
        live_sglang,
        "validate_pinned_sglang_lifecycle_timing_proof_artifact",
        validate,
    )
    monkeypatch.setattr(
        gpu_hour_authority,
        "validate_formal_serving_execution_proof_artifact",
        lambda path, **_kwargs: execution_payloads[path],
    )
    monkeypatch.setattr(
        gpu_hour_authority,
        "_load_formal_serving_execution_proof_artifact",
        lambda path: execution_artifacts[path],
    )
    monkeypatch.setattr(
        gpu_hour_authority,
        "_load_lifecycle_timing_proof_artifact",
        lambda path: lifecycle_artifacts[path],
    )
    original_lifecycle_union = gpu_hour_authority._validate_serving_lifecycle_timing

    def validate_lifecycle_union(*, proof_path, topology_mode, **kwargs):
        if topology_mode == "tp1_dp1":
            return original_lifecycle_union(
                proof_path=proof_path,
                topology_mode=topology_mode,
                **kwargs,
            )
        verified = verified_by_path[proof_path]
        projection = gpu_hour_authority._ValidatedLifecycleTiming(
            proof_kind="distributed",
            **{
                field: getattr(verified, field)
                for field in gpu_hour_authority._ValidatedLifecycleTiming.__dataclass_fields__
                if field != "proof_kind"
            },
        )
        return (
            CanonicalJsonProofBinding.bind(proof_path),
            lifecycle_artifacts[proof_path].replay_reservation,
            projection,
        )

    monkeypatch.setattr(
        gpu_hour_authority,
        "_validate_serving_lifecycle_timing",
        validate_lifecycle_union,
    )
    output = (tmp_path / "gpu-hour-source.json").resolve()
    envelope = materialize_stage_gpu_hour_envelope_from_lifecycle_proofs(
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=tuple(proof_inputs),
        source_manifest_output_path=str(output),
        now_ns=2_000_000_000,
    )
    return (
        lock,
        runtime_manifest,
        inventory,
        materialization,
        tuple(proof_inputs),
        verified_by_path,
        output,
        envelope,
    )


def test_single_tp1_charges_fixed_two_gpu_reservation(
    tmp_path,
    monkeypatch,
) -> None:
    *_, envelope = _case(
        tmp_path,
        gangs=(("GPU-0",),),
        starts=(10_000_000_000,),
        monkeypatch=monkeypatch,
    )
    assert envelope.schema_version == 2
    assert envelope.estimate.compute_gpu_hours == pytest.approx(1.0)
    assert envelope.estimate.estimated_wall_hours == pytest.approx(1.0)
    assert envelope.estimate.retry_reserve_gpu_hours == pytest.approx(0.1)
    assert envelope.estimate.reserved_gpu_hours == pytest.approx(2.1)


def test_evidence_tail_is_reserved_once_and_never_counted_as_compute(
    tmp_path,
    monkeypatch,
) -> None:
    *_, envelope = _case(
        tmp_path,
        gangs=(("GPU-0",),),
        starts=(10_000_000_000,),
        monkeypatch=monkeypatch,
        evidence_tail_ns=_NS_PER_HOUR // 2,
    )
    assert envelope.estimate.compute_gpu_hours == pytest.approx(1.0)
    assert envelope.estimate.estimated_wall_hours == pytest.approx(1.0)
    assert envelope.estimate.evidence_reserve_gpu_hours == pytest.approx(1.0)
    assert envelope.estimate.profile_reserve_gpu_hours == 0.0
    assert envelope.estimate.retry_reserve_gpu_hours == pytest.approx(0.1)
    assert envelope.estimate.reserved_gpu_hours == pytest.approx(3.1)


def test_profile_row_records_profile_phase_without_double_charging_it(
    tmp_path,
    monkeypatch,
) -> None:
    *_, source_path, envelope = _case(
        tmp_path,
        gangs=(("GPU-0",),),
        starts=(10_000_000_000,),
        monkeypatch=monkeypatch,
        telemetry_detail="profile",
    )
    assert envelope.estimate.compute_gpu_hours == pytest.approx(1.0)
    assert envelope.estimate.estimated_wall_hours == pytest.approx(1.0)
    assert envelope.estimate.profile_reserve_gpu_hours == 0.0
    assert envelope.estimate.evidence_reserve_gpu_hours == 0.0
    assert envelope.estimate.retry_reserve_gpu_hours == pytest.approx(0.1)
    assert envelope.estimate.reserved_gpu_hours == pytest.approx(2.1)
    source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_path).reopen()
    )
    row = source.observations[0]
    edges = dict(row.phase_edges_ns)
    assert edges["execution_started_ns"] < edges["server_ready_ns"]
    assert edges["server_ready_ns"] < edges["warmup_finished_ns"]
    assert gpu_hour_authority._gpu_process_occupied_ns(row) == (
        edges["process_exited_ns"] - edges["execution_started_ns"]
    )
    assert envelope.estimate.compute_gpu_hours == pytest.approx(
        gpu_hour_authority._gpu_process_occupied_ns(row) / _NS_PER_HOUR
    )


def _prospective_projection_case(
    tmp_path,
    monkeypatch,
) -> tuple[
    object,
    FormalRuntimeAuthorityManifest,
    GpuInventory,
    StageMaterializationReceipt,
    LifecycleGpuHourSourceManifest,
    Path,
    StageGpuHourEnvelope,
    StageMaterializationReceipt,
]:
    (
        lock,
        runtime_manifest,
        inventory,
        _source_materialization,
        _proof_inputs,
        _verified,
        source_path,
        _source_envelope,
    ) = _case(
        tmp_path / "source",
        gangs=(("GPU-0",),) * 4,
        starts=tuple(1_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(4)),
        monkeypatch=monkeypatch,
    )
    source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_path).reopen()
    )

    def cell(block: int, *, phase: str) -> MaterializedCell:
        return MaterializedCell(
            stage="E3b",
            method_role="LightCone",
            model="Qwen/Qwen3-8B",
            backend="DFlash",
            task="production_slo_power_prefix",
            publication_policy="first_ready",
            recipe_sha256=_sha("prospective-recipe"),
            dimensions=tuple(
                sorted(
                    {
                        "block": block,
                        "block_phase": phase,
                        "topology": "tp1_dp1",
                    }.items()
                )
            ),
        )

    pilot_cells = tuple(
        sorted(
            (cell(block, phase="excluded_pilot") for block in range(4)),
            key=lambda row: row.cell_id,
        )
    )
    pilot_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3b",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("prospective-upstream"),),
        source_decision_sha256=_sha("prospective-pilot-source"),
        materialization_rule="test_exact_four_excluded_pilot_blocks",
        expected_cell_count=4,
        cells=pilot_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    pilot_observations = tuple(
        sorted(
            (
                replace(
                    observation,
                    materialized_cell_id=pilot_cell.cell_id,
                    wave_index=index,
                )
                for index, (pilot_cell, observation) in enumerate(
                    zip(pilot_cells, source.observations, strict=True)
                )
            ),
            key=lambda row: (row.wave_index, row.materialized_cell_id),
        )
    )
    pilot_source = replace(
        source,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=pilot_materialization.sha256,
        inventory_sha256=inventory.sha256,
        observations=pilot_observations,
        schedule_sha256=gpu_hour_authority._schedule_sha256(pilot_observations),
    )
    final_cells = tuple(
        sorted(
            (cell(block, phase="final") for block in range(4, 16)),
            key=lambda row: row.cell_id,
        )
    )
    final_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3b",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("prospective-pilot-coverage"),),
        source_decision_sha256=_sha("prospective-signed-power"),
        materialization_rule="test_exact_twelve_final_blocks",
        expected_cell_count=12,
        cells=final_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    pilot_source_path = (tmp_path / "prospective-pilot-source.json").resolve()
    publish_canonical_json_no_replace(pilot_source_path, pilot_source.to_dict())
    pilot_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=pilot_materialization.sha256,
        signed_pilot_receipt_sha256=pilot_source.sha256,
        schedule_sha256=pilot_source.schedule_sha256,
        estimate=gpu_hour_authority._estimate(pilot_source),
    )
    return (
        lock,
        runtime_manifest,
        inventory,
        pilot_materialization,
        pilot_source,
        pilot_source_path,
        pilot_envelope,
        final_materialization,
    )


def test_excluded_pilots_project_exact_final_prefix_without_double_count(
    tmp_path,
    monkeypatch,
) -> None:
    (
        lock,
        _runtime_manifest_value,
        inventory,
        pilot_materialization,
        pilot_source,
        pilot_source_path,
        _pilot_envelope,
        final_materialization,
    ) = _prospective_projection_case(tmp_path, monkeypatch)

    projected, mapping_sha256 = gpu_hour_authority._project_final_cost(
        pilot_materialization=pilot_materialization,
        pilot_source=pilot_source,
        final_materialization=final_materialization,
    )
    actual = gpu_hour_authority._cost_from_actual_observations(
        category="actual_tuning",
        observations=pilot_source.observations,
        inventory_gpu_count=2,
    )
    assert actual.cell_count == 4
    assert projected.cell_count == 12
    assert actual.compute_gpu_ns == 4 * _NS_PER_HOUR
    assert projected.compute_gpu_ns == 12 * _NS_PER_HOUR
    assert projected.profile_reserve_gpu_ns == 0
    assert gpu_hour_authority._scaled_ceiling(0, numerator=10_000, denominator=1) == 0

    prospective = gpu_hour_authority.ProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="prospective_gpu_hour_source_manifest",
        protocol_sha256=gpu_hour_authority.PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=_sha("prospective-runtime-member"),
        stage="E3b",
        final_materialization_receipt_sha256=final_materialization.sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_source_manifest=CanonicalJsonProofBinding.bind(pilot_source_path),
        one_shot_source_manifest=None,
        prospective_authority_sha256=_sha("prospective-authority"),
        signed_power_authority_sha256=_sha("prospective-signed-power"),
        signed_power_challenge_sha256=_sha("prospective-power-challenge"),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        mapping_sha256=mapping_sha256,
        costs=(actual, projected),
    )
    estimate = gpu_hour_authority._estimate_prospective_manifest(prospective)
    assert estimate.compute_gpu_hours == pytest.approx(16.0)
    assert estimate.estimated_wall_hours == pytest.approx(16.0)
    assert estimate.profile_reserve_gpu_hours == 0.0
    assert estimate.retry_reserve_gpu_hours == pytest.approx(1.6)
    assert estimate.reserved_gpu_hours == pytest.approx(33.6)


def test_two_simultaneous_tp1_and_three_sequential_tp2_have_exact_hours(
    tmp_path,
    monkeypatch,
) -> None:
    *_, parallel = _case(
        tmp_path / "parallel",
        gangs=(("GPU-0",), ("GPU-1",)),
        starts=(1_000_000_000, 1_000_000_000),
        monkeypatch=monkeypatch,
    )
    assert parallel.estimate.compute_gpu_hours == pytest.approx(2.0)
    assert parallel.estimate.estimated_wall_hours == pytest.approx(1.0)
    assert parallel.estimate.reserved_gpu_hours == pytest.approx(2.2)

    *_, tp2 = _case(
        tmp_path / "tp2",
        gangs=(
            ("GPU-0", "GPU-1"),
            ("GPU-0", "GPU-1"),
            ("GPU-0", "GPU-1"),
        ),
        starts=(
            1_000_000_000,
            2 * _NS_PER_HOUR,
            4 * _NS_PER_HOUR,
        ),
        monkeypatch=monkeypatch,
    )
    assert tp2.estimate.compute_gpu_hours == pytest.approx(6.0)
    assert tp2.estimate.estimated_wall_hours == pytest.approx(3.0)
    assert tp2.estimate.reserved_gpu_hours == pytest.approx(6.6)


def test_persisted_source_rejects_real_timing_relabelled_to_another_cell(
    tmp_path,
    monkeypatch,
) -> None:
    (
        lock,
        runtime_manifest,
        inventory,
        materialization,
        _proof_inputs,
        _verified,
        source_path,
        _envelope,
    ) = _case(
        tmp_path,
        gangs=(("GPU-0",), ("GPU-0",)),
        starts=(1_000_000_000, 2 * _NS_PER_HOUR),
        monkeypatch=monkeypatch,
    )
    source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_path).reopen()
    )
    left, right = source.observations
    observations = tuple(
        sorted(
            (
                replace(left, materialized_cell_id=right.materialized_cell_id),
                replace(right, materialized_cell_id=left.materialized_cell_id),
            ),
            key=lambda row: (row.wave_index, row.materialized_cell_id),
        )
    )
    relabelled = replace(
        source,
        observations=observations,
        schedule_sha256=gpu_hour_authority._schedule_sha256(observations),
    )
    relabelled_path = (tmp_path / "relabelled-source.json").resolve()
    publish_canonical_json_no_replace(relabelled_path, relabelled.to_dict())
    relabelled_envelope = StageGpuHourEnvelope(
        schema_version=2,
        protocol_lock_sha256=lock.sha256,
        materialization_receipt_sha256=materialization.sha256,
        signed_pilot_receipt_sha256=relabelled.sha256,
        schedule_sha256=relabelled.schedule_sha256,
        estimate=gpu_hour_authority._estimate(relabelled),
    )
    with pytest.raises(ValueError, match="proof differs from source"):
        revalidate_persisted_stage_gpu_hour_source_manifest(
            str(relabelled_path),
            envelope=relabelled_envelope,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=2_000_000_000,
        )


def test_revalidation_deep_rejects_changed_timing_and_foreign_runtime_member(
    tmp_path,
    monkeypatch,
) -> None:
    (
        lock,
        runtime_manifest,
        inventory,
        materialization,
        proof_inputs,
        verified,
        source_path,
        envelope,
    ) = _case(
        tmp_path,
        gangs=(("GPU-0",),),
        starts=(1_000_000_000,),
        monkeypatch=monkeypatch,
    )
    reopened = revalidate_stage_gpu_hour_source_manifest(
        str(source_path),
        envelope=envelope,
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        proof_inputs=proof_inputs,
        now_ns=2_000_000_000,
    )
    assert isinstance(reopened, LifecycleGpuHourSourceManifest)
    persisted = revalidate_persisted_stage_gpu_hour_source_manifest(
        str(source_path),
        envelope=envelope,
        protocol_lock=lock,
        formal_runtime_authority_manifest=runtime_manifest,
        materialization=materialization,
        inventory=inventory,
        now_ns=2_000_000_000,
    )
    assert persisted == reopened

    proof_path = proof_inputs[0].lifecycle_proof_artifact_path
    original = verified[proof_path]
    altered = object.__new__(VerifiedPinnedSglangLifecycleTimingProof)
    for name in original.__dataclass_fields__:
        object.__setattr__(altered, name, getattr(original, name))
    edges, durations = _phase_values(1_000_000_000, 2 * _NS_PER_HOUR)
    object.__setattr__(altered, "phase_edges_ns", edges)
    object.__setattr__(altered, "phase_durations_ns", durations)
    verified[proof_path] = altered
    with pytest.raises(ValueError, match="differs from lifecycle proofs"):
        revalidate_stage_gpu_hour_source_manifest(
            str(source_path),
            envelope=envelope,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            proof_inputs=proof_inputs,
            now_ns=2_000_000_000,
        )
    with pytest.raises(ValueError, match="proof differs from source"):
        revalidate_persisted_stage_gpu_hour_source_manifest(
            str(source_path),
            envelope=envelope,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=2_000_000_000,
        )

    foreign_member = replace(
        runtime_manifest.member("gpu_hour_budget_reducer"),
        runner_sha256=_sha("foreign-gpu-hour-runner"),
    )
    foreign_manifest = replace(
        runtime_manifest,
        members=tuple(
            foreign_member if row.member_id == foreign_member.member_id else row
            for row in runtime_manifest.members
        ),
    )
    foreign_lock = replace(
        lock, formal_runtime_authority_manifest_sha256=foreign_manifest.sha256
    )
    foreign_materialization = replace(
        materialization, protocol_lock_sha256=foreign_lock.sha256
    )
    with pytest.raises(
        FormalGpuHourLifecycleBlocked,
        match="gpu_hour_budget_reducer_source_identity_mismatch",
    ):
        materialize_stage_gpu_hour_envelope_from_lifecycle_proofs(
            protocol_lock=foreign_lock,
            formal_runtime_authority_manifest=foreign_manifest,
            materialization=foreign_materialization,
            inventory=inventory,
            proof_inputs=proof_inputs,
            source_manifest_output_path=str((tmp_path / "foreign.json").resolve()),
            now_ns=2_000_000_000,
        )


def test_staged_projection_blocks_unmeasured_strata_and_uses_same_stratum_only(
    tmp_path,
    monkeypatch,
) -> None:
    (
        lock,
        _runtime_manifest,
        inventory,
        original_materialization,
        _proof_inputs,
        _verified,
        original_source_path,
        _envelope,
    ) = _case(
        tmp_path / "staged",
        gangs=(("GPU-0",), ("GPU-1",)),
        starts=(10_000_000_000, 10_000_000_000),
        monkeypatch=monkeypatch,
    )
    cells = tuple(
        sorted(
            (
                replace(
                    cell,
                    dimensions=(
                        ("registry_cell_id", f"e3a-source-{index}"),
                        ("science", "same"),
                    ),
                )
                for index, cell in enumerate(original_materialization.cells)
            ),
            key=lambda row: row.cell_id,
        )
    )
    materialization = replace(
        original_materialization,
        cells=cells,
        expected_cell_count=len(cells),
    )
    original_source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(original_source_path).reopen()
    )
    observations = tuple(
        replace(row, materialized_cell_id=cell.cell_id)
        for row, cell in zip(original_source.observations, cells, strict=True)
    )
    completed_observations = (observations[0],)
    completed_source = replace(
        original_source,
        materialization_receipt_sha256=materialization.sha256,
        observations=completed_observations,
        schedule_sha256=gpu_hour_authority._schedule_sha256(completed_observations),
    )
    completed_path = (tmp_path / "completed-staged-source.json").resolve()
    publish_canonical_json_no_replace(completed_path, completed_source.to_dict())
    completed_binding = CanonicalJsonProofBinding.bind(completed_path)

    blocked = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=(
            completed_source.runtime_authority_member_sha256
        ),
        materialization=materialization,
        inventory=inventory,
        completed_source_binding=None,
        completed_source=None,
    )
    assert blocked.status == "BLOCKED"
    assert blocked.actual_completed.cell_count == 0
    assert blocked.projected_remaining is None
    assert blocked.total is None
    assert blocked.minimum_pilot_cell_ids == (cells[0].cell_id,)

    blocked_path = (tmp_path / "blocked-staged-source.json").resolve()
    published_blocked, published_envelope = (
        gpu_hour_authority.materialize_staged_prospective_gpu_hour_envelope(
            protocol_lock=lock,
            formal_runtime_authority_manifest=_runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            source_manifest_output_path=str(blocked_path),
            now_ns=2_000_000_000,
        )
    )
    assert published_blocked == blocked
    assert published_envelope is None
    with pytest.raises(RuntimeError, match="target already exists"):
        gpu_hour_authority.materialize_staged_prospective_gpu_hour_envelope(
            protocol_lock=lock,
            formal_runtime_authority_manifest=_runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            source_manifest_output_path=str(blocked_path),
            now_ns=2_000_000_000,
        )

    ready = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=(
            completed_source.runtime_authority_member_sha256
        ),
        materialization=materialization,
        inventory=inventory,
        completed_source_binding=completed_binding,
        completed_source=completed_source,
    )
    assert ready.status == "READY"
    assert ready.minimum_pilot_cell_ids == ()
    assert ready.actual_completed.cell_count == 1
    assert ready.projected_remaining is not None
    assert ready.projected_remaining.cell_count == 1
    assert ready.total is not None
    assert ready.total.cell_count == 2
    assert ready.total.compute_gpu_ns == (
        ready.actual_completed.compute_gpu_ns + ready.projected_remaining.compute_gpu_ns
    )
    pilot_row = completed_source.observations[0]
    pilot_edges = dict(pilot_row.phase_edges_ns)
    assert pilot_edges["execution_started_ns"] < pilot_edges["server_ready_ns"]
    assert pilot_edges["server_ready_ns"] < pilot_edges["warmup_finished_ns"]
    assert ready.actual_completed.compute_gpu_ns == (
        gpu_hour_authority._gpu_process_occupied_ns(pilot_row)
        * pilot_row.gang_gpu_count
    )
    assert ready.projected_remaining.compute_gpu_ns == (
        ready.actual_completed.compute_gpu_ns
    )
    assert ready.actual_completed.profile_reserve_gpu_ns == 0
    assert ready.projected_remaining.profile_reserve_gpu_ns == 0
    assert (
        gpu_hour_authority.StagedProspectiveGpuHourSourceManifest.from_dict(
            ready.to_dict()
        )
        == ready
    )
    tampered_ready = ready.to_dict()
    tampered_ready["actual_completed"]["cell_count"] = 2
    tampered_ready["manifest_sha256"] = gpu_hour_authority.content_sha256(
        {
            key: value
            for key, value in tampered_ready.items()
            if key != "manifest_sha256"
        }
    )
    with pytest.raises(ValueError, match="actual coverage differs"):
        gpu_hour_authority.StagedProspectiveGpuHourSourceManifest.from_dict(
            tampered_ready
        )
    assert gpu_hour_authority._staged_prospective_envelope(ready).schema_version == 2

    split_cells = tuple(
        sorted(
            (
                cells[0],
                replace(
                    cells[1],
                    dimensions=(
                        ("registry_cell_id", "e3a-source-other"),
                        ("science", "other"),
                    ),
                ),
            ),
            key=lambda row: row.cell_id,
        )
    )
    split_materialization = replace(materialization, cells=split_cells)
    completed_cell = next(
        cell for cell in split_cells if dict(cell.dimensions)["science"] == "same"
    )
    split_observation = replace(
        completed_observations[0],
        materialized_cell_id=completed_cell.cell_id,
    )
    split_source = replace(
        completed_source,
        materialization_receipt_sha256=split_materialization.sha256,
        observations=(split_observation,),
        schedule_sha256=gpu_hour_authority._schedule_sha256((split_observation,)),
    )
    split_path = (tmp_path / "split-stratum-source.json").resolve()
    publish_canonical_json_no_replace(split_path, split_source.to_dict())
    split = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=(
            completed_source.runtime_authority_member_sha256
        ),
        materialization=split_materialization,
        inventory=inventory,
        completed_source_binding=CanonicalJsonProofBinding.bind(split_path),
        completed_source=split_source,
    )
    assert split.status == "BLOCKED"
    assert split.actual_completed.cell_count == 1
    assert split.projected_remaining is None
    assert split.minimum_pilot_cell_ids == tuple(
        row.cell_ids[0] for row in split.strata if row.status == "UNMEASURED"
    )
    assert split.minimum_pilot_cell_ids != ()

    foreign_source = replace(
        completed_source,
        materialization_receipt_sha256=_sha("foreign-materialization"),
    )
    with pytest.raises(ValueError, match="source lineage differs"):
        gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
            protocol_lock=lock,
            runtime_authority_member_sha256=(
                completed_source.runtime_authority_member_sha256
            ),
            materialization=materialization,
            inventory=inventory,
            completed_source_binding=completed_binding,
            completed_source=foreign_source,
        )


@pytest.mark.parametrize(
    ("stage", "varying_dimension", "expected_strata"),
    (
        ("E3a", "registry_cell_id", 1),
        ("TTS-Cal", "block", 1),
        ("E1", "search_axis", 2),
        ("E2", "search_axis", 2),
        ("E4", "search_axis", 2),
        ("E1a", "search_axis", 2),
    ),
)
def test_staged_projection_protocol_covers_every_early_dag_stage(
    stage: str,
    varying_dimension: str,
    expected_strata: int,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    original = _materialization(lock.sha256, 2)
    cells = tuple(
        sorted(
            (
                replace(
                    cell,
                    stage=stage,
                    dimensions=tuple(
                        sorted(
                            {
                                varying_dimension: index,
                                **(
                                    {"replicate_phase": "technical"}
                                    if stage == "TTS-Cal"
                                    else {}
                                ),
                            }.items()
                        )
                    ),
                )
                for index, cell in enumerate(original.cells)
            ),
            key=lambda row: row.cell_id,
        )
    )
    materialization = replace(
        original,
        stage=stage,
        cells=cells,
        expected_cell_count=len(cells),
    )
    blocked = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization=materialization,
        inventory=_inventory(),
        completed_source_binding=None,
        completed_source=None,
    )
    assert blocked.stage == stage
    assert blocked.status == "BLOCKED"
    assert len(blocked.strata) == expected_strata
    assert len(blocked.minimum_pilot_cell_ids) == expected_strata
    assert blocked.minimum_pilot_cell_ids == tuple(
        sorted(row.cell_ids[0] for row in blocked.strata)
    )
    assert blocked.to_dict()["status"] == "BLOCKED"


def test_staged_ready_source_deep_reopens_path_bound_lifecycle_proofs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        lock,
        runtime_manifest,
        inventory,
        materialization,
        _proof_inputs,
        _verified,
        completed_source_path,
        _actual_envelope,
    ) = _case(
        tmp_path / "deep-staged",
        gangs=(("GPU-0",), ("GPU-1",)),
        starts=(10_000_000_000, 10_000_000_000),
        monkeypatch=monkeypatch,
    )
    staged_source_path = (tmp_path / "staged-ready-source.json").resolve()
    source, envelope = (
        gpu_hour_authority.materialize_staged_prospective_gpu_hour_envelope(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            completed_source_manifest_path=str(completed_source_path),
            source_manifest_output_path=str(staged_source_path),
            now_ns=2_000_000_000,
        )
    )
    assert source.status == "READY"
    assert source.actual_completed.cell_count == 2
    assert source.projected_remaining is not None
    assert source.projected_remaining.cell_count == 0
    assert envelope is not None
    assert (
        gpu_hour_authority.revalidate_persisted_staged_prospective_gpu_hour_source_manifest(
            str(staged_source_path),
            envelope=envelope,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=2_000_000_001,
        )
        == source
    )

    Path(source.completed_source_manifest.absolute_path).write_text(
        '{"kind":"tampered-lifecycle-source"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields differ|binding changed"):
        gpu_hour_authority.revalidate_persisted_staged_prospective_gpu_hour_source_manifest(
            str(staged_source_path),
            envelope=envelope,
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=materialization,
            inventory=inventory,
            now_ns=2_000_000_002,
        )


def test_formal_launch_caps_preserve_parallel_provider_wave_and_preconsumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _lock,
        _runtime_manifest_value,
        _inventory_value,
        materialization,
        _proof_inputs,
        _verified,
        source_path,
        _envelope,
    ) = _case(
        tmp_path / "launch-caps-observed",
        gangs=(("GPU-0",), ("GPU-1",)),
        starts=(10_000_000_000, 10_000_000_000),
        monkeypatch=monkeypatch,
    )
    source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_path).reopen()
    )
    schedule = gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
        source,
        materialization,
    )

    assert schedule.launchable_cell_ids == ()
    assert schedule.preconsumed_compute_gpu_ns == 2 * _NS_PER_HOUR
    assert schedule.preconsumed_provider_reserved_gpu_ns == 2 * _NS_PER_HOUR
    assert {row.provider_reserved_gpu_count for row in schedule.cell_caps} == {1}
    assert len({row.wave_group_sha256 for row in schedule.cell_caps}) == 1
    assert (
        gpu_hour_authority.FormalLaunchCapSchedule.from_dict(schedule.to_dict())
        == schedule
    )

    tampered = schedule.to_dict()
    assert isinstance(tampered["cell_caps"], list)
    tampered["cell_caps"][0]["provider_reserved_gpu_count"] = 2
    with pytest.raises(ValueError, match="charge differs|provider wave"):
        gpu_hour_authority.FormalLaunchCapSchedule.from_dict(tampered)


def test_staged_launch_caps_authorize_only_same_stratum_remaining_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        lock,
        runtime_manifest,
        inventory,
        _fixture_materialization,
        _proof_inputs,
        _verified,
        fixture_source_path,
        _envelope,
    ) = _case(
        tmp_path / "launch-caps-staged-fixture",
        gangs=(("GPU-0",),),
        starts=(10_000_000_000,),
        monkeypatch=monkeypatch,
    )
    cells = tuple(
        sorted(
            (
                MaterializedCell(
                    stage="E3a",
                    method_role="Static",
                    model="Qwen/Qwen3-8B",
                    backend="DFlash",
                    task="LiveCodeBench",
                    publication_policy="none",
                    recipe_sha256=None,
                    dimensions=(("registry_cell_id", _sha(f"repeat-{index}")),),
                )
                for index in range(3)
            ),
            key=lambda row: row.cell_id,
        )
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E3a",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("preflight-coverage"),),
        source_decision_sha256=lock.formal_workload_e3a_authorization_sha256,
        materialization_rule="test_three_same_stratum_rows",
        expected_cell_count=3,
        cells=cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    completed = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(fixture_source_path).reopen()
    )
    observation = replace(
        completed.observations[0],
        materialized_cell_id=cells[0].cell_id,
    )
    completed = replace(
        completed,
        materialization_receipt_sha256=materialization.sha256,
        observations=(observation,),
        schedule_sha256=gpu_hour_authority._schedule_sha256((observation,)),
    )
    completed_path = (tmp_path / "launch-caps-staged-completed.json").resolve()
    publish_canonical_json_no_replace(completed_path, completed.to_dict())
    staged = gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
        protocol_lock=lock,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        materialization=materialization,
        inventory=inventory,
        completed_source_binding=CanonicalJsonProofBinding.bind(completed_path),
        completed_source=completed,
    )
    assert staged.status == "READY"
    schedule = gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
        staged,
        materialization,
    )

    assert schedule.launchable_cell_ids == tuple(
        sorted(cell.cell_id for cell in cells[1:])
    )
    assert schedule.cap_for(cells[0].cell_id).disposition == "PRECONSUMED"
    assert all(
        schedule.cap_for(cell.cell_id).provider_reserved_gpu_count == 2
        for cell in cells[1:]
    )
    # Projection has no interference authority in its source graph, so it is
    # deliberately isolated.  It cannot silently claim a paired TP1 wave.
    projected_rows = tuple(schedule.cap_for(cell.cell_id) for cell in cells[1:])
    assert len({row.wave_group_sha256 for row in projected_rows}) == len(projected_rows)
    assert staged.projected_remaining is not None
    assert schedule.retry_reserve_gpu_ns == (
        staged.projected_remaining.retry_reserve_gpu_ns
    )
    assert schedule.maximum_compute_gpu_ns == (
        staged.total.compute_gpu_ns + staged.projected_remaining.retry_reserve_gpu_ns
    )
    for row in schedule.cell_caps:
        assert row.maximum_compute_gpu_ns_per_attempt == (
            row.process_hard_timeout_ns_per_attempt * row.gpu_count
        )
        assert row.maximum_provider_reserved_gpu_ns_per_attempt == (
            row.provider_wave_hard_timeout_ns_per_attempt
            * row.provider_reserved_gpu_count
        )


def test_prospective_launch_caps_charge_pilots_once_and_only_unlock_finals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        lock,
        _runtime_manifest_value,
        inventory,
        pilot_materialization,
        pilot_source,
        pilot_source_path,
        _pilot_envelope,
        final_materialization,
    ) = _prospective_projection_case(tmp_path, monkeypatch)
    actual = gpu_hour_authority._cost_from_actual_observations(
        category="actual_tuning",
        observations=pilot_source.observations,
        inventory_gpu_count=2,
    )
    projected, mapping_sha256 = gpu_hour_authority._project_final_cost(
        pilot_materialization=pilot_materialization,
        pilot_source=pilot_source,
        final_materialization=final_materialization,
    )
    source = gpu_hour_authority.ProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="prospective_gpu_hour_source_manifest",
        protocol_sha256=gpu_hour_authority.PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=_sha("prospective-runtime-member"),
        stage="E3b",
        final_materialization_receipt_sha256=final_materialization.sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_source_manifest=CanonicalJsonProofBinding.bind(pilot_source_path),
        one_shot_source_manifest=None,
        prospective_authority_sha256=_sha("prospective-authority"),
        signed_power_authority_sha256=_sha("prospective-power"),
        signed_power_challenge_sha256=_sha("prospective-power-challenge"),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        mapping_sha256=mapping_sha256,
        costs=(actual, projected),
    )
    schedule = gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
        source,
        final_materialization,
        pilot_materialization=pilot_materialization,
    )

    assert schedule.launchable_cell_ids == tuple(
        cell.cell_id for cell in final_materialization.cells
    )
    assert schedule.preconsumed_compute_gpu_ns == actual.compute_gpu_ns
    assert schedule.launchable_compute_gpu_ns == projected.compute_gpu_ns
    assert schedule.retry_reserve_gpu_ns == projected.retry_reserve_gpu_ns
    assert all(
        row.disposition == "LAUNCHABLE" and row.provider_reserved_gpu_count == 2
        for row in schedule.cell_caps
    )
    with pytest.raises(ValueError, match="exact pilots"):
        gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
            source,
            final_materialization,
        )
    e5_source = replace(
        source,
        stage="E5",
        one_shot_source_manifest=source.pilot_source_manifest,
        costs=(*source.costs, replace(actual, category="actual_one_shot")),
    )
    with pytest.raises(
        gpu_hour_authority.FormalGpuHourLifecycleBlocked,
        match="e5_dedicated_failure_launch_cap_required",
    ):
        gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
            e5_source,
            final_materialization,
            pilot_materialization=pilot_materialization,
        )


def test_preflight_legacy_1_plus_1_plus_8_never_publishes_available_budget(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "forbidden-preflight-source.json").resolve()
    with pytest.raises(
        FormalGpuHourLifecycleBlocked,
        match="preflight_qualification_lifecycle_cost_schedule_missing",
    ):
        gpu_hour_authority.materialize_preflight_gpu_hour_envelope(
            protocol_lock=None,  # type: ignore[arg-type]
            formal_runtime_authority_manifest=None,  # type: ignore[arg-type]
            final_evidence=None,
            inventory=None,  # type: ignore[arg-type]
            interference_lifecycle_proof_inputs=(),
            source_manifest_output_path=str(output),
            now_ns=1,
        )
    assert not output.exists()


def test_e4_profiler_rejects_serving_gpu_hours_staging_and_launch_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    inventory = _inventory()
    profiler = _materialize_e4_profiler_diagnostic(
        protocol_lock_sha256=lock.sha256,
        upstream_local_receipt_sha256=_sha("e4-local"),
        source_decision_sha256=_sha("e4-selection"),
        selected_configuration_sha256=_sha("e4-configuration"),
        model="Qwen/Qwen3.6-35B-A3B",
        lightcone_recipe_sha256=_sha("e4-lightcone"),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    with pytest.raises(
        FormalGpuHourLifecycleBlocked,
        match="e4_profiler_dedicated_lifecycle_cost_proof_missing",
    ):
        materialize_stage_gpu_hour_envelope_from_lifecycle_proofs(
            protocol_lock=lock,
            formal_runtime_authority_manifest=runtime_manifest,
            materialization=profiler,
            inventory=inventory,
            proof_inputs=(),
            source_manifest_output_path=str(tmp_path / "forbidden-profiler.json"),
            now_ns=2_000_000_000,
        )
    with pytest.raises(
        FormalGpuHourLifecycleBlocked,
        match="e4_profiler_dedicated_lifecycle_cost_proof_missing",
    ):
        gpu_hour_authority._derive_staged_prospective_gpu_hour_source(
            protocol_lock=lock,
            runtime_authority_member_sha256=runtime_manifest.member(
                "gpu_hour_budget_reducer"
            ).sha256,
            materialization=profiler,
            inventory=inventory,
            completed_source_binding=None,
            completed_source=None,
        )

    (
        _fixture_lock,
        _fixture_runtime,
        _fixture_inventory,
        _fixture_materialization,
        _fixture_inputs,
        _fixture_verified,
        source_path,
        _fixture_envelope,
    ) = _case(
        tmp_path / "ordinary-source",
        gangs=(("GPU-0",),),
        starts=(10_000_000_000,),
        monkeypatch=monkeypatch,
    )
    ordinary = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(source_path).reopen()
    )
    with pytest.raises(
        FormalGpuHourLifecycleBlocked,
        match="e4_profiler_dedicated_lifecycle_cost_proof_missing",
    ):
        gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
            ordinary,
            profiler,
        )


def test_e6_ordinary_serving_lifecycle_cannot_cost_model_preflights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    inventory = _inventory()
    preflights = tuple(
        MaterializedCell(
            stage="E6",
            method_role="Target-only",
            model=f"example/model-{index}",
            backend="NEXTN",
            task="immutable_metadata_interface_and_fit_preflight",
            publication_policy="none",
            recipe_sha256=None,
            dimensions=(("topology", "tp2_dp1"),),
        )
        for index in range(2)
    )

    def block_cell(block: int, phase: str) -> MaterializedCell:
        return MaterializedCell(
            stage="E6",
            method_role="Static",
            model="example/model-0",
            backend="NEXTN",
            task="heldout_acceptance",
            publication_policy="none",
            recipe_sha256=None,
            dimensions=(
                ("block", block),
                ("block_phase", phase),
                ("topology", "tp2_dp1"),
            ),
        )

    pilot_cells = tuple(
        sorted(
            (*preflights, *(block_cell(index, "excluded_pilot") for index in range(4))),
            key=lambda row: row.cell_id,
        )
    )
    pilot_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E6",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("e5-confirmation"),),
        source_decision_sha256=_sha("e6-pilot-source"),
        materialization_rule="test_two_preflights_plus_four_excluded_pilots",
        expected_cell_count=len(pilot_cells),
        cells=pilot_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    starts = tuple(
        10_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(len(pilot_cells))
    )
    with pytest.raises(
        gpu_hour_authority.FormalGpuHourLifecycleBlocked,
        match="e6_model_compatibility_lifecycle_cost_proof_missing",
    ):
        _case(
            tmp_path / "e6-pilot-source",
            gangs=tuple(("GPU-0", "GPU-1") for _cell in pilot_cells),
            starts=starts,
            monkeypatch=monkeypatch,
            lock_override=lock,
            runtime_manifest_override=runtime_manifest,
            inventory_override=inventory,
            materialization_override=pilot_materialization,
        )


def test_e5_ordinary_serving_lifecycle_cannot_cost_failure_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    inventory = _inventory()
    failure_cell = MaterializedCell(
        stage="E5",
        method_role="Static",
        model="Qwen/Qwen3-8B",
        backend="DFlash",
        task="deterministic_failure_injection",
        publication_policy="diagnostic_only",
        recipe_sha256=_sha("e5-failure-recipe"),
        dimensions=(("failure_test_index", 0),),
    )
    materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E5",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("e5-final-upstream"),),
        source_decision_sha256=_sha("e5-final-source"),
        materialization_rule="test_single_failure_row_must_fail_closed",
        expected_cell_count=1,
        cells=(failure_cell,),
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    with pytest.raises(
        FormalGpuHourLifecycleBlocked,
        match="e5_dedicated_failure_lifecycle_cost_proof_required",
    ):
        _case(
            tmp_path / "ordinary-e5-failure",
            gangs=(("GPU-0",),),
            starts=(10_000_000_000,),
            monkeypatch=monkeypatch,
            lock_override=lock,
            runtime_manifest_override=runtime_manifest,
            inventory_override=inventory,
            materialization_override=materialization,
        )


def test_e5_failure_cost_source_is_exact_actual_only_and_caps_are_preconsumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_manifest = _runtime_manifest()
    lock = replace(
        _protocol_lock(),
        formal_runtime_authority_manifest_sha256=runtime_manifest.sha256,
    )
    inventory = _inventory()

    def headline(block: int, phase: str) -> MaterializedCell:
        return MaterializedCell(
            stage="E5",
            method_role="Static",
            model="Qwen/Qwen3-8B",
            backend="DFlash",
            task="production_slo_power_prefix",
            publication_policy="first_ready",
            recipe_sha256=_sha("e5-cap-recipe"),
            dimensions=(
                ("block", block),
                ("block_phase", phase),
                ("topology", "tp1_dp1"),
            ),
        )

    pilot_cells = tuple(
        sorted(
            (headline(block, "excluded_pilot") for block in range(4)),
            key=lambda row: row.cell_id,
        )
    )
    pilot_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E5",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("e5-cap-upstream"),),
        source_decision_sha256=_sha("e5-cap-pilots"),
        materialization_rule="test_exact_four_e5_excluded_pilots",
        expected_cell_count=4,
        cells=pilot_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    *_, pilot_source_path, _pilot_envelope = _case(
        tmp_path / "pilots",
        gangs=(("GPU-0",),) * 4,
        starts=tuple(10_000_000_000 + index * 2 * _NS_PER_HOUR for index in range(4)),
        monkeypatch=monkeypatch,
        lock_override=lock,
        runtime_manifest_override=runtime_manifest,
        inventory_override=inventory,
        materialization_override=pilot_materialization,
    )
    pilot_source = LifecycleGpuHourSourceManifest.from_dict(
        CanonicalJsonProofBinding.bind(pilot_source_path).reopen()
    )
    failure_cells = tuple(
        MaterializedCell(
            stage="E5",
            method_role="LightCone",
            model="Qwen/Qwen3-8B",
            backend="DFlash",
            task="deterministic_failure_injection",
            publication_policy="diagnostic_only",
            recipe_sha256=_sha("e5-cap-recipe"),
            dimensions=(("failure_test_index", index),),
        )
        for index in range(264)
    )
    final_cells = tuple(
        sorted((*failure_cells, headline(4, "final")), key=lambda row: row.cell_id)
    )
    final_materialization = StageMaterializationReceipt(
        schema_version=1,
        stage="E5",
        protocol_lock_sha256=lock.sha256,
        upstream_receipt_sha256s=(_sha("e5-cap-upstream"),),
        source_decision_sha256=_sha("e5-cap-final"),
        materialization_rule="test_one_final_plus_exact_264_failure_rows",
        expected_cell_count=len(final_cells),
        cells=final_cells,
        gpu_hours=GpuHourEstimate.unmeasured(),
    )
    observations = []
    for wave_index, cell in enumerate(
        sorted(failure_cells, key=lambda row: row.cell_id)
    ):
        started_ns = 100_000_000_000 + wave_index * 100
        lifecycle_semantic = _sha(f"e5-cost-proof:{wave_index}")
        projection = FormalE5FailureLifecycleCostProjection(
            schema_version=1,
            kind="formal_e5_failure_lifecycle_cost_projection",
            proof_artifact_sha256=lifecycle_semantic,
            raw_lifecycle_receipt_sha256=_sha(f"e5-cost-raw:{wave_index}"),
            formal_failure_execution_binding_sha256=_sha(
                f"e5-cost-failure-binding:{wave_index}"
            ),
            failure_subject_sha256=_sha(f"e5-cost-subject:{wave_index}"),
            materialized_cell_id=cell.cell_id,
            serving_execution_binding_sha256=_sha(
                f"e5-cost-serving-binding:{wave_index}"
            ),
            serving_execution_plan_sha256=_sha(f"e5-cost-serving-plan:{wave_index}"),
            assignment_sha256=_sha(f"e5-cost-assignment:{wave_index}"),
            inventory_sha256=inventory.sha256,
            registry_sha256=lock.registry_sha256,
            root_manifest_sha256=lock.offline_release_trust_root_sha256,
            run_nonce_sha256=_sha(f"e5-cost-run:{wave_index}"),
            topology_mode="tp1_dp1",
            gpu_uuids=("GPU-0",),
            server_process_group_id=wave_index + 1,
            formal_launch_admission_sha256=_sha(f"e5-cost-admission:{wave_index}"),
            formal_launch_consumption_sha256=_sha(
                f"e5-cost-launch-consumption:{wave_index}"
            ),
            budget_consumption_sha256=_sha(f"e5-cost-budget-consumption:{wave_index}"),
            raw_failure_terminal_sha256=_sha(f"e5-cost-terminal:{wave_index}"),
            recovery_receipt_sha256=_sha(f"e5-cost-recovery:{wave_index}"),
            execution_started_ns=started_ns,
            process_exited_ns=started_ns + 10,
            process_group_empty_checked_ns=started_ns + 12,
            gpu_release_ns=started_ns + 20,
            evidence_flush_finished_ns=started_ns + 25,
            compute_gpu_ns=10,
            provider_reserved_gpu_ns=40,
            evidence_gpu_ns=10,
        )
        reservation_sha256 = _sha(f"e5-cost-reservation:{wave_index}")
        observations.append(
            gpu_hour_authority.E5FailureGpuHourObservation(
                materialized_cell_id=cell.cell_id,
                failure_execution_rebuild_input=CanonicalJsonProofBinding(
                    absolute_path=str(
                        (tmp_path / f"failure-rebuild-{wave_index}.json").resolve()
                    ),
                    raw_sha256=_sha(f"e5-cost-rebuild-raw:{wave_index}"),
                    semantic_sha256=_sha(f"e5-cost-rebuild-semantic:{wave_index}"),
                    size=2,
                ),
                lifecycle_proof=CanonicalJsonProofBinding(
                    absolute_path=str(
                        (tmp_path / f"failure-proof-{wave_index}.json").resolve()
                    ),
                    raw_sha256=_sha(f"e5-cost-proof-raw:{wave_index}"),
                    semantic_sha256=lifecycle_semantic,
                    size=2,
                ),
                projection=projection,
                control_envelope_sha256=_sha(f"e5-cost-control:{wave_index}"),
                replay_reservation=ChallengeReplayReservationBinding(
                    schema_version=1,
                    kind="lightcone_challenge_replay_reservation_binding",
                    path=str(
                        (tmp_path / f"reservation-{reservation_sha256}.json").resolve()
                    ),
                    reservation_sha256=reservation_sha256,
                    raw_sha256=_sha(f"e5-cost-reservation-raw:{wave_index}"),
                    size=2,
                    reserved_ns=started_ns,
                    challenge_sha256s=(_sha(f"e5-cost-challenge:{wave_index}"),),
                ),
                wave_index=wave_index,
            )
        )
    observation_tuple = tuple(observations)
    failure_source = gpu_hour_authority.E5FailureGpuHourSourceManifest(
        schema_version=1,
        kind="e5_failure_gpu_hour_source_manifest",
        protocol_sha256=(gpu_hour_authority.E5_FAILURE_GPU_HOUR_SOURCE_PROTOCOL_SHA256),
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        runtime_authority_manifest_sha256=runtime_manifest.sha256,
        materialization_receipt_sha256=final_materialization.sha256,
        inventory_sha256=inventory.sha256,
        registry_sha256=lock.registry_sha256,
        root_manifest_sha256=lock.offline_release_trust_root_sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        observations=observation_tuple,
        cost=gpu_hour_authority._e5_failure_cost(observation_tuple),
        schedule_sha256=gpu_hour_authority._e5_failure_schedule_sha256(
            observation_tuple
        ),
    )
    failure_source_path = (tmp_path / "e5-failure-cost-source.json").resolve()
    publish_canonical_json_no_replace(failure_source_path, failure_source.to_dict())
    assert (
        gpu_hour_authority.E5FailureGpuHourSourceManifest.from_dict(
            CanonicalJsonProofBinding.bind(failure_source_path).reopen()
        )
        == failure_source
    )
    actual_tuning = gpu_hour_authority._cost_from_actual_observations(
        category="actual_tuning",
        observations=pilot_source.observations,
        inventory_gpu_count=2,
    )
    projected, mapping_sha256 = gpu_hour_authority._project_final_cost(
        pilot_materialization=pilot_materialization,
        pilot_source=pilot_source,
        final_materialization=final_materialization,
    )
    source = gpu_hour_authority.ProspectiveGpuHourSourceManifest(
        schema_version=1,
        kind="prospective_gpu_hour_source_manifest",
        protocol_sha256=gpu_hour_authority.PROSPECTIVE_GPU_HOUR_PROTOCOL_SHA256,
        protocol_lock_sha256=lock.sha256,
        runtime_authority_member_sha256=runtime_manifest.member(
            "gpu_hour_budget_reducer"
        ).sha256,
        stage="E5",
        final_materialization_receipt_sha256=final_materialization.sha256,
        pilot_materialization_receipt_sha256=pilot_materialization.sha256,
        pilot_source_manifest=CanonicalJsonProofBinding.bind(pilot_source_path),
        one_shot_source_manifest=CanonicalJsonProofBinding.bind(failure_source_path),
        prospective_authority_sha256=_sha("e5-cap-authority"),
        signed_power_authority_sha256=_sha("e5-cap-power"),
        signed_power_challenge_sha256=_sha("e5-cap-power-challenge"),
        inventory_sha256=inventory.sha256,
        hardware_envelope_sha256=pilot_source.hardware_envelope_sha256,
        mapping_sha256=mapping_sha256,
        costs=(actual_tuning, projected, failure_source.cost),
    )
    schedule = gpu_hour_authority.derive_and_validate_formal_launch_cap_schedule(
        source,
        final_materialization,
        pilot_materialization=pilot_materialization,
    )
    assert schedule.launchable_cell_ids == (headline(4, "final").cell_id,)
    assert sum(row.disposition == "PRECONSUMED" for row in schedule.cell_caps) == 264
    assert schedule.preconsumed_compute_gpu_ns == (
        actual_tuning.compute_gpu_ns + failure_source.cost.compute_gpu_ns
    )
    assert failure_source.cost.retry_reserve_gpu_ns == 0
    assert failure_source.cost.profile_reserve_gpu_ns == 0
    with pytest.raises(ValueError, match="reuses proof/control authority"):
        gpu_hour_authority.E5FailureGpuHourSourceManifest(
            **{
                field: getattr(failure_source, field)
                for field in failure_source.__dataclass_fields__
                if field != "observations"
            },
            observations=(
                failure_source.observations[0],
                replace(
                    failure_source.observations[1],
                    control_envelope_sha256=(
                        failure_source.observations[0].control_envelope_sha256
                    ),
                ),
                *failure_source.observations[2:],
            ),
        )
