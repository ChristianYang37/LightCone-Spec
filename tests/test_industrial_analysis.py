from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_native_terminal_provider import (
    FakeAdminTransport,
    _rehash_terminal,
    _server_request,
)
from test_native_terminal_provider import (
    _run as _run_native_terminal,
)

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.cli.main import main
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.industrial_analysis import (
    _INDUSTRIAL_DOCTOR_CHECKS,
    BoundArtifact,
    IndustrialBlockEvidence,
    IndustrialCellEvidence,
    _load_budget_observation,
    _validate_allocation_free_performance,
    _validate_industrial_doctor,
    _validate_industrial_gpu_attestation,
    _validate_run_row,
    reduce_confirmation_family_power,
    reduce_e2_stage_from_raw,
    reduce_industrial_schema_v3,
)
from lightcone_spec.experiments.planning import (
    AnalysisDependenceUnit,
    BudgetJobKind,
    ConfirmationFamilyPowerReductionArtifact,
    DispositionStatus,
    E1GeometryIdentity,
    E1ParetoArtifact,
    EvidenceDependenceMap,
    ExpectedMaximumCount,
    ExperimentBudget,
    FamilyActivationArtifact,
    P99AnchorStatus,
    ScenarioMilliseconds,
    derive_confirmation_family,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
    reduce_e2_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    confirmation_family_power_reduction_artifact_from_dict,
    family_activation_artifact_from_dict,
    family_activation_artifact_to_dict,
)
from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.orchestration.native_terminal import (
    NativeTerminalRunBinding,
    canonical_sha256,
)
from lightcone_spec.telemetry import (
    DEFAULT_EVIDENCE_WRITER_POLICY,
    OUTPUT_HASH_FORMAT,
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
    load_completed_evidence,
)

_PHYSICAL_GPU_UUIDS = ("GPU-analysis-a", "GPU-analysis-b")
_BUDGET_OBSERVATION_COMPONENTS = (
    "startup_model_load",
    "compile_jit_graph_prewarm",
    "excluded_warmup",
    "scored_arrival",
    "drain",
    "reset_finalization",
    "evidence_flush_shutdown",
    "soak",
    "failure_injection",
    "retry",
    "profiler",
    "download_compile_reservation",
)


def _gpu_inventory(*, gpu_count: int = 2) -> GpuInventory:
    gpu_uuids = tuple(
        sorted(
            _PHYSICAL_GPU_UUIDS
            + tuple(f"GPU-analysis-{index}" for index in range(2, gpu_count))
        )
    )
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id="analysis-host",
            model="A100",
            memory_bytes=80_000 * 1024 * 1024,
            compute_capability=(8, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root="root-0",
            numa_node=0,
            interconnects=("nvlink",),
            peer_access_class="nvlink",
            clock_policy="locked",
            power_limit_watts=300.0,
            thermal_limit_celsius=85.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=("analysis-nvlink",),
        )
        for index, uuid in enumerate(gpu_uuids[:gpu_count])
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="analysis-nvlink",
                host_id="analysis-host",
                gpu_uuids=tuple(device.uuid for device in devices),
                fabric="nvlink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=content_sha256(
            {"inventory": "analysis", "gpu_count": gpu_count}
        ),
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_native_terminal_artifact(
    writer: EvidenceWriter,
    *,
    method: str,
    run_nonce_sha256: str,
    execution_plan_sha256: str,
    rank_config_sha256: str,
    request_id: str,
    output_token_ids: tuple[int, ...],
) -> dict[str, object]:
    """Exercise the real provider and persist its canonical begin/reset/final chain."""

    binding = NativeTerminalRunBinding(
        run_id=writer.run_id,
        run_nonce_sha256=run_nonce_sha256,
        execution_plan_sha256=execution_plan_sha256,
        rank_config_sha256=rank_config_sha256,
        attempt_id=writer.attempt_id,
        session_id=f"standalone-{writer.run_id}",
        session_epoch=1,
        previous_run_id=None,
        challenge_nonce_sha256=content_sha256(
            {"challenge": writer.run_id, "attempt": writer.attempt_id}
        ),
        method=method,
        warmup_request_ids=(),
        scored_request_ids=(request_id,),
    )
    scored_request = _server_request(
        request_id,
        inputs=(1,),
        outputs=output_token_ids,
    )

    def align_adapted_terminal(value: dict[str, object]) -> None:
        if method not in {"tts", "l0"}:
            return
        request_round_rows = value["request_round_rows"]
        performance = value["performance_counters"]
        assert isinstance(request_round_rows, dict)
        assert isinstance(performance, dict)
        rounds = request_round_rows["rounds"]
        assert isinstance(rounds, list) and len(rounds) == 1
        round_row = rounds[0]
        assert isinstance(round_row, dict)
        accepted = max(len(output_token_ids) - 1, 0)
        round_row.update(
            {
                "verify_len": accepted,
                "accepted_drafts": accepted,
                "committed_tokens": len(output_token_ids),
            }
        )
        unsigned_round = dict(round_row)
        unsigned_round.pop("round_sha256", None)
        round_row["round_sha256"] = canonical_sha256(unsigned_round)
        performance.update(
            {
                "accepted_drafts": accepted,
                "committed_tokens": len(output_token_ids),
                "verified_drafts": accepted,
                "accepted_drafts_per_verify": float(accepted),
                "committed_tokens_per_verify": float(len(output_token_ids)),
                "verified_drafts_per_verify": float(accepted),
                "verification_waste": 0,
            }
        )
        _rehash_terminal(value)

    transport = FakeAdminTransport(
        binding=binding,
        warmup=(),
        scored=(scored_request,),
        terminal_mutator=align_adapted_terminal,
    )
    _, _, _, terminal = asyncio.run(_run_native_terminal(transport))
    assert terminal.binding == binding
    assert terminal.requests == (scored_request,)
    return writer.persist_native_terminal_artifact(
        terminal.to_artifact(warmup_requests=())
    )


def _write_json(path: Path, value: object) -> BoundArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return BoundArtifact(path=path, sha256=_file_sha256(path))


def _validate_durable_test_binding(reference: BoundArtifact) -> None:
    if not reference.path.is_file() or _file_sha256(reference.path) != reference.sha256:
        raise RuntimeError("test budget observation is not durably content-bound")


def _assert_registered_terminal_chain(
    *,
    terminal_path: Path,
    prepared_receipt: Path,
    budget_observation: BoundArtifact,
) -> None:
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    observation = json.loads(budget_observation.path.read_text(encoding="utf-8"))
    assert terminal["prepared_receipt_sha256"] == _file_sha256(prepared_receipt)
    assert observation["terminal_evidence_sha256"] == _file_sha256(prepared_receipt)
    assert (
        terminal["budget_observation"]["budget_observation_sha256"]
        == (observation["budget_observation_sha256"])
    )
    assert terminal["writer_policy"] == DEFAULT_EVIDENCE_WRITER_POLICY.to_dict()
    assert terminal["writer_policy_sha256"] == DEFAULT_EVIDENCE_WRITER_POLICY.sha256


def _write_budget_observation(path: Path, value: dict[str, object]) -> BoundArtifact:
    reference = _write_json(path, value)
    semantic_sha256 = value.get("budget_observation_sha256")
    if not isinstance(semantic_sha256, str):
        raise TypeError("test budget observation lacks its semantic digest")
    Path(f"{path}.sha256").write_text(
        f"{semantic_sha256}\n",
        encoding="ascii",
    )
    return reference


def _write_bound_json(path: Path, value: object) -> None:
    _write_json(path, value)
    Path(f"{path}.sha256").write_text(
        content_sha256(value) + "\n",
        encoding="utf-8",
    )


def _zero_budget(cell_id: str) -> ExperimentBudget:
    zero = ScenarioMilliseconds(0, 0, 0)
    zero_count = ExpectedMaximumCount(0, 0)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell_id,
        experiment="E3b",
        method="target_only",
        workload_class=WorkloadClass.HEADLINE,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=zero,
        compile_jit_graph_prewarm=zero,
        excluded_warmup=zero,
        excluded_warmup_requests=zero_count,
        scored_arrival=zero,
        request_deadline=zero,
        drain=zero,
        reset_finalization=zero,
        evidence_flush_shutdown=zero,
        output_tokens=zero_count,
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=zero,
        retry_allowance=0,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=1,
        topology="tp1_dp1",
        reserved_gpu_ms=zero,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=zero,
    )


def _execution_budget(cell: ExperimentCell) -> ExperimentBudget:
    def milliseconds(value: int) -> ScenarioMilliseconds:
        return ScenarioMilliseconds(value, value, value)

    zero = milliseconds(0)
    observed_wall_ms = 1_022
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=milliseconds(10),
        compile_jit_graph_prewarm=zero,
        excluded_warmup=zero,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=milliseconds(1_000),
        request_deadline=milliseconds(1_000),
        drain=milliseconds(10),
        reset_finalization=milliseconds(1),
        evidence_flush_shutdown=milliseconds(1),
        output_tokens=ExpectedMaximumCount(100, 100),
        minimum_completed_requests=1,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=zero,
        retry_allowance=0,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=milliseconds(observed_wall_ms * cell.resources.gpu_count),
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=milliseconds(
            observed_wall_ms * len(_PHYSICAL_GPU_UUIDS)
        ),
    )


def _budget_observation_value(
    budget: ExperimentBudget,
    *,
    terminal_receipt_sha256: str,
    fixed_instance_gpu_count: int = len(_PHYSICAL_GPU_UUIDS),
) -> dict[str, object]:
    observed_rows = [
        [name, getattr(budget, name).registered]
        for name in _BUDGET_OBSERVATION_COMPONENTS
    ]
    observed_wall_ms = sum(row[1] for row in observed_rows)
    measured_gpu_ms = observed_wall_ms * budget.gpu_count
    fixed_instance_billed_gpu_ms = observed_wall_ms * fixed_instance_gpu_count
    content = {
        "schema_version": 1,
        "budget": asdict(budget),
        "observed_component_ms": observed_rows,
        "measured_gpu_ms": measured_gpu_ms,
        "fixed_instance_billed_gpu_ms": fixed_instance_billed_gpu_ms,
        "terminal_evidence_sha256": terminal_receipt_sha256,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "industrial_budget_observation_receipt_v1",
        "experiment_budget_sha256": budget.sha256,
        "budget_observation_sha256": content_sha256(content),
        **{name: content[name] for name in content if name != "schema_version"},
        "observed_wall_ms": observed_wall_ms,
        "registered_wall_delta_ms": (observed_wall_ms - budget.wall_time.registered),
        "registered_gpu_delta_ms": (measured_gpu_ms - budget.compute_gpu_ms.registered),
        "registered_billed_delta_ms": (
            fixed_instance_billed_gpu_ms
            - budget.fixed_instance_billed_gpu_ms.registered
        ),
        "gpu_measurement_semantics": ("exclusive_reserved_gang_wall_ms_x_gpu_count"),
        "fixed_instance_billing_semantics": "whole_inventory_wall_clock_v1",
    }


def _hardware_envelope() -> HardwareEnvelope:
    return HardwareEnvelope(
        gpu_clock_mhz_min=1_500.0,
        gpu_clock_mhz_max=2_100.0,
        memory_clock_mhz_min=1_000.0,
        memory_clock_mhz_max=1_500.0,
        temperature_c_max=80.0,
        power_watts_min=100.0,
        power_watts_max=600.0,
        power_state="P0",
    )


def _passing_doctor(
    registry: ExperimentRegistry,
    *,
    inventory_authority: GpuInventory | None = None,
) -> dict:
    del registry
    inventory_authority = inventory_authority or _gpu_inventory()
    devices = [
        {
            "uuid": device.uuid,
            "name": device.model,
            "memory_total_mib": device.memory_bytes // (1024 * 1024),
            "driver_version": "580.65.06",
            "compute_capability": ".".join(
                str(component) for component in device.compute_capability
            ),
            "pci_bus_id": device.pci_bus_id,
        }
        for device in inventory_authority.devices
    ]
    gpu_rows = [f"GPU{index}" for index in range(len(devices))]
    topology = {
        "gpu_rows": gpu_rows,
        "pairs": [
            {
                "left": left,
                "right": right,
                "link": "NV18",
                "reciprocal_link": "NV18",
            }
            for left_index, left in enumerate(gpu_rows)
            for right in gpu_rows[left_index + 1 :]
        ],
        "parse_error": None,
    }
    checks = {name: {"status": "PASS"} for name in _INDUSTRIAL_DOCTOR_CHECKS}
    checks["gpu_identity"]["observed"] = devices
    checks["gpu_topology"]["observed"] = topology
    runtime_manifest_sha256 = "a" * 64
    inventory = "two exact registry GPU inventory rows"
    return {
        "schema_version": 1,
        "status": "PASS",
        "readiness": {
            "status": "PASS",
            "pass_count": len(checks),
            "fail_count": 0,
            "unknown_count": 0,
        },
        "checks": checks,
        "runtime_manifest": {
            "valid": True,
            "sha256": runtime_manifest_sha256,
            "sidecar_sha256": runtime_manifest_sha256,
            "error": None,
        },
        "roots": {
            "project": "/runtime/lightcone-spec",
            "patched_sglang": "/runtime/sglang",
            "distinct": True,
        },
        "source_tree": {
            "path": "/runtime/sglang",
            "is_git_checkout": True,
            "root_matches_toplevel": True,
            "head": "b" * 40,
            "tree": PINNED_SGLANG_TREE,
            "dirty": False,
            "pinned_ancestor": True,
            "patch_commits": PINNED_SGLANG_PATCH_COUNT,
        },
        "gpu": {
            "inventory": inventory,
            "parsed_inventory": {"devices": devices, "parse_error": None},
            "parsed_topology": topology,
            "visible_gpu_count": len(devices),
            "gpu_pool_visible": True,
            "two_gpu_visible": len(devices) == 2,
        },
        "commands": {"nvidia_smi": inventory},
        "compatibility": {
            "status": "PASS",
            "manifest_sha256": runtime_manifest_sha256,
            "sglang_commit": PINNED_SGLANG_COMMIT,
            "sglang_tree": PINNED_SGLANG_TREE,
            "patch_count": PINNED_SGLANG_PATCH_COUNT,
            "single_node_only": True,
            "multi_node_supported": False,
        },
    }


def _synthetic_attestation(
    doctor: BoundArtifact,
    artifact,
) -> dict:
    return {
        "schema_version": 1,
        "kind": "industrial_gpu_attestation",
        "status": "PASS",
        "doctor_report_sha256": doctor.sha256,
        "registry_sha256": artifact.registry_sha256,
        "experiment": artifact.experiment,
        "runtime_sha256": artifact.runtime_sha256,
        "split_sha256": artifact.split_sha256,
        "inventory_sha256": artifact.inventory_sha256,
        "inventory_source_receipt_sha256": (artifact.inventory_source_receipt_sha256),
        "fixed_instance_gpu_count": artifact.fixed_instance_gpu_count,
        "inventory_host_id": artifact.inventory_host_id,
        "confirmation_family_sha256": artifact.confirmation_family_sha256,
        "pilot_activation_sha256": artifact.pilot_activation_sha256,
        "final_activation_sha256": artifact.final_activation_sha256,
        "confirmation_plan_sha256": artifact.confirmation_plan_sha256,
        "evidence_dependence_map_sha256": (artifact.evidence_dependence_map_sha256),
        "patched_sglang_tree": artifact.patched_sglang_tree,
        "model_lock_sha256": artifact.model_lock_sha256,
        "hardware_envelope_sha256": artifact.hardware_envelope_sha256,
        "pilot_evidence_sha256": artifact.pilot_evidence_sha256,
        "completed_pilot_cells_sha256": artifact.completed_pilot_cells_sha256,
        "gpu_uuids": sorted(
            {
                gpu_uuid
                for binding in artifact.run_bindings
                for gpu_uuid in binding.gpu_uuids
            }
        ),
        "terminal_receipt_sha256s": list(artifact.terminal_receipt_sha256s),
        "qualification_lock_sha256s": list(artifact.qualification_lock_sha256s),
        "hardware_receipt_sha256s": list(artifact.hardware_receipt_sha256s),
        "budget_observation_sha256s": list(artifact.budget_observation_sha256s),
        "run_bindings": [asdict(binding) for binding in artifact.run_bindings],
    }


def _round(run_id: str, request_id: str) -> RoundRecord:
    return RoundRecord(
        run_id=run_id,
        request_id=request_id,
        round_index=0,
        generated_tokens_before=0,
        prefix_len_before=1,
        verify_len=8,
        accepted_drafts=7,
        committed_tokens=8,
        target_calls=1,
        proposal_source_version=0,
        kv_source_versions="[]",
    )


def _update(run_id: str, request_id: str) -> UpdateRecord:
    return UpdateRecord(
        run_id=run_id,
        cohort_sha256=content_sha256({"cohort": request_id}),
        parameter_layout_sha256=content_sha256({"layout": "selected"}),
        update_index=0,
        request_ids=json.dumps([request_id]),
        prefix_len_before="[1]",
        prefix_len_min=1,
        prefix_len_max=1,
        prefix_len_mean=1.0,
        source_round=0,
        source_version=0,
        optimizer_step=1,
        published_version=1,
        candidate_status="published",
        loss=0.1,
        gradient_norm=0.1,
        reconstruction_ok=True,
        reconstruction_max_abs=0.0,
        reconstruction_relative_rms=0.0,
        reconstruction_top1_match=1.0,
        reconstruction_mean_kl=0.0,
        supervision_nonempty=True,
        trainable_parameters=8,
        training_cuda_ms=1.0,
        optimizer_cuda_ms=1.0,
        merge_cuda_ms=1.0,
        publish_cuda_ms=1.0,
        barrier_cuda_ms=None,
        exposed_update_ms=1.0,
        overlap_ratio=0.5,
        online_hint_error=None,
        online_ensemble_entropy=None,
        online_effective_experts=None,
        online_expert_probabilities=None,
        online_cumulative_losses=None,
        online_expert_gradient_norms=None,
    )


def _performance(run_id: str, cell: ExperimentCell) -> PerformanceRecord:
    adapted = cell.identity.method in {"tts", "l0"}
    return PerformanceRecord(
        run_id=run_id,
        prompt_id="prompt",
        method=cell.identity.method,
        repetition_block=int(cell.identity.block),
        region="score",
        concurrency=1,
        generated_bucket_start=0,
        generated_bucket_end=100,
        at_risk_requests=1,
        output_tokens=100,
        elapsed_s=1.0,
        decode_goodput_tps=100.0,
        itl_p50_ms=1.0,
        itl_p95_ms=1.0,
        itl_p99_ms=1.0,
        survival_weighted_accepted_prefix=None,
        accepted_drafts_per_verify=None,
        committed_tokens_per_verify=None,
        verified_drafts_per_verify=None,
        verification_waste=None,
        target_calls_per_output_token=1.0,
        batch_fill=1.0,
        queue_occupancy=0.0,
        gpu_busy=0.9,
        sm_utilization=0.8,
        dram_utilization=0.5,
        target_estimated_mfu=None,
        peak_hbm_bytes=1_000,
        kv_bytes=500,
        optimizer_bytes=100 if adapted else 0,
        adaptation_memory_ledger=None,
        trainable_parameters=8 if adapted else 0,
        training_cuda_ms=1.0 if adapted else None,
        optimizer_cuda_ms=1.0 if adapted else None,
        merge_cuda_ms=1.0 if adapted else None,
        publish_cuda_ms=1.0 if adapted else None,
        barrier_cuda_ms=None,
        exposed_update_ms=1.0 if adapted else None,
        main_side_overlap_ratio=0.5 if adapted else None,
        graph_replay_hit_rate=1.0,
        updates_launched=1 if adapted else 0,
        updates_published=1 if adapted else 0,
        exactness_violations=0,
        version_mismatches=0,
        fallbacks=0,
        nonfinite_updates=0,
        oom_events=0,
        retractions=0,
        admission_rejections=0,
        timeouts=0,
        cancellations=0,
        offered_requests=1,
        admitted_requests=1,
        completed_requests=1,
        unfinished_requests=0,
        power_watts=300.0,
        gpu_clock_mhz=1_800.0,
        memory_clock_mhz=1_200.0,
        temperature_c=70.0,
        throttling_reasons="[]",
        communicator_failures=0,
        evidence_backpressure_events=0,
        evidence_dropped_rows=0,
    )


def _slice_cells(
    registry: ExperimentRegistry,
    *,
    final_block_count: int,
) -> dict[int, dict[str, ExperimentCell]]:
    selected: dict[int, dict[str, ExperimentCell]] = {}
    for block in PILOT_BLOCKS + FINAL_BLOCKS[:final_block_count]:
        rows = {
            cell.identity.method: cell
            for cell in registry.cells_for("E3b")
            if cell.runnable
            and cell.identity.block == block
            and cell.identity.context == 4096
            and cell.identity.regime == "long_input_short_output"
            and cell.identity.arrival == "closed_loop_c1"
            and cell.identity.variant
            in {
                "excluded_pilot:concurrency_one:matched",
                "final_candidate:concurrency_one:matched",
            }
        }
        assert set(rows) == set(CORE_METHODS)
        selected[block] = rows
    return selected


def _goodput(
    block: int,
    method: str,
    *,
    l0_pilot_multipliers: tuple[float, float, float, float],
) -> float:
    if method == "target_only":
        return 90.0
    if method == "static":
        return 100.0
    if method == "tts":
        return 101.0
    return 103.0 * l0_pilot_multipliers[block] if block in PILOT_BLOCKS else 104.0


def _build_evidence(
    tmp_path: Path,
    *,
    final_block_count: int = 12,
    l0_pilot_multipliers: tuple[float, float, float, float] = (
        0.99,
        1.01,
        1.00,
        1.02,
    ),
) -> tuple[
    ExperimentRegistry,
    FamilyActivationArtifact,
    FamilyActivationArtifact,
    ConfirmationFamilyPowerReductionArtifact,
    tuple[IndustrialBlockEvidence, ...],
    HardwareEnvelope,
]:
    registry = build_industrial_registry(
        gpu_uuids=("logical-rank-slot-0", "logical-rank-slot-1"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    runtime_sha256 = content_sha256({"runtime": "analysis-test"})
    split_sha256 = content_sha256({"split": "analysis-test"})
    envelope = _hardware_envelope()
    block_cells = _slice_cells(registry, final_block_count=final_block_count)
    family = derive_confirmation_family(
        registry,
        cell_id=block_cells[PILOT_BLOCKS[0]]["target_only"].cell_id,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        trace_sha256=content_sha256({"trace_pack": "analysis-test"}),
        sampling_sha256=content_sha256({"sampling": "greedy"}),
        hardware_envelope_sha256=content_sha256(envelope),
    )
    pilot_activation = materialize_confirmation_pilots(registry, family)
    block_evidence: list[IndustrialBlockEvidence] = []

    for block, methods in block_cells.items():
        request_id = f"request-{block}"
        request_ids_sha256 = content_sha256([request_id])
        identities = {
            "corpus_sha256": content_sha256({"corpus": block}),
            "arrival_trace_sha256": content_sha256({"trace": block}),
            "request_ids_sha256": request_ids_sha256,
            "sampling_profile_sha256": content_sha256({"sampling": "greedy"}),
            "model_lock_sha256": content_sha256({"model": "qwen3-8b"}),
        }
        cell_evidence: list[IndustrialCellEvidence] = []
        for method in CORE_METHODS:
            cell = methods[method]
            budget = _execution_budget(cell)
            physical_gpu_uuids = tuple(
                _PHYSICAL_GPU_UUIDS[registry.gpu_uuids.index(logical_slot)]
                for logical_slot in cell.resources.gpu_uuids
            )
            run_id = f"analysis-{block}-{method.replace('_', '-')}"
            evidence_root = Path(cell.resources.evidence_root)
            output_token_ids = tuple(range(100, 200))
            output_token_ids_json = json.dumps(
                output_token_ids,
                separators=(",", ":"),
            )
            output_token_ids_sha256 = hashlib.sha256(
                output_token_ids_json.encode("utf-8")
            ).hexdigest()
            run_nonce_sha256 = content_sha256({"nonce": cell.cell_id, "run": run_id})
            rank_config_sha256 = content_sha256(
                {"rank_config": cell.cell_id, "rank": 0}
            )
            writer = EvidenceWriter(
                evidence_root,
                run_id=run_id,
                rank=0,
                process_id=block * 10 + CORE_METHODS.index(method) + 1,
                registered_policy=DEFAULT_EVIDENCE_WRITER_POLICY,
            )
            native_artifact_binding = _persist_native_terminal_artifact(
                writer,
                method=method,
                run_nonce_sha256=run_nonce_sha256,
                execution_plan_sha256=runtime_sha256,
                rank_config_sha256=rank_config_sha256,
                request_id=request_id,
                output_token_ids=output_token_ids,
            )
            adapted = method in {"tts", "l0"}
            expected_rounds = 1 if adapted else 0
            expected_updates = 1 if adapted else 0
            workload_contract = (
                f"industrial_{method}"
                if method in {"target_only", "static"}
                else "industrial_adapted"
            )
            writer.write(
                RunRecord(
                    run_id=run_id,
                    manifest_sha256=registry.sha256,
                    config_sha256=cell.cell_id,
                    method=method,
                    model_pair=cell.identity.model,
                    repetition_block=block,
                    started_ns=1_000_000,
                    completed_ns=2_000_000_000,
                    status="complete",
                    industrial_cell_id=cell.cell_id,
                    experiment_budget_sha256=budget.sha256,
                    rank_config_sha256=rank_config_sha256,
                    runtime_sha256=runtime_sha256,
                    split_sha256=split_sha256,
                    **identities,
                    patched_sglang_tree=PINNED_SGLANG_TREE,
                    run_nonce_sha256=run_nonce_sha256,
                    topology_sha256=content_sha256(
                        {
                            "schema_version": 1,
                            "cell_id": cell.cell_id,
                            "topology": cell.identity.topology,
                            "gpu_uuids": list(physical_gpu_uuids),
                            "tensor_parallel_size": 1,
                            "data_parallel_size": 1,
                            "world_size": 1,
                        }
                    ),
                    tensor_parallel_size=1,
                    data_parallel_size=1,
                    world_size=1,
                    rank=0,
                    expected_request_rows=1,
                    expected_round_rows=expected_rounds,
                    expected_update_rows=expected_updates,
                    expected_performance_rows=1,
                    workload_contract=workload_contract,
                    native_terminal_artifact_path=str(native_artifact_binding["path"]),
                    native_terminal_artifact_size=int(native_artifact_binding["size"]),
                    native_terminal_raw_sha256=str(
                        native_artifact_binding["raw_sha256"]
                    ),
                    native_terminal_sha256=str(
                        native_artifact_binding["terminal_sha256"]
                    ),
                    trusted_attester_policy_sha256=str(
                        native_artifact_binding["trusted_attester_policy_sha256"]
                    ),
                )
            )
            arrival_ns = 1_000_000
            completion_ns = arrival_ns + round(
                100
                / _goodput(
                    block,
                    method,
                    l0_pilot_multipliers=l0_pilot_multipliers,
                )
                * 1_000_000_000
            )
            writer.write(
                RequestRecord(
                    run_id=run_id,
                    request_id=request_id,
                    prompt_id="prompt",
                    method=method,
                    repetition_block=block,
                    concurrency=1,
                    input_tokens=128,
                    output_tokens=100,
                    output_hash_format=OUTPUT_HASH_FORMAT,
                    output_sha256=output_token_ids_sha256,
                    ttft_ms=1.0,
                    finished=True,
                    stop_reason="length",
                    output_token_ids=output_token_ids_json,
                    output_token_ids_sha256=output_token_ids_sha256,
                    outcome_status="completed",
                    arrival_ns=arrival_ns,
                    queue_enter_ns=arrival_ns,
                    admitted_ns=arrival_ns,
                    first_token_ns=arrival_ns + 1_000_000,
                    completed_ns=completion_ns,
                    token_timestamps_ns=(
                        None
                        if block == FINAL_BLOCKS[0] and method == "tts"
                        else json.dumps(
                            [arrival_ns + 1_000_000 * index for index in range(1, 101)]
                        )
                    ),
                    inter_token_ms=(
                        None
                        if block == FINAL_BLOCKS[0] and method == "tts"
                        else json.dumps([1.0] * 99)
                    ),
                    token_timing_coverage=(
                        98 / 99 if block == FINAL_BLOCKS[0] and method == "tts" else 1.0
                    ),
                    coalesced_intervals=(
                        1 if block == FINAL_BLOCKS[0] and method == "tts" else 0
                    ),
                    admission_code="admitted",
                    retry_attempt=0,
                )
            )
            if adapted:
                writer.write(_round(run_id, request_id))
                writer.write(_update(run_id, request_id))
            writer.write(_performance(run_id, cell))
            _, prepared_receipt = writer.prepare_close()
            terminal_path = evidence_root / f"{run_id}.rank0.complete.json"
            prepared_receipt_sha256 = _file_sha256(prepared_receipt)
            budget_observation = _write_budget_observation(
                evidence_root
                / f"{run_id}.rank0.budget-observation"
                / "observation.json",
                _budget_observation_value(
                    budget,
                    terminal_receipt_sha256=prepared_receipt_sha256,
                ),
            )
            writer.publish_close(
                validate_post_binding=lambda reference=budget_observation: (
                    _validate_durable_test_binding(reference)
                )
            )
            _assert_registered_terminal_chain(
                terminal_path=terminal_path,
                prepared_receipt=prepared_receipt,
                budget_observation=budget_observation,
            )
            terminal = BoundArtifact(
                path=terminal_path,
                sha256=_file_sha256(terminal_path),
            )
            hardware = _write_json(
                evidence_root / f"{run_id}.hardware.json",
                {
                    "schema_version": 1,
                    "kind": "industrial_hardware_receipt",
                    "registry_sha256": registry.sha256,
                    "runtime_sha256": runtime_sha256,
                    "split_sha256": split_sha256,
                    "cell_id": cell.cell_id,
                    "block": block,
                    "topology_sha256": content_sha256(
                        {
                            "schema_version": 1,
                            "cell_id": cell.cell_id,
                            "topology": cell.identity.topology,
                            "gpu_uuids": list(physical_gpu_uuids),
                            "tensor_parallel_size": 1,
                            "data_parallel_size": 1,
                            "world_size": 1,
                        }
                    ),
                    "hardware_envelope_sha256": content_sha256(envelope),
                    "terminal_receipt_sha256s": [terminal.sha256],
                    "rank_contexts": [
                        {
                            "rank": 0,
                            "gpu_uuid": physical_gpu_uuids[0],
                            "power_state": "P0",
                            "background_processes": [],
                        }
                    ],
                },
            )
            cell_evidence.append(
                IndustrialCellEvidence(
                    cell_id=cell.cell_id,
                    terminal_receipts=(terminal,),
                    hardware_receipt=hardware,
                    budget_observation=budget_observation,
                    diagnostic_lineage_identity=True,
                )
            )
        qualification = _write_json(
            tmp_path / "qualification" / f"block-{block}.json",
            {
                "schema_version": 1,
                "kind": "industrial_request_qualification_lock",
                "registry_sha256": registry.sha256,
                "runtime_sha256": runtime_sha256,
                "split_sha256": split_sha256,
                "block": block,
                **identities,
                "rows": [
                    {
                        "request_id": request_id,
                        "prompt_bucket": "short",
                        "eligible": True,
                    }
                ],
            },
        )
        block_evidence.append(
            IndustrialBlockEvidence(
                block=block,
                cells=tuple(cell_evidence),
                qualification_lock=qualification,
            )
        )
    evidence = tuple(block_evidence)
    plan = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot_activation,
        blocks=tuple(block for block in evidence if block.block in PILOT_BLOCKS),
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        confirmation_data_visible=False,
    )
    assert plan.selected_final_blocks == (final_block_count or None)
    final_activation = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=plan,
        pilot_activation=pilot_activation,
    )
    return (
        registry,
        pilot_activation,
        final_activation,
        plan,
        evidence,
        envelope,
    )


def _build_e2_stage_evidence(
    tmp_path: Path,
) -> tuple[
    ExperimentRegistry,
    ExperimentReceipt,
    E1ParetoArtifact,
    tuple[IndustrialCellEvidence, ...],
    HardwareEnvelope,
]:
    registry = build_industrial_registry(
        gpu_uuids=("logical-rank-slot-0", "logical-rank-slot-1"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    seed_cell = next(
        cell
        for cell in registry.cells_for("E2")
        if cell.runnable
        and cell.identity.method == "tts"
        and "halving_stage=0:" in cell.identity.variant
    )
    runtime_sha256 = content_sha256({"runtime": "e2-raw-test"})
    split_sha256 = content_sha256({"split": "e2-raw-test"})
    pareto = E1ParetoArtifact(
        schema_version=1,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        e1_activation_sha256=content_sha256({"e1": "activation"}),
        reducer_evidence_sha256=content_sha256({"e1": "raw-evidence"}),
        common_load_sha256=content_sha256({"e1": "common-load"}),
        surviving_geometries=(E1GeometryIdentity.from_cell(seed_cell),),
        selection_state="sealed_before_e2_unblinding",
    )
    receipt = ExperimentReceipt(
        experiment="E1",
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        completed_cells_sha256=content_sha256({"E1": "completed"}),
        dependency_receipts=(LockedOutput("E3a", content_sha256({"E3a": "receipt"})),),
        outputs=(
            LockedOutput("common_downstream_load", pareto.common_load_sha256),
            LockedOutput("dflash_pareto_set", pareto.sha256),
        ),
    )
    activation = reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=0,
    )
    selected = {
        cell.cell_id: cell
        for cell in registry.cells_for("E2")
        if cell.cell_id in set(activation.plan.activated_cell_ids)
    }
    envelope = _hardware_envelope()
    request_id = "e2-raw-request"
    output_token_ids = tuple(range(100, 200))
    output_token_ids_json = json.dumps(output_token_ids, separators=(",", ":"))
    output_token_ids_sha256 = hashlib.sha256(
        output_token_ids_json.encode("utf-8")
    ).hexdigest()
    identities = {
        "corpus_sha256": content_sha256({"corpus": "e2-common"}),
        "arrival_trace_sha256": content_sha256({"arrival": "e2-common"}),
        "request_ids_sha256": content_sha256([request_id]),
        "sampling_profile_sha256": content_sha256({"sampling": "greedy"}),
        "model_lock_sha256": content_sha256({"model": "qwen3-8b"}),
    }
    result: list[IndustrialCellEvidence] = []
    for index, cell in enumerate(selected.values()):
        method = cell.identity.method
        adapted = method in {"tts", "l0"}
        budget = _execution_budget(cell)
        physical_gpu_uuids = tuple(
            _PHYSICAL_GPU_UUIDS[registry.gpu_uuids.index(logical_slot)]
            for logical_slot in cell.resources.gpu_uuids
        )
        topology_sha256 = content_sha256(
            {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "topology": cell.identity.topology,
                "gpu_uuids": list(physical_gpu_uuids),
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "world_size": 1,
            }
        )
        run_id = f"e2-raw-{index}-{method.replace('_', '-')}"
        evidence_root = Path(cell.resources.evidence_root)
        run_nonce_sha256 = content_sha256({"nonce": cell.cell_id, "run": run_id})
        rank_config_sha256 = content_sha256({"rank_config": cell.cell_id, "rank": 0})
        writer = EvidenceWriter(
            evidence_root,
            run_id=run_id,
            rank=0,
            process_id=10_000 + index,
            registered_policy=DEFAULT_EVIDENCE_WRITER_POLICY,
        )
        native_artifact_binding = _persist_native_terminal_artifact(
            writer,
            method=method,
            run_nonce_sha256=run_nonce_sha256,
            execution_plan_sha256=runtime_sha256,
            rank_config_sha256=rank_config_sha256,
            request_id=request_id,
            output_token_ids=output_token_ids,
        )
        writer.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=registry.sha256,
                config_sha256=cell.cell_id,
                method=method,
                model_pair=cell.identity.model,
                repetition_block=cell.identity.block,
                started_ns=1_000_000,
                completed_ns=2_000_000_000,
                status="complete",
                industrial_cell_id=cell.cell_id,
                experiment_budget_sha256=budget.sha256,
                rank_config_sha256=rank_config_sha256,
                runtime_sha256=runtime_sha256,
                split_sha256=split_sha256,
                **identities,
                patched_sglang_tree=PINNED_SGLANG_TREE,
                run_nonce_sha256=run_nonce_sha256,
                topology_sha256=topology_sha256,
                tensor_parallel_size=1,
                data_parallel_size=1,
                world_size=1,
                rank=0,
                expected_request_rows=1,
                expected_round_rows=1 if adapted else 0,
                expected_update_rows=1 if adapted else 0,
                expected_performance_rows=1,
                workload_contract=(
                    "industrial_adapted" if adapted else f"industrial_{method}"
                ),
                native_terminal_artifact_path=str(native_artifact_binding["path"]),
                native_terminal_artifact_size=int(native_artifact_binding["size"]),
                native_terminal_raw_sha256=str(native_artifact_binding["raw_sha256"]),
                native_terminal_sha256=str(native_artifact_binding["terminal_sha256"]),
                trusted_attester_policy_sha256=str(
                    native_artifact_binding["trusted_attester_policy_sha256"]
                ),
            )
        )
        arrival_ns = 1_000_000
        goodput = 90.0 if method == "target_only" else 100.0
        if adapted:
            goodput = 102.0
        completion_ns = arrival_ns + round(100 / goodput * 1_000_000_000)
        writer.write(
            RequestRecord(
                run_id=run_id,
                request_id=request_id,
                prompt_id="prompt",
                method=method,
                repetition_block=cell.identity.block,
                concurrency=1,
                input_tokens=128,
                output_tokens=100,
                output_hash_format=OUTPUT_HASH_FORMAT,
                output_sha256=output_token_ids_sha256,
                ttft_ms=1.0,
                finished=True,
                stop_reason="length",
                output_token_ids=output_token_ids_json,
                output_token_ids_sha256=output_token_ids_sha256,
                outcome_status="completed",
                arrival_ns=arrival_ns,
                queue_enter_ns=arrival_ns,
                admitted_ns=arrival_ns,
                first_token_ns=arrival_ns + 1_000_000,
                completed_ns=completion_ns,
                token_timestamps_ns=json.dumps(
                    [arrival_ns + 1_000_000 * item for item in range(1, 101)]
                ),
                inter_token_ms=json.dumps([1.0] * 99),
                token_timing_coverage=1.0,
                coalesced_intervals=0,
                admission_code="admitted",
                retry_attempt=0,
            )
        )
        if adapted:
            writer.write(_round(run_id, request_id))
            writer.write(_update(run_id, request_id))
        writer.write(_performance(run_id, cell))
        _, prepared_receipt = writer.prepare_close()
        terminal_path = evidence_root / f"{run_id}.rank0.complete.json"
        prepared_receipt_sha256 = _file_sha256(prepared_receipt)
        budget_observation = _write_budget_observation(
            evidence_root / f"{run_id}.rank0.budget-observation" / "observation.json",
            _budget_observation_value(
                budget,
                terminal_receipt_sha256=prepared_receipt_sha256,
            ),
        )
        writer.publish_close(
            validate_post_binding=lambda reference=budget_observation: (
                _validate_durable_test_binding(reference)
            )
        )
        _assert_registered_terminal_chain(
            terminal_path=terminal_path,
            prepared_receipt=prepared_receipt,
            budget_observation=budget_observation,
        )
        terminal = BoundArtifact(
            path=terminal_path,
            sha256=_file_sha256(terminal_path),
        )
        hardware = _write_json(
            evidence_root / f"{run_id}.hardware.json",
            {
                "schema_version": 1,
                "kind": "industrial_hardware_receipt",
                "registry_sha256": registry.sha256,
                "runtime_sha256": runtime_sha256,
                "split_sha256": split_sha256,
                "cell_id": cell.cell_id,
                "block": cell.identity.block,
                "topology_sha256": topology_sha256,
                "hardware_envelope_sha256": content_sha256(envelope),
                "terminal_receipt_sha256s": [terminal.sha256],
                "rank_contexts": [
                    {
                        "rank": 0,
                        "gpu_uuid": physical_gpu_uuids[0],
                        "power_state": "P0",
                        "background_processes": [],
                    }
                ],
            },
        )
        result.append(
            IndustrialCellEvidence(
                cell_id=cell.cell_id,
                terminal_receipts=(terminal,),
                hardware_receipt=hardware,
                budget_observation=budget_observation,
                diagnostic_lineage_identity=True,
            )
        )
    return registry, receipt, pareto, tuple(result), envelope


def _bound_reference(reference: BoundArtifact) -> dict[str, str]:
    return {"path": str(reference.path), "sha256": reference.sha256}


def _analysis_manifest(
    tmp_path: Path,
    *,
    registry: ExperimentRegistry,
    pilot_activation: FamilyActivationArtifact,
    final_activation: FamilyActivationArtifact,
    reduction: ConfirmationFamilyPowerReductionArtifact,
    evidence: tuple[IndustrialBlockEvidence, ...],
    envelope: HardwareEnvelope,
    name: str,
    gpu_attestation: BoundArtifact | None = None,
    doctor_report: BoundArtifact | None = None,
) -> Path:
    cache_root = Path(registry.cells[0].resources.cache_root).parents[1]
    evidence_root = Path(registry.cells[0].resources.evidence_root).parents[1]
    registry_path = tmp_path / f"{name}-registry.json"
    assert (
        main(
            [
                "build-industrial-registry",
                "--logical-gpu-slot",
                *registry.gpu_uuids,
                "--cache-root",
                str(cache_root),
                "--evidence-root",
                str(evidence_root),
                "--output",
                str(registry_path),
            ]
        )
        == 0
    )
    registry_value = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry_value["registry_sha256"] == registry.sha256
    pilot_activation_value = family_activation_artifact_to_dict(pilot_activation)
    pilot_activation_path = tmp_path / f"{name}-pilot-activation.json"
    _write_bound_json(pilot_activation_path, pilot_activation_value)
    final_activation_value = family_activation_artifact_to_dict(final_activation)
    final_activation_path = tmp_path / f"{name}-final-activation.json"
    _write_bound_json(final_activation_path, final_activation_value)
    inventory = _gpu_inventory()
    inventory_value = inventory.to_dict()
    inventory_path = tmp_path / f"{name}-gpu-inventory.json"
    _write_bound_json(inventory_path, inventory_value)
    inventory_binding = {
        "path": str(inventory_path),
        "sha256": content_sha256(inventory_value),
    }

    def manifest_block(block: IndustrialBlockEvidence) -> dict[str, object]:
        return {
            "block": block.block,
            "qualification_lock": _bound_reference(block.qualification_lock),
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "terminal_receipts": [
                        _bound_reference(receipt) for receipt in cell.terminal_receipts
                    ],
                    "hardware_receipt": _bound_reference(cell.hardware_receipt),
                    "budget_observation": _bound_reference(cell.budget_observation),
                }
                for cell in block.cells
            ],
        }

    power_manifest = {
        "schema_version": 2,
        "kind": "industrial_family_power_manifest",
        "registry_artifact": {
            "path": str(registry_path),
            "sha256": content_sha256(registry_value),
        },
        "pilot_activation": {
            "path": str(pilot_activation_path),
            "sha256": content_sha256(pilot_activation_value),
        },
        "gpu_inventory": inventory_binding,
        "hardware_envelope": asdict(envelope),
        "blocks": [
            manifest_block(block) for block in evidence if block.block in PILOT_BLOCKS
        ],
    }
    power_manifest_path = tmp_path / f"{name}-power-manifest.json"
    _write_bound_json(power_manifest_path, power_manifest)
    manifest = {
        "schema_version": 3,
        "kind": "industrial_analysis_manifest",
        "registry_artifact": {
            "path": str(registry_path),
            "sha256": content_sha256(registry_value),
        },
        "pilot_activation": {
            "path": str(pilot_activation_path),
            "sha256": content_sha256(pilot_activation_value),
        },
        "final_activation": {
            "path": str(final_activation_path),
            "sha256": content_sha256(final_activation_value),
        },
        "confirmation_power_manifest": {
            "path": str(power_manifest_path),
            "sha256": content_sha256(power_manifest),
        },
        "gpu_inventory": inventory_binding,
        "evidence_alias_manifests": [],
        "evidence_dependence_map": None,
        "gpu_attestation": (
            None if gpu_attestation is None else _bound_reference(gpu_attestation)
        ),
        "doctor_report": (
            None if doctor_report is None else _bound_reference(doctor_report)
        ),
        "hardware_envelope": asdict(envelope),
        "bootstrap": {"repetitions": 300, "seed": 17},
        "blocks": [manifest_block(block) for block in evidence],
    }
    manifest_path = tmp_path / f"{name}-manifest.json"
    _write_bound_json(manifest_path, manifest)
    return manifest_path


@pytest.fixture(scope="module")
def evidence_bundle(tmp_path_factory: pytest.TempPathFactory):
    return _build_evidence(tmp_path_factory.mktemp("industrial-analysis"))


def _singleton_dependence_map(
    *,
    pilot_activation: FamilyActivationArtifact,
    final_activation: FamilyActivationArtifact,
) -> EvidenceDependenceMap:
    active = pilot_activation.activated_cell_ids + final_activation.activated_cell_ids
    units = tuple(
        AnalysisDependenceUnit(
            unit_sha256=content_sha256({"direct_observation_cell_id": cell_id}),
            source_cell_id=cell_id,
            member_cell_ids=(cell_id,),
        )
        for cell_id in active
    )
    return EvidenceDependenceMap(
        schema_version=1,
        units=tuple(sorted(units, key=lambda row: row.unit_sha256)),
    )


def _replace_with_unverified_alias(
    dependence_map: EvidenceDependenceMap,
    *,
    source_cell_id: str,
    target_cell_id: str,
) -> EvidenceDependenceMap:
    alias = AnalysisDependenceUnit(
        unit_sha256=content_sha256(
            {
                "unverified_alias_source": source_cell_id,
                "unverified_alias_target": target_cell_id,
            }
        ),
        source_cell_id=source_cell_id,
        member_cell_ids=tuple(sorted((source_cell_id, target_cell_id))),
    )
    units = (
        alias,
        *(
            unit
            for unit in dependence_map.units
            if source_cell_id not in unit.member_cell_ids
            and target_cell_id not in unit.member_cell_ids
        ),
    )
    return EvidenceDependenceMap(
        schema_version=1,
        units=tuple(sorted(units, key=lambda row: row.unit_sha256)),
    )


def test_budget_observation_requires_a_positive_registered_factor_anchor(
    tmp_path: Path,
) -> None:
    budget = _zero_budget("a" * 64)
    terminal_sha256 = "b" * 64
    observed_rows = [[name, 0] for name in _BUDGET_OBSERVATION_COMPONENTS]
    observation_content = {
        "schema_version": 1,
        "budget": asdict(budget),
        "observed_component_ms": observed_rows,
        "measured_gpu_ms": 0,
        "fixed_instance_billed_gpu_ms": 0,
        "terminal_evidence_sha256": terminal_sha256,
    }
    value = {
        "schema_version": 1,
        "artifact_kind": "industrial_budget_observation_receipt_v1",
        "experiment_budget_sha256": budget.sha256,
        "budget_observation_sha256": content_sha256(observation_content),
        "budget": asdict(budget),
        "observed_component_ms": observed_rows,
        "measured_gpu_ms": 0,
        "fixed_instance_billed_gpu_ms": 0,
        "terminal_evidence_sha256": terminal_sha256,
        "observed_wall_ms": 0,
        "registered_wall_delta_ms": 0,
        "registered_gpu_delta_ms": 0,
        "registered_billed_delta_ms": 0,
        "gpu_measurement_semantics": ("exclusive_reserved_gang_wall_ms_x_gpu_count"),
        "fixed_instance_billing_semantics": "whole_inventory_wall_clock_v1",
    }
    reference = _write_json(tmp_path / "observation.json", value)
    with pytest.raises(ValueError, match="positive scenario anchors"):
        _load_budget_observation(
            reference,
            experiment_budget_sha256=budget.sha256,
            terminal_receipt_sha256=terminal_sha256,
            fixed_instance_gpu_count=2,
        )


def test_tp1_on_two_gpu_instance_underbilling_fails_exact_factor(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, _, _, _, evidence, _ = evidence_bundle
    cell = evidence[0].cells[0]
    value = json.loads(cell.budget_observation.path.read_text(encoding="utf-8"))
    assert value["budget"]["gpu_count"] == 1
    observed_wall_ms = value["observed_wall_ms"]
    assert value["fixed_instance_billed_gpu_ms"] == observed_wall_ms * 2
    value["fixed_instance_billed_gpu_ms"] = observed_wall_ms
    registered_billed_ms = value["budget"]["fixed_instance_billed_gpu_ms"]["registered"]
    value["registered_billed_delta_ms"] = observed_wall_ms - registered_billed_ms
    observation_content = {
        "schema_version": value["schema_version"],
        "budget": value["budget"],
        "observed_component_ms": value["observed_component_ms"],
        "measured_gpu_ms": value["measured_gpu_ms"],
        "fixed_instance_billed_gpu_ms": value["fixed_instance_billed_gpu_ms"],
        "terminal_evidence_sha256": value["terminal_evidence_sha256"],
    }
    value["budget_observation_sha256"] = content_sha256(observation_content)
    tampered = _write_json(tmp_path / "tp1-two-gpu-underbill.json", value)
    with pytest.raises(ValueError, match="derived accounting"):
        _load_budget_observation(
            tampered,
            experiment_budget_sha256=value["experiment_budget_sha256"],
            terminal_receipt_sha256=value["terminal_evidence_sha256"],
            fixed_instance_gpu_count=2,
        )

    registry_cell = next(row for row in registry.cells if row.cell_id == cell.cell_id)
    registered = _execution_budget(registry_cell)
    coordinated_budget = replace(
        registered,
        fixed_instance_billed_gpu_ms=registered.wall_time.scale(1),
    )
    coordinated_value = _budget_observation_value(
        coordinated_budget,
        terminal_receipt_sha256=value["terminal_evidence_sha256"],
        fixed_instance_gpu_count=1,
    )
    coordinated = _write_json(
        tmp_path / "tp1-coordinated-one-gpu-budget.json",
        coordinated_value,
    )
    with pytest.raises(ValueError, match="bound inventory count"):
        _load_budget_observation(
            coordinated,
            experiment_budget_sha256=coordinated_budget.sha256,
            terminal_receipt_sha256=value["terminal_evidence_sha256"],
            fixed_instance_gpu_count=2,
        )


def test_e2_stage_reducer_rebuilds_metrics_and_rejects_bare_prior(
    tmp_path: Path,
) -> None:
    registry, receipt, pareto, cells, envelope = _build_e2_stage_evidence(tmp_path)
    reduction = reduce_e2_stage_from_raw(
        registry=registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=0,
        cells=cells,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        confirmation_data_visible=False,
    )
    assert reduction.survivor_receipt.status == "SURVIVORS"
    assert reduction.activation == reduce_e2_activation(
        registry,
        e1_receipt=receipt,
        pareto=pareto,
        stage_index=0,
    )
    assert {
        binding.cell_id for binding in reduction.stage_evidence.run_bindings
    } == set(reduction.activation.plan.activated_cell_ids)
    assert all(
        row.min_tts_l0_static_goodput_ratio == pytest.approx(1.02)
        and row.confidence_lower_goodput_ratio == pytest.approx(1.02)
        and row.hbm_bytes == 1_000
        and row.p99_itl_us == 1_000
        and row.exposed_update_us == 1_000
        and row.minimum_published_updates == 1
        for row in reduction.stage_evidence.evaluations
    )
    with pytest.raises(TypeError, match="prior raw reduction"):
        reduce_e2_activation(
            registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=1,
            prior_reduction=reduction.survivor_receipt,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly cover"):
        reduce_e2_stage_from_raw(
            registry=registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=0,
            cells=cells[:-1],
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            confirmation_data_visible=False,
        )


def test_family_power_reducer_uses_only_excluded_terminal_evidence(
    evidence_bundle,
) -> None:
    registry, pilots, _, plan, evidence, envelope = evidence_bundle
    pilot_blocks = tuple(block for block in evidence if block.block in PILOT_BLOCKS)
    assert (
        reduce_confirmation_family_power(
            registry=registry,
            pilot_activation=pilots,
            blocks=pilot_blocks,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            confirmation_data_visible=False,
        )
        == plan
    )
    with pytest.raises(ValueError, match="visible confirmation data"):
        reduce_confirmation_family_power(
            registry=registry,
            pilot_activation=pilots,
            blocks=pilot_blocks,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            confirmation_data_visible=True,
        )
    with pytest.raises(ValueError, match="exactly four excluded pilot blocks"):
        reduce_confirmation_family_power(
            registry=registry,
            pilot_activation=pilots,
            blocks=(*pilot_blocks, evidence[len(PILOT_BLOCKS)]),
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            confirmation_data_visible=False,
        )

    original = pilot_blocks[0].cells[0]
    value = json.loads(original.hardware_receipt.path.read_text(encoding="utf-8"))
    value["rank_contexts"][0]["power_state"] = "P1"
    invalid = replace(
        original,
        hardware_receipt=_write_json(
            original.hardware_receipt.path.parent / "invalid-pilot-hardware.json",
            value,
        ),
    )
    invalid_block = replace(
        pilot_blocks[0],
        cells=(invalid, *pilot_blocks[0].cells[1:]),
    )
    with pytest.raises(ValueError, match="invalid hardware or safety evidence"):
        reduce_confirmation_family_power(
            registry=registry,
            pilot_activation=pilots,
            blocks=(invalid_block, *pilot_blocks[1:]),
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            confirmation_data_visible=False,
        )


def test_industrial_attestation_contract_binds_doctor_gpu_and_run_chain(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, *_ = evidence_bundle
    doctor_value = _passing_doctor(registry)
    doctor = _write_json(tmp_path / "doctor.json", doctor_value)
    _validate_industrial_doctor(
        doctor,
        inventory_authority=_gpu_inventory(),
    )

    mismatched_manifest = json.loads(json.dumps(doctor_value))
    mismatched_manifest["compatibility"]["manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="runtime-manifest digests"):
        _validate_industrial_doctor(
            _write_json(
                tmp_path / "doctor-manifest-mismatch.json", mismatched_manifest
            ),
            inventory_authority=_gpu_inventory(),
        )

    mismatched_gpu = json.loads(json.dumps(doctor_value))
    mismatched_gpu["gpu"]["parsed_inventory"]["devices"][0]["uuid"] = "GPU-other"
    mismatched_gpu["checks"]["gpu_identity"]["observed"][0]["uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="complete GPU inventory"):
        _validate_industrial_doctor(
            _write_json(tmp_path / "doctor-gpu-mismatch.json", mismatched_gpu),
            inventory_authority=_gpu_inventory(),
        )

    expected_chain = {
        "registry_sha256": registry.sha256,
        "terminal_receipt_sha256s": ["d" * 64],
        "hardware_receipt_sha256s": ["e" * 64],
        "run_bindings": [
            {
                "run_id": "run-1",
                "run_nonce_sha256": "f" * 64,
                "config_sha256": "1" * 64,
                "rank_config_sha256s": ["2" * 64],
                "topology_sha256": "3" * 64,
            }
        ],
    }
    attestation_value = {
        "schema_version": 1,
        "kind": "industrial_gpu_attestation",
        "status": "PASS",
        "doctor_report_sha256": doctor.sha256,
        **expected_chain,
    }
    attestation = _write_json(tmp_path / "attestation.json", attestation_value)
    _validate_industrial_gpu_attestation(
        attestation,
        doctor_report=doctor,
        expected_chain=expected_chain,
    )
    tampered = json.loads(json.dumps(attestation_value))
    tampered["terminal_receipt_sha256s"] = ["9" * 64]
    with pytest.raises(ValueError, match="exact doctor/run evidence chain"):
        _validate_industrial_gpu_attestation(
            _write_json(tmp_path / "attestation-tampered.json", tampered),
            doctor_report=doctor,
            expected_chain=expected_chain,
        )


def test_industrial_doctor_binds_complete_eight_gpu_inventory(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, *_ = evidence_bundle
    inventory = _gpu_inventory(gpu_count=8)
    doctor = _write_json(
        tmp_path / "doctor-eight-gpu.json",
        _passing_doctor(registry, inventory_authority=inventory),
    )
    _validate_industrial_doctor(doctor, inventory_authority=inventory)

    reduced = _gpu_inventory(gpu_count=4)
    with pytest.raises(ValueError, match="complete GPU inventory"):
        _validate_industrial_doctor(doctor, inventory_authority=reduced)


def test_static_terminal_evidence_has_no_round_or_update_trace_state(
    evidence_bundle,
) -> None:
    registry, _, _, plan, evidence, _ = evidence_bundle
    block = evidence[0]
    static_reference = next(
        reference
        for reference in block.cells
        if next(
            cell for cell in registry.cells if cell.cell_id == reference.cell_id
        ).identity.method
        == "static"
    )
    receipt_reference = static_reference.terminal_receipts[0]
    receipt = json.loads(receipt_reference.path.read_text(encoding="utf-8"))
    completed = load_completed_evidence(
        receipt_reference.path.parent,
        run_id=receipt["run_id"],
        rank=receipt["rank"],
    )
    assert completed is not None
    assert "round" not in completed
    assert "update" not in completed
    run = pq.read_table(completed["run"]).to_pylist()[0]
    assert run["expected_round_rows"] == 0
    assert run["expected_update_rows"] == 0

    claimed_trace = {**run, "expected_round_rows": 1}
    cell = next(
        cell for cell in registry.cells if cell.cell_id == static_reference.cell_id
    )
    with pytest.raises(ValueError, match="detail-table coverage"):
        _validate_run_row(
            claimed_trace,
            registry=registry,
            family=plan.family,
            cell=cell,
            rank=0,
        )
    with pytest.raises(ValueError, match="experiment_budget_sha256"):
        _validate_run_row(
            {**run, "experiment_budget_sha256": None},
            registry=registry,
            family=plan.family,
            cell=cell,
            rank=0,
        )
    with pytest.raises(ValueError, match="registry/runtime identity"):
        _validate_run_row(
            {**run, "workload_contract": "industrial_preflight_static"},
            registry=registry,
            family=plan.family,
            cell=cell,
            rank=0,
        )
    for field, value in (
        ("session_plan_sha256", "1" * 64),
        ("session_open_receipt_sha256", "2" * 64),
        ("reset_receipt_sha256", "3" * 64),
        ("session_epoch", 1),
    ):
        with pytest.raises(ValueError, match="pre-mutation release"):
            _validate_run_row(
                {**run, field: value},
                registry=registry,
                family=plan.family,
                cell=cell,
                rank=0,
            )
    with pytest.raises(ValueError, match="pre-mutation release"):
        _validate_run_row(
            {**run, "session_close_receipt_sha256": "4" * 64},
            registry=registry,
            family=plan.family,
            cell=cell,
            rank=0,
        )

    performance = pq.read_table(completed["performance"]).to_pylist()[0]
    _validate_allocation_free_performance(performance, method="static")
    for field, value in (
        ("optimizer_bytes", 1),
        ("trainable_parameters", 1),
        ("updates_launched", 1),
        ("updates_published", 1),
        ("training_cuda_ms", 0.0),
        ("exposed_update_ms", 0.0),
    ):
        with pytest.raises(ValueError, match="Target-only/Static performance"):
            _validate_allocation_free_performance(
                {**performance, field: value},
                method="static",
            )


def test_fully_matching_synthetic_attestation_remains_unmeasured(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    diagnostic = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
        bootstrap_seed=23,
    )
    doctor = _write_json(tmp_path / "synthetic-doctor.json", _passing_doctor(registry))
    attestation = _write_json(
        tmp_path / "synthetic-attestation.json",
        _synthetic_attestation(doctor, diagnostic.artifact),
    )
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        gpu_attestation=attestation,
        doctor_report=doctor,
        bootstrap_repetitions=100,
        bootstrap_seed=23,
    )
    assert reduction.artifact.status == "UNRESOLVED"
    assert reduction.artifact.gpu_evidence == "UNMEASURED"
    assert reduction.artifact.reasons == ("gpu_attestation:untrusted_attester",)
    assert reduction.artifact.gpu_attestation_sha256 == attestation.sha256
    assert reduction.artifact.doctor_report_sha256 == doctor.sha256


def test_analyze_industrial_cli_preserves_untrusted_attestation_gate(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    diagnostic = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
        bootstrap_seed=23,
    )
    doctor = _write_json(tmp_path / "synthetic-doctor.json", _passing_doctor(registry))
    attestation = _write_json(
        tmp_path / "synthetic-attestation.json",
        _synthetic_attestation(doctor, diagnostic.artifact),
    )
    manifest = _analysis_manifest(
        tmp_path,
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        reduction=plan,
        evidence=evidence,
        envelope=envelope,
        name="synthetic-attestation",
        gpu_attestation=attestation,
        doctor_report=doctor,
    )
    output = tmp_path / "synthetic-attestation-reducer.json"
    assert (
        main(
            [
                "analyze-industrial",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    cli_artifact = json.loads(output.read_text(encoding="utf-8"))
    assert cli_artifact["status"] == "UNRESOLVED"
    assert cli_artifact["gpu_evidence"] == "UNMEASURED"
    assert cli_artifact["reasons"] == ["gpu_attestation:untrusted_attester"]


def test_reducer_derives_only_unattested_diagnostics_from_cpu_terminal_rows(
    evidence_bundle,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        bootstrap_repetitions=300,
        bootstrap_seed=17,
    )
    repeated = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        bootstrap_repetitions=300,
        bootstrap_seed=17,
    )

    artifact = reduction.artifact
    assert artifact.status == "UNRESOLVED"
    assert artifact.gpu_evidence == "UNMEASURED"
    assert artifact.reasons == ("gpu_attestation:missing",)
    assert artifact.gpu_attestation_sha256 is None
    assert artifact.doctor_report_sha256 is None
    assert artifact.sha256 == repeated.artifact.sha256
    assert artifact.power_plan is not None
    assert artifact.power_plan.selected_final_blocks == 12
    assert tuple(row.name for row in artifact.primary_contrasts) == (
        "l0_vs_static",
        "l0_vs_tts",
    )
    assert all(len(row.block_ids) == 12 for row in artifact.primary_contrasts)
    slo_by_method = {row.method: row.slo for row in artifact.methods}
    assert not slo_by_method["tts"].passed
    assert all(
        slo_by_method[method].passed for method in ("target_only", "static", "l0")
    )
    assert all(
        row.aggregate_latency_p99.status == "UNRESOLVED" for row in artifact.methods
    )
    assert len(artifact.terminal_receipt_sha256s) == 16 * 4
    assert len(artifact.budget_observation_sha256s) == 16 * 4
    bound_physical = {
        gpu_uuid for binding in artifact.run_bindings for gpu_uuid in binding.gpu_uuids
    }
    assert bound_physical <= set(_PHYSICAL_GPU_UUIDS)
    assert bound_physical.isdisjoint(registry.gpu_uuids)
    assert artifact.to_dict()["kind"] == "industrial_schema_v3_reducer"

    hierarchical = reduction.hierarchical_block_request_bootstrap(
        "l0",
        "latency_ms",
        np.mean,
        repetitions=100,
        seed=9,
    )
    whole_time = reduction.whole_time_block_bootstrap(
        "l0",
        "latency_ms",
        np.mean,
        repetitions=100,
        seed=9,
    )
    assert hierarchical.independent_units == ("block", "request")
    assert whole_time.independent_units == ("time_block",)
    with pytest.raises(ValueError, match="refuses to impute"):
        reduction.hierarchical_block_request_bootstrap(
            "tts",
            "within_request_p99_itl_ms",
            np.mean,
            repetitions=100,
        )


def test_family_artifacts_gate_exact_pilots_plan_sha_and_activated_cells(
    evidence_bundle,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    dispositions = list(pilot_activation.dispositions)
    pilot_index = next(
        index
        for index, row in enumerate(dispositions)
        if row.status is DispositionStatus.ACTIVATED
    )
    dispositions[pilot_index] = replace(
        dispositions[pilot_index],
        status=DispositionStatus.DEFERRED,
        reason_code="forged_missing_pilot",
    )
    forged_pilots = replace(pilot_activation, dispositions=tuple(dispositions))
    with pytest.raises(ValueError, match="pilot activation is not reducer-generated"):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=forged_pilots,
            final_activation=final_activation,
            confirmation_reduction=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            bootstrap_repetitions=100,
        )

    stale_final = replace(final_activation, power_plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="final activation is not reducer-generated"):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilot_activation,
            final_activation=stale_final,
            confirmation_reduction=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            bootstrap_repetitions=100,
        )

    with pytest.raises(ValueError, match="raw pilot manifest"):
        replace(
            plan,
            plan=replace(plan.plan, pilot_evidence_sha256="c" * 64),
        )

    deferred_cell_id = next(
        row.cell_id
        for row in final_activation.dispositions
        if row.status is DispositionStatus.DEFERRED
    )
    unactivated_cell = replace(evidence[-1].cells[0], cell_id=deferred_cell_id)
    unactivated_block = replace(
        evidence[-1],
        cells=(unactivated_cell, *evidence[-1].cells[1:]),
    )
    with pytest.raises(
        ValueError,
        match="direct evidence plus raw aliases must exactly cover activation",
    ):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilot_activation,
            final_activation=final_activation,
            confirmation_reduction=plan,
            blocks=(*evidence[:-1], unactivated_block),
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            bootstrap_repetitions=100,
        )


def test_underpowered_family_uses_only_pilots_and_produces_no_final_analysis(
    tmp_path: Path,
) -> None:
    registry, pilots, final, plan, evidence, envelope = _build_evidence(
        tmp_path,
        final_block_count=0,
        l0_pilot_multipliers=(0.5, 1.5, 0.7, 1.3),
    )
    assert plan.status == "UNDERPOWERED"
    assert plan.selected_final_prefix == ()
    assert final.activated_cell_ids == ()

    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilots,
        final_activation=final,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )

    assert reduction.artifact.status == "UNRESOLVED"
    assert reduction.artifact.gpu_evidence == "UNMEASURED"
    assert reduction.artifact.reasons == (
        "confirmation_family:underpowered",
        "gpu_attestation:missing",
    )
    assert reduction.artifact.methods == ()
    assert reduction.artifact.primary_contrasts == ()
    assert reduction.artifact.holm_family == ()
    assert len(reduction.artifact.run_bindings) == len(PILOT_BLOCKS) * len(CORE_METHODS)


def test_dependence_map_rekeys_units_but_rejects_unverified_aliases(
    evidence_bundle,
) -> None:
    registry, pilots, final, plan, evidence, envelope = evidence_bundle
    dependence_map = _singleton_dependence_map(
        pilot_activation=pilots,
        final_activation=final,
    )
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilots,
        final_activation=final,
        confirmation_reduction=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        evidence_dependence_map=dependence_map,
        bootstrap_repetitions=100,
        bootstrap_seed=41,
    )

    assert reduction.artifact.evidence_dependence_map_sha256 == dependence_map.sha256
    methods = {row.method: row for row in reduction.artifact.methods}
    assert all(len(method.block_ids) == 12 for method in methods.values())
    assert len(reduction._request_metrics["l0"]) == 12
    assert all(
        row.independent_unit == "evidence_dependence_component"
        and len(row.block_ids) == 12
        for row in reduction.artifact.primary_contrasts
    )
    interval = reduction.hierarchical_block_request_bootstrap(
        "l0",
        "latency_ms",
        np.mean,
        repetitions=100,
        seed=5,
    )
    assert interval.independent_units == ("evidence_dependence_unit", "request")

    incomplete_map = EvidenceDependenceMap(
        schema_version=1,
        units=dependence_map.units[:-1],
    )
    with pytest.raises(ValueError, match="exactly one dependence unit"):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilots,
            final_activation=final,
            confirmation_reduction=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            evidence_dependence_map=incomplete_map,
            bootstrap_repetitions=100,
        )

    by_id = {cell.cell_id: cell for cell in registry.cells}
    final_l0 = tuple(
        cell_id
        for cell_id in final.activated_cell_ids
        if by_id[cell_id].identity.method == "l0"
    )[:2]
    unverified_final_alias = _replace_with_unverified_alias(
        dependence_map,
        source_cell_id=final_l0[0],
        target_cell_id=final_l0[1],
    )
    with pytest.raises(ValueError, match="evidence-recomputed alias receipts"):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilots,
            final_activation=final,
            confirmation_reduction=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            evidence_dependence_map=unverified_final_alias,
            bootstrap_repetitions=100,
        )

    pilot_l0 = tuple(
        cell_id
        for cell_id in pilots.activated_cell_ids
        if by_id[cell_id].identity.method == "l0"
    )[:2]
    pilot_alias_map = _replace_with_unverified_alias(
        dependence_map,
        source_cell_id=pilot_l0[0],
        target_cell_id=pilot_l0[1],
    )
    with pytest.raises(ValueError, match="evidence-recomputed alias receipts"):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilots,
            final_activation=final,
            confirmation_reduction=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            evidence_dependence_map=pilot_alias_map,
            bootstrap_repetitions=100,
        )


def test_reducer_fails_closed_on_missing_paired_method_or_rank(evidence_bundle) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilot_activation,
            final_activation=final_activation,
            confirmation_reduction=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            gpu_attestation=BoundArtifact(Path("missing.json"), "a" * 64),
            bootstrap_repetitions=100,
        )

    missing_method = replace(evidence[-1], cells=evidence[-1].cells[:-1])
    with pytest.raises(
        ValueError,
        match="direct evidence plus raw aliases must exactly cover activation",
    ):
        reduce_industrial_schema_v3(
            registry=registry,
            pilot_activation=pilot_activation,
            final_activation=final_activation,
            confirmation_reduction=plan,
            blocks=(*evidence[:-1], missing_method),
            hardware_envelope=envelope,
            inventory=_gpu_inventory(),
            bootstrap_repetitions=100,
        )

    with pytest.raises(ValueError, match="terminal rank receipts"):
        replace(evidence[-1].cells[0], terminal_receipts=())
    with pytest.raises(TypeError, match="bound budget observation"):
        replace(evidence[-1].cells[0], budget_observation=None)


def test_hardware_invalidation_suppresses_all_contrasts(
    evidence_bundle,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    original_cell = evidence[-1].cells[0]
    source = json.loads(original_cell.hardware_receipt.path.read_text(encoding="utf-8"))
    source["rank_contexts"][0]["power_state"] = "P1"
    invalid_hardware = _write_json(
        original_cell.hardware_receipt.path.parent / "invalid-hardware.json",
        source,
    )
    invalid_cell = replace(original_cell, hardware_receipt=invalid_hardware)
    invalid_block = replace(
        evidence[-1],
        cells=(invalid_cell, *evidence[-1].cells[1:]),
    )

    reduction = reduce_industrial_schema_v3(
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        confirmation_reduction=plan,
        blocks=(*evidence[:-1], invalid_block),
        hardware_envelope=envelope,
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    assert reduction.artifact.status == "UNRESOLVED"
    assert reduction.artifact.gpu_evidence == "INVALIDATED"
    assert reduction.artifact.primary_contrasts == ()
    assert reduction.artifact.holm_family == ()
    assert any(reason.startswith("hardware:") for reason in reduction.artifact.reasons)


def test_analyze_industrial_cli_uses_only_bound_manifest_evidence(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    manifest_path = _analysis_manifest(
        tmp_path,
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        reduction=plan,
        evidence=evidence,
        envelope=envelope,
        name="unattested",
    )
    output = tmp_path / "reducer.json"
    assert (
        main(
            [
                "analyze-industrial",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["kind"] == "industrial_schema_v3_reducer"
    assert artifact["status"] == "UNRESOLVED"
    assert artifact["gpu_evidence"] == "UNMEASURED"
    assert artifact["reasons"] == ["gpu_attestation:missing"]
    assert artifact["identity"]["gpu_attestation_sha256"] is None
    assert artifact["identity"]["doctor_report_sha256"] is None
    assert artifact["identity"]["registry_sha256"] == registry.sha256
    assert Path(f"{output}.sha256").read_text(encoding="utf-8").strip() == (
        content_sha256(artifact)
    )

    injected = json.loads(manifest_path.read_text(encoding="utf-8"))
    injected["metrics"] = {"l0_goodput": 1e30}
    injected_path = tmp_path / "injected-summary.json"
    _write_bound_json(injected_path, injected)
    with pytest.raises(ValueError, match="manifest fields do not match schema"):
        main(
            [
                "analyze-industrial",
                "--manifest",
                str(injected_path),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )


def test_analyze_industrial_cli_writes_unresolved_and_returns_nonzero(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    original = evidence[-1].cells[0]
    hardware = json.loads(original.hardware_receipt.path.read_text(encoding="utf-8"))
    hardware["rank_contexts"][0]["power_state"] = "P1"
    invalid_receipt = _write_json(
        original.hardware_receipt.path.parent / "cli-invalid-hardware.json",
        hardware,
    )
    invalid_cell = replace(original, hardware_receipt=invalid_receipt)
    invalid_block = replace(
        evidence[-1],
        cells=(invalid_cell, *evidence[-1].cells[1:]),
    )
    invalid_evidence = (*evidence[:-1], invalid_block)
    manifest_path = _analysis_manifest(
        tmp_path,
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        reduction=plan,
        evidence=invalid_evidence,
        envelope=envelope,
        name="unresolved",
    )
    output = tmp_path / "unresolved.json"
    assert (
        main(
            [
                "analyze-industrial",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "UNRESOLVED"
    assert artifact["gpu_evidence"] == "INVALIDATED"
    assert artifact["primary_contrasts"] == []
    assert Path(f"{output}.sha256").is_file()


def test_confirmation_family_power_cli_reduces_only_bound_pilot_evidence(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    analysis_path = _analysis_manifest(
        tmp_path,
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        reduction=plan,
        evidence=evidence,
        envelope=envelope,
        name="family-power-source",
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    power_manifest = {
        "schema_version": 2,
        "kind": "industrial_family_power_manifest",
        "registry_artifact": analysis["registry_artifact"],
        "pilot_activation": analysis["pilot_activation"],
        "gpu_inventory": analysis["gpu_inventory"],
        "hardware_envelope": analysis["hardware_envelope"],
        "blocks": [
            block for block in analysis["blocks"] if block["block"] in PILOT_BLOCKS
        ],
    }
    manifest_path = tmp_path / "family-power-manifest.json"
    _write_bound_json(manifest_path, power_manifest)
    output = tmp_path / "family-power-plan.json"
    assert (
        main(
            [
                "reduce-confirmation-family-power",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    reduced = confirmation_family_power_reduction_artifact_from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert reduced == plan
    final_output = tmp_path / "family-final-activation.json"
    assert (
        main(
            [
                "materialize-confirmation-prefix",
                "--power-manifest",
                str(manifest_path),
                "--output",
                str(final_output),
            ]
        )
        == 0
    )
    assert (
        family_activation_artifact_from_dict(
            json.loads(final_output.read_text(encoding="utf-8"))
        )
        == final_activation
    )

    forged = dict(power_manifest)
    forged["pilot_scores"] = {"l0": 1e30}
    forged_path = tmp_path / "forged-family-power-manifest.json"
    _write_bound_json(forged_path, forged)
    with pytest.raises(ValueError, match="manifest fields do not match schema"):
        main(
            [
                "reduce-confirmation-family-power",
                "--manifest",
                str(forged_path),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )
