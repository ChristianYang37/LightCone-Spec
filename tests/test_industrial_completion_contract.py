from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from test_native_terminal_provider import (
    FakeAdminTransport,
    _server_request,
)
from test_native_terminal_provider import (
    _run as _run_native_terminal,
)

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.cli.main import (
    _completed_industrial_cells,
    _industrial_completion_activation_contract,
    _industrial_physical_assignment_from_dict,
    main,
)
from lightcone_spec.experiments.evidence import evidence_files_sha256
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.planning import (
    ZERO_COUNT,
    ZERO_MILLISECONDS,
    BudgetJobKind,
    BudgetObservationReceipt,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.planning_artifacts import experiment_budget_to_dict
from lightcone_spec.experiments.registry import (
    INDUSTRIAL_EXPERIMENT_ORDER,
    CellStatus,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    WorkloadClass,
    build_industrial_registry,
)
from lightcone_spec.experiments.stage_activation import (
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    RegistryStageActivationArtifact,
    materialize_registry_stage_activation,
    release_dispatch_rejection_reason,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
from lightcone_spec.orchestration.native_terminal import NativeTerminalRunBinding
from lightcone_spec.telemetry import (
    DEFAULT_EVIDENCE_WRITER_POLICY,
    OUTPUT_HASH_FORMAT,
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
)


def _sha(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _receipt_prefix(
    registry: ExperimentRegistry,
    experiment: str,
) -> tuple[ExperimentReceipt, ...]:
    receipts: list[ExperimentReceipt] = []
    target_index = INDUSTRIAL_EXPERIMENT_ORDER.index(experiment)
    for name in INDUSTRIAL_EXPERIMENT_ORDER[:target_index]:
        definition = registry.definition(name)
        receipt = registry.make_receipt(
            name,
            {
                output: _sha({"stage": name, "output": output})
                for output in definition.locked_outputs
            },
            runtime_sha256=_sha({"stage": name, "artifact": "runtime"}),
            split_sha256=_sha({"stage": name, "artifact": "split"}),
            completed_cells_sha256=_sha({"stage": name, "artifact": "completed_cells"}),
            dependencies=tuple(receipts),
        )
        receipts.append(receipt)
    return tuple(receipts)


def _write_bound(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(_sha(value) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(*, gpu_count: int = 2) -> GpuInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-physical-{index}",
            host_id="fixture-host",
            model="A100",
            memory_bytes=80_000_000_000,
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
            allowed_topology_groups=("fixture-nvlink-group",),
        )
        for index in range(gpu_count)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id="fixture-nvlink-group",
                host_id="fixture-host",
                gpu_uuids=tuple(device.uuid for device in devices),
                fabric="nvlink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=_sha({"inventory": "fixture", "count": gpu_count}),
    )


def _budget(cell: ExperimentCell) -> ExperimentBudget:
    compile_time = (
        ScenarioMilliseconds(1, 1, 1)
        if cell.resources.workload_class.value == "compile"
        else ZERO_MILLISECONDS
    )
    scored_time = (
        ZERO_MILLISECONDS
        if cell.identity.experiment == "preflight"
        else ScenarioMilliseconds(1, 1, 1)
    )
    wall = compile_time + scored_time
    gpu_time = wall.scale(cell.resources.gpu_count)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=(
            BudgetJobKind.COMPILE
            if cell.resources.workload_class.value == "compile"
            else BudgetJobKind.STANDARD
        ),
        startup_model_load=ZERO_MILLISECONDS,
        compile_jit_graph_prewarm=compile_time,
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ZERO_COUNT,
        scored_arrival=scored_time,
        request_deadline=scored_time,
        drain=ZERO_MILLISECONDS,
        reset_finalization=ZERO_MILLISECONDS,
        evidence_flush_shutdown=ZERO_MILLISECONDS,
        output_tokens=(
            ExpectedMaximumCount(0, 0)
            if cell.identity.experiment == "preflight"
            else ExpectedMaximumCount(1, 1)
        ),
        minimum_completed_requests=(
            0 if cell.identity.experiment == "preflight" else 1
        ),
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=ZERO_MILLISECONDS,
        retry_allowance=0,
        profiler=ZERO_MILLISECONDS,
        download_compile_reservation=ZERO_MILLISECONDS,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_time,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=wall.scale(2),
    )


def _physical_assignment(
    cell: ExperimentCell,
    *,
    cell_index: int,
    budget: ExperimentBudget,
) -> IndustrialPhysicalAssignment:
    inventory = _inventory()
    gpu_uuids = tuple(
        inventory.devices[rank].uuid for rank in range(cell.resources.gpu_count)
    )
    tensor_parallel_size = 2 if cell.identity.topology == "tp2_dp1" else 1
    data_parallel_size = cell.resources.gpu_count // tensor_parallel_size
    rank_groups = tuple(
        gpu_uuids[replica * tensor_parallel_size : (replica + 1) * tensor_parallel_size]
        for replica in range(data_parallel_size)
    )
    return IndustrialPhysicalAssignment(
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        dispatch_plan_sha256=_sha({"dispatch": cell.cell_id}),
        experiment_budget_sha256=budget.sha256,
        budget_plan_sha256=_sha({"budget-plan": cell.cell_id}),
        capacity_authority_sha256=_sha({"capacity-authority": cell.cell_id}),
        budget_materialization_authority_sha256=_sha(
            {"budget-materialization-authority": cell.cell_id}
        ),
        assignment_sha256=_sha({"assignment": cell.cell_id}),
        work_item_sha256=_sha({"work-item": cell.cell_id}),
        gpu_uuids=gpu_uuids,
        rank_groups=rank_groups,
        ports=tuple(30_000 + cell_index * 10 + index for index in range(3)),
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        fixed_instance_gpu_count=2,
        host_id="fixture-host",
        topology_group_ids=tuple(
            (() if tensor_parallel_size == 1 else ("fixture-nvlink-group",))
            for _ in range(data_parallel_size)
        ),
    )


def _write_budget_observation(
    root: Path,
    *,
    run_id: str,
    budget: ExperimentBudget,
    terminal_receipt_sha256: str,
    fixed_instance_gpu_count: int,
) -> tuple[Path, str]:
    component_names = (
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
    observed = tuple(
        (name, getattr(budget, name).registered) for name in component_names
    )
    observed_wall_ms = sum(value for _, value in observed)
    observation = BudgetObservationReceipt(
        schema_version=1,
        budget=budget,
        observed_component_ms=observed,
        measured_gpu_ms=observed_wall_ms * budget.gpu_count,
        fixed_instance_billed_gpu_ms=(observed_wall_ms * fixed_instance_gpu_count),
        terminal_evidence_sha256=terminal_receipt_sha256,
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": "industrial_budget_observation_receipt_v1",
        "experiment_budget_sha256": budget.sha256,
        "budget_observation_sha256": observation.sha256,
        "budget": asdict(budget),
        "observed_component_ms": [list(row) for row in observed],
        "measured_gpu_ms": observation.measured_gpu_ms,
        "fixed_instance_billed_gpu_ms": observation.fixed_instance_billed_gpu_ms,
        "terminal_evidence_sha256": terminal_receipt_sha256,
        "observed_wall_ms": observation.observed_wall_ms,
        "registered_wall_delta_ms": observation.registered_wall_delta_ms,
        "registered_gpu_delta_ms": observation.registered_gpu_delta_ms,
        "registered_billed_delta_ms": observation.registered_billed_delta_ms,
        "gpu_measurement_semantics": ("exclusive_reserved_gang_wall_ms_x_gpu_count"),
        "fixed_instance_billing_semantics": "whole_inventory_wall_clock_v1",
    }
    directory = root / f"{run_id}.rank0.budget-observation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "observation.json"
    path.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    Path(f"{path}.sha256").write_text(observation.sha256 + "\n", encoding="utf-8")
    return path, observation.sha256


def _topology(
    cell: ExperimentCell,
    assignment: IndustrialPhysicalAssignment,
    topology_receipt_sha256: str,
) -> tuple[int, int, int, str]:
    tensor_parallel_size = assignment.tensor_parallel_size
    data_parallel_size = assignment.data_parallel_size
    world_size = len(assignment.gpu_uuids)
    digest = _sha(
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "topology": cell.identity.topology,
            "topology_receipt_sha256": topology_receipt_sha256,
            "physical_assignment_sha256": assignment.assignment_sha256,
            "physical_binding_sha256": assignment.sha256,
            "physical_host_id": assignment.host_id,
            "physical_gpu_uuids": list(assignment.gpu_uuids),
            "physical_rank_groups": [list(group) for group in assignment.rank_groups],
            "physical_ports": list(assignment.ports),
            "topology_group_ids": [
                list(group) for group in assignment.topology_group_ids
            ],
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "world_size": world_size,
        }
    )
    return tensor_parallel_size, data_parallel_size, world_size, digest


def _locked_cell(
    cell: ExperimentCell,
    *,
    assignment: IndustrialPhysicalAssignment,
    budget: ExperimentBudget,
    topology_receipt_sha256: str,
) -> dict[str, object]:
    request_ids = [f"request-{cell.cell_id[:16]}"]
    method = cell.identity.method
    workload_contract = (
        f"industrial_preflight_{method}"
        if cell.identity.experiment == "preflight"
        else (
            f"industrial_{method}"
            if method in {"target_only", "static"}
            else "industrial_adapted"
        )
    )
    contract = {
        "cell_id": cell.cell_id,
        "request_ids": request_ids,
        "expected_request_rows": len(request_ids),
        "expected_round_rows": 0 if method in {"target_only", "static"} else 1,
        "expected_update_rows": 0 if method in {"target_only", "static"} else 1,
        "expected_performance_rows": 1,
        "request_ids_sha256": _sha(request_ids),
        "corpus_sha256": _sha({"kind": "corpus", "cell_id": cell.cell_id}),
        "arrival_trace_sha256": _sha(
            {"kind": "arrival_trace", "cell_id": cell.cell_id}
        ),
        "sampling_profile_sha256": _sha(
            {"kind": "sampling_profile", "cell_id": cell.cell_id}
        ),
        "model_lock_sha256": _sha({"kind": "model_lock", "cell_id": cell.cell_id}),
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "workload_contract": workload_contract,
        "physical_assignment": assignment.to_dict(),
        "physical_binding_sha256": assignment.sha256,
        "topology_receipt_sha256": topology_receipt_sha256,
        "experiment_budget_sha256": budget.sha256,
        "experiment_budget": experiment_budget_to_dict(budget),
        "execution_plan_sha256": _sha({"execution-plan": cell.cell_id}),
        "execution_split_sha256": _sha({"execution-split": cell.cell_id}),
        "rank_config_sha256s": None,
    }
    if cell.identity.experiment != "preflight":
        contract["rank_config_sha256s"] = [
            _sha({"kind": "rank_config", "cell_id": cell.cell_id, "rank": rank})
            for rank in range(len(assignment.gpu_uuids))
        ]
    return contract


def _required_preflight_checks(task: str) -> dict[str, str]:
    names = {
        "environment_and_patch_preflight": {
            "identity",
            "environment",
            "patch_apply",
            "compile",
            "patch_tests",
            "compatibility",
        },
        "exactness_memory_telemetry_preflight": {
            "exactness",
            "memory",
            "telemetry",
            "target_only_allocation",
            "static_allocation",
        },
        "simultaneous_single_gpu_interference": {
            "isolated",
            "simultaneous",
            "hardware",
            "paired_blocks",
        },
    }[task]
    return {name: "PASS" for name in sorted(names)}


def _performance(
    run_id: str,
    cell: ExperimentCell,
    *,
    itl_ms: float | None = 1.0,
) -> PerformanceRecord:
    return PerformanceRecord(
        run_id=run_id,
        prompt_id="industrial-contract",
        method=cell.identity.method,
        repetition_block=cell.identity.block,
        region="industrial-contract",
        concurrency=1,
        generated_bucket_start=0,
        generated_bucket_end=1,
        at_risk_requests=1,
        output_tokens=1,
        elapsed_s=1.0,
        decode_goodput_tps=1.0,
        itl_p50_ms=itl_ms,
        itl_p95_ms=itl_ms,
        itl_p99_ms=itl_ms,
        survival_weighted_accepted_prefix=None,
        accepted_drafts_per_verify=None,
        committed_tokens_per_verify=None,
        verified_drafts_per_verify=None,
        verification_waste=None,
        target_calls_per_output_token=1.0,
        batch_fill=1.0,
        queue_occupancy=0.0,
        gpu_busy=None,
        sm_utilization=None,
        dram_utilization=None,
        target_estimated_mfu=None,
        peak_hbm_bytes=1,
        kv_bytes=0,
        optimizer_bytes=0,
        adaptation_memory_ledger=None,
        trainable_parameters=0,
        training_cuda_ms=None,
        optimizer_cuda_ms=None,
        merge_cuda_ms=None,
        publish_cuda_ms=None,
        barrier_cuda_ms=None,
        exposed_update_ms=None,
        main_side_overlap_ratio=None,
        graph_replay_hit_rate=None,
        updates_launched=0,
        updates_published=0,
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
        communicator_failures=0,
        evidence_backpressure_events=0,
        evidence_dropped_rows=0,
    )


def _request(
    run_id: str,
    cell: ExperimentCell,
    request_id: str,
    output_sha256: str,
    *,
    ttft_ms: float | None = 1.0,
    finished: bool = True,
) -> RequestRecord:
    token_ids = [int(output_sha256[:8], 16)] if finished else []
    serialized_token_ids = json.dumps(token_ids, separators=(",", ":"))
    canonical_output_sha256 = hashlib.sha256(
        serialized_token_ids.encode("utf-8")
    ).hexdigest()
    return RequestRecord(
        run_id=run_id,
        request_id=request_id,
        prompt_id="industrial-contract",
        method=cell.identity.method,
        repetition_block=cell.identity.block,
        concurrency=1,
        input_tokens=1,
        output_tokens=1 if finished else 0,
        output_hash_format=OUTPUT_HASH_FORMAT,
        output_sha256=canonical_output_sha256,
        ttft_ms=ttft_ms,
        finished=finished,
        stop_reason="length" if finished else "cancelled_before_first_token",
        output_token_ids=serialized_token_ids,
        output_token_ids_sha256=canonical_output_sha256,
        outcome_status="completed" if finished else "cancelled",
        arrival_ns=1,
        admitted_ns=1,
        completed_ns=2,
    )


def _round(run_id: str, request_id: str) -> RoundRecord:
    return RoundRecord(
        run_id=run_id,
        request_id=request_id,
        round_index=0,
        generated_tokens_before=0,
        prefix_len_before=1,
        verify_len=1,
        accepted_drafts=0,
        committed_tokens=1,
        target_calls=1,
        proposal_source_version=0,
        kv_source_versions="[]",
    )


def _build_registry(tmp_path: Path) -> tuple[Path, ExperimentRegistry]:
    registry_path = tmp_path / "registry.json"
    cache_root = str(tmp_path / "cache")
    evidence_root = str(tmp_path / "evidence")
    assert (
        main(
            [
                "build-industrial-registry",
                "--logical-gpu-slot",
                "logical-test-slot-a",
                "logical-test-slot-b",
                "--cache-root",
                cache_root,
                "--evidence-root",
                evidence_root,
                "--output",
                str(registry_path),
            ]
        )
        == 0
    )
    registry = build_industrial_registry(
        gpu_uuids=("logical-test-slot-a", "logical-test-slot-b"),
        cache_root=cache_root,
        evidence_root=evidence_root,
    )
    return registry_path, registry


def _preflight_bundle(
    tmp_path: Path,
    *,
    invalid_attestation: bool = False,
    mismatched_rank_output: bool = False,
    not_applicable_task: str | None = None,
) -> dict[str, object]:
    registry_path, registry = _build_registry(tmp_path)
    # This legacy fixture exercises the non-serving preflight disposition
    # contract.  The registered interference cells now use the serving/native
    # terminal path and have their own focused scheduler/authority fixtures.
    registry = replace(
        registry,
        cells=tuple(
            cell.with_status(
                CellStatus.NOT_APPLICABLE,
                reason_code="fixture_serving_calibration_covered_separately",
                reason=(
                    "The focused non-serving preflight fixture excludes the "
                    "separately tested serving calibration path."
                ),
            )
            if cell.identity.experiment == "preflight"
            and cell.identity.task == "simultaneous_single_gpu_interference"
            else cell
            for cell in registry.cells
        ),
    )
    inventory = _inventory()
    inventory_path = tmp_path / "gpu-inventory.json"
    _write_bound(inventory_path, inventory.to_dict())
    if (
        not_applicable_task is not None
        and not_applicable_task != "simultaneous_single_gpu_interference"
    ):
        source = next(
            cell
            for cell in registry.cells_for("preflight")
            if cell.identity.task == not_applicable_task
        )
        replacement = source.with_status(
            CellStatus.NOT_APPLICABLE,
            reason_code="fixture_not_applicable",
            reason="The focused completion fixture marks this cell N/A.",
        )
        registry = replace(
            registry,
            cells=tuple(
                replacement if cell.cell_id == source.cell_id else cell
                for cell in registry.cells
            ),
        )
    runtime = {"schema_version": 1, "kind": "industrial_runtime_test"}
    runtime_path = tmp_path / "runtime.json"
    _write_bound(runtime_path, runtime)
    runtime_sha256 = _sha(runtime)

    cells = tuple(
        cell
        for cell in registry.cells_for("preflight")
        if release_dispatch_rejection_reason(cell) is None
    )
    budgets = {cell.cell_id: _budget(cell) for cell in cells}
    assignments = {
        cell.cell_id: _physical_assignment(
            cell,
            cell_index=index,
            budget=budgets[cell.cell_id],
        )
        for index, cell in enumerate(cells)
    }
    topology_receipts = {
        cell.cell_id: _sha({"topology-receipts": cell.cell_id}) for cell in cells
    }
    contracts = {
        cell.cell_id: _locked_cell(
            cell,
            assignment=assignments[cell.cell_id],
            budget=budgets[cell.cell_id],
            topology_receipt_sha256=topology_receipts[cell.cell_id],
        )
        for cell in cells
    }
    split = {
        "schema_version": 1,
        "kind": "industrial_locked_split",
        "registry_sha256": registry.sha256,
        "experiment": "preflight",
        "cells": [contracts[cell.cell_id] for cell in cells],
    }
    split_path = tmp_path / "split.json"
    _write_bound(split_path, split)
    split_sha256 = _sha(split)
    activation = materialize_registry_stage_activation(
        registry,
        experiment="preflight",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    assert set(activation.activated_cell_ids) == {cell.cell_id for cell in cells}
    _, activation_dispositions, activation_binding = (
        _industrial_completion_activation_contract(
            registry,
            experiment="preflight",
            runtime_sha256=runtime_sha256,
            split_sha256=split_sha256,
            direct_dependency_receipt_sha256=None,
            activation_artifact=activation,
            family_activations=(),
            family_power_reductions=(),
        )
    )

    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        contract = contracts[cell.cell_id]
        assignment = assignments[cell.cell_id]
        budget = budgets[cell.cell_id]
        request_id = str(contract["request_ids"][0])
        run_id = f"preflight-{cell.cell_id[:16]}"
        run_nonce_sha256 = _sha(
            {
                "kind": "run_nonce",
                "cell_id": cell.cell_id,
            }
        )
        tp_size, dp_size, world_size, topology_sha256 = _topology(
            cell,
            assignment,
            topology_receipts[cell.cell_id],
        )
        for rank, gpu_uuid in enumerate(assignment.gpu_uuids):
            root = Path(cell.resources.evidence_root)
            root.mkdir(parents=True, exist_ok=True)
            source = root / f"{run_id}.rank{rank}.source.json"
            _write_bound(
                source,
                {
                    "schema_version": 1,
                    "cell_id": cell.cell_id,
                    "rank": rank,
                    "probe": "raw",
                },
            )
            checks = _required_preflight_checks(cell.identity.task)
            if invalid_attestation and cell_index == 0 and rank == 0:
                checks.pop(next(iter(checks)))
            attestation = {
                "schema_version": 1,
                "kind": "industrial_preflight_attestation",
                "status": "PASS",
                "registry_sha256": registry.sha256,
                "cell_id": cell.cell_id,
                "runtime_sha256": runtime_sha256,
                "split_sha256": split_sha256,
                "run_nonce_sha256": run_nonce_sha256,
                "topology_sha256": topology_sha256,
                "rank": rank,
                "gpu_uuid": gpu_uuid,
                "checks": checks,
                "source_files": [str(source)],
                "source_evidence_sha256": evidence_files_sha256((source,)),
            }
            attestation_path = root / f"{run_id}.rank{rank}.attestation.json"
            _write_bound(attestation_path, attestation)
            attestation_sha256 = _sha(attestation)

            writer = EvidenceWriter(
                root,
                run_id=run_id,
                rank=rank,
                process_id=cell_index * 10 + rank + 1,
                checkpoint_interval_s=None,
            )
            writer.write(
                RunRecord(
                    run_id=run_id,
                    manifest_sha256=registry.sha256,
                    config_sha256=cell.cell_id,
                    method=cell.identity.method,
                    model_pair=cell.identity.model,
                    repetition_block=cell.identity.block,
                    started_ns=1,
                    completed_ns=2,
                    status="complete",
                    industrial_cell_id=cell.cell_id,
                    runtime_sha256=str(contract["execution_plan_sha256"]),
                    split_sha256=str(contract["execution_split_sha256"]),
                    corpus_sha256=str(contract["corpus_sha256"]),
                    arrival_trace_sha256=str(contract["arrival_trace_sha256"]),
                    request_ids_sha256=str(contract["request_ids_sha256"]),
                    sampling_profile_sha256=str(contract["sampling_profile_sha256"]),
                    model_lock_sha256=str(contract["model_lock_sha256"]),
                    patched_sglang_tree=PINNED_SGLANG_TREE,
                    run_nonce_sha256=run_nonce_sha256,
                    topology_sha256=topology_sha256,
                    tensor_parallel_size=tp_size,
                    data_parallel_size=dp_size,
                    world_size=world_size,
                    rank=rank,
                    expected_request_rows=int(contract["expected_request_rows"]),
                    expected_round_rows=int(contract["expected_round_rows"]),
                    expected_update_rows=int(contract["expected_update_rows"]),
                    expected_performance_rows=int(
                        contract["expected_performance_rows"]
                    ),
                    workload_contract=str(contract["workload_contract"]),
                    experiment_budget_sha256=budget.sha256,
                    preflight_attestation_sha256=attestation_sha256,
                )
            )
            output_variant = (
                rank if mismatched_rank_output and cell_index == 0 and rank == 1 else 0
            )
            writer.write(
                _request(
                    run_id,
                    cell,
                    request_id,
                    _sha(
                        {
                            "kind": "request_output",
                            "request_id": request_id,
                            "variant": output_variant,
                        }
                    ),
                )
            )
            writer.write(_performance(run_id, cell))
            evidence = writer.close()
            receipt_path = root / f"{run_id}.rank{rank}.complete.json"
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "evidence_root": str(root),
                    "run_id": run_id,
                    "rank": rank,
                    "evidence_sha256": evidence_files_sha256(evidence.values()),
                    "terminal_receipt_sha256": _file_sha256(receipt_path),
                    "physical_gpu_uuid": gpu_uuid,
                    "physical_binding_sha256": assignment.sha256,
                    "experiment_budget_sha256": budget.sha256,
                    "budget_observation_status": "NOT_APPLICABLE",
                    "budget_observation_reason_code": (
                        "preflight_or_non_serving_execution"
                    ),
                    "budget_observation_path": None,
                    "budget_observation_sha256": None,
                    "preflight_attestation_path": str(attestation_path),
                    "preflight_attestation_sha256": attestation_sha256,
                    "status": "MEASURED",
                }
            )
    rows.extend(
        activation_dispositions[cell.cell_id]
        for cell in registry.cells_for("preflight")
        if cell.cell_id not in activation.activated_cell_ids
    )

    completed = {
        "schema_version": 4,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry.sha256,
        "experiment": "preflight",
        "runtime_sha256": runtime_sha256,
        "split_sha256": split_sha256,
        "split_contract": split,
        "activation_binding": activation_binding,
        "inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "rows": rows,
    }
    completed_path = tmp_path / "completed.json"
    _write_bound(completed_path, completed)
    locked_output = tmp_path / "runtime-envelope.json"
    _write_bound(locked_output, {"schema_version": 1, "status": "PASS"})
    return {
        "registry": registry,
        "registry_path": registry_path,
        "inventory": inventory,
        "inventory_path": inventory_path,
        "runtime": runtime,
        "runtime_path": runtime_path,
        "split": split,
        "split_path": split_path,
        "activation": activation,
        "completed": completed,
        "completed_path": completed_path,
        "locked_output": locked_output,
    }


def _serving_bundle(tmp_path: Path) -> dict[str, object]:
    registry = build_industrial_registry(
        gpu_uuids=("logical-serving-slot-a", "logical-serving-slot-b"),
        cache_root=str(tmp_path / "serving-cache"),
        evidence_root=str(tmp_path / "serving-evidence"),
    )
    inventory = _inventory()
    stage = "E3a"
    active_candidates = tuple(
        cell
        for cell in registry.cells_for(stage)
        if release_dispatch_rejection_reason(cell) is None
        and cell.resources.gpu_count == 1
        and cell.identity.concurrency == 1
    )
    assert active_candidates
    selected = active_candidates[0]
    registry = replace(
        registry,
        cells=tuple(
            cell.with_status(
                CellStatus.BLOCKED,
                reason_code="fixture_reduced_scope",
                reason="The focused completion fixture executes one exact cell.",
            )
            if cell.identity.experiment == stage
            and cell.cell_id != selected.cell_id
            and release_dispatch_rejection_reason(cell) is None
            else cell
            for cell in registry.cells
        ),
    )
    dependency_receipts = _receipt_prefix(registry, stage)
    direct_dependency_sha256 = dependency_receipts[-1].sha256
    runtime_sha256 = _sha({"runtime": stage})
    cells = tuple(
        cell
        for cell in registry.cells_for(stage)
        if release_dispatch_rejection_reason(cell) is None
    )
    budgets = {cell.cell_id: _budget(cell) for cell in cells}
    assignments = {
        cell.cell_id: _physical_assignment(
            cell,
            cell_index=index,
            budget=budgets[cell.cell_id],
        )
        for index, cell in enumerate(cells)
    }
    topology_receipts = {
        cell.cell_id: _sha({"topology-receipts": cell.cell_id}) for cell in cells
    }
    contracts = {
        cell.cell_id: _locked_cell(
            cell,
            assignment=assignments[cell.cell_id],
            budget=budgets[cell.cell_id],
            topology_receipt_sha256=topology_receipts[cell.cell_id],
        )
        for cell in cells
    }
    split = {
        "schema_version": 1,
        "kind": "industrial_locked_split",
        "registry_sha256": registry.sha256,
        "experiment": stage,
        "cells": [contracts[cell.cell_id] for cell in cells],
    }
    split_sha256 = _sha(split)
    activation = materialize_registry_stage_activation(
        registry,
        experiment=stage,
        dependency_receipts=dependency_receipts,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
    )
    assert set(activation.activated_cell_ids) == {cell.cell_id for cell in cells}
    _, dispositions, activation_binding = _industrial_completion_activation_contract(
        registry,
        experiment=stage,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        direct_dependency_receipt_sha256=direct_dependency_sha256,
        activation_artifact=activation,
        family_activations=(),
        family_power_reductions=(),
    )
    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        contract = contracts[cell.cell_id]
        assignment = assignments[cell.cell_id]
        budget = budgets[cell.cell_id]
        run_id = f"serving-{cell.cell_id[:16]}"
        request_id = str(contract["request_ids"][0])
        _, _, world_size, topology_sha256 = _topology(
            cell,
            assignment,
            topology_receipts[cell.cell_id],
        )
        assert world_size == 1
        root = Path(cell.resources.evidence_root)
        writer = EvidenceWriter(
            root,
            run_id=run_id,
            rank=0,
            process_id=cell_index + 100,
            registered_policy=DEFAULT_EVIDENCE_WRITER_POLICY,
        )
        run_nonce_sha256 = _sha({"nonce": cell.cell_id})
        output_seed_sha256 = _sha({"output": request_id})
        output_token_ids = (int(output_seed_sha256[:8], 16),)
        native_run_binding = NativeTerminalRunBinding(
            run_id=run_id,
            run_nonce_sha256=run_nonce_sha256,
            execution_plan_sha256=str(contract["execution_plan_sha256"]),
            rank_config_sha256=str(contract["rank_config_sha256s"][0]),
            attempt_id=writer.attempt_id,
            session_id=f"standalone-{cell_index}",
            session_epoch=1,
            previous_run_id=None,
            challenge_nonce_sha256=_sha({"challenge": cell.cell_id}),
            method=cell.identity.method,
            warmup_request_ids=(),
            scored_request_ids=(request_id,),
        )
        scored_request = _server_request(
            request_id,
            inputs=(1,),
            outputs=output_token_ids,
        )
        native_transport = FakeAdminTransport(
            binding=native_run_binding,
            warmup=(),
            scored=(scored_request,),
        )
        _, _, _, native_terminal = asyncio.run(_run_native_terminal(native_transport))
        native_artifact_binding = writer.persist_native_terminal_artifact(
            native_terminal.to_artifact(warmup_requests=())
        )
        writer.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=registry.sha256,
                config_sha256=cell.cell_id,
                method=cell.identity.method,
                model_pair=cell.identity.model,
                repetition_block=cell.identity.block,
                started_ns=1,
                completed_ns=2,
                status="complete",
                industrial_cell_id=cell.cell_id,
                rank_config_sha256=str(contract["rank_config_sha256s"][0]),
                runtime_sha256=str(contract["execution_plan_sha256"]),
                split_sha256=str(contract["execution_split_sha256"]),
                corpus_sha256=str(contract["corpus_sha256"]),
                arrival_trace_sha256=str(contract["arrival_trace_sha256"]),
                request_ids_sha256=str(contract["request_ids_sha256"]),
                sampling_profile_sha256=str(contract["sampling_profile_sha256"]),
                model_lock_sha256=str(contract["model_lock_sha256"]),
                patched_sglang_tree=PINNED_SGLANG_TREE,
                run_nonce_sha256=run_nonce_sha256,
                topology_sha256=topology_sha256,
                tensor_parallel_size=1,
                data_parallel_size=1,
                world_size=1,
                rank=0,
                expected_request_rows=1,
                expected_round_rows=0,
                expected_update_rows=0,
                expected_performance_rows=1,
                workload_contract=str(contract["workload_contract"]),
                experiment_budget_sha256=budget.sha256,
                preflight_attestation_sha256=None,
                native_terminal_artifact_path=str(native_artifact_binding["path"]),
                native_terminal_artifact_size=int(native_artifact_binding["size"]),
                native_terminal_raw_sha256=str(native_artifact_binding["raw_sha256"]),
                native_terminal_sha256=str(native_artifact_binding["terminal_sha256"]),
                trusted_attester_policy_sha256=str(
                    native_artifact_binding["trusted_attester_policy_sha256"]
                ),
            )
        )
        writer.write(
            _request(
                run_id,
                cell,
                request_id,
                output_seed_sha256,
            )
        )
        writer.write(_performance(run_id, cell))
        evidence, prepared_path = writer.prepare_close()
        prepared_sha256 = _file_sha256(prepared_path)
        observation_path, observation_sha256 = _write_budget_observation(
            root,
            run_id=run_id,
            budget=budget,
            terminal_receipt_sha256=prepared_sha256,
            fixed_instance_gpu_count=assignment.fixed_instance_gpu_count,
        )

        def validate_post_binding(
            path: Path = observation_path,
            sha256: str = observation_sha256,
        ) -> None:
            assert path.is_file() and not path.is_symlink()
            assert Path(f"{path}.sha256").read_text(encoding="utf-8") == (sha256 + "\n")

        assert (
            writer.publish_close(validate_post_binding=validate_post_binding)
            == evidence
        )
        terminal_path = root / f"{run_id}.rank0.complete.json"
        terminal_sha256 = _file_sha256(terminal_path)
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        assert terminal["prepared_receipt_sha256"] == prepared_sha256
        assert terminal["writer_policy"] == (DEFAULT_EVIDENCE_WRITER_POLICY.to_dict())
        assert terminal["writer_policy_sha256"] == (
            DEFAULT_EVIDENCE_WRITER_POLICY.sha256
        )
        assert terminal["budget_observation"]["budget_observation_sha256"] == (
            observation_sha256
        )
        rows.append(
            {
                "cell_id": cell.cell_id,
                "evidence_root": str(root),
                "run_id": run_id,
                "rank": 0,
                "evidence_sha256": evidence_files_sha256(evidence.values()),
                "terminal_receipt_sha256": terminal_sha256,
                "physical_gpu_uuid": assignment.gpu_uuids[0],
                "physical_binding_sha256": assignment.sha256,
                "experiment_budget_sha256": budget.sha256,
                "budget_observation_status": "OBSERVED",
                "budget_observation_reason_code": None,
                "budget_observation_path": str(observation_path),
                "budget_observation_sha256": observation_sha256,
                "preflight_attestation_path": None,
                "preflight_attestation_sha256": None,
                "status": "MEASURED",
            }
        )
    rows.extend(
        dispositions[cell.cell_id]
        for cell in registry.cells_for(stage)
        if cell.cell_id not in activation.activated_cell_ids
    )
    completed = {
        "schema_version": 4,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry.sha256,
        "experiment": stage,
        "runtime_sha256": runtime_sha256,
        "split_sha256": split_sha256,
        "split_contract": split,
        "activation_binding": activation_binding,
        "inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "rows": rows,
    }
    completed_path = tmp_path / "serving-completed.json"
    _write_bound(completed_path, completed)
    return {
        "registry": registry,
        "inventory": inventory,
        "stage": stage,
        "runtime_sha256": runtime_sha256,
        "split": split,
        "direct_dependency_receipt": dependency_receipts[-1],
        "direct_dependency_sha256": direct_dependency_sha256,
        "activation": activation,
        "completed": completed,
        "completed_path": completed_path,
    }


def _validate_bundle(
    bundle: dict[str, object], path: Path
) -> tuple[tuple[str, ...], str]:
    registry = bundle["registry"]
    runtime = bundle["runtime"]
    split = bundle["split"]
    inventory = bundle["inventory"]
    activation = bundle["activation"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(split, dict)
    assert isinstance(inventory, GpuInventory)
    assert isinstance(activation, RegistryStageActivationArtifact)
    completed, digest = _completed_industrial_cells(
        str(path),
        registry,
        experiment="preflight",
        runtime_sha256=_sha(runtime),
        split_sha256=_sha(split),
        split_contract=split,
        require_industrial_contract=True,
        activation_artifact=activation,
        inventory=inventory,
    )
    assert digest is not None
    return completed, digest


def _validate_serving_bundle(
    bundle: dict[str, object], path: Path
) -> tuple[tuple[str, ...], str]:
    registry = bundle["registry"]
    split = bundle["split"]
    inventory = bundle["inventory"]
    activation = bundle["activation"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(split, dict)
    assert isinstance(inventory, GpuInventory)
    assert isinstance(activation, RegistryStageActivationArtifact)
    completed, digest = _completed_industrial_cells(
        str(path),
        registry,
        experiment=str(bundle["stage"]),
        runtime_sha256=str(bundle["runtime_sha256"]),
        split_sha256=_sha(split),
        split_contract=split,
        require_industrial_contract=True,
        direct_dependency_receipt_sha256=str(bundle["direct_dependency_sha256"]),
        activation_artifact=activation,
        inventory=inventory,
    )
    assert digest is not None
    return completed, digest


def _rebind_generic_activation(
    bundle: dict[str, object],
    completed: dict[str, object],
) -> RegistryStageActivationArtifact:
    registry = bundle["registry"]
    prior = bundle["activation"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(prior, RegistryStageActivationArtifact)
    activation = materialize_registry_stage_activation(
        registry,
        experiment=str(completed["experiment"]),
        dependency_receipts=prior.dependency_receipts,
        runtime_sha256=str(completed["runtime_sha256"]),
        split_sha256=str(completed["split_sha256"]),
    )
    _, _, binding = _industrial_completion_activation_contract(
        registry,
        experiment=activation.experiment,
        runtime_sha256=activation.runtime_sha256,
        split_sha256=activation.split_sha256,
        direct_dependency_receipt_sha256=(activation.direct_dependency_receipt_sha256),
        activation_artifact=activation,
        family_activations=(),
        family_power_reductions=(),
    )
    completed["activation_binding"] = binding
    return activation


def test_physical_assignment_parser_requires_schema3_raw_budget_authority(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    completed = bundle["completed"]
    assert isinstance(completed, dict)
    split = completed["split_contract"]
    assert isinstance(split, dict)
    cells = split["cells"]
    assert isinstance(cells, list) and cells
    assignment = cells[0]["physical_assignment"]
    assert isinstance(assignment, dict)

    restored = _industrial_physical_assignment_from_dict(copy.deepcopy(assignment))
    assert restored.to_dict() == assignment
    assert restored.to_dict()["schema_version"] == 3

    missing = copy.deepcopy(assignment)
    missing.pop("budget_materialization_authority_sha256")
    with pytest.raises(ValueError, match="fields differ from schema"):
        _industrial_physical_assignment_from_dict(missing)

    legacy = copy.deepcopy(assignment)
    legacy["schema_version"] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        _industrial_physical_assignment_from_dict(legacy)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_raw_budget_authority", "fields differ from schema"),
        ("legacy_schema", "identity mismatch"),
        ("foreign_billing", "billing (semantics )?mismatch"),
    ),
)
def test_completion_contract_requires_exact_physical_assignment_schema3(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bundle = _serving_bundle(tmp_path)
    tampered = copy.deepcopy(bundle["completed"])
    split = tampered["split_contract"]
    assignment = split["cells"][0]["physical_assignment"]
    if mutation == "missing_raw_budget_authority":
        assignment.pop("budget_materialization_authority_sha256")
    elif mutation == "legacy_schema":
        assignment["schema_version"] = 2
    else:
        assignment["fixed_instance_billing_semantics"] = "per_assigned_gpu"
    tampered["split_sha256"] = _sha(split)
    activation = _rebind_generic_activation(bundle, tampered)
    path = tmp_path / f"physical-assignment-{mutation}.json"
    _write_bound(path, tampered)

    with pytest.raises(ValueError, match=message):
        _validate_serving_bundle(
            {**bundle, "split": split, "activation": activation},
            path,
        )


def test_preflight_stage_has_no_executable_compile_assignment(tmp_path: Path) -> None:
    bundle = _preflight_bundle(tmp_path)
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    completed, digest = _validate_bundle(bundle, completed_path)
    registry = bundle["registry"]
    activation = bundle["activation"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(activation, RegistryStageActivationArtifact)
    assert activation.status == "BLOCKED"
    assert activation.activated_cell_ids == ()
    compile_cell = next(
        cell
        for cell in registry.cells_for("preflight")
        if cell.resources.workload_class is WorkloadClass.COMPILE
    )
    compile_disposition = next(
        row for row in activation.dispositions if row.cell_id == compile_cell.cell_id
    )
    assert (
        compile_disposition.reason_code
        == RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
    )
    assert not Path(compile_cell.resources.evidence_root).exists()
    assert set(completed) == set(activation.activated_cell_ids)
    assert digest == _sha(bundle["completed"])

    forged_disposition = copy.deepcopy(bundle["completed"])
    forged_disposition["rows"][0]["reason_code"] = "caller_claimed_success"
    forged_path = tmp_path / "forged-disposition.json"
    _write_bound(forged_path, forged_disposition)
    with pytest.raises(ValueError, match="exact immutable disposition"):
        _validate_bundle(bundle, forged_path)


def test_blocked_preflight_rejects_forged_activation_and_measured_rows(
    tmp_path: Path,
) -> None:
    bundle = _preflight_bundle(tmp_path)
    registry = bundle["registry"]
    assert isinstance(registry, ExperimentRegistry)

    cases: list[tuple[str, dict[str, object], str]] = []

    forged_activation = copy.deepcopy(bundle["completed"])
    forged_activation["activation_binding"]["dispositions_sha256"] = "0" * 64
    cases.append(
        (
            "forged-activation",
            forged_activation,
            "activation binding is missing or forged",
        )
    )

    forged_measured = copy.deepcopy(bundle["completed"])
    compile_cell = next(
        cell
        for cell in registry.cells_for("preflight")
        if cell.resources.workload_class is WorkloadClass.COMPILE
    )
    compile_row = next(
        row for row in forged_measured["rows"] if row["cell_id"] == compile_cell.cell_id
    )
    compile_row["status"] = "MEASURED"
    cases.append(
        (
            "forged-compile-measured-row",
            forged_measured,
            "exact immutable disposition",
        )
    )

    forged_reason = copy.deepcopy(bundle["completed"])
    compile_row = next(
        row for row in forged_reason["rows"] if row["cell_id"] == compile_cell.cell_id
    )
    compile_row["reason_code"] = "caller_rehashed_compile_success"
    cases.append(
        (
            "forged-compile-disposition",
            forged_reason,
            "exact immutable disposition",
        )
    )

    for name, artifact, message in cases:
        path = tmp_path / f"{name}.json"
        _write_bound(path, artifact)
        with pytest.raises(ValueError, match=message):
            _validate_bundle(bundle, path)


def test_serving_completion_requires_exact_observed_budget_receipt(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    completed, _ = _validate_serving_bundle(bundle, completed_path)
    registry = bundle["registry"]
    activation = bundle["activation"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(activation, RegistryStageActivationArtifact)
    assert set(completed) == set(activation.activated_cell_ids)

    missing = copy.deepcopy(bundle["completed"])
    measured = next(row for row in missing["rows"] if row["status"] == "MEASURED")
    measured["budget_observation_path"] = None
    measured["budget_observation_sha256"] = None
    missing_path = tmp_path / "missing-serving-observation.json"
    _write_bound(missing_path, missing)
    with pytest.raises(ValueError, match="exact budget observation"):
        _validate_serving_bundle(bundle, missing_path)

    observed_row = next(
        row for row in bundle["completed"]["rows"] if row["status"] == "MEASURED"
    )
    observation_path = Path(str(observed_row["budget_observation_path"]))
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["fixed_instance_billed_gpu_ms"] -= 1
    observation_path.write_text(
        json.dumps(observation, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="accounting is inconsistent"):
        _validate_serving_bundle(bundle, completed_path)


def test_two_gpu_inventory_rejects_coordinated_one_gpu_underbilling(
    tmp_path: Path,
) -> None:
    bundle = _serving_bundle(tmp_path)
    registry = bundle["registry"]
    assert isinstance(registry, ExperimentRegistry)
    tampered = copy.deepcopy(bundle["completed"])
    split = tampered["split_contract"]
    contract = split["cells"][0]
    cell = next(row for row in registry.cells if row.cell_id == contract["cell_id"])
    one_gpu_inventory = _inventory(gpu_count=1)
    registered_budget = _budget(cell)
    one_gpu_budget = replace(
        registered_budget,
        fixed_instance_billed_gpu_ms=registered_budget.wall_time.scale(1),
    )
    assignment = contract["physical_assignment"]
    assignment["inventory_sha256"] = one_gpu_inventory.sha256
    assignment["inventory_source_receipt_sha256"] = (
        one_gpu_inventory.source_receipt_sha256
    )
    assignment["fixed_instance_gpu_count"] = 1
    assignment["experiment_budget_sha256"] = one_gpu_budget.sha256
    contract["physical_binding_sha256"] = _sha(assignment)
    contract["experiment_budget_sha256"] = one_gpu_budget.sha256
    contract["experiment_budget"] = experiment_budget_to_dict(one_gpu_budget)
    measured = next(
        row
        for row in tampered["rows"]
        if row.get("cell_id") == cell.cell_id and row.get("status") == "MEASURED"
    )
    measured["physical_binding_sha256"] = contract["physical_binding_sha256"]
    measured["experiment_budget_sha256"] = one_gpu_budget.sha256
    evidence_root = Path(str(measured["evidence_root"]))
    terminal = json.loads(
        (evidence_root / f"{measured['run_id']}.rank0.complete.json").read_text(
            encoding="utf-8"
        )
    )
    observation_path, observation_sha256 = _write_budget_observation(
        evidence_root,
        run_id=str(measured["run_id"]),
        budget=one_gpu_budget,
        terminal_receipt_sha256=str(terminal["prepared_receipt_sha256"]),
        fixed_instance_gpu_count=1,
    )
    measured["budget_observation_path"] = str(observation_path)
    measured["budget_observation_sha256"] = observation_sha256
    tampered["split_sha256"] = _sha(split)
    rebound_activation = _rebind_generic_activation(bundle, tampered)
    path = tmp_path / "coordinated-one-gpu-underbill.json"
    _write_bound(path, tampered)
    with pytest.raises(ValueError, match="differs from the bound GPU inventory"):
        _validate_serving_bundle(
            {**bundle, "split": split, "activation": rebound_activation},
            path,
        )


def test_blocked_and_not_applicable_outcomes_are_explicit_and_exact(
    tmp_path: Path,
) -> None:
    bundle = _preflight_bundle(
        tmp_path,
        not_applicable_task="simultaneous_single_gpu_interference",
    )
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    completed, _ = _validate_bundle(bundle, completed_path)
    registry = bundle["registry"]
    activation = bundle["activation"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(activation, RegistryStageActivationArtifact)
    assert set(completed) == set(activation.activated_cell_ids)

    forged = copy.deepcopy(bundle["completed"])
    disposition = next(row for row in forged["rows"] if row.get("status") == "N/A")
    disposition["reason_code"] = "edited_after_materialization"
    forged_path = tmp_path / "forged-disposition.json"
    _write_bound(forged_path, forged)
    with pytest.raises(ValueError, match="exact immutable disposition"):
        _validate_bundle(bundle, forged_path)

    missing = copy.deepcopy(bundle["completed"])
    missing["rows"] = [row for row in missing["rows"] if row.get("status") != "N/A"]
    missing_path = tmp_path / "missing-disposition.json"
    _write_bound(missing_path, missing)
    with pytest.raises(ValueError, match="do not cover every declared stage cell"):
        _validate_bundle(bundle, missing_path)

    extra = copy.deepcopy(bundle["completed"])
    extra["rows"].append(dict(disposition))
    extra_path = tmp_path / "extra-disposition.json"
    _write_bound(extra_path, extra)
    with pytest.raises(ValueError, match="exact immutable disposition"):
        _validate_bundle(bundle, extra_path)


def test_pre_token_latency_fields_persist_null_without_imputation(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry(
        gpu_uuids=("GPU-null-a", "GPU-null-b"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    cell = next(
        cell
        for cell in registry.cells_for("E3a")
        if cell.runnable and cell.identity.method == "target_only"
    )
    run_id = "nullable-latencies"
    writer = EvidenceWriter(tmp_path / "nullable", run_id=run_id, rank=0)
    writer.write(
        RunRecord(
            run_id=run_id,
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            method="target_only",
            model_pair="test-pair",
            repetition_block=0,
            started_ns=1,
            completed_ns=2,
            status="complete",
        )
    )
    writer.write(
        _request(
            run_id,
            cell,
            "cancelled-request",
            "c" * 64,
            ttft_ms=None,
            finished=False,
        )
    )
    writer.write(_performance(run_id, cell, itl_ms=None))
    evidence = writer.close()
    assert pq.read_table(evidence["request"], columns=["ttft_ms"]).to_pylist() == [
        {"ttft_ms": None}
    ]
    assert pq.read_table(
        evidence["performance"],
        columns=["itl_p50_ms", "itl_p95_ms", "itl_p99_ms"],
    ).to_pylist() == [{"itl_p50_ms": None, "itl_p95_ms": None, "itl_p99_ms": None}]
