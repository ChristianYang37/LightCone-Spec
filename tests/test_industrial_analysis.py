from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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
from lightcone_spec.doctor import _project_runtime_source, _project_tree
from lightcone_spec.experiments.evidence import evidence_files_sha256
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
)
from lightcone_spec.experiments.industrial_analysis import (
    _INDUSTRIAL_DOCTOR_CHECKS,
    BoundArtifact,
    E3bLongContextRawFamilyInput,
    IndustrialBlockEvidence,
    IndustrialCellEvidence,
    IndustrialReduction,
    MethodReduction,
    _alias_analysis_budget,
    _BlockReduction,
    _guard_preregistered_p99_analysis,
    _load_budget_observation,
    _load_cell,
    _LoadedCell,
    _mark_e2_confidence_pareto,
    _replay_cell_execution_identity,
    _RequestTerminalTimingUnavailable,
    _validate_allocation_free_performance,
    _validate_industrial_doctor,
    _validate_industrial_gpu_attestation,
    _validate_run_row,
    reduce_confirmation_family_power,
    reduce_e2_stage_from_raw,
    reduce_e3b_long_context_from_raw,
    reduce_industrial_schema_v3,
)
from lightcone_spec.experiments.long_context_analysis import (
    E3B_CONTEXT_GRID,
    E3bReductionStatus,
)
from lightcone_spec.experiments.planning import (
    AnalysisDependenceUnit,
    BudgetJobKind,
    ConfirmationFamilyPowerReductionArtifact,
    DispositionStatus,
    E1GeometryIdentity,
    E1ParetoArtifact,
    E2CandidateEvaluation,
    EvidenceDependenceMap,
    ExpectedMaximumCount,
    ExperimentBudget,
    FamilyActivationArtifact,
    P99AnchorStatus,
    ScenarioMilliseconds,
    derive_confirmation_family,
    family_pilot_block_id,
    materialize_confirmation_pilots,
    materialize_confirmation_prefix,
    reduce_e2_activation,
)
from lightcone_spec.experiments.planning_artifacts import (
    confirmation_family_power_reduction_artifact_from_dict,
    confirmation_family_power_reduction_artifact_to_dict,
    experiment_budget_to_dict,
    family_activation_artifact_from_dict,
    family_activation_artifact_to_dict,
)
from lightcone_spec.experiments.registry import (
    CONFIRMATION_METHOD_ROLES,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    ExperimentCell,
    ExperimentReceipt,
    ExperimentRegistry,
    LockedOutput,
    WorkloadClass,
    build_industrial_registry,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.runtime_metrics import RuntimeMetricStatus
from lightcone_spec.experiments.statistics import (
    HardwareEnvelope,
    SloRequest,
    account_slo,
    guard_p99_claim,
)
from lightcone_spec.orchestration.industrial import IndustrialPhysicalAssignment
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _simulate_clean_project_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    def clean_project_tree(root: Path) -> dict[str, object]:
        value = _project_tree(root)
        if root.resolve() == PROJECT_ROOT:
            value["dirty"] = False
        return value

    monkeypatch.setattr("lightcone_spec.doctor._project_tree", clean_project_tree)


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


def _analysis_physical_assignment(
    cell: ExperimentCell,
    *,
    budget: ExperimentBudget,
    inventory: GpuInventory,
    physical_gpu_uuids: tuple[str, ...],
    cell_index: int,
) -> IndustrialPhysicalAssignment:
    shape = {
        "tp1_dp1": (1, 1),
        "tp2_dp1": (2, 1),
        "two_replica_tp1_dp2": (1, 2),
        "two_gpu_host": (1, 2),
        "two_independent_tp1": (1, 2),
    }.get(cell.identity.topology)
    if shape is None or shape[0] * shape[1] != len(physical_gpu_uuids):
        raise ValueError("analysis fixture has no exact physical gang shape")
    tensor_parallel_size, data_parallel_size = shape
    rank_groups = tuple(
        physical_gpu_uuids[
            replica * tensor_parallel_size : (replica + 1) * tensor_parallel_size
        ]
        for replica in range(data_parallel_size)
    )
    return IndustrialPhysicalAssignment(
        inventory_sha256=inventory.sha256,
        inventory_source_receipt_sha256=inventory.source_receipt_sha256,
        dispatch_plan_sha256=content_sha256({"analysis_dispatch": cell.cell_id}),
        experiment_budget_sha256=budget.sha256,
        budget_plan_sha256=content_sha256({"analysis_budget_plan": cell.cell_id}),
        capacity_authority_sha256=content_sha256({"analysis_capacity": cell.cell_id}),
        budget_materialization_authority_sha256=content_sha256(
            {"analysis_budget_authority": cell.cell_id}
        ),
        assignment_sha256=content_sha256({"analysis_assignment": cell.cell_id}),
        work_item_sha256=content_sha256({"analysis_work_item": cell.cell_id}),
        gpu_uuids=physical_gpu_uuids,
        rank_groups=rank_groups,
        ports=tuple(32_000 + cell_index * 4 + index for index in range(3)),
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        fixed_instance_gpu_count=len(inventory.devices),
        host_id=inventory.host_ids[0],
        topology_group_ids=tuple(
            (
                ()
                if tensor_parallel_size == 1
                else (inventory.topology_groups[0].group_id,)
            )
            for _ in range(data_parallel_size)
        ),
    )


def _analysis_locked_cell_contract(
    cell: ExperimentCell,
    *,
    request_id: str,
    identities: dict[str, str],
    budget: ExperimentBudget,
    assignment: IndustrialPhysicalAssignment,
) -> dict[str, object]:
    adapted = cell.identity.method in {"tts", "l0"}
    workload_contract = (
        f"industrial_{cell.identity.method}"
        if cell.identity.method in {"target_only", "static"}
        else "industrial_adapted"
    )
    return {
        "cell_id": cell.cell_id,
        "request_ids": [request_id],
        "expected_request_rows": 1,
        "expected_round_rows": 1 if adapted else 0,
        "expected_update_rows": 1 if adapted else 0,
        "expected_performance_rows": 1,
        "request_ids_sha256": identities["request_ids_sha256"],
        "corpus_sha256": identities["corpus_sha256"],
        "arrival_trace_sha256": identities["arrival_trace_sha256"],
        "sampling_profile_sha256": identities["sampling_profile_sha256"],
        "model_lock_sha256": identities["model_lock_sha256"],
        "patched_sglang_tree": PINNED_SGLANG_TREE,
        "workload_contract": workload_contract,
        "rank_config_sha256s": [
            content_sha256({"rank_config": cell.cell_id, "rank": rank})
            for rank in range(len(assignment.gpu_uuids))
        ],
        "physical_assignment": assignment.to_dict(),
        "physical_binding_sha256": assignment.sha256,
        "topology_receipt_sha256": content_sha256(
            {"analysis_topology_receipt": cell.cell_id}
        ),
        "experiment_budget_sha256": budget.sha256,
        "experiment_budget": experiment_budget_to_dict(budget),
        "execution_plan_sha256": content_sha256(
            {"analysis_execution_plan": cell.cell_id}
        ),
        "execution_split_sha256": content_sha256(
            {"analysis_execution_split": cell.cell_id}
        ),
    }


def _analysis_activation_binding(
    *,
    activation_round: str,
    activations: tuple[FamilyActivationArtifact, ...],
    reductions: tuple[ConfirmationFamilyPowerReductionArtifact, ...],
) -> dict[str, object]:
    latest_by_family: dict[str, FamilyActivationArtifact] = {}
    for activation in activations:
        prior = latest_by_family.get(activation.family.sha256)
        if prior is None or activation.activation_round == "final_prefix":
            latest_by_family[activation.family.sha256] = activation
    disposition_rows = [
        {
            "cell_id": row.cell_id,
            "status": row.status.value,
            "reason_code": row.reason_code,
        }
        for activation in latest_by_family.values()
        for row in activation.dispositions
    ]
    return {
        "schema_version": 1,
        "kind": "industrial_stage_activation_binding",
        "stage_activation_sha256": None,
        "family_activation_sha256s": sorted(
            activation.sha256 for activation in activations
        ),
        "family_power_reduction_sha256s": sorted(
            reduction.sha256 for reduction in reductions
        ),
        "direct_dependency_receipt_sha256": content_sha256(
            {"analysis_dependency": "E3a"}
        ),
        "activation_round": activation_round,
        "dispositions_sha256": content_sha256(
            sorted(disposition_rows, key=lambda row: row["cell_id"])
        ),
    }


def _write_analysis_completion_contract(
    path: Path,
    *,
    registry: ExperimentRegistry,
    runtime_sha256: str,
    split_contract: dict[str, object],
    inventory: GpuInventory,
    activation_binding: dict[str, object],
    rows: tuple[dict[str, object], ...],
) -> BoundArtifact:
    value = {
        "schema_version": 4,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry.sha256,
        "experiment": "E3b",
        "runtime_sha256": runtime_sha256,
        "split_sha256": content_sha256(split_contract),
        "split_contract": split_contract,
        "activation_binding": activation_binding,
        "inventory_sha256": inventory.sha256,
        "inventory_source_receipt_sha256": inventory.source_receipt_sha256,
        "rows": list(rows),
    }
    reference = _write_json(path, value)
    Path(f"{path}.sha256").write_text(
        f"{content_sha256(value)}\n",
        encoding="ascii",
    )
    return reference


def _attach_completion_contract(
    evidence: tuple[IndustrialBlockEvidence, ...],
    *,
    cell_ids: frozenset[str],
    completion_contract: BoundArtifact,
) -> tuple[IndustrialBlockEvidence, ...]:
    return tuple(
        replace(
            block,
            cells=tuple(
                replace(
                    cell,
                    completion_contract=completion_contract,
                    diagnostic_lineage_identity=False,
                )
                if cell.cell_id in cell_ids
                else cell
                for cell in block.cells
            ),
        )
        for block in evidence
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
    project_root = Path(__file__).resolve().parents[1]
    project_source_tree = _project_tree(project_root)
    project_source_tree["dirty"] = False
    return {
        "schema_version": 2,
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
            "project": str(project_root),
            "patched_sglang": "/runtime/sglang",
            "distinct": True,
        },
        "project_source_tree": project_source_tree,
        "project_runtime_source": _project_runtime_source(project_root),
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
            scientific_role_for_cell(registry, cell): cell
            for cell in registry.cells_for("E3b")
            if cell.identity.block == block
            and cell.identity.context == 4096
            and cell.identity.regime == "long_input_short_output"
            and cell.identity.arrival == "closed_loop_c1"
            and ":matched:role=" in (cell.identity.variant or "")
        }
        assert set(rows) == set(CONFIRMATION_METHOD_ROLES)
        selected[block] = rows
    return selected


def _goodput(
    block: int,
    scientific_role: str,
    *,
    lightcone_pilot_multipliers: tuple[float, float, float, float],
) -> float:
    if scientific_role == "target_only":
        return 90.0
    if scientific_role == "static":
        return 100.0
    if scientific_role == "tts":
        return 101.0
    if scientific_role == "l0_naive":
        return 102.0
    return (
        103.0 * lightcone_pilot_multipliers[block] if block in PILOT_BLOCKS else 104.0
    )


def _build_evidence(
    tmp_path: Path,
    *,
    final_block_count: int = 12,
    lightcone_pilot_multipliers: tuple[float, float, float, float] = (
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
    envelope = _hardware_envelope()
    block_cells = _slice_cells(registry, final_block_count=final_block_count)
    inventory = _gpu_inventory()
    identities_by_block: dict[int, dict[str, str]] = {}
    budgets: dict[str, ExperimentBudget] = {}
    assignments: dict[str, IndustrialPhysicalAssignment] = {}
    locked_cells: dict[str, dict[str, object]] = {}
    cell_index = 0
    for block, methods in block_cells.items():
        request_id = f"request-{block}"
        identities = {
            "corpus_sha256": content_sha256({"corpus": block}),
            "arrival_trace_sha256": content_sha256({"trace": block}),
            "request_ids_sha256": content_sha256([request_id]),
            "sampling_profile_sha256": content_sha256({"sampling": "greedy"}),
            "model_lock_sha256": content_sha256({"model": "qwen3-8b"}),
        }
        identities_by_block[block] = identities
        for scientific_role in CONFIRMATION_METHOD_ROLES:
            cell = methods[scientific_role]
            budget = _execution_budget(cell)
            physical_gpu_uuids = tuple(
                _PHYSICAL_GPU_UUIDS[registry.gpu_uuids.index(logical_slot)]
                for logical_slot in cell.resources.gpu_uuids
            )
            assignment = _analysis_physical_assignment(
                cell,
                budget=budget,
                inventory=inventory,
                physical_gpu_uuids=physical_gpu_uuids,
                cell_index=cell_index,
            )
            budgets[cell.cell_id] = budget
            assignments[cell.cell_id] = assignment
            locked_cells[cell.cell_id] = _analysis_locked_cell_contract(
                cell,
                request_id=request_id,
                identities=identities,
                budget=budget,
                assignment=assignment,
            )
            cell_index += 1
    split_contract: dict[str, object] = {
        "schema_version": 1,
        "kind": "industrial_locked_split",
        "registry_sha256": registry.sha256,
        "experiment": "E3b",
        "cells": [locked_cells[cell_id] for cell_id in sorted(locked_cells)],
    }
    split_sha256 = content_sha256(split_contract)
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
    completion_rows: dict[str, dict[str, object]] = {}

    for block, methods in block_cells.items():
        request_id = f"request-{block}"
        identities = identities_by_block[block]
        cell_evidence: list[IndustrialCellEvidence] = []
        for scientific_role in CONFIRMATION_METHOD_ROLES:
            cell = methods[scientific_role]
            method = cell.identity.method
            budget = budgets[cell.cell_id]
            assignment = assignments[cell.cell_id]
            contract = locked_cells[cell.cell_id]
            physical_gpu_uuids = assignment.gpu_uuids
            run_id = f"analysis-{block}-{scientific_role.replace('_', '-')}"
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
            rank_config_sha256 = str(contract["rank_config_sha256s"][0])
            writer = EvidenceWriter(
                evidence_root,
                run_id=run_id,
                rank=0,
                process_id=(
                    block * 10 + CONFIRMATION_METHOD_ROLES.index(scientific_role) + 1
                ),
                registered_policy=DEFAULT_EVIDENCE_WRITER_POLICY,
            )
            native_artifact_binding = _persist_native_terminal_artifact(
                writer,
                method=method,
                run_nonce_sha256=run_nonce_sha256,
                execution_plan_sha256=str(contract["execution_plan_sha256"]),
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
                    runtime_sha256=str(contract["execution_plan_sha256"]),
                    split_sha256=str(contract["execution_split_sha256"]),
                    **identities,
                    patched_sglang_tree=PINNED_SGLANG_TREE,
                    run_nonce_sha256=run_nonce_sha256,
                    topology_sha256=content_sha256(
                        {
                            "schema_version": 1,
                            "cell_id": cell.cell_id,
                            "topology": cell.identity.topology,
                            "gpu_uuids": list(physical_gpu_uuids),
                            "tensor_parallel_size": assignment.tensor_parallel_size,
                            "data_parallel_size": assignment.data_parallel_size,
                            "world_size": len(assignment.gpu_uuids),
                        }
                    ),
                    tensor_parallel_size=assignment.tensor_parallel_size,
                    data_parallel_size=assignment.data_parallel_size,
                    world_size=len(assignment.gpu_uuids),
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
                    scientific_role,
                    lightcone_pilot_multipliers=lightcone_pilot_multipliers,
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
            budget_observation_value = _budget_observation_value(
                budget,
                terminal_receipt_sha256=prepared_receipt_sha256,
            )
            budget_observation = _write_budget_observation(
                evidence_root
                / f"{run_id}.rank0.budget-observation"
                / "observation.json",
                budget_observation_value,
            )
            completed_evidence = writer.publish_close(
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
                    "runtime_sha256": contract["execution_plan_sha256"],
                    "split_sha256": contract["execution_split_sha256"],
                    "cell_id": cell.cell_id,
                    "block": block,
                    "topology_sha256": content_sha256(
                        {
                            "schema_version": 1,
                            "cell_id": cell.cell_id,
                            "topology": cell.identity.topology,
                            "gpu_uuids": list(physical_gpu_uuids),
                            "tensor_parallel_size": assignment.tensor_parallel_size,
                            "data_parallel_size": assignment.data_parallel_size,
                            "world_size": len(assignment.gpu_uuids),
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
            completion_rows[cell.cell_id] = {
                "cell_id": cell.cell_id,
                "evidence_root": cell.resources.evidence_root,
                "run_id": run_id,
                "rank": 0,
                "evidence_sha256": evidence_files_sha256(completed_evidence.values()),
                "terminal_receipt_sha256": terminal.sha256,
                "physical_gpu_uuid": physical_gpu_uuids[0],
                "physical_binding_sha256": assignment.sha256,
                "experiment_budget_sha256": budget.sha256,
                "budget_observation_status": "OBSERVED",
                "budget_observation_reason_code": None,
                "budget_observation_path": str(budget_observation.path),
                "budget_observation_sha256": budget_observation_value[
                    "budget_observation_sha256"
                ],
                "preflight_attestation_path": None,
                "preflight_attestation_sha256": None,
                "status": "MEASURED",
            }
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
    pilot_cell_ids = frozenset(pilot_activation.activated_cell_ids)
    pilot_completion = _write_analysis_completion_contract(
        tmp_path / "completion" / "pilot-completed.json",
        registry=registry,
        runtime_sha256=runtime_sha256,
        split_contract=split_contract,
        inventory=inventory,
        activation_binding=_analysis_activation_binding(
            activation_round="excluded_pilots",
            activations=(pilot_activation,),
            reductions=(),
        ),
        rows=tuple(completion_rows[cell_id] for cell_id in sorted(pilot_cell_ids)),
    )
    evidence = _attach_completion_contract(
        evidence,
        cell_ids=pilot_cell_ids,
        completion_contract=pilot_completion,
    )
    plan = reduce_confirmation_family_power(
        registry=registry,
        pilot_activation=pilot_activation,
        blocks=tuple(block for block in evidence if block.block in PILOT_BLOCKS),
        hardware_envelope=envelope,
        inventory=inventory,
        confirmation_data_visible=False,
    )
    assert plan.selected_final_blocks == (final_block_count or None)
    final_activation = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=plan,
        pilot_activation=pilot_activation,
    )
    final_cell_ids = frozenset(final_activation.activated_cell_ids)
    if final_cell_ids:
        final_completion = _write_analysis_completion_contract(
            tmp_path / "completion" / "final-completed.json",
            registry=registry,
            runtime_sha256=runtime_sha256,
            split_contract=split_contract,
            inventory=inventory,
            activation_binding=_analysis_activation_binding(
                activation_round="final_prefix",
                activations=(pilot_activation, final_activation),
                reductions=(plan,),
            ),
            rows=tuple(completion_rows[cell_id] for cell_id in sorted(final_cell_ids)),
        )
        evidence = _attach_completion_contract(
            evidence,
            cell_ids=final_cell_ids,
            completion_contract=final_completion,
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
        if scientific_role_for_cell(registry, cell) == "lc_candidate"
        and "halving_stage=0:" in cell.identity.variant
    )
    runtime_sha256 = content_sha256({"runtime": "e2-raw-test"})
    split_sha256 = content_sha256({"split": "e2-raw-test"})
    pareto = E1ParetoArtifact(
        schema_version=2,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        e1_activation_sha256=content_sha256({"e1": "activation"}),
        reducer_evidence_sha256=content_sha256({"e1": "raw-evidence"}),
        surviving_geometries=(
            E1GeometryIdentity.from_cell(seed_cell, registry=registry),
        ),
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
            LockedOutput(
                "common_downstream_load",
                content_sha256({"e1": "untrusted-common-load"}),
            ),
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
    runtime_metrics_manifest: dict[str, str] | None = None,
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
        assert all(cell.completion_contract is not None for cell in block.cells)
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
                    "completion_contract": _bound_reference(cell.completion_contract),
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
    if runtime_metrics_manifest is not None:
        manifest["runtime_metrics_manifest"] = runtime_metrics_manifest
    manifest_path = tmp_path / f"{name}-manifest.json"
    _write_bound_json(manifest_path, manifest)
    return manifest_path


def test_default_recipe_authorities_block_formal_five_role_reduction() -> None:
    registry = build_industrial_registry()
    rows = {
        scientific_role_for_cell(registry, cell): cell
        for cell in registry.cells_for("E3b")
        if cell.identity.block == PILOT_BLOCKS[0]
        and cell.identity.context == 4096
        and cell.identity.regime == "long_input_short_output"
        and cell.identity.arrival == "closed_loop_c1"
        and ":matched:" in (cell.identity.variant or "")
    }
    assert tuple(sorted(rows)) == (
        "l0_naive",
        "lightcone_template",
        "static",
        "target_only",
        "tts",
    )
    assert rows["target_only"].runnable
    assert rows["static"].runnable
    assert not rows["tts"].runnable
    assert not rows["l0_naive"].runnable
    assert not rows["lightcone_template"].runnable
    assert rows["tts"].reason_code == "tts_official_recipe_unavailable"
    assert rows["l0_naive"].reason_code == "tts_official_recipe_unavailable"
    assert rows["lightcone_template"].reason_code == "sealed_e2_recipe_receipt_required"


def test_typed_method_reductions_cover_exactly_five_scientific_roles() -> None:
    slo = account_slo(
        (
            SloRequest(
                request_id="request",
                prompt_bucket="short",
                eligible=True,
                completed=True,
                error=False,
                ttft_ms=1.0,
                within_request_p99_itl_ms=1.0,
            ),
        )
    )
    p99 = guard_p99_claim(
        "anchor",
        completed_requests=1,
        observed_p99_ms=None,
        minimum_completions=10_000,
        preregistered_anchor_locked=False,
    )
    reductions = tuple(
        MethodReduction(
            method=role,
            block_ids=("final-0", "final-1"),
            mean_output_goodput_tps=100.0,
            mean_slo_qualified_goodput_tps=99.0,
            slo=slo,
            aggregate_latency_p99=p99,
        )
        for role in CONFIRMATION_METHOD_ROLES
    )
    assert tuple(row.method for row in reductions) == CONFIRMATION_METHOD_ROLES


def test_e2_confidence_pareto_preserves_two_fixed_reference_contrasts() -> None:
    def evaluation(
        label: str,
        *,
        tts_lower: float,
        static_lower: float,
        hbm_bytes: int,
    ) -> E2CandidateEvaluation:
        return E2CandidateEvaluation(
            candidate_id=content_sha256({"candidate": label}),
            evidence_sha256=content_sha256({"evidence": label}),
            safety_passed=True,
            confidence_pareto=False,
            lc_vs_tts_goodput_ratio=tts_lower,
            lc_vs_tts_confidence_lower_goodput_ratio=tts_lower,
            lc_vs_static_goodput_ratio=static_lower,
            lc_vs_static_confidence_lower_goodput_ratio=static_lower,
            hbm_bytes=hbm_bytes,
            p99_itl_us=1_000,
            exposed_update_us=1_000,
            minimum_launched_updates=1,
            minimum_published_updates=1,
            safety_reason_codes=(),
        )

    rows = _mark_e2_confidence_pareto(
        (
            evaluation("tts-strong", tts_lower=1.10, static_lower=1.01, hbm_bytes=10),
            evaluation(
                "static-strong", tts_lower=1.01, static_lower=1.10, hbm_bytes=10
            ),
            evaluation("dominated", tts_lower=1.00, static_lower=1.00, hbm_bytes=20),
        )
    )
    by_id = {row.candidate_id: row for row in rows}
    assert by_id[content_sha256({"candidate": "tts-strong"})].confidence_pareto
    assert by_id[content_sha256({"candidate": "static-strong"})].confidence_pareto
    assert not by_id[content_sha256({"candidate": "dominated"})].confidence_pareto


@pytest.fixture(scope="module")
def evidence_bundle(tmp_path_factory: pytest.TempPathFactory):
    registry = build_industrial_registry()
    formal_roles = {
        scientific_role_for_cell(registry, cell): cell
        for cell in registry.cells_for("E3b")
        if cell.identity.block == PILOT_BLOCKS[0]
        and cell.identity.context == 4096
        and cell.identity.regime == "long_input_short_output"
        and cell.identity.arrival == "closed_loop_c1"
        and ":matched:" in (cell.identity.variant or "")
    }
    if any(not formal_roles[role].runnable for role in CONFIRMATION_METHOD_ROLES):
        pytest.skip(
            "formal industrial raw evidence remains BLOCKED until the frozen TTS "
            "recipe and sealed LightCone receipt authorities exist"
        )
    return _build_evidence(tmp_path_factory.mktemp("industrial-analysis"))


def _e3b_raw_family_inputs_from_template(
    evidence_bundle,
) -> tuple[ExperimentRegistry, tuple[E3bLongContextRawFamilyInput, ...]]:
    registry, _, _, template, blocks, _ = evidence_bundle
    result: list[E3bLongContextRawFamilyInput] = []
    for context in E3B_CONTEXT_GRID:
        family = replace(template.family, context=context)
        pilot = materialize_confirmation_pilots(registry, family)
        power_sizing = replace(
            template.plan.power_sizing,
            pilot_block_ids=tuple(
                family_pilot_block_id(family, block) for block in PILOT_BLOCKS
            ),
        )
        pilot_cells = {
            (cell.identity.block, scientific_role_for_cell(registry, cell)): cell
            for cell in registry.cells
            if cell.cell_id in set(pilot.activated_cell_ids)
        }
        run_bindings = tuple(
            sorted(
                (
                    replace(
                        binding,
                        cell_id=pilot_cells[
                            (
                                int(binding.scientific_unit.rsplit("_", 1)[1]),
                                binding.scientific_role,
                            )
                        ].cell_id,
                        config_sha256=pilot_cells[
                            (
                                int(binding.scientific_unit.rsplit("_", 1)[1]),
                                binding.scientific_role,
                            )
                        ].cell_id,
                    )
                    for binding in template.run_bindings
                ),
                key=lambda binding: binding.cell_id,
            )
        )
        power_plan = replace(
            template.plan,
            family=family,
            pilot_activation_sha256=pilot.sha256,
            completed_pilot_cells_sha256=content_sha256(
                tuple(sorted(pilot.activated_cell_ids))
            ),
            power_sizing=power_sizing,
        )
        reduction = replace(template, plan=power_plan, run_bindings=run_bindings)
        final = materialize_confirmation_prefix(
            registry,
            family=family,
            reduction=reduction,
            pilot_activation=pilot,
        )
        result.append(
            E3bLongContextRawFamilyInput(
                pilot_activation=pilot,
                final_activation=final,
                confirmation_reduction=reduction,
                blocks=blocks,
            )
        )
    return registry, tuple(result)


def _synthetic_e3b_loaded_reduction(
    *,
    registry: ExperimentRegistry,
    raw: E3bLongContextRawFamilyInput,
    missing_request_role: str | None = None,
    missing_round_role: str | None = None,
    uses_evidence_dependence_units: bool = False,
) -> IndustrialReduction:
    by_id = {cell.cell_id: cell for cell in registry.cells}
    activated = tuple(
        by_id[cell_id] for cell_id in raw.final_activation.activated_cell_ids
    )
    blocks: list[_BlockReduction] = []
    goodput_by_role = {
        "target_only": 90.0,
        "static": 100.0,
        "tts": 105.0,
        "l0_naive": 107.0,
        "lightcone": 110.0,
    }
    for block in raw.confirmation_reduction.selected_final_prefix:
        loaded: dict[str, _LoadedCell] = {}
        for scientific_role in CONFIRMATION_METHOD_ROLES:
            cell = next(
                value
                for value in activated
                if value.identity.block == block
                and scientific_role_for_cell(registry, value) == scientific_role
            )
            request_id = f"e3b-{raw.confirmation_reduction.family.context}-{block}"
            arrival_ns = 1_000_000
            completed_ns = arrival_ns + round(
                100 / goodput_by_role[scientific_role] * 1_000_000_000
            )
            request_rows = (
                ()
                if missing_request_role == scientific_role
                else (
                    {
                        "request_id": request_id,
                        "output_tokens": 100,
                        "finished": True,
                        "outcome_status": "completed",
                        "arrival_ns": arrival_ns,
                        "completed_ns": completed_ns,
                        "first_token_ns": None,
                        "ttft_ms": None,
                        "token_timestamps_ns": None,
                        "inter_token_ms": None,
                    },
                )
            )
            rounds = (
                (
                    {
                        "request_id": request_id,
                        "round_index": 0,
                        "accepted_drafts": {
                            "tts": 6,
                            "l0_naive": 7,
                            "lightcone": 8,
                        }[scientific_role],
                        "target_calls": 1,
                    },
                )
                if scientific_role in {"tts", "l0_naive", "lightcone"}
                and scientific_role != missing_round_role
                else ()
            )
            digest = content_sha256(
                {
                    "context": raw.confirmation_reduction.family.context,
                    "block": block,
                    "scientific_role": scientific_role,
                }
            )
            loaded[scientific_role] = _LoadedCell(
                cell=cell,
                observation_source_cell_id=cell.cell_id,
                evidence_alias_reduction_sha256=None,
                run_rows=(),
                request_rows=request_rows,
                performance_rows_by_rank=(),
                update_rows_by_rank=(),
                terminal_receipt_sha256s=(digest,),
                hardware_receipt_sha256=digest,
                physical_gpu_uuids=("GPU-analysis-a",),
                experiment_budget_sha256=digest,
                inventory_sha256=digest,
                inventory_source_receipt_sha256=digest,
                fixed_instance_gpu_count=2,
                physical_host_id="analysis-host",
                budget_observation_sha256=digest,
                hardware_validity=(),
                itl_timestamp_authority_path=None,
                round_rows_by_rank=(rounds,),
            )
        blocks.append(
            _BlockReduction(
                block=block,
                qualification_sha256=content_sha256({"qualification": block}),
                cells=MappingProxyType(loaded),
                request_metrics=MappingProxyType({}),
                goodput_tps=MappingProxyType({}),
                slo_goodput_tps=MappingProxyType({}),
                slo_requests=MappingProxyType({}),
            )
        )
    return IndustrialReduction(
        artifact=SimpleNamespace(reasons=("gpu_attestation:missing",)),
        _request_metrics=MappingProxyType({}),
        _uses_evidence_dependence_units=uses_evidence_dependence_units,
        _loaded_blocks=tuple(blocks),
    )


def _replace_e3b_raw_family_identity(
    raw: E3bLongContextRawFamilyInput,
    *,
    family,
) -> E3bLongContextRawFamilyInput:
    pilot = replace(raw.pilot_activation, family=family)
    power_sizing = replace(
        raw.confirmation_reduction.plan.power_sizing,
        pilot_block_ids=tuple(
            family_pilot_block_id(family, block) for block in PILOT_BLOCKS
        ),
    )
    power_plan = replace(
        raw.confirmation_reduction.plan,
        family=family,
        pilot_activation_sha256=pilot.sha256,
        power_sizing=power_sizing,
    )
    reduction = replace(raw.confirmation_reduction, plan=power_plan)
    final = replace(
        raw.final_activation,
        family=family,
        power_plan_sha256=power_plan.sha256,
    )
    return replace(
        raw,
        pilot_activation=pilot,
        final_activation=final,
        confirmation_reduction=reduction,
    )


def test_e3b_raw_stage_reopens_exact_eight_contexts_and_preserves_evidence_level(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)
    calls: list[str] = []

    def reopen(**kwargs):
        raw = next(
            value
            for value in families
            if value.confirmation_reduction == kwargs["confirmation_reduction"]
        )
        assert kwargs["blocks"] is raw.blocks
        calls.append(raw.confirmation_reduction.family.sha256)
        return _synthetic_e3b_loaded_reduction(registry=registry, raw=raw)

    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        reopen,
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=families,
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
        bootstrap_seed=17,
    )

    assert tuple(sorted(calls)) == tuple(
        sorted(value.confirmation_reduction.family.sha256 for value in families)
    )
    assert (
        tuple(sorted(value.confirmation_reduction.family.context for value in families))
        == E3B_CONTEXT_GRID
    )
    assert artifact.status == "UNRESOLVED"
    assert artifact.evidence_level == "RAW_DIAGNOSTIC_OBSERVED_UNATTESTED"
    assert artifact.final_block_ids == FINAL_BLOCKS[:12]
    reductions = {value.name: value.reduction for value in artifact.reductions}
    for name in (
        "committed_token_goodput:lightcone:tts",
        "committed_token_goodput:lightcone:static",
        "committed_token_goodput:l0_naive:tts",
        "committed_token_goodput:lightcone:l0_naive",
        "accepted_length:lightcone:tts",
        "accepted_length:l0_naive:tts",
        "accepted_length:lightcone:l0_naive",
    ):
        assert reductions[name].status is E3bReductionStatus.OBSERVED
    static_accepted = reductions["accepted_length:lightcone:static"]
    assert static_accepted.status is E3bReductionStatus.UNRESOLVED
    assert static_accepted.reason_code == "e3b_adapted_round_source_missing_or_invalid"
    assert static_accepted.curve_points is None
    assert (
        "observations"
        not in inspect.signature(reduce_e3b_long_context_from_raw).parameters
    )
    assert artifact.sha256 == content_sha256(artifact.to_dict())
    with pytest.raises(ValueError, match="swaps its registered plan"):
        replace(
            artifact,
            reductions=(
                replace(
                    artifact.reductions[0], reduction=artifact.reductions[1].reduction
                ),
                *artifact.reductions[1:],
            ),
        )


def test_e3b_raw_stage_missing_request_is_named_unresolved(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)

    def reopen(**kwargs):
        raw = next(
            value
            for value in families
            if value.confirmation_reduction == kwargs["confirmation_reduction"]
        )
        return _synthetic_e3b_loaded_reduction(
            registry=registry,
            raw=raw,
            missing_request_role=(
                "lightcone"
                if raw.confirmation_reduction.family.context == 1024
                else None
            ),
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        reopen,
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=families,
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    reductions = {value.name: value.reduction for value in artifact.reductions}
    for baseline in ("tts", "static", "l0_naive"):
        row = reductions[f"committed_token_goodput:lightcone:{baseline}"]
        assert row.status is E3bReductionStatus.UNRESOLVED
        assert row.reason_code == "e3b_raw_request_source_missing"
        assert row.curve_points is None


def test_e3b_raw_stage_missing_terminal_timestamp_is_typed_stage_unresolved(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)

    def reopen(**kwargs):
        raw = next(
            value
            for value in families
            if value.confirmation_reduction == kwargs["confirmation_reduction"]
        )
        if raw.confirmation_reduction.family.context == 1024:
            raise _RequestTerminalTimingUnavailable
        return _synthetic_e3b_loaded_reduction(registry=registry, raw=raw)

    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        reopen,
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=families,
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    assert artifact.evidence_level == "RAW_UNRESOLVED"
    assert artifact.reductions == ()
    assert artifact.reasons == (
        "context-1024:e3b_goodput_terminal_timestamp_unavailable",
    )


def test_e3b_raw_stage_missing_adapted_round_is_named_unresolved(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)

    def reopen(**kwargs):
        raw = next(
            value
            for value in families
            if value.confirmation_reduction == kwargs["confirmation_reduction"]
        )
        return _synthetic_e3b_loaded_reduction(
            registry=registry,
            raw=raw,
            missing_round_role=(
                "lightcone"
                if raw.confirmation_reduction.family.context == 1024
                else None
            ),
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        reopen,
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=families,
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    reductions = {value.name: value.reduction for value in artifact.reductions}
    accepted = reductions["accepted_length:lightcone:tts"]
    assert accepted.status is E3bReductionStatus.UNRESOLVED
    assert accepted.reason_code == "e3b_adapted_round_source_missing_or_invalid"
    assert accepted.curve_points is None
    assert reductions["committed_token_goodput:lightcone:static"].status is (
        E3bReductionStatus.OBSERVED
    )


def test_e3b_raw_stage_propagates_bound_request_tamper(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)
    target = next(
        value
        for value in families
        if value.confirmation_reduction.family.context == 4096
    )
    first_block = target.blocks[0]
    first_cell = first_block.cells[0]
    first_terminal = first_cell.terminal_receipts[0]
    tampered_terminal = replace(first_terminal, sha256="0" * 64)
    tampered_cell = replace(
        first_cell,
        terminal_receipts=(tampered_terminal, *first_cell.terminal_receipts[1:]),
    )
    tampered_block = replace(
        first_block,
        cells=(tampered_cell, *first_block.cells[1:]),
    )
    tampered_target = replace(
        target,
        blocks=(tampered_block, *target.blocks[1:]),
    )
    tampered_families = tuple(
        tampered_target if value is target else value for value in families
    )
    real_reducer = reduce_industrial_schema_v3

    def reopen(**kwargs):
        raw = next(
            value
            for value in tampered_families
            if value.confirmation_reduction == kwargs["confirmation_reduction"]
        )
        if raw.confirmation_reduction.family.context == 4096:
            return real_reducer(**kwargs)
        return _synthetic_e3b_loaded_reduction(registry=registry, raw=raw)

    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        reopen,
    )
    with pytest.raises(ValueError, match="completion contract differs"):
        reduce_e3b_long_context_from_raw(
            registry=registry,
            families=tampered_families,
            hardware_envelope=evidence_bundle[-1],
            inventory=_gpu_inventory(),
            bootstrap_repetitions=100,
        )


def test_e3b_raw_stage_rejects_prefix_mismatch_before_raw_reopen(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)
    target = families[-1]
    power_sizing = replace(
        target.confirmation_reduction.plan.power_sizing,
        selected_final_blocks=13,
        power_grid=tuple(
            replace(cell, power=0.0) if cell.final_blocks == 12 else cell
            for cell in target.confirmation_reduction.plan.power_sizing.power_grid
        ),
    )
    plan = replace(
        target.confirmation_reduction.plan,
        power_sizing=power_sizing,
        selected_final_blocks=13,
        selected_final_prefix=FINAL_BLOCKS[:13],
    )
    mismatched = replace(
        target,
        confirmation_reduction=replace(target.confirmation_reduction, plan=plan),
    )
    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        lambda **_: pytest.fail("prefix mismatch reached raw evidence"),
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=(*families[:-1], mismatched),
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    assert artifact.reasons == ("e3b_context_families_use_different_final_prefixes",)
    assert artifact.final_block_ids is None
    assert artifact.reductions == ()


def test_e3b_raw_stage_rejects_cross_block_alias_dependence(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)
    target = families[0]
    dependence = _singleton_dependence_map(
        pilot_activation=target.pilot_activation,
        final_activation=target.final_activation,
    )
    by_id = {cell.cell_id: cell for cell in registry.cells}
    final_lightcone = tuple(
        cell_id
        for cell_id in target.final_activation.activated_cell_ids
        if scientific_role_for_cell(registry, by_id[cell_id]) == "lightcone"
    )[:2]
    dependence = _replace_with_unverified_alias(
        dependence,
        source_cell_id=final_lightcone[0],
        target_cell_id=final_lightcone[1],
    )
    dependent = replace(target, evidence_dependence_map=dependence)
    dependent_families = (dependent, *families[1:])

    def reopen(**kwargs):
        raw = next(
            value
            for value in dependent_families
            if value.confirmation_reduction == kwargs["confirmation_reduction"]
        )
        return _synthetic_e3b_loaded_reduction(
            registry=registry,
            raw=raw,
            uses_evidence_dependence_units=(raw is dependent),
        )

    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        reopen,
    )
    artifact = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=dependent_families,
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    assert artifact.reductions == ()
    assert artifact.evidence_level == "RAW_UNRESOLVED"
    assert artifact.reasons == ("context-1024:e3b_cross_block_evidence_dependence",)


def test_e3b_context_axis_and_coverage_fail_closed_before_raw_reopen(
    evidence_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, families = _e3b_raw_family_inputs_from_template(evidence_bundle)
    monkeypatch.setattr(
        "lightcone_spec.experiments.industrial_analysis.reduce_industrial_schema_v3",
        lambda **_: pytest.fail("invalid context identity reached raw evidence"),
    )
    missing = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=families[:-1],
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    assert missing.evidence_level == "RAW_UNRESOLVED"
    assert missing.reasons == ("e3b_registered_context_family_coverage_incomplete",)

    cross_axis_family = replace(
        families[-1].confirmation_reduction.family, width_panel="deployment_optimal"
    )
    cross_axis = _replace_e3b_raw_family_identity(
        families[-1], family=cross_axis_family
    )
    rejected = reduce_e3b_long_context_from_raw(
        registry=registry,
        families=(*families[:-1], cross_axis),
        hardware_envelope=evidence_bundle[-1],
        inventory=_gpu_inventory(),
        bootstrap_repetitions=100,
    )
    assert rejected.reasons == ("e3b_context_families_cross_a_registered_axis",)
    assert rejected.reductions == ()


def test_analyze_e3b_long_context_cli_exports_typed_incomplete_raw_stage(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot, final, reduction, evidence, envelope = evidence_bundle
    family_manifest = _analysis_manifest(
        tmp_path,
        registry=registry,
        pilot_activation=pilot,
        final_activation=final,
        reduction=reduction,
        evidence=evidence,
        envelope=envelope,
        name="e3b-long-context-single-family",
    )
    family_value = json.loads(family_manifest.read_text(encoding="utf-8"))
    stage_value = {
        "schema_version": 1,
        "kind": "industrial_e3b_long_context_analysis_manifest",
        "family_manifests": [
            {
                "path": str(family_manifest),
                "sha256": content_sha256(family_value),
            }
        ],
        "bootstrap": {"repetitions": 300, "seed": 17},
    }
    manifest = tmp_path / "e3b-long-context-manifest.json"
    _write_bound_json(manifest, stage_value)
    output = tmp_path / "e3b-long-context-reducer.json"

    assert (
        main(
            [
                "analyze-e3b-long-context",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "UNRESOLVED"
    assert artifact["evidence_level"] == "RAW_UNRESOLVED"
    assert artifact["reasons"] == ["e3b_registered_context_family_coverage_incomplete"]
    assert artifact["reductions"] == []
    assert len(artifact["raw_family_input_sha256s"]) == 1

    tampered = json.loads(json.dumps(stage_value))
    tampered["family_manifests"][0]["sha256"] = "0" * 64
    tampered_path = tmp_path / "e3b-long-context-tampered.json"
    _write_bound_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        main(
            [
                "analyze-e3b-long-context",
                "--manifest",
                str(tampered_path),
                "--output",
                str(tmp_path / "must-not-exist.json"),
            ]
        )


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
    registry = build_industrial_registry()
    registry_cell = next(
        cell
        for cell in registry.cells_for("E3b")
        if cell.runnable
        and cell.identity.method == "target_only"
        and cell.resources.gpu_count == 1
        and cell.identity.topology == "tp1_dp1"
    )
    budget = replace(
        _zero_budget(registry_cell.cell_id),
        workload_class=registry_cell.resources.workload_class,
    )
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
            cell=registry_cell,
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
            cell=next(row for row in registry.cells if row.cell_id == cell.cell_id),
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
            cell=registry_cell,
            experiment_budget_sha256=coordinated_budget.sha256,
            terminal_receipt_sha256=value["terminal_evidence_sha256"],
            fixed_instance_gpu_count=2,
        )


def test_budget_observation_rejects_foreign_registry_cell(
    evidence_bundle,
) -> None:
    registry, _, _, _, evidence, _ = evidence_bundle
    reference = evidence[0].cells[0]
    value = json.loads(reference.budget_observation.path.read_text(encoding="utf-8"))
    foreign = next(
        cell
        for cell in registry.cells_for("E3b")
        if cell.runnable and cell.cell_id != reference.cell_id
    )
    with pytest.raises(ValueError, match="differs from its registry cell"):
        _load_budget_observation(
            reference.budget_observation,
            cell=foreign,
            experiment_budget_sha256=value["experiment_budget_sha256"],
            terminal_receipt_sha256=value["terminal_evidence_sha256"],
            fixed_instance_gpu_count=2,
        )


def test_p99_analysis_requires_one_locked_preregistered_anchor() -> None:
    registry = build_industrial_registry()
    cell = next(
        row
        for row in registry.cells_for("E3b")
        if row.runnable and row.identity.method == "target_only"
    )
    minimum = 12_000
    locked = replace(
        _execution_budget(cell),
        job_kind=BudgetJobKind.P99_ANCHOR,
        minimum_completed_requests=minimum,
        p99_anchor_status=P99AnchorStatus.LOCKED,
    )
    completed = (87.0,) * minimum
    claimable = _guard_preregistered_p99_analysis(
        family_experiment="E3b",
        method="target_only",
        analysis_budgets=(locked,),
        independent_observations=((locked, completed),),
    )
    assert claimable.claimable
    assert claimable.minimum_completions == minimum
    assert claimable.completed_requests == minimum
    assert claimable.observed_p99_ms == 87.0

    below_floor = _guard_preregistered_p99_analysis(
        family_experiment="E3b",
        method="target_only",
        analysis_budgets=(locked,),
        independent_observations=((locked, completed[:-1]),),
    )
    assert below_floor.status == "UNRESOLVED"
    assert below_floor.observed_p99_ms is None

    standard = replace(
        locked,
        job_kind=BudgetJobKind.STANDARD,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
    )
    standard_result = _guard_preregistered_p99_analysis(
        family_experiment="E3b",
        method="target_only",
        analysis_budgets=(standard,),
        independent_observations=((standard, completed),),
    )
    assert standard_result.status == "UNRESOLVED"
    assert standard_result.observed_p99_ms is None

    unresolved_anchor = replace(
        locked,
        p99_anchor_status=P99AnchorStatus.REQUIRED_UNRESOLVED,
    )
    unresolved_result = _guard_preregistered_p99_analysis(
        family_experiment="E3b",
        method="target_only",
        analysis_budgets=(unresolved_anchor,),
        independent_observations=((unresolved_anchor, completed),),
    )
    assert unresolved_result.status == "UNRESOLVED"
    assert unresolved_result.observed_p99_ms is None

    distinct_anchor = replace(
        locked,
        output_tokens=ExpectedMaximumCount(101, 101),
    )
    cross_anchor = _guard_preregistered_p99_analysis(
        family_experiment="E3b",
        method="target_only",
        analysis_budgets=(locked, distinct_anchor),
        independent_observations=(
            (locked, completed),
            (distinct_anchor, completed),
        ),
    )
    assert cross_anchor.status == "UNRESOLVED"
    assert cross_anchor.observed_p99_ms is None


def test_alias_analysis_uses_target_raw_budget_without_erasing_source() -> None:
    registry = build_industrial_registry()
    source_cell, target_cell = tuple(
        row
        for row in registry.cells_for("E3b")
        if row.runnable and row.identity.method == "target_only"
    )[:2]
    source_budget = _execution_budget(source_cell)
    target_budget = _execution_budget(target_cell)
    assert source_budget != target_budget
    analysis_budget = _alias_analysis_budget(
        observed_source_budget=source_budget,
        source_budget=source_budget,
        target_budget=target_budget,
    )
    assert analysis_budget is target_budget
    assert source_budget.cell_id == source_cell.cell_id
    assert analysis_budget.cell_id == target_cell.cell_id
    with pytest.raises(ValueError, match="source budget observation"):
        _alias_analysis_budget(
            observed_source_budget=target_budget,
            source_budget=source_budget,
            target_budget=target_budget,
        )


@pytest.mark.parametrize("diagnostic_lineage_identity", (False, True))
def test_formal_replay_rejects_missing_or_diagnostic_completion_identity(
    diagnostic_lineage_identity: bool,
    tmp_path: Path,
) -> None:
    reference = IndustrialCellEvidence(
        cell_id="0" * 64,
        terminal_receipts=(
            BoundArtifact(path=tmp_path / "terminal.json", sha256="1" * 64),
        ),
        hardware_receipt=BoundArtifact(
            path=tmp_path / "hardware.json",
            sha256="2" * 64,
        ),
        budget_observation=BoundArtifact(
            path=tmp_path / "budget.json",
            sha256="3" * 64,
        ),
        completion_contract=None,
        diagnostic_lineage_identity=diagnostic_lineage_identity,
    )
    with pytest.raises(
        ValueError,
        match="formal raw evidence requires its schema-v4 completion contract",
    ):
        _replay_cell_execution_identity(
            reference,
            registry=object(),  # type: ignore[arg-type]
            family=object(),  # type: ignore[arg-type]
            cell=object(),  # type: ignore[arg-type]
            inventory=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(
        ValueError,
        match="formal raw evidence requires its schema-v4 completion contract",
    ):
        _load_cell(
            reference,
            registry=object(),  # type: ignore[arg-type]
            family=object(),  # type: ignore[arg-type]
            cells_by_id={},
            envelope=object(),  # type: ignore[arg-type]
            inventory=object(),  # type: ignore[arg-type]
        )


def test_e2_stage_reducer_rebuilds_metrics_and_rejects_bare_prior(
    tmp_path: Path,
) -> None:
    pytest.skip(
        "E2 raw positive reduction awaits source-owned optimizer/stride/width values"
    )
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
        row.lc_vs_tts_goodput_ratio == pytest.approx(1.02)
        and row.lc_vs_tts_confidence_lower_goodput_ratio == pytest.approx(1.02)
        and row.lc_vs_static_goodput_ratio == pytest.approx(1.02)
        and row.lc_vs_static_confidence_lower_goodput_ratio == pytest.approx(1.02)
        and row.hbm_bytes == 1_000
        and row.p99_itl_us == 1_000
        and row.exposed_update_us == 1_000
        and row.minimum_launched_updates == 1
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


def test_e2_raw_stage_reducer_stops_at_unregistered_common_load_authority() -> None:
    registry = build_industrial_registry()
    seed_cell = next(
        cell
        for cell in registry.cells_for("E2")
        if scientific_role_for_cell(registry, cell) == "lc_candidate"
        and "halving_stage=0:" in cell.identity.variant
    )
    runtime_sha256 = content_sha256({"runtime": "e2-minima-block"})
    split_sha256 = content_sha256({"split": "e2-minima-block"})
    pareto = E1ParetoArtifact(
        schema_version=2,
        registry_sha256=registry.sha256,
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        e1_activation_sha256=content_sha256({"e1": "activation"}),
        reducer_evidence_sha256=content_sha256({"e1": "raw-evidence"}),
        surviving_geometries=(
            E1GeometryIdentity.from_cell(seed_cell, registry=registry),
        ),
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
            LockedOutput(
                "common_downstream_load",
                content_sha256({"e1": "untrusted-common-load"}),
            ),
            LockedOutput("dflash_pareto_set", pareto.sha256),
        ),
    )
    with pytest.raises(
        ValueError,
        match="E2 raw stage reduction is BLOCKED: e1_common_load_authority_unregistered",
    ):
        reduce_e2_stage_from_raw(
            registry=registry,
            e1_receipt=receipt,
            pareto=pareto,
            stage_index=0,
            cells=(),
            hardware_envelope=_hardware_envelope(),
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

    extended_pci_doctor = json.loads(json.dumps(doctor_value))
    for rows in (
        extended_pci_doctor["gpu"]["parsed_inventory"]["devices"],
        extended_pci_doctor["checks"]["gpu_identity"]["observed"],
    ):
        for row in rows:
            row["pci_bus_id"] = ("0000" + row["pci_bus_id"]).upper()
    _validate_industrial_doctor(
        _write_json(tmp_path / "doctor-extended-pci.json", extended_pci_doctor),
        inventory_authority=_gpu_inventory(),
    )

    aliased_pci_doctor = json.loads(json.dumps(extended_pci_doctor))
    for rows in (
        aliased_pci_doctor["gpu"]["parsed_inventory"]["devices"],
        aliased_pci_doctor["checks"]["gpu_identity"]["observed"],
    ):
        rows[1]["pci_bus_id"] = "0000:01:00.0"
    with pytest.raises(ValueError, match="complete GPU inventory"):
        _validate_industrial_doctor(
            _write_json(tmp_path / "doctor-aliased-pci.json", aliased_pci_doctor),
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


@pytest.mark.parametrize(
    "mutation",
    ("schema_v1", "missing_runtime_source", "tampered_runtime_source"),
)
def test_industrial_doctor_rejects_stale_project_source_identity(
    mutation: str,
    tmp_path: Path,
) -> None:
    inventory = _gpu_inventory()
    report = _passing_doctor(
        build_industrial_registry(),
        inventory_authority=inventory,
    )
    if mutation == "schema_v1":
        report["schema_version"] = 1
        message = "schema-v2 PASS doctor"
    elif mutation == "missing_runtime_source":
        del report["project_runtime_source"]
        message = "LightCone source identity is not exact"
    else:
        report["project_runtime_source"]["content_sha256"] = "c" * 64
        message = "LightCone source identity is not exact"
    with pytest.raises(ValueError, match=message):
        _validate_industrial_doctor(
            _write_json(tmp_path / f"doctor-{mutation}.json", report),
            inventory_authority=inventory,
        )


def test_industrial_doctor_reopens_current_project_cleanliness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inventory = _gpu_inventory()
    report = _passing_doctor(
        build_industrial_registry(),
        inventory_authority=inventory,
    )

    def dirty_project_tree(root: Path) -> dict[str, object]:
        value = _project_tree(root)
        if root.resolve() == PROJECT_ROOT:
            value["dirty"] = True
        return value

    monkeypatch.setattr("lightcone_spec.doctor._project_tree", dirty_project_tree)
    with pytest.raises(ValueError, match="LightCone source identity is not exact"):
        _validate_industrial_doctor(
            _write_json(tmp_path / "doctor-dirty-project.json", report),
            inventory_authority=inventory,
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
    execution_identity = _replay_cell_execution_identity(
        static_reference,
        registry=registry,
        family=plan.family,
        cell=cell,
        inventory=_gpu_inventory(),
    )
    with pytest.raises(ValueError, match="detail-table coverage"):
        _validate_run_row(
            claimed_trace,
            registry=registry,
            family=execution_identity,
            cell=cell,
            rank=0,
        )
    with pytest.raises(ValueError, match="experiment_budget_sha256"):
        _validate_run_row(
            {**run, "experiment_budget_sha256": None},
            registry=registry,
            family=execution_identity,
            cell=cell,
            rank=0,
        )
    with pytest.raises(ValueError, match="registry/runtime identity"):
        _validate_run_row(
            {**run, "workload_contract": "industrial_preflight_static"},
            registry=registry,
            family=execution_identity,
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
                family=execution_identity,
                cell=cell,
                rank=0,
            )
    with pytest.raises(ValueError, match="pre-mutation release"):
        _validate_run_row(
            {**run, "session_close_receipt_sha256": "4" * 64},
            registry=registry,
            family=execution_identity,
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
        "lightcone_vs_tts",
        "lightcone_vs_static",
    )
    assert tuple(row.name for row in artifact.secondary_contrasts) == (
        "l0_naive_vs_tts",
        "lightcone_vs_l0_naive",
    )
    assert all(len(row.block_ids) == 12 for row in artifact.primary_contrasts)
    slo_by_method = {row.method: row.slo for row in artifact.methods}
    assert not slo_by_method["tts"].passed
    assert all(
        slo_by_method[method].passed
        for method in ("target_only", "static", "l0_naive", "lightcone")
    )
    assert all(
        row.aggregate_latency_p99.status == "UNRESOLVED" for row in artifact.methods
    )
    assert all(
        row.aggregate_latency_p99.observed_p99_ms is None for row in artifact.methods
    )
    assert len(artifact.terminal_receipt_sha256s) == 16 * 5
    assert len(artifact.budget_observation_sha256s) == 16 * 5
    bound_physical = {
        gpu_uuid for binding in artifact.run_bindings for gpu_uuid in binding.gpu_uuids
    }
    assert bound_physical <= set(_PHYSICAL_GPU_UUIDS)
    assert bound_physical.isdisjoint(registry.gpu_uuids)
    assert artifact.to_dict()["kind"] == "industrial_schema_v3_reducer"

    hierarchical = reduction.hierarchical_block_request_bootstrap(
        "lightcone",
        "latency_ms",
        np.mean,
        repetitions=100,
        seed=9,
    )
    whole_time = reduction.whole_time_block_bootstrap(
        "lightcone",
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


@pytest.mark.skip(
    reason=(
        "formal family power cannot consume TTS/L0-naive/LightCone evidence until "
        "their recipe authorities are unblocked"
    )
)
def test_underpowered_family_uses_only_pilots_and_produces_no_final_analysis(
    tmp_path: Path,
) -> None:
    registry, pilots, final, plan, evidence, envelope = _build_evidence(
        tmp_path,
        final_block_count=0,
        lightcone_pilot_multipliers=(0.5, 1.5, 0.7, 1.3),
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
    assert reduction.artifact.secondary_contrasts == ()
    assert reduction.artifact.holm_family == ()
    assert len(reduction.artifact.run_bindings) == len(PILOT_BLOCKS) * len(
        CONFIRMATION_METHOD_ROLES
    )
    runtime = reduction.artifact.runtime_metrics
    assert runtime.status is RuntimeMetricStatus.UNRESOLVED
    assert runtime.authority_sha256 is None
    assert runtime.reduction_sha256 is None
    assert runtime.expected_run_ids == tuple(
        sorted(binding.run_id for binding in reduction.artifact.run_bindings)
    )
    assert runtime.observations
    assert all(
        row.status is RuntimeMetricStatus.UNRESOLVED for row in runtime.observations
    )
    assert all(row.value is None for row in runtime.observations)
    exported = reduction.artifact.to_dict()["evidence"]["runtime_metrics"]
    assert reduction.artifact.to_dict()["schema_version"] == 2
    assert (
        reduction.artifact.to_dict()["evidence"]["runtime_metrics_sha256"]
        == runtime.sha256
    )
    assert exported["status"] == "UNRESOLVED"
    assert all(row["value"] is None for row in exported["observations"])


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
    assert len(reduction._request_metrics["lightcone"]) == 12
    assert all(
        row.independent_unit == "evidence_dependence_component"
        and len(row.block_ids) == 12
        for row in (
            *reduction.artifact.primary_contrasts,
            *reduction.artifact.secondary_contrasts,
        )
    )
    interval = reduction.hierarchical_block_request_bootstrap(
        "lightcone",
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
    final_lightcone = tuple(
        cell_id
        for cell_id in final.activated_cell_ids
        if scientific_role_for_cell(registry, by_id[cell_id]) == "lightcone"
    )[:2]
    unverified_final_alias = _replace_with_unverified_alias(
        dependence_map,
        source_cell_id=final_lightcone[0],
        target_cell_id=final_lightcone[1],
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

    pilot_lightcone = tuple(
        cell_id
        for cell_id in pilots.activated_cell_ids
        if scientific_role_for_cell(registry, by_id[cell_id]) == "lightcone"
    )[:2]
    pilot_alias_map = _replace_with_unverified_alias(
        dependence_map,
        source_cell_id=pilot_lightcone[0],
        target_cell_id=pilot_lightcone[1],
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
    assert reduction.artifact.secondary_contrasts == ()
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
    runtime_metrics = artifact["evidence"]["runtime_metrics"]
    assert runtime_metrics["status"] == "UNRESOLVED"
    assert runtime_metrics["authority_sha256"] is None
    assert runtime_metrics["reduction_sha256"] is None
    assert runtime_metrics["source_sha256s"] == []
    assert runtime_metrics["observations"]
    assert all(
        row["status"] == "UNRESOLVED" and row["value"] is None
        for row in runtime_metrics["observations"]
    )
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


def test_analyze_industrial_cli_replays_path_bound_runtime_metrics(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, pilot_activation, final_activation, plan, evidence, envelope = (
        evidence_bundle
    )
    terminal_path = evidence[0].cells[0].terminal_receipts[0].path
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    native_binding = terminal.get("native_terminal_artifact")
    assert isinstance(native_binding, dict)
    native_name = native_binding.get("path")
    assert isinstance(native_name, str)
    native_path = terminal_path.parent / native_name
    runtime_source_manifest = {
        "schema_version": 1,
        "kind": "runtime_metrics_raw_source_manifest",
        "compile_sources": [],
        "fresh_process_sources": [],
        "native_sources": [{"artifact": str(native_path)}],
    }
    runtime_source_path = tmp_path / "runtime-metrics-raw.json"
    _write_bound_json(runtime_source_path, runtime_source_manifest)
    manifest_path = _analysis_manifest(
        tmp_path,
        registry=registry,
        pilot_activation=pilot_activation,
        final_activation=final_activation,
        reduction=plan,
        evidence=evidence,
        envelope=envelope,
        name="runtime-source",
        runtime_metrics_manifest={
            "path": str(runtime_source_path),
            "sha256": content_sha256(runtime_source_manifest),
        },
    )
    output = tmp_path / "runtime-source-reducer.json"

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
    runtime = artifact["evidence"]["runtime_metrics"]
    assert runtime["status"] == "UNRESOLVED"
    assert isinstance(runtime["authority_sha256"], str)
    assert isinstance(runtime["reduction_sha256"], str)
    assert len(runtime["source_sha256s"]) == 1
    untrusted = [
        row
        for row in runtime["observations"]
        if row["metric"] == "graph_replay_hit_rate"
        and row["source_kind"] == "native_terminal"
    ]
    assert len(untrusted) == 1
    assert untrusted[0]["status"] == "UNRESOLVED"
    assert untrusted[0]["value"] is None
    assert untrusted[0]["reason_code"] == ("release_trusted_runtime_source_required")
    assert untrusted[0]["release_trusted"] is False

    runtime_source_path.write_text(
        json.dumps(
            {**runtime_source_manifest, "schema_version": 2},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rejected_output = tmp_path / "tampered-runtime-must-not-exist.json"
    with pytest.raises(ValueError, match="sidecar"):
        main(
            [
                "analyze-industrial",
                "--manifest",
                str(manifest_path),
                "--output",
                str(rejected_output),
            ]
        )
    assert not rejected_output.exists()


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
    assert artifact["secondary_contrasts"] == []
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
    assert all(binding.schema_version == 2 for binding in reduced.run_bindings)
    assert all(
        binding.runtime_sha256 == reduced.family.runtime_sha256
        and binding.split_sha256 == reduced.family.split_sha256
        and binding.execution_plan_sha256 is not None
        and binding.execution_split_sha256 is not None
        for binding in reduced.run_bindings
    )
    assert any(
        binding.execution_plan_sha256 != binding.runtime_sha256
        or binding.execution_split_sha256 != binding.split_sha256
        for binding in reduced.run_bindings
    )

    old_mixed_domain = confirmation_family_power_reduction_artifact_to_dict(reduced)
    for binding in old_mixed_domain["run_bindings"]:
        del binding["execution_plan_sha256"]
        del binding["execution_split_sha256"]
    with pytest.raises(ValueError, match="fields differ"):
        confirmation_family_power_reduction_artifact_from_dict(old_mixed_domain)

    swapped_lineage = confirmation_family_power_reduction_artifact_to_dict(reduced)
    swapped_lineage["run_bindings"][0]["runtime_sha256"] = swapped_lineage[
        "run_bindings"
    ][0]["execution_plan_sha256"]
    swapped_lineage["run_bindings"][0]["split_sha256"] = swapped_lineage[
        "run_bindings"
    ][0]["execution_split_sha256"]
    with pytest.raises(ValueError, match="family substantive run provenance"):
        confirmation_family_power_reduction_artifact_from_dict(swapped_lineage)
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
    forged["pilot_scores"] = {"lightcone": 1e30}
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
