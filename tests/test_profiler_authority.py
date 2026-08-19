from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lightcone_spec.experiments.gpu_pool import (
    GpuAssignment,
    registry_pool_work_item,
)
from lightcone_spec.experiments.planning import (
    ZERO_COUNT,
    ZERO_MILLISECONDS,
    BudgetJobKind,
    ExperimentBudget,
    P99AnchorStatus,
    ScenarioMilliseconds,
)
from lightcone_spec.experiments.profiler_authority import (
    PROFILER_RAW_PROFILE_MISSING_REASON,
    PROFILER_REQUIRED_METRIC_MISSING_REASON,
    PROFILER_TERMINAL_POINTER_MISSING_REASON,
    PROFILER_TOOL_UNAVAILABLE_REASON,
    ProfilerAuthorityBlocked,
    bind_profiler_plan_authority,
    reduce_profiler_terminal_authority,
    release_profiler_plan,
    require_profiler_execution_authority,
    revalidate_profiler_plan_authority,
)
from lightcone_spec.experiments.registry import (
    WorkloadClass,
)
from lightcone_spec.experiments.registry import (
    build_legacy_industrial_registry as build_industrial_registry,
)


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _inputs(variant: str = "nsight_compute"):
    registry = build_industrial_registry()
    cell = next(
        row
        for row in registry.cells_for("E4")
        if row.resources.workload_class is WorkloadClass.PROFILE
        and row.identity.variant == variant
    )
    item = registry_pool_work_item(cell, estimated_duration_seconds=1.0)
    assignment = GpuAssignment(
        work_item=item,
        gpu_uuids=("GPU-A", "GPU-B"),
        rank_groups=(("GPU-A", "GPU-B"),),
        ports=tuple(31_000 + index for index in range(item.claim.port_count)),
    )
    profiler = ScenarioMilliseconds(100, 100, 100)
    gpu_time = ScenarioMilliseconds(200, 200, 200)
    budget = ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment="E4",
        method=cell.identity.method,
        workload_class=WorkloadClass.PROFILE,
        job_kind=BudgetJobKind.PROFILER,
        startup_model_load=ZERO_MILLISECONDS,
        compile_jit_graph_prewarm=ZERO_MILLISECONDS,
        excluded_warmup=ZERO_MILLISECONDS,
        excluded_warmup_requests=ZERO_COUNT,
        scored_arrival=ZERO_MILLISECONDS,
        request_deadline=ZERO_MILLISECONDS,
        drain=ZERO_MILLISECONDS,
        reset_finalization=ZERO_MILLISECONDS,
        evidence_flush_shutdown=ZERO_MILLISECONDS,
        output_tokens=ZERO_COUNT,
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=ZERO_MILLISECONDS,
        failure_injection=ZERO_MILLISECONDS,
        retry=ZERO_MILLISECONDS,
        retry_allowance=0,
        profiler=profiler,
        download_compile_reservation=ZERO_MILLISECONDS,
        gpu_count=2,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_time,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=gpu_time,
    )
    return registry, cell, assignment, budget


def _binding(tmp_path: Path, variant: str = "nsight_compute"):
    registry, cell, assignment, budget = _inputs(variant)
    subject_sha256 = "a" * 64
    plan = release_profiler_plan(
        registry,
        cell,
        assignment,
        budget,
        subject_plan_sha256=subject_sha256,
    )
    path = _write_json(tmp_path / f"{variant}-plan.json", plan.to_dict())
    binding = bind_profiler_plan_authority(
        path,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
        subject_plan_sha256=subject_sha256,
    )
    return registry, cell, assignment, budget, plan, path, binding


def _terminal(plan, binding, assignment, budget, raw_profile: Path):
    metrics = {name: 1.0 for name in plan.tool_contract.required_metrics}
    return {
        "schema_version": 1,
        "kind": "e4_profiler_terminal_pointer",
        "plan_sha256": plan.sha256,
        "authority_sha256": binding.sha256,
        "assignment_sha256": assignment.sha256,
        "budget_sha256": budget.sha256,
        "tool_contract_sha256": plan.tool_contract.sha256,
        "tool_version_sha256": plan.tool_version_sha256,
        "process_id": 1234,
        "process_start_ns": 100,
        "physical_gpu_uuids": list(plan.physical_gpu_uuids),
        "concurrent_headline_processes": 0,
        "terminal_status": "COMPLETE",
        "isolated_process": True,
        "headline_eligible": False,
        "raw_profiles": [
            {
                "role": plan.tool_contract.raw_profile_role,
                "path": str(raw_profile),
                "sha256": hashlib.sha256(raw_profile.read_bytes()).hexdigest(),
            }
        ],
        "metrics": metrics,
    }


@pytest.mark.parametrize(
    ("variant", "tool", "role"),
    [
        ("nvtx", "nsys", "nsys_report"),
        ("nsight_systems", "nsys", "nsys_report"),
        ("nsight_compute", "ncu", "ncu_report"),
    ],
)
def test_release_profiler_contracts_are_exact_and_non_headline(
    variant: str,
    tool: str,
    role: str,
) -> None:
    registry, cell, assignment, budget = _inputs(variant)
    plan = release_profiler_plan(
        registry,
        cell,
        assignment,
        budget,
        subject_plan_sha256="a" * 64,
    )

    assert plan.tool_contract.tool == tool
    assert plan.tool_contract.raw_profile_role == role
    assert plan.tool_contract.required_metrics
    assert plan.tool_contract.command_template[-1] == "{subject_argv}"
    assert plan.physical_gpu_uuids == ("GPU-A", "GPU-B")
    assert plan.profiler_duration_ms == 100
    assert plan.isolated_process and plan.exclusive_host
    assert plan.headline_eligible is False
    assert plan.tool_path is plan.tool_version is plan.tool_version_sha256 is None


def test_plan_bind_revalidates_and_blocks_before_profiler_launch(
    tmp_path: Path,
) -> None:
    registry, cell, assignment, budget, plan, _, binding = _binding(tmp_path)

    result = revalidate_profiler_plan_authority(
        binding,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    assert result.plan == plan
    assert result.status == "BLOCKED"
    assert result.reason == PROFILER_TOOL_UNAVAILABLE_REASON
    assert result.tool_version is None
    with pytest.raises(ProfilerAuthorityBlocked) as captured:
        require_profiler_execution_authority(
            binding,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
        )
    assert captured.value.reason == PROFILER_TOOL_UNAVAILABLE_REASON


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(headline_eligible=True), "headline evidence"),
        (
            lambda row: row["tool_contract"].update(required_metrics=[]),
            "differs from release derivation",
        ),
        (
            lambda row: row.update(profiler_duration_ms=0),
            "explicitly positive",
        ),
        (
            lambda row: row.update(physical_gpu_uuids=["GPU-A"]),
            "two exact physical GPUs",
        ),
    ],
)
def test_plan_rejects_caller_headline_metric_budget_and_assignment(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    registry, cell, assignment, budget = _inputs()
    plan = release_profiler_plan(
        registry,
        cell,
        assignment,
        budget,
        subject_plan_sha256="a" * 64,
    ).to_dict()
    mutation(plan)
    path = _write_json(tmp_path / "forged-plan.json", plan)
    with pytest.raises(ValueError, match=message):
        bind_profiler_plan_authority(
            path,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
            subject_plan_sha256="a" * 64,
        )


def test_plan_rejects_symlink_and_raw_byte_rehash(tmp_path: Path) -> None:
    registry, cell, assignment, budget, plan, path, binding = _binding(tmp_path)
    link = tmp_path / "plan-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="resolved and non-symlink"):
        bind_profiler_plan_authority(
            link,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
            subject_plan_sha256="a" * 64,
        )

    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh raw replay"):
        revalidate_profiler_plan_authority(
            binding,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
        )


def test_missing_terminal_is_named_blocked_with_none_values(tmp_path: Path) -> None:
    registry, cell, assignment, budget, plan, _, binding = _binding(tmp_path)
    result = reduce_profiler_terminal_authority(
        binding,
        None,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    assert result.status == "BLOCKED"
    assert result.reason == PROFILER_TERMINAL_POINTER_MISSING_REASON
    assert result.terminal_pointer_sha256 is None
    assert result.raw_profile_sha256s is None
    assert result.metrics == tuple(
        (name, None) for name in plan.tool_contract.required_metrics
    )


def test_complete_raw_terminal_still_blocks_on_empty_source_allowlist(
    tmp_path: Path,
) -> None:
    registry, cell, assignment, budget, plan, _, binding = _binding(tmp_path)
    raw_profile = tmp_path / "profile.ncu-rep"
    raw_profile.write_bytes(b"raw-ncu-profile")
    pointer = _write_json(
        tmp_path / "terminal.json",
        _terminal(plan, binding, assignment, budget, raw_profile),
    )
    result = reduce_profiler_terminal_authority(
        binding,
        pointer,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    assert result.status == "BLOCKED"
    assert result.reason == PROFILER_TOOL_UNAVAILABLE_REASON
    assert result.raw_profile_sha256s is None
    assert all(value is None for _, value in result.metrics)


def test_terminal_missing_metric_or_profile_is_named_blocked(tmp_path: Path) -> None:
    registry, cell, assignment, budget, plan, _, binding = _binding(tmp_path)
    raw_profile = tmp_path / "profile.ncu-rep"
    raw_profile.write_bytes(b"raw-ncu-profile")
    terminal = _terminal(plan, binding, assignment, budget, raw_profile)
    terminal["metrics"].pop(plan.tool_contract.required_metrics[0])
    pointer = _write_json(tmp_path / "missing-metric.json", terminal)
    result = reduce_profiler_terminal_authority(
        binding,
        pointer,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    assert result.reason == PROFILER_REQUIRED_METRIC_MISSING_REASON
    assert all(value is None for _, value in result.metrics)

    terminal = _terminal(plan, binding, assignment, budget, raw_profile)
    terminal["raw_profiles"] = []
    pointer = _write_json(tmp_path / "missing-profile.json", terminal)
    result = reduce_profiler_terminal_authority(
        binding,
        pointer,
        registry=registry,
        cell=cell,
        assignment=assignment,
        budget=budget,
    )
    assert result.reason == PROFILER_RAW_PROFILE_MISSING_REASON
    assert result.raw_profile_sha256s is None


def test_terminal_rejects_profile_tamper_and_headline_promotion(tmp_path: Path) -> None:
    registry, cell, assignment, budget, plan, _, binding = _binding(tmp_path)
    raw_profile = tmp_path / "profile.ncu-rep"
    raw_profile.write_bytes(b"raw-ncu-profile")
    terminal = _terminal(plan, binding, assignment, budget, raw_profile)
    raw_profile.write_bytes(b"tampered")
    pointer = _write_json(tmp_path / "tampered-profile.json", terminal)
    with pytest.raises(ValueError, match="hash differs"):
        reduce_profiler_terminal_authority(
            binding,
            pointer,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
        )

    raw_profile.write_bytes(b"raw-ncu-profile")
    promoted = deepcopy(_terminal(plan, binding, assignment, budget, raw_profile))
    promoted["headline_eligible"] = True
    pointer = _write_json(tmp_path / "promoted.json", promoted)
    with pytest.raises(ValueError, match="exact plan"):
        reduce_profiler_terminal_authority(
            binding,
            pointer,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
        )

    wrong_gpu = deepcopy(_terminal(plan, binding, assignment, budget, raw_profile))
    wrong_gpu["physical_gpu_uuids"] = ["GPU-A", "GPU-C"]
    pointer = _write_json(tmp_path / "wrong-gpu.json", wrong_gpu)
    with pytest.raises(ValueError, match="exact plan"):
        reduce_profiler_terminal_authority(
            binding,
            pointer,
            registry=registry,
            cell=cell,
            assignment=assignment,
            budget=budget,
        )
