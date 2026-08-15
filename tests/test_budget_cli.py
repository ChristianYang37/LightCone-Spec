from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lightcone_spec.cli.main import main
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.load import (
    FrozenSamplingParameters,
    ProductionLoadPlan,
    ProductionWindow,
    RequestTemplate,
    closed_loop_corpus,
)
from lightcone_spec.experiments.planning import (
    BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
    ZERO_MILLISECONDS,
    BudgetDispositionStatus,
    BudgetJobKind,
    BudgetJobPolicy,
    BudgetLoadBinding,
    BudgetPolicy,
    CapacityEnvelope,
    CellCapacityRequirement,
    P99AnchorStatus,
    ScenarioMilliseconds,
    budget_inventory_identity_from_gpu_inventory,
    materialize_industrial_budgets,
)
from lightcone_spec.experiments.planning_artifacts import (
    budget_load_binding_to_dict,
    budget_plan_from_dict,
    budget_plan_to_dict,
    budget_policy_to_dict,
    capacity_envelope_to_dict,
)
from lightcone_spec.experiments.registry import (
    ExperimentRegistry,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.stage_activation import (
    RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
    RegistryStageDispositionStatus,
    registry_stage_activation_from_dict,
)


def _canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _write_bound(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(_canonical_sha256(value) + "\n", encoding="utf-8")


def _scenario(
    optimistic: int, registered: int | None = None, quota: int | None = None
) -> ScenarioMilliseconds:
    return ScenarioMilliseconds(
        optimistic,
        optimistic if registered is None else registered,
        optimistic if quota is None else quota,
    )


def _registry(tmp_path: Path) -> tuple[Path, ExperimentRegistry]:
    cache_root = str(tmp_path / "cache")
    evidence_root = str(tmp_path / "evidence")
    path = tmp_path / "registry.json"
    assert (
        main(
            [
                "build-industrial-registry",
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


def _physical_inventory(gpu_count: int) -> GpuInventory:
    group_id = f"same-host-fabric-{gpu_count}"
    uuids = tuple(f"GPU-budget-cli-{index:03d}" for index in range(gpu_count))
    devices = tuple(
        GpuDevice(
            uuid=uuid,
            host_id="budget-cli-host",
            model="H100-SXM",
            memory_bytes=80 * 1024**3,
            compute_capability=(9, 0),
            pci_bus_id=f"0000:{index + 1:02x}:00.0",
            pci_root=f"root-{index // 4}",
            numa_node=index // 4,
            interconnects=("NVLink4", "PCIe5"),
            peer_access_class="NVSwitch",
            clock_policy="locked-1980MHz",
            power_limit_watts=700.0,
            thermal_limit_celsius=83.0,
            availability=GpuAvailability.READY,
            reserved_processes=(),
            allowed_topology_groups=(group_id,),
        )
        for index, uuid in enumerate(uuids)
    )
    return GpuInventory(
        schema_version=1,
        devices=devices,
        topology_groups=(
            GpuTopologyGroup(
                group_id=group_id,
                host_id="budget-cli-host",
                gpu_uuids=uuids,
                fabric="NVSwitch",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=content_sha256({"budget-cli-inventory-gpus": gpu_count}),
    )


def _budget_policy() -> BudgetPolicy:
    rows = []
    for job_kind in sorted(BudgetJobKind, key=lambda value: value.value):
        rows.append(
            BudgetJobPolicy(
                job_kind=job_kind,
                startup_model_load=_scenario(100, 200, 300),
                compile_jit_graph_prewarm=(
                    _scenario(400, 500, 600)
                    if job_kind is BudgetJobKind.COMPILE
                    else ZERO_MILLISECONDS
                ),
                reset_finalization=_scenario(10, 20, 30),
                evidence_flush_shutdown=_scenario(10, 20, 30),
                retry=ZERO_MILLISECONDS,
                retry_allowance=0,
                download_compile_reservation=(
                    _scenario(500, 600, 700)
                    if job_kind is BudgetJobKind.DOWNLOAD
                    else ZERO_MILLISECONDS
                ),
                reserved_gpu_overhead=_scenario(5, 10, 15),
            )
        )
    return BudgetPolicy(
        schema_version=1,
        policy_name="registered-budget-cli-policy",
        reducer_protocol_sha256=BUDGET_MATERIALIZATION_PROTOCOL_SHA256,
        job_policies=tuple(rows),
    )


def _load_binding(cell_id: str) -> BudgetLoadBinding:
    sampling = FrozenSamplingParameters.from_mapping({"temperature": 0.0, "top_p": 1.0})
    templates = tuple(
        RequestTemplate(
            input_token_ids=(index + 1,),
            requested_output_tokens=32,
            sampling=sampling,
        )
        for index in range(4)
    )
    scored = closed_loop_corpus(
        templates,
        namespace="budget-cli-preflight-score",
        split="tuning",
        concurrency=1,
        cohort_count=1,
        cohort_popularity="uniform",
        cohort_seed=20260811,
    )
    plans = tuple(
        ProductionLoadPlan(
            warmup=None,
            scored=scored,
            window=ProductionWindow(
                warmup_duration_us=0,
                arrival_duration_us=arrival_ms * 1_000,
                request_deadline_us=deadline_ms * 1_000,
                drain_duration_us=drain_ms * 1_000,
            ),
        )
        for arrival_ms, deadline_ms, drain_ms in (
            (1_000, 5_000, 100),
            (2_000, 6_000, 200),
            (3_000, 7_000, 300),
        )
    )
    return BudgetLoadBinding(
        cell_id=cell_id,
        job_kind=BudgetJobKind.STANDARD,
        optimistic_load=plans[0],
        registered_load=plans[1],
        quota_envelope_load=plans[2],
        minimum_completed_requests=4,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
    )


def _budget_authority(
    tmp_path: Path,
) -> tuple[
    Path,
    ExperimentRegistry,
    object,
    tuple[BudgetLoadBinding, ...],
    BudgetPolicy,
    Path,
    tuple[Path, ...],
]:
    registry_path, registry = _registry(tmp_path)
    runtime_path = tmp_path / "budget-preflight-runtime.json"
    split_path = tmp_path / "budget-preflight-split.json"
    _write_bound(runtime_path, {"runtime": "budget-preflight"})
    _write_bound(split_path, {"split": "budget-preflight"})
    activation_manifest = {
        "schema_version": 1,
        "kind": "industrial_registry_stage_activation_manifest",
        "registry_artifact": str(registry_path),
        "experiment": "preflight",
        "runtime_artifact": str(runtime_path),
        "split_artifact": str(split_path),
        "dependency_receipts": [],
    }
    activation_manifest_path = tmp_path / "budget-activation-manifest.json"
    _write_bound(activation_manifest_path, activation_manifest)
    activation_path = tmp_path / "budget-activation.json"
    assert (
        main(
            [
                "materialize-stage-activation",
                "--manifest",
                str(activation_manifest_path),
                "--output",
                str(activation_path),
            ]
        )
        == 0
    )
    activation = registry_stage_activation_from_dict(
        json.loads(activation_path.read_text(encoding="utf-8"))
    )
    assert activation.status == "AVAILABLE"
    assert activation.activated_cell_ids
    cells_by_id = {cell.cell_id: cell for cell in registry.cells_for("preflight")}
    activated_cells = tuple(
        cells_by_id[cell_id] for cell_id in activation.activated_cell_ids
    )
    assert {cell.identity.method for cell in activated_cells} == {"static"}
    assert {cell.identity.task for cell in activated_cells} == {
        "simultaneous_single_gpu_interference"
    }
    policy = _budget_policy()
    policy_path = tmp_path / "budget-policy.json"
    _write_bound(policy_path, budget_policy_to_dict(policy))
    bindings = tuple(
        _load_binding(cell_id) for cell_id in activation.activated_cell_ids
    )
    binding_paths = []
    for index, binding in enumerate(bindings):
        path = tmp_path / f"budget-load-{index:02d}.json"
        _write_bound(path, budget_load_binding_to_dict(binding))
        binding_paths.append(path)
    return (
        registry_path,
        registry,
        activation,
        bindings,
        policy,
        policy_path,
        tuple(binding_paths),
    )


def _write_inventory(tmp_path: Path, gpu_count: int) -> tuple[GpuInventory, Path]:
    inventory = _physical_inventory(gpu_count)
    path = tmp_path / f"inventory-{gpu_count}.json"
    _write_bound(path, inventory.to_dict())
    return inventory, path


def _write_capacity(
    path: Path,
    *,
    inventory: GpuInventory,
    cell_ids: tuple[str, ...],
    provider_quota_gpu_ms: int = 10**15,
    host_free_bytes: int = 10**15,
    host_quota_bytes: int = 10**15,
) -> tuple[CapacityEnvelope, Path]:
    envelope = CapacityEnvelope(
        schema_version=1,
        budget_inventory_sha256=budget_inventory_identity_from_gpu_inventory(
            inventory
        ).sha256,
        provider_quota_gpu_ms=provider_quota_gpu_ms,
        host_free_bytes=host_free_bytes,
        host_quota_bytes=host_quota_bytes,
        cell_requirements=tuple(
            CellCapacityRequirement(
                cell_id=cell_id,
                maximum_evidence_bytes=1_000,
                model_staging_bytes=2_000,
                compile_overlay_bytes=3_000,
            )
            for cell_id in sorted(cell_ids)
        ),
        source_receipt_sha256=content_sha256(
            {
                "capacity": path.name,
                "provider_quota_gpu_ms": provider_quota_gpu_ms,
                "host_free_bytes": host_free_bytes,
                "host_quota_bytes": host_quota_bytes,
            }
        ),
    )
    _write_bound(path, capacity_envelope_to_dict(envelope))
    return envelope, path


def _raw_budget_args(
    *,
    registry_path: Path,
    activation_path: Path,
    inventory_path: Path,
    policy_path: Path,
    binding_paths: tuple[Path, ...],
    capacity_path: Path,
) -> list[str]:
    args = [
        "--registry",
        str(registry_path),
        "--activation-plan",
        str(activation_path),
        "--inventory",
        str(inventory_path),
        "--budget-policy",
        str(policy_path),
        "--capacity-envelope",
        str(capacity_path),
    ]
    for path in binding_paths:
        args.extend(("--budget-load-binding", str(path)))
    return args


def test_materialize_budget_cli_reports_diagnostics_for_one_to_sixteen_gpus(
    tmp_path: Path,
) -> None:
    (
        registry_path,
        _,
        activation,
        _,
        _,
        policy_path,
        binding_paths,
    ) = _budget_authority(tmp_path)
    activation_path = tmp_path / "budget-activation-manifest.json"
    for gpu_count in (1, 2, 4, 8, 16):
        inventory, inventory_path = _write_inventory(tmp_path, gpu_count)
        _, capacity_path = _write_capacity(
            tmp_path / f"capacity-{gpu_count}.json",
            inventory=inventory,
            cell_ids=activation.activated_cell_ids,
        )
        output = tmp_path / f"budget-plan-{gpu_count}.json"
        assert (
            main(
                [
                    "materialize-industrial-budgets",
                    *_raw_budget_args(
                        registry_path=registry_path,
                        activation_path=activation_path,
                        inventory_path=inventory_path,
                        policy_path=policy_path,
                        binding_paths=binding_paths,
                        capacity_path=capacity_path,
                    ),
                    "--output",
                    str(output),
                ]
            )
            == 42
        )
        plan = budget_plan_from_dict(json.loads(output.read_text(encoding="utf-8")))
        assert plan.status == "UNRESOLVED"
        assert plan.inventory.gpu_count == gpu_count
        assert len(plan.diagnostic_budgets) == len(activation.activated_cell_ids)
        assert {row.reason_code for row in plan.dispositions} == {
            "capacity_raw_authority_missing"
        }
        assert all(
            budget.fixed_instance_billed_gpu_ms == budget.wall_time.scale(gpu_count)
            for budget in plan.diagnostic_budgets
        )


def test_materialize_budget_cli_fails_closed_for_missing_quota_and_disk(
    tmp_path: Path,
) -> None:
    (
        registry_path,
        _,
        activation,
        _,
        _,
        policy_path,
        binding_paths,
    ) = _budget_authority(tmp_path)
    activation_path = tmp_path / "budget-activation-manifest.json"
    inventory, inventory_path = _write_inventory(tmp_path, 2)

    cases = (
        (
            "missing-load",
            binding_paths[:-1],
            activation.activated_cell_ids,
            {},
            {"missing_load_semantics", "capacity_budget_coverage_incomplete"},
        ),
        (
            "missing-capacity",
            binding_paths,
            activation.activated_cell_ids[:-1],
            {},
            {
                "capacity_requirement_missing",
                "capacity_requirement_coverage_incomplete",
            },
        ),
        (
            "quota",
            binding_paths,
            activation.activated_cell_ids,
            {"provider_quota_gpu_ms": 0},
            {"capacity_provider_quota_exceeded"},
        ),
        (
            "disk",
            binding_paths,
            activation.activated_cell_ids,
            {"host_free_bytes": 0},
            {"capacity_host_disk_exceeded"},
        ),
    )
    for name, selected_bindings, capacity_cell_ids, capacity_kwargs, reasons in cases:
        _, capacity_path = _write_capacity(
            tmp_path / f"capacity-{name}.json",
            inventory=inventory,
            cell_ids=capacity_cell_ids,
            **capacity_kwargs,
        )
        output = tmp_path / f"budget-plan-{name}.json"
        assert (
            main(
                [
                    "materialize-industrial-budgets",
                    *_raw_budget_args(
                        registry_path=registry_path,
                        activation_path=activation_path,
                        inventory_path=inventory_path,
                        policy_path=policy_path,
                        binding_paths=selected_bindings,
                        capacity_path=capacity_path,
                    ),
                    "--output",
                    str(output),
                ]
            )
            == 42
        )
        plan = budget_plan_from_dict(json.loads(output.read_text(encoding="utf-8")))
        assert plan.status == "UNRESOLVED"
        assert len(plan.diagnostic_budgets) == len(selected_bindings)
        assert {
            row.reason_code
            for row in plan.dispositions
            if row.status is BudgetDispositionStatus.UNRESOLVED
        } == reasons


def test_budget_consumers_rematerialize_and_bind_physical_scheduler_authority(
    tmp_path: Path,
) -> None:
    (
        registry_path,
        registry,
        activation,
        bindings,
        policy,
        policy_path,
        binding_paths,
    ) = _budget_authority(tmp_path)
    activation_path = tmp_path / "budget-activation-manifest.json"
    inventory, inventory_path = _write_inventory(tmp_path, 2)
    capacity, capacity_path = _write_capacity(
        tmp_path / "capacity-ready.json",
        inventory=inventory,
        cell_ids=activation.activated_cell_ids,
    )
    plan_path = tmp_path / "budget-plan-ready.json"
    raw_args = _raw_budget_args(
        registry_path=registry_path,
        activation_path=activation_path,
        inventory_path=inventory_path,
        policy_path=policy_path,
        binding_paths=binding_paths,
        capacity_path=capacity_path,
    )
    assert (
        main(
            [
                "materialize-industrial-budgets",
                *raw_args,
                "--output",
                str(plan_path),
            ]
        )
        == 42
    )

    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("budget-cli-serial-envelope")
    )
    envelope_path = tmp_path / "interference-envelope.json"
    _write_bound(envelope_path, envelope.to_dict())
    report_path = tmp_path / "budget-report.json"
    status = main(
        [
            "estimate-industrial-budget",
            *raw_args,
            "--interference-envelope",
            str(envelope_path),
            "--budget-plan",
            str(plan_path),
            "--output",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scheduler_gpu_inventory_sha256"] == inventory.sha256
    assert report["interference_envelope_sha256"] == envelope.sha256
    assert "capacity_raw_authority_missing" in report["unresolved_assumptions"]
    assert status == (42 if report["unresolved_assumptions"] else 0)

    with pytest.raises(ValueError, match="capacity_raw_authority_missing"):
        main(
            [
                "plan-industrial-dispatch",
                *raw_args,
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(plan_path),
                "--output",
                str(tmp_path / "blocked-dispatch.json"),
            ]
        )

    other_policy = replace(policy, policy_name="different-valid-policy")
    other_plan = materialize_industrial_budgets(
        registry,
        activations=(activation,),
        load_bindings=bindings,
        policy=other_policy,
        inventory=budget_inventory_identity_from_gpu_inventory(inventory),
        capacity_envelope=capacity,
    )
    other_plan_path = tmp_path / "different-valid-budget-plan.json"
    _write_bound(other_plan_path, budget_plan_to_dict(other_plan))
    with pytest.raises(ValueError, match="first-party rematerialization"):
        main(
            [
                "estimate-industrial-budget",
                *raw_args,
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(other_plan_path),
                "--output",
                str(tmp_path / "rejected-report.json"),
            ]
        )
    with pytest.raises(ValueError, match="first-party rematerialization"):
        main(
            [
                "plan-industrial-dispatch",
                *raw_args,
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(other_plan_path),
                "--output",
                str(tmp_path / "rejected-dispatch.json"),
            ]
        )

    tampered_policy_path = tmp_path / "tampered-policy.json"
    policy_wire = budget_policy_to_dict(policy)
    _write_bound(tampered_policy_path, policy_wire)
    policy_wire["policy_name"] = "tampered-after-sidecar"
    tampered_policy_path.write_text(
        json.dumps(policy_wire, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar is missing or invalid"):
        main(
            [
                "materialize-industrial-budgets",
                *_raw_budget_args(
                    registry_path=registry_path,
                    activation_path=activation_path,
                    inventory_path=inventory_path,
                    policy_path=tampered_policy_path,
                    binding_paths=binding_paths,
                    capacity_path=capacity_path,
                ),
                "--output",
                str(tmp_path / "tampered-output.json"),
            ]
        )

    with pytest.raises(SystemExit):
        main(
            [
                "estimate-industrial-budget",
                *raw_args,
                "--budget-plan",
                str(plan_path),
                "--output",
                str(tmp_path / "missing-envelope-report.json"),
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "estimate-industrial-budget",
                *raw_args,
                "--interference-envelope",
                str(envelope_path),
                "--budgets",
                str(plan_path),
                "--output",
                str(tmp_path / "legacy-budget-report.json"),
            ]
        )


def test_budget_cli_requires_paired_raw_capacity_authority_paths(
    tmp_path: Path,
) -> None:
    (
        registry_path,
        _,
        activation,
        _,
        _,
        policy_path,
        binding_paths,
    ) = _budget_authority(tmp_path)
    inventory, inventory_path = _write_inventory(tmp_path, 2)
    _, capacity_path = _write_capacity(
        tmp_path / "capacity.json",
        inventory=inventory,
        cell_ids=activation.activated_cell_ids,
    )
    raw_args = _raw_budget_args(
        registry_path=registry_path,
        activation_path=tmp_path / "budget-activation-manifest.json",
        inventory_path=inventory_path,
        policy_path=policy_path,
        binding_paths=binding_paths,
        capacity_path=capacity_path,
    )
    plan_path = tmp_path / "budget-plan.json"
    assert (
        main(
            [
                "materialize-industrial-budgets",
                *raw_args,
                "--output",
                str(plan_path),
            ]
        )
        == 42
    )
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("paired-capacity-cli-envelope")
    )
    envelope_path = tmp_path / "interference-envelope.json"
    _write_bound(envelope_path, envelope.to_dict())
    commands = (
        ("materialize-industrial-budgets", ()),
        (
            "estimate-industrial-budget",
            (
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(plan_path),
            ),
        ),
        (
            "plan-industrial-dispatch",
            (
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(plan_path),
            ),
        ),
    )
    one_sided_authority = (
        ("--capacity-manifest", str(tmp_path / "capacity-manifest.json")),
        (
            "--capacity-verification-receipt",
            str(tmp_path / "capacity-verification.json"),
        ),
    )
    for command, command_args in commands:
        for flag, path in one_sided_authority:
            with pytest.raises(
                ValueError,
                match=(
                    "capacity manifest and verification receipt must be supplied "
                    "together"
                ),
            ):
                main(
                    [
                        command,
                        *raw_args,
                        *command_args,
                        flag,
                        path,
                        "--output",
                        str(tmp_path / f"{command}-{flag[2:]}.json"),
                    ]
                )


def test_budget_materialization_blocks_without_reducer_owned_activation(
    tmp_path: Path,
) -> None:
    (
        registry_path,
        _,
        activation,
        _,
        _,
        policy_path,
        binding_paths,
    ) = _budget_authority(tmp_path)
    inventory, inventory_path = _write_inventory(tmp_path, 2)
    _, capacity_path = _write_capacity(
        tmp_path / "capacity.json",
        inventory=inventory,
        cell_ids=activation.activated_cell_ids,
    )
    args = _raw_budget_args(
        registry_path=registry_path,
        activation_path=tmp_path / "budget-activation-manifest.json",
        inventory_path=inventory_path,
        policy_path=policy_path,
        binding_paths=binding_paths,
        capacity_path=capacity_path,
    )
    activation_index = args.index("--activation-plan")
    del args[activation_index : activation_index + 2]
    output = tmp_path / "preflight-budget-plan.json"
    assert (
        main(
            [
                "materialize-industrial-budgets",
                *args,
                "--output",
                str(output),
            ]
        )
        == 42
    )
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["status"] == "BLOCKED"
    assert decision["reason_code"] == "reducer_owned_activation_manifest_missing"


def test_preflight_compile_activation_and_budget_fail_closed_without_runner(
    tmp_path: Path,
) -> None:
    registry_path, _ = _registry(tmp_path)
    runtime_path = tmp_path / "preflight-runtime.json"
    split_path = tmp_path / "preflight-split.json"
    _write_bound(runtime_path, {"runtime": "preflight-test"})
    _write_bound(split_path, {"split": "preflight-test"})
    manifest = {
        "schema_version": 1,
        "kind": "industrial_registry_stage_activation_manifest",
        "registry_artifact": str(registry_path),
        "experiment": "preflight",
        "runtime_artifact": str(runtime_path),
        "split_artifact": str(split_path),
        "dependency_receipts": [],
    }
    manifest_path = tmp_path / "preflight-activation-manifest.json"
    _write_bound(manifest_path, manifest)
    activation_path = tmp_path / "preflight-activation.json"
    assert (
        main(
            [
                "materialize-stage-activation",
                "--manifest",
                str(manifest_path),
                "--output",
                str(activation_path),
            ]
        )
        == 0
    )
    activation = registry_stage_activation_from_dict(
        json.loads(activation_path.read_text(encoding="utf-8"))
    )
    assert activation.experiment == "preflight"
    assert activation.status == "AVAILABLE"
    assert activation.activated_cell_ids
    assert {
        row.reason_code
        for row in activation.dispositions
        if row.status is RegistryStageDispositionStatus.BLOCKED
    } == {
        RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE,
        "release_preflight_method_unsupported",
    }
    assert all(
        row.cell_id not in activation.activated_cell_ids
        for row in activation.dispositions
        if row.reason_code == RELEASE_COMPILE_ASSIGNMENT_CONTRACT_UNAVAILABLE
    )

    policy = _budget_policy()
    policy_path = tmp_path / "preflight-budget-policy.json"
    _write_bound(policy_path, budget_policy_to_dict(policy))
    inventory, inventory_path = _write_inventory(tmp_path, 2)
    _, capacity_path = _write_capacity(
        tmp_path / "preflight-capacity.json",
        inventory=inventory,
        cell_ids=activation.activated_cell_ids,
    )
    raw_args = [
        "--registry",
        str(registry_path),
        "--activation-plan",
        str(manifest_path),
        "--inventory",
        str(inventory_path),
        "--budget-policy",
        str(policy_path),
        "--capacity-envelope",
        str(capacity_path),
    ]
    plan_path = tmp_path / "preflight-budget-plan.json"
    assert (
        main(
            [
                "materialize-industrial-budgets",
                *raw_args,
                "--output",
                str(plan_path),
            ]
        )
        == 42
    )
    plan = budget_plan_from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    assert plan.status == "UNRESOLVED"
    assert set(plan.activated_cell_ids) == set(activation.activated_cell_ids)

    with pytest.raises(ValueError, match="bound raw activation manifest"):
        main(
            [
                "materialize-industrial-budgets",
                "--registry",
                str(registry_path),
                "--activation-plan",
                str(activation_path),
                "--inventory",
                str(inventory_path),
                "--budget-policy",
                str(policy_path),
                "--capacity-envelope",
                str(capacity_path),
                "--output",
                str(tmp_path / "serialized-activation-budget.json"),
            ]
        )

    forged_manifest = {**manifest, "activated_cell_ids": activation.activated_cell_ids}
    forged_manifest_path = tmp_path / "caller-cell-list-manifest.json"
    _write_bound(forged_manifest_path, forged_manifest)
    with pytest.raises(ValueError, match="manifest fields differ"):
        main(
            [
                "materialize-stage-activation",
                "--manifest",
                str(forged_manifest_path),
                "--output",
                str(tmp_path / "forged-activation.json"),
            ]
        )
