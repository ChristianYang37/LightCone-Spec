from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evidence_alias_authority import _alias_artifact
from test_evidence_alias_authority import _manifest as _raw_alias_manifest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.cli.main import (
    _analysis_cells,
    _load_e2_stage_manifest,
    _load_stage_activation_plan,
    main,
)
from lightcone_spec.experiments.evidence import evidence_files_sha256
from lightcone_spec.experiments.gpu_pool import (
    GpuAvailability,
    GpuDevice,
    GpuInventory,
    GpuTopologyGroup,
    InterferenceEnvelope,
)
from lightcone_spec.experiments.industrial_analysis import (
    RawEvidenceAliasManifest,
    raw_evidence_alias_manifest_to_dict,
)
from lightcone_spec.experiments.planning import (
    CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256,
    E2_HALVING_PROTOCOL_SHA256,
    BudgetJobKind,
    CellDisposition,
    ConfirmationFamilyPowerPlan,
    ConfirmationFamilyPowerReductionArtifact,
    DispositionStatus,
    ExpectedMaximumCount,
    ExperimentBudget,
    P99AnchorStatus,
    RawEvidenceRunBinding,
    ReducerActivationArtifact,
    ScenarioMilliseconds,
    StageActivationPlan,
    build_evidence_dependence_map,
    derive_confirmation_family,
    family_pilot_block_id,
    materialize_confirmation_prefix,
)
from lightcone_spec.experiments.planning_artifacts import (
    confirmation_family_identity_to_dict,
    confirmation_family_power_reduction_artifact_to_dict,
    evidence_alias_reduction_artifact_from_dict,
    evidence_alias_reduction_artifact_to_dict,
    evidence_dependence_map_from_dict,
    evidence_dependence_map_to_dict,
    experiment_budget_sequence_to_dict,
    family_activation_artifact_from_dict,
    family_activation_artifact_to_dict,
    reducer_activation_artifact_to_dict,
)
from lightcone_spec.experiments.registry import (
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    WorkloadClass,
    build_legacy_industrial_registry,
    content_sha256,
    scientific_role_for_cell,
)
from lightcone_spec.experiments.statistics import (
    HardwareEnvelope,
    PilotBlock,
    preregister_power_sizing,
)
from lightcone_spec.telemetry import (
    OUTPUT_HASH_FORMAT,
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RunRecord,
)


def _sha(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _write_bound(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(_sha(value) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _performance(
    run_id: str, method: str, *, repetition_block: int = 0
) -> PerformanceRecord:
    return PerformanceRecord(
        run_id=run_id,
        prompt_id="preflight",
        method=method,
        repetition_block=repetition_block,
        region="preflight",
        concurrency=1,
        generated_bucket_start=0,
        generated_bucket_end=1,
        at_risk_requests=1,
        output_tokens=1,
        elapsed_s=1.0,
        decode_goodput_tps=1.0,
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
        offered_requests=1,
        admitted_requests=1,
        completed_requests=1,
        unfinished_requests=0,
        admission_rejections=0,
        timeouts=0,
        cancellations=0,
    )


def _completed_stage(
    tmp_path: Path,
    registry: dict,
    experiment: str,
) -> Path:
    root = tmp_path / f"{experiment}-evidence"
    rows = []
    cells = [
        cell
        for cell in registry["registry"]["cells"]
        if cell["identity"]["experiment"] == experiment
        and cell["status"] == "UNMEASURED"
    ]
    for index, cell in enumerate(cells):
        cell_id = _sha(cell["identity"])
        method = cell["identity"]["method"]
        run_id = f"{experiment.lower()}-{index}"
        writer = EvidenceWriter(
            root,
            run_id=run_id,
            rank=0,
            process_id=index + 1,
            checkpoint_interval_s=None,
        )
        writer.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=registry["registry_sha256"],
                config_sha256=cell_id,
                method=method,
                model_pair=cell["identity"]["model"],
                repetition_block=0,
                started_ns=1,
                completed_ns=2,
                status="complete",
            )
        )
        token_ids = json.dumps([index], separators=(",", ":"))
        token_ids_sha256 = hashlib.sha256(token_ids.encode("utf-8")).hexdigest()
        writer.write(
            RequestRecord(
                run_id=run_id,
                request_id=f"request-{index}",
                prompt_id="preflight",
                method=method,
                repetition_block=0,
                concurrency=1,
                input_tokens=1,
                output_tokens=1,
                output_hash_format=OUTPUT_HASH_FORMAT,
                output_sha256=token_ids_sha256,
                ttft_ms=1.0,
                finished=True,
                stop_reason="length",
                output_token_ids=token_ids,
                output_token_ids_sha256=token_ids_sha256,
            )
        )
        writer.write(_performance(run_id, method))
        evidence = writer.close()
        receipt = root / f"{run_id}.rank0.complete.json"
        rows.append(
            {
                "cell_id": cell_id,
                "evidence_root": str(root),
                "run_id": run_id,
                "rank": 0,
                "evidence_sha256": evidence_files_sha256(evidence.values()),
                "terminal_receipt_sha256": _file_sha(receipt),
                "status": "MEASURED",
            }
        )
    completed = {
        "schema_version": 1,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry["registry_sha256"],
        "rows": rows,
    }
    path = tmp_path / f"{experiment}-completed.json"
    _write_bound(path, completed)
    return path


def _build_registry(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "registry.json"
    assert (
        main(
            [
                "build-industrial-registry",
                "--legacy-diagnostic",
                "--logical-gpu-slot",
                "logical-rank-slot-a",
                "logical-rank-slot-b",
                "--cache-root",
                str(tmp_path / "runtime-cache"),
                "--evidence-root",
                str(tmp_path / "artifacts"),
                "--output",
                str(path),
            ]
        )
        == 0
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _pool_inputs(tmp_path: Path, registry: dict) -> tuple[Path, Path, Path]:
    uuids = ("GPU-pool-000", "GPU-pool-001")
    inventory = GpuInventory(
        schema_version=1,
        devices=tuple(
            GpuDevice(
                uuid=uuid,
                host_id="test-host",
                model="test-gpu",
                memory_bytes=80 * 1024**3,
                compute_capability=(9, 0),
                pci_bus_id=f"0000:0{index + 1}:00.0",
                pci_root="root-0",
                numa_node=0,
                interconnects=("NVLink",),
                peer_access_class="nvlink",
                clock_policy="locked",
                power_limit_watts=700.0,
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
                host_id="test-host",
                gpu_uuids=uuids,
                fabric="NVLink",
                bandwidth_class="high",
            ),
        ),
        source_receipt_sha256=content_sha256("test-inventory"),
    )
    inventory_path = tmp_path / "gpu-inventory.json"
    _write_bound(inventory_path, inventory.to_dict())
    envelope = InterferenceEnvelope.serial(
        source_receipt_sha256=content_sha256("serial-interference")
    )
    envelope_path = tmp_path / "interference-envelope.json"
    _write_bound(envelope_path, envelope.to_dict())
    cell = next(
        row
        for row in registry["registry"]["cells"]
        if row["identity"]["experiment"] == "preflight"
        and row["identity"]["method"] == "target_only"
    )
    cell_id = _sha(cell["identity"])
    zero = ScenarioMilliseconds(0, 0, 0)
    compile_time = ScenarioMilliseconds(1_000, 2_000, 3_000)
    gpu_time = compile_time.scale(2)
    budget = ExperimentBudget(
        schema_version=1,
        cell_id=cell_id,
        experiment="preflight",
        method="target_only",
        workload_class=WorkloadClass.COMPILE,
        job_kind=BudgetJobKind.COMPILE,
        startup_model_load=zero,
        compile_jit_graph_prewarm=compile_time,
        excluded_warmup=zero,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=zero,
        request_deadline=zero,
        drain=zero,
        reset_finalization=zero,
        evidence_flush_shutdown=zero,
        output_tokens=ExpectedMaximumCount(0, 0),
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=zero,
        retry_allowance=0,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=2,
        topology=cell["identity"]["topology"],
        reserved_gpu_ms=gpu_time,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=gpu_time,
    )
    budget_path = tmp_path / "budgets.json"
    _write_bound(budget_path, experiment_budget_sequence_to_dict((budget,)))
    return inventory_path, envelope_path, budget_path


def _standard_budget(cell) -> ExperimentBudget:
    zero = ScenarioMilliseconds(0, 0, 0)
    startup = ScenarioMilliseconds(1_000, 2_000, 3_000)
    gpu_time = startup.scale(cell.resources.gpu_count)
    return ExperimentBudget(
        schema_version=1,
        cell_id=cell.cell_id,
        experiment=cell.identity.experiment,
        method=cell.identity.method,
        workload_class=cell.resources.workload_class,
        job_kind=BudgetJobKind.STANDARD,
        startup_model_load=startup,
        compile_jit_graph_prewarm=zero,
        excluded_warmup=zero,
        excluded_warmup_requests=ExpectedMaximumCount(0, 0),
        scored_arrival=zero,
        request_deadline=zero,
        drain=zero,
        reset_finalization=zero,
        evidence_flush_shutdown=zero,
        output_tokens=ExpectedMaximumCount(0, 0),
        minimum_completed_requests=0,
        p99_anchor_status=P99AnchorStatus.NOT_REQUIRED,
        soak=zero,
        failure_injection=zero,
        retry=zero,
        retry_allowance=0,
        profiler=zero,
        download_compile_reservation=zero,
        gpu_count=cell.resources.gpu_count,
        topology=cell.identity.topology,
        reserved_gpu_ms=gpu_time,
        measured_gpu_ms=None,
        fixed_instance_billed_gpu_ms=startup.scale(2),
    )


def test_dispatch_rejects_bare_budget_sequence_without_raw_materialization(
    tmp_path: Path,
) -> None:
    registry_path, registry = _build_registry(tmp_path)
    inventory_path, envelope_path, budget_path = _pool_inputs(tmp_path, registry)
    first_plan = tmp_path / "preflight-plan.json"
    with pytest.raises(SystemExit):
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--inventory",
                str(inventory_path),
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(budget_path),
                "--output",
                str(first_plan),
            ]
        )
    assert not first_plan.exists()
    with pytest.raises(SystemExit):
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--inventory",
                str(inventory_path),
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(budget_path),
                "--budget-policy",
                str(tmp_path / "policy.json"),
                "--capacity-envelope",
                str(tmp_path / "capacity.json"),
                "--confirmation-block-plan",
                str(tmp_path / "legacy-confirmation.json"),
                "--output",
                str(tmp_path / "legacy-plan.json"),
            ]
        )


def test_stage_sealing_is_blocked_before_cpu_evidence_can_mint_a_receipt(
    tmp_path: Path,
) -> None:
    registry_path, registry = _build_registry(tmp_path)
    inventory_path, _, _ = _pool_inputs(tmp_path, registry)
    runtime = tmp_path / "runtime.json"
    split = tmp_path / "split.json"
    locked = tmp_path / "runtime-envelope.json"
    _write_bound(runtime, {"runtime": "test"})
    _write_bound(split, {"split": "preflight"})
    _write_bound(locked, {"runtime_envelope": "passed"})
    completed = _completed_stage(tmp_path, registry, "preflight")
    output = tmp_path / "forged-receipt.json"
    with pytest.raises(ValueError, match="schema-version-4"):
        main(
            [
                "seal-industrial-stage",
                "--registry",
                str(registry_path),
                "--experiment",
                "preflight",
                "--runtime-artifact",
                str(runtime),
                "--split-artifact",
                str(split),
                "--completed-cells",
                str(completed),
                "--inventory",
                str(inventory_path),
                "--interference-calibration-authority",
                str(tmp_path / "must-not-be-opened.json"),
                "--locked-output",
                f"runtime_envelope={locked}",
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_registry_and_receipt_edits_fail_closed(tmp_path: Path) -> None:
    registry_path, registry = _build_registry(tmp_path)
    inventory_path, envelope_path, budget_path = _pool_inputs(tmp_path, registry)
    registry["registry"]["name"] = "edited-after-generation"
    _write_bound(registry_path, registry)
    with pytest.raises(ValueError, match="edited after generation"):
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--inventory",
                str(inventory_path),
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(budget_path),
                "--budget-policy",
                str(tmp_path / "policy.json"),
                "--capacity-envelope",
                str(tmp_path / "capacity.json"),
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )


def test_dispatch_rejects_boolean_substitution_for_interference_envelope(
    tmp_path: Path,
) -> None:
    registry_path, registry = _build_registry(tmp_path)
    inventory_path, _, budget_path = _pool_inputs(tmp_path, registry)
    forged = {
        "schema_version": 1,
        "kind": "two_gpu_interference_gate",
        "status": "PASS",
        "registry_sha256": registry["registry_sha256"],
        "gpu_uuids": ["GPU-aaaaaaaa", "GPU-bbbbbbbb"],
    }
    forged_path = tmp_path / "forged.json"
    _write_bound(forged_path, forged)
    with pytest.raises(ValueError, match="interference envelope fields differ"):
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--inventory",
                str(inventory_path),
                "--interference-envelope",
                str(forged_path),
                "--budget-plan",
                str(budget_path),
                "--budget-policy",
                str(tmp_path / "policy.json"),
                "--capacity-envelope",
                str(tmp_path / "capacity.json"),
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )


def test_registry_uses_logical_slots_and_rejects_physical_uuid_flag(
    tmp_path: Path,
) -> None:
    output = tmp_path / "default-registry.json"
    assert (
        main(
            [
                "build-industrial-registry",
                "--cache-root",
                str(tmp_path / "cache"),
                "--evidence-root",
                str(tmp_path / "evidence"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["schema_version"] == 3
    assert value["parameters"]["logical_gpu_slots"] == [
        "logical-rank-slot-0",
        "logical-rank-slot-1",
    ]
    assert "gpu_uuids" not in value["parameters"]
    with pytest.raises(SystemExit):
        main(
            [
                "build-industrial-registry",
                "--gpu-uuid",
                "GPU-physical-a",
                "GPU-physical-b",
                "--output",
                str(tmp_path / "physical-registry.json"),
            ]
        )


def test_registry_cli_accepts_arbitrary_logical_rank_slots(tmp_path: Path) -> None:
    output = tmp_path / "four-slot-registry.json"
    logical_slots = tuple(f"logical-rank-slot-{index}" for index in range(4))
    assert (
        main(
            [
                "build-industrial-registry",
                "--logical-gpu-slot",
                *logical_slots,
                "--cache-root",
                str(tmp_path / "cache"),
                "--evidence-root",
                str(tmp_path / "evidence"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["parameters"]["logical_gpu_slots"] == list(logical_slots)
    assert value["registry"]["cells"]


def _confirmation_family_inputs(tmp_path: Path):
    logical_slots = ("logical-rank-slot-a", "logical-rank-slot-b")
    cache_root = str(tmp_path / "runtime-cache")
    evidence_root = str(tmp_path / "artifacts")
    registry_path, _ = _build_registry(tmp_path)
    registry = build_legacy_industrial_registry(
        gpu_uuids=logical_slots,
        cache_root=cache_root,
        evidence_root=evidence_root,
    )
    source = next(cell for cell in registry.cells_for("E3b") if cell.runnable)
    family = derive_confirmation_family(
        registry,
        cell_id=source.cell_id,
        runtime_sha256=_sha("family-runtime"),
        split_sha256=_sha("family-split"),
        trace_sha256=_sha("family-trace"),
        sampling_sha256=_sha("family-sampling"),
        hardware_envelope_sha256=_sha("family-hardware"),
    )
    family_path = tmp_path / "family.json"
    _write_bound(family_path, confirmation_family_identity_to_dict(family))
    return registry_path, registry, family, family_path


def _family_power_reduction(registry, family, pilot, power):
    evidence_sha256 = _sha("content-bound-pilot-evidence")
    selected = power.selected_final_blocks
    assert selected is not None
    plan = ConfirmationFamilyPowerPlan(
        schema_version=1,
        family=family,
        pilot_activation_sha256=pilot.sha256,
        completed_pilot_cells_sha256=content_sha256(
            tuple(sorted(pilot.activated_cell_ids))
        ),
        pilot_evidence_sha256=evidence_sha256,
        power_sizing=power,
        status="POWERED",
        selected_final_blocks=selected,
        selected_final_prefix=FINAL_BLOCKS[:selected],
        reason_code="registered_family_power_target_met",
        selection_state="sealed_before_confirmation_unblinding",
    )
    cells = {cell.cell_id: cell for cell in registry.cells}

    def digest(kind: str, cell_id: str) -> str:
        return _sha({"kind": kind, "cell_id": cell_id})

    bindings = tuple(
        RawEvidenceRunBinding(
            schema_version=3,
            cell_id=cell_id,
            experiment=family.experiment,
            method=cells[cell_id].identity.method,
            scientific_role=scientific_role_for_cell(registry, cells[cell_id]),
            scientific_unit=f"excluded_pilot_{cells[cell_id].identity.block}",
            config_sha256=digest("config", cell_id),
            rank_config_sha256s=(digest("rank-config", cell_id),),
            run_id=f"cli-family-{cell_id}",
            rank_count=1,
            model_pair=family.model,
            runtime_sha256=family.runtime_sha256,
            split_sha256=family.split_sha256,
            corpus_sha256=digest("corpus", cell_id),
            arrival_trace_sha256=family.trace_sha256,
            request_ids_sha256=digest("requests", cell_id),
            sampling_profile_sha256=family.sampling_sha256,
            model_lock_sha256=digest("model-lock", cell_id),
            patched_sglang_tree=PINNED_SGLANG_TREE,
            run_nonce_sha256=digest("nonce", cell_id),
            topology_sha256=digest("topology", cell_id),
            experiment_budget_sha256=digest("budget", cell_id),
            physical_gpu_uuids=(f"GPU-cli-family-{cell_id}",),
            terminal_receipt_sha256s=(digest("terminal", cell_id),),
            hardware_receipt_sha256=digest("hardware", cell_id),
            budget_observation_sha256=digest("observation", cell_id),
            execution_plan_sha256=digest("execution-plan", cell_id),
            execution_split_sha256=digest("execution-split", cell_id),
        )
        for cell_id in sorted(pilot.activated_cell_ids)
    )
    return ConfirmationFamilyPowerReductionArtifact(
        schema_version=2,
        plan=plan,
        inventory_sha256=_sha("cli-family-inventory"),
        inventory_source_receipt_sha256=_sha("cli-family-inventory-source"),
        fixed_instance_gpu_count=2,
        inventory_host_id="cli-family-host",
        raw_evidence_manifest_sha256=evidence_sha256,
        terminal_receipt_sha256s=tuple(
            sorted(
                receipt
                for binding in bindings
                for receipt in binding.terminal_receipt_sha256s
            )
        ),
        hardware_receipt_sha256s=tuple(
            sorted(binding.hardware_receipt_sha256 for binding in bindings)
        ),
        budget_observation_sha256s=tuple(
            sorted(binding.budget_observation_sha256 for binding in bindings)
        ),
        run_bindings=bindings,
        reducer_protocol_sha256=(CONFIRMATION_FAMILY_POWER_REDUCER_PROTOCOL_SHA256),
        data_source="excluded_pilots_only",
        confirmation_data_visible=False,
    )


def test_e2_cli_rejects_serialized_summary_without_raw_stage_manifest(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "caller-authored-e2-reduction.json"
    _write_bound(
        summary_path,
        {
            "schema_version": 1,
            "activation": {"sha256": _sha("activation")},
            "stage_evidence": {"evaluations": []},
            "survivor_receipt": {"survivor_candidate_ids": []},
        },
    )

    with pytest.raises(ValueError, match="E2 raw stage manifest fields"):
        main(
            [
                "reduce-e2-successive-halving",
                "--manifest",
                str(summary_path),
                "--output",
                str(tmp_path / "e2-reduction.json"),
            ]
        )

    cell_id = _sha("caller-selected-e2-cell")
    activation = ReducerActivationArtifact(
        schema_version=1,
        plan=StageActivationPlan(
            registry_sha256=_sha("e2-registry"),
            experiment="E2",
            dependency_receipt_sha256=_sha("e1-receipt"),
            runtime_sha256=_sha("e2-runtime"),
            split_sha256=_sha("e2-split"),
            source_selection_sha256=_sha("caller-selection"),
            activation_round="halving_1",
            status="AVAILABLE",
            activated_cell_ids=(cell_id,),
            not_applicable_cell_ids=(),
            blocked_cell_ids=(),
            deferred_cell_ids=(),
            reason_code="caller_authored_activation",
        ),
        reducer_protocol_sha256=E2_HALVING_PROTOCOL_SHA256,
        dispositions=(
            CellDisposition(
                cell_id=cell_id,
                status=DispositionStatus.ACTIVATED,
                reason_code="caller_selected",
            ),
        ),
    )
    activation_path = tmp_path / "caller-authored-e2-activation.json"
    _write_bound(
        activation_path,
        reducer_activation_artifact_to_dict(activation),
    )
    with pytest.raises(ValueError, match="bound raw activation manifest"):
        _load_stage_activation_plan(str(activation_path))


def test_e2_cli_rejects_old_schema_and_requires_path_only_itl_authority(
    tmp_path: Path,
) -> None:
    old_manifest = tmp_path / "old-e2-stage.json"
    _write_bound(
        old_manifest,
        {
            "schema_version": 2,
            "kind": "industrial_e2_stage_reduction_manifest",
            "registry_artifact": None,
            "e1_receipt": None,
            "pareto": None,
            "stage_index": 0,
            "prior_stage_manifest": None,
            "gpu_inventory": None,
            "hardware_envelope": {},
            "cells": [],
        },
    )
    with pytest.raises(ValueError, match="identity is invalid"):
        _load_e2_stage_manifest(old_manifest)

    float_manifest = tmp_path / "float-e2-stage.json"
    float_value = json.loads(old_manifest.read_text(encoding="utf-8"))
    float_value["schema_version"] = 3.0
    _write_bound(float_manifest, float_value)
    with pytest.raises(ValueError, match="identity is invalid"):
        _load_e2_stage_manifest(float_manifest)

    legacy_cell = {
        "cell_id": _sha("legacy-e2-cell"),
        "terminal_receipts": [],
        "hardware_receipt": {},
        "budget_observation": {},
    }
    with pytest.raises(ValueError, match="cell fields do not match schema"):
        _analysis_cells(old_manifest.resolve(), [legacy_cell])


def test_confirmation_family_reducers_emit_exact_pilots_and_final_prefix(
    tmp_path: Path,
) -> None:
    registry_path, registry, family, family_path = _confirmation_family_inputs(tmp_path)
    pilot_path = tmp_path / "family-pilots.json"
    assert (
        main(
            [
                "materialize-confirmation-pilots",
                "--registry",
                str(registry_path),
                "--family",
                str(family_path),
                "--output",
                str(pilot_path),
            ]
        )
        == 0
    )
    pilot = family_activation_artifact_from_dict(
        json.loads(pilot_path.read_text(encoding="utf-8"))
    )
    assert pilot.activation_round == "excluded_pilots"
    if not pilot.activated_cell_ids:
        assert {
            row.reason_code
            for row in pilot.dispositions
            if row.status is DispositionStatus.BLOCKED
        } >= {
            "tts_official_recipe_unavailable",
            "sealed_e2_recipe_receipt_required",
        }
        return
    assert len(pilot.activated_cell_ids) == 20

    multipliers = (0.99, 1.01, 1.00, 1.02)
    power = preregister_power_sizing(
        tuple(
            PilotBlock(
                block_id=family_pilot_block_id(family, block),
                static_goodput=100.0,
                tts_goodput=101.0,
                l0_goodput=103.0 * multiplier,
            )
            for block, multiplier in zip(PILOT_BLOCKS, multipliers, strict=True)
        )
    )
    plan = _family_power_reduction(registry, family, pilot, power)
    power_path = tmp_path / "family-power.json"
    _write_bound(
        power_path,
        confirmation_family_power_reduction_artifact_to_dict(plan),
    )
    final_path = tmp_path / "family-final.json"
    with pytest.raises(ValueError, match="family power manifest fields"):
        main(
            [
                "materialize-confirmation-prefix",
                "--power-manifest",
                str(power_path),
                "--output",
                str(final_path),
            ]
        )
    final_fixture = materialize_confirmation_prefix(
        registry,
        family=family,
        reduction=plan,
        pilot_activation=pilot,
    )
    _write_bound(final_path, family_activation_artifact_to_dict(final_fixture))
    final = family_activation_artifact_from_dict(
        json.loads(final_path.read_text(encoding="utf-8"))
    )
    assert final.activation_round == "final_prefix"
    assert len(final.activated_cell_ids) == 5 * plan.selected_final_blocks

    inventory_path, envelope_path, _ = _pool_inputs(
        tmp_path,
        json.loads(registry_path.read_text(encoding="utf-8")),
    )
    by_cell_id = {cell.cell_id: cell for cell in registry.cells}
    budget_path = tmp_path / "family-final-budgets.json"
    _write_bound(
        budget_path,
        experiment_budget_sequence_to_dict(
            tuple(
                _standard_budget(by_cell_id[cell_id])
                for cell_id in final.activated_cell_ids
            )
        ),
    )
    report_path = tmp_path / "family-final-budget-report.json"
    with pytest.raises(ValueError, match="family power manifest fields"):
        main(
            [
                "estimate-industrial-budget",
                "--registry",
                str(registry_path),
                "--family-activation",
                str(pilot_path),
                "--family-activation",
                str(final_path),
                "--family-power-plan",
                str(power_path),
                "--inventory",
                str(inventory_path),
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(budget_path),
                "--budget-policy",
                str(tmp_path / "policy.json"),
                "--capacity-envelope",
                str(tmp_path / "capacity.json"),
                "--output",
                str(report_path),
            ]
        )
    assert not report_path.exists()
    with pytest.raises(SystemExit):
        main(
            [
                "estimate-industrial-budget",
                "--registry",
                str(registry_path),
                "--confirmation-plan",
                str(final_path),
                "--inventory",
                str(inventory_path),
                "--interference-envelope",
                str(envelope_path),
                "--budget-plan",
                str(budget_path),
                "--budget-policy",
                str(tmp_path / "policy.json"),
                "--capacity-envelope",
                str(tmp_path / "capacity.json"),
                "--output",
                str(tmp_path / "legacy-budget-report.json"),
            ]
        )


def test_alias_cli_replays_raw_authority_and_dependence_uses_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path, registry_value = _build_registry(tmp_path)
    inventory_path, _, _ = _pool_inputs(tmp_path, registry_value)
    hardware_value = {
        "gpu_clock_mhz_min": 1500.0,
        "gpu_clock_mhz_max": 2100.0,
        "memory_clock_mhz_min": 1000.0,
        "memory_clock_mhz_max": 1500.0,
        "temperature_c_max": 80.0,
        "power_watts_min": 100.0,
        "power_watts_max": 600.0,
        "power_state": "P0",
        "allowed_throttling_reasons": [],
        "allowed_background_processes": [],
    }
    hardware_path = tmp_path / "hardware-envelope.json"
    _write_bound(hardware_path, hardware_value)
    raw_manifest = _raw_alias_manifest(tmp_path)
    raw_manifest_path = tmp_path / "raw-alias-manifest.json"
    _write_bound(
        raw_manifest_path,
        raw_evidence_alias_manifest_to_dict(raw_manifest),
    )
    reduced = _alias_artifact()
    replayed: list[str] = []

    def replay_alias(*, registry, manifest, hardware_envelope, inventory):
        assert isinstance(manifest, RawEvidenceAliasManifest)
        assert manifest == raw_manifest
        assert registry.sha256 == registry_value["registry_sha256"]
        assert hardware_envelope == HardwareEnvelope(
            gpu_clock_mhz_min=1500.0,
            gpu_clock_mhz_max=2100.0,
            memory_clock_mhz_min=1000.0,
            memory_clock_mhz_max=1500.0,
            temperature_c_max=80.0,
            power_watts_min=100.0,
            power_watts_max=600.0,
            power_state="P0",
        )
        assert isinstance(inventory, GpuInventory)
        replayed.append(manifest.sha256)
        return reduced

    monkeypatch.setattr(
        "lightcone_spec.cli.main.reduce_evidence_alias",
        replay_alias,
    )
    reduction_path = tmp_path / "alias-reduction.json"
    assert (
        main(
            [
                "validate-evidence-alias",
                "--manifest",
                str(raw_manifest_path),
                "--registry",
                str(registry_path),
                "--inventory",
                str(inventory_path),
                "--hardware-envelope",
                str(hardware_path),
                "--output",
                str(reduction_path),
            ]
        )
        == 0
    )
    assert replayed == [raw_manifest.sha256]
    assert (
        evidence_alias_reduction_artifact_from_dict(
            json.loads(reduction_path.read_text(encoding="utf-8"))
        )
        == reduced
    )

    direct = build_evidence_dependence_map(
        direct_observation_cell_ids=(reduced.source_cell_id,), aliases=()
    )
    direct_path = tmp_path / "direct-map.json"
    _write_bound(direct_path, evidence_dependence_map_to_dict(direct))
    output = tmp_path / "dependence-map.json"
    assert (
        main(
            [
                "build-evidence-dependence-map",
                "--direct-map",
                str(direct_path),
                "--alias-reduction",
                str(reduction_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    dependence = evidence_dependence_map_from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert dependence.independent_unit_count == 1
    assert dependence.units[0].member_cell_ids == tuple(
        sorted((reduced.source_cell_id, reduced.target_cell_id))
    )


def test_alias_cli_rejects_legacy_flags_receipts_and_reduction_summaries(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy-alias-receipt.json"
    _write_bound(
        legacy_path,
        {
            "schema_version": 1,
            "artifact_kind": "evidence_alias_receipt",
            "artifact_sha256": _sha("legacy-alias-receipt"),
            "source": {},
            "target": {},
        },
    )
    reduction_path = tmp_path / "caller-authored-alias-reduction.json"
    _write_bound(
        reduction_path,
        evidence_alias_reduction_artifact_to_dict(_alias_artifact()),
    )
    with pytest.raises(SystemExit):
        main(
            [
                "validate-evidence-alias",
                "--alias",
                str(legacy_path),
                "--output",
                str(tmp_path / "legacy-output.json"),
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "build-evidence-dependence-map",
                "--direct-map",
                str(reduction_path),
                "--alias",
                str(legacy_path),
                "--output",
                str(tmp_path / "legacy-dependence.json"),
            ]
        )
    for manifest_path in (legacy_path, reduction_path):
        with pytest.raises(ValueError, match="ambiguous schema"):
            main(
                [
                    "validate-evidence-alias",
                    "--manifest",
                    str(manifest_path),
                    "--registry",
                    str(manifest_path),
                    "--inventory",
                    str(manifest_path),
                    "--hardware-envelope",
                    str(manifest_path),
                    "--output",
                    str(tmp_path / f"rejected-{manifest_path.name}"),
                ]
            )


def test_analyze_industrial_forwards_raw_alias_manifests_to_formal_reducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_manifest = _raw_alias_manifest(tmp_path)
    runtime_metrics_authority = object()
    loaded = (
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
        (object(),),
        (raw_manifest,),
        None,
        None,
        None,
        100,
        17,
        runtime_metrics_authority,
    )
    monkeypatch.setattr(
        "lightcone_spec.cli.main._load_industrial_analysis_manifest",
        lambda _path: loaded,
    )
    artifact_value = {"kind": "raw_alias_forwarding_probe"}
    artifact = SimpleNamespace(
        sha256=_sha(artifact_value),
        to_dict=lambda: artifact_value,
    )
    received: list[tuple[tuple[RawEvidenceAliasManifest, ...], object]] = []

    def reduce_probe(**kwargs):
        received.append(
            (
                kwargs["evidence_alias_manifests"],
                kwargs["runtime_metrics_authority"],
            )
        )
        return SimpleNamespace(artifact=artifact)

    monkeypatch.setattr(
        "lightcone_spec.cli.main.reduce_industrial_schema_v3",
        reduce_probe,
    )
    output = tmp_path / "analysis-output.json"
    assert (
        main(
            [
                "analyze-industrial",
                "--manifest",
                str(tmp_path / "analysis-manifest.json"),
                "--output",
                str(output),
            ]
        )
        == 42
    )
    assert received == [((raw_manifest,), runtime_metrics_authority)]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact_value
