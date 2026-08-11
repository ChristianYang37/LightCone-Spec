from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from lightcone_spec import (
    PINNED_SGLANG_COMMIT,
    PINNED_SGLANG_PATCH_COUNT,
    PINNED_SGLANG_TREE,
)
from lightcone_spec.cli.main import main
from lightcone_spec.experiments.industrial_analysis import (
    _INDUSTRIAL_DOCTOR_CHECKS,
    BoundArtifact,
    IndustrialBlockEvidence,
    IndustrialCellEvidence,
    _validate_industrial_doctor,
    _validate_industrial_gpu_attestation,
    _validate_run_row,
    industrial_completed_pilot_cells_sha256,
    industrial_pilot_evidence_sha256,
    reduce_industrial_schema_v3,
)
from lightcone_spec.experiments.registry import (
    CORE_METHODS,
    FINAL_BLOCKS,
    PILOT_BLOCKS,
    ConfirmationBlockPlan,
    ExperimentCell,
    ExperimentRegistry,
    build_industrial_registry,
    content_sha256,
)
from lightcone_spec.experiments.statistics import HardwareEnvelope
from lightcone_spec.telemetry import (
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
    load_completed_evidence,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> BoundArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return BoundArtifact(path=path, sha256=_file_sha256(path))


def _write_bound_json(path: Path, value: object) -> None:
    _write_json(path, value)
    Path(f"{path}.sha256").write_text(
        content_sha256(value) + "\n",
        encoding="utf-8",
    )


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


def _passing_doctor(registry: ExperimentRegistry) -> dict:
    devices = [
        {
            "uuid": gpu_uuid,
            "name": "NVIDIA H200",
            "memory_total_mib": 141_000,
            "driver_version": "580.65.06",
            "compute_capability": "9.0",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
        }
        for index, gpu_uuid in enumerate(registry.gpu_uuids)
    ]
    topology = {
        "gpu_rows": ["GPU0", "GPU1"],
        "pair_link": "NV18",
        "reciprocal_link": "NV18",
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
            "two_gpu_visible": True,
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
    registry: ExperimentRegistry,
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
        "confirmation_plan_sha256": artifact.confirmation_plan_sha256,
        "patched_sglang_tree": artifact.patched_sglang_tree,
        "model_lock_sha256": artifact.model_lock_sha256,
        "hardware_envelope_sha256": artifact.hardware_envelope_sha256,
        "pilot_evidence_sha256": artifact.pilot_evidence_sha256,
        "completed_pilot_cells_sha256": artifact.completed_pilot_cells_sha256,
        "gpu_uuids": list(registry.gpu_uuids),
        "terminal_receipt_sha256s": list(artifact.terminal_receipt_sha256s),
        "qualification_lock_sha256s": list(artifact.qualification_lock_sha256s),
        "hardware_receipt_sha256s": list(artifact.hardware_receipt_sha256s),
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


def _slice_cells(registry: ExperimentRegistry) -> dict[int, dict[str, ExperimentCell]]:
    selected: dict[int, dict[str, ExperimentCell]] = {}
    for block in PILOT_BLOCKS + FINAL_BLOCKS[:12]:
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


def _goodput(block: int, method: str) -> float:
    if method == "target_only":
        return 90.0
    if method == "static":
        return 100.0
    if method == "tts":
        return 101.0
    pilot_multiplier = (0.99, 1.01, 1.00, 1.02)
    return 103.0 * pilot_multiplier[block] if block in PILOT_BLOCKS else 104.0


def _build_evidence(
    tmp_path: Path,
) -> tuple[
    ExperimentRegistry,
    ConfirmationBlockPlan,
    tuple[IndustrialBlockEvidence, ...],
    HardwareEnvelope,
]:
    registry = build_industrial_registry(
        gpu_uuids=("GPU-analysis-a", "GPU-analysis-b"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    runtime_sha256 = content_sha256({"runtime": "analysis-test"})
    split_sha256 = content_sha256({"split": "analysis-test"})
    envelope = _hardware_envelope()
    block_cells = _slice_cells(registry)
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
            run_id = f"analysis-{block}-{method.replace('_', '-')}"
            evidence_root = Path(cell.resources.evidence_root)
            writer = EvidenceWriter(
                evidence_root,
                run_id=run_id,
                rank=0,
                process_id=block * 10 + CORE_METHODS.index(method) + 1,
                checkpoint_interval_s=None,
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
                    rank_config_sha256=content_sha256(
                        {"rank_config": cell.cell_id, "rank": 0}
                    ),
                    runtime_sha256=runtime_sha256,
                    split_sha256=split_sha256,
                    **identities,
                    patched_sglang_tree=PINNED_SGLANG_TREE,
                    run_nonce_sha256=content_sha256(
                        {"nonce": cell.cell_id, "run": run_id}
                    ),
                    topology_sha256=content_sha256(
                        {
                            "schema_version": 1,
                            "cell_id": cell.cell_id,
                            "topology": cell.identity.topology,
                            "gpu_uuids": list(cell.resources.gpu_uuids),
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
                )
            )
            arrival_ns = 1_000_000
            completion_ns = arrival_ns + round(
                100 / _goodput(block, method) * 1_000_000_000
            )
            output_token_ids = tuple(range(100, 200))
            output_token_ids_json = json.dumps(output_token_ids, separators=(",", ":"))
            output_token_ids_sha256 = hashlib.sha256(
                output_token_ids_json.encode("utf-8")
            ).hexdigest()
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
            writer.close()
            terminal_path = evidence_root / f"{run_id}.rank0.complete.json"
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
                            "gpu_uuids": list(cell.resources.gpu_uuids),
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
                            "gpu_uuid": cell.resources.gpu_uuids[0],
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
    plan = ConfirmationBlockPlan(
        registry_sha256=registry.sha256,
        experiment="E3b",
        runtime_sha256=runtime_sha256,
        split_sha256=split_sha256,
        pilot_evidence_sha256=industrial_pilot_evidence_sha256(evidence),
        completed_pilot_cells_sha256=(
            industrial_completed_pilot_cells_sha256(evidence)
        ),
        status="POWERED",
        selected_final_blocks=12,
        reason_code="pilot_power_locked",
    )
    return registry, plan, evidence, envelope


def _bound_reference(reference: BoundArtifact) -> dict[str, str]:
    return {"path": str(reference.path), "sha256": reference.sha256}


def _analysis_manifest(
    tmp_path: Path,
    *,
    registry: ExperimentRegistry,
    plan: ConfirmationBlockPlan,
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
                "--gpu-uuid",
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
    plan_path = tmp_path / f"{name}-plan.json"
    _write_bound_json(plan_path, asdict(plan))
    manifest = {
        "schema_version": 1,
        "kind": "industrial_analysis_manifest",
        "registry_artifact": {
            "path": str(registry_path),
            "sha256": content_sha256(registry_value),
        },
        "confirmation_plan": {
            "path": str(plan_path),
            "sha256": content_sha256(asdict(plan)),
        },
        "gpu_attestation": (
            None if gpu_attestation is None else _bound_reference(gpu_attestation)
        ),
        "doctor_report": (
            None if doctor_report is None else _bound_reference(doctor_report)
        ),
        "hardware_envelope": asdict(envelope),
        "bootstrap": {"repetitions": 300, "seed": 17},
        "blocks": [
            {
                "block": block.block,
                "qualification_lock": _bound_reference(block.qualification_lock),
                "cells": [
                    {
                        "cell_id": cell.cell_id,
                        "terminal_receipts": [
                            _bound_reference(receipt)
                            for receipt in cell.terminal_receipts
                        ],
                        "hardware_receipt": _bound_reference(cell.hardware_receipt),
                    }
                    for cell in block.cells
                ],
            }
            for block in evidence
        ],
    }
    manifest_path = tmp_path / f"{name}-manifest.json"
    _write_bound_json(manifest_path, manifest)
    return manifest_path


@pytest.fixture(scope="module")
def evidence_bundle(tmp_path_factory: pytest.TempPathFactory):
    return _build_evidence(tmp_path_factory.mktemp("industrial-analysis"))


def test_industrial_attestation_contract_binds_doctor_gpu_and_run_chain(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, _, _, _ = evidence_bundle
    doctor_value = _passing_doctor(registry)
    doctor = _write_json(tmp_path / "doctor.json", doctor_value)
    _validate_industrial_doctor(doctor, registry=registry)

    mismatched_manifest = json.loads(json.dumps(doctor_value))
    mismatched_manifest["compatibility"]["manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="runtime-manifest digests"):
        _validate_industrial_doctor(
            _write_json(
                tmp_path / "doctor-manifest-mismatch.json", mismatched_manifest
            ),
            registry=registry,
        )

    mismatched_gpu = json.loads(json.dumps(doctor_value))
    mismatched_gpu["gpu"]["parsed_inventory"]["devices"][0]["uuid"] = "GPU-other"
    mismatched_gpu["checks"]["gpu_identity"]["observed"][0]["uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="two registry GPU UUIDs"):
        _validate_industrial_doctor(
            _write_json(tmp_path / "doctor-gpu-mismatch.json", mismatched_gpu),
            registry=registry,
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


def test_static_terminal_evidence_has_no_round_or_update_trace_state(
    evidence_bundle,
) -> None:
    registry, plan, evidence, _ = evidence_bundle
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
            plan=plan,
            cell=cell,
            rank=0,
        )


def test_fully_matching_synthetic_attestation_remains_unmeasured(
    evidence_bundle,
    tmp_path: Path,
) -> None:
    registry, plan, evidence, envelope = evidence_bundle
    diagnostic = reduce_industrial_schema_v3(
        registry=registry,
        confirmation_plan=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        bootstrap_repetitions=100,
        bootstrap_seed=23,
    )
    doctor = _write_json(tmp_path / "synthetic-doctor.json", _passing_doctor(registry))
    attestation = _write_json(
        tmp_path / "synthetic-attestation.json",
        _synthetic_attestation(registry, doctor, diagnostic.artifact),
    )
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        confirmation_plan=plan,
        blocks=evidence,
        hardware_envelope=envelope,
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

    manifest = _analysis_manifest(
        tmp_path,
        registry=registry,
        plan=plan,
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
    registry, plan, evidence, envelope = evidence_bundle
    reduction = reduce_industrial_schema_v3(
        registry=registry,
        confirmation_plan=plan,
        blocks=evidence,
        hardware_envelope=envelope,
        bootstrap_repetitions=300,
        bootstrap_seed=17,
    )
    repeated = reduce_industrial_schema_v3(
        registry=registry,
        confirmation_plan=plan,
        blocks=evidence,
        hardware_envelope=envelope,
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


def test_reducer_fails_closed_on_missing_paired_method_or_rank(evidence_bundle) -> None:
    registry, plan, evidence, envelope = evidence_bundle
    with pytest.raises(ValueError, match="must be supplied together"):
        reduce_industrial_schema_v3(
            registry=registry,
            confirmation_plan=plan,
            blocks=evidence,
            hardware_envelope=envelope,
            gpu_attestation=BoundArtifact(Path("missing.json"), "a" * 64),
            bootstrap_repetitions=100,
        )

    missing_method = replace(evidence[-1], cells=evidence[-1].cells[:-1])
    with pytest.raises(ValueError, match="exactly Target-only/Static/TTS/L0"):
        reduce_industrial_schema_v3(
            registry=registry,
            confirmation_plan=plan,
            blocks=(*evidence[:-1], missing_method),
            hardware_envelope=envelope,
            bootstrap_repetitions=100,
        )

    with pytest.raises(ValueError, match="terminal rank receipts"):
        replace(evidence[-1].cells[0], terminal_receipts=())


def test_hardware_invalidation_suppresses_all_contrasts(
    evidence_bundle,
) -> None:
    registry, plan, evidence, envelope = evidence_bundle
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
        confirmation_plan=plan,
        blocks=(*evidence[:-1], invalid_block),
        hardware_envelope=envelope,
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
    registry, plan, evidence, envelope = evidence_bundle
    manifest_path = _analysis_manifest(
        tmp_path,
        registry=registry,
        plan=plan,
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
    registry, plan, evidence, envelope = evidence_bundle
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
        plan=plan,
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
