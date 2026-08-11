from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from lightcone_spec import PINNED_SGLANG_TREE
from lightcone_spec.cli.main import _completed_industrial_cells, main
from lightcone_spec.experiments.evidence import evidence_files_sha256
from lightcone_spec.experiments.registry import (
    CellStatus,
    ExperimentCell,
    ExperimentRegistry,
    build_industrial_registry,
)
from lightcone_spec.telemetry import (
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


def _write_bound(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{path}.sha256").write_text(_sha(value) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology(cell: ExperimentCell) -> tuple[int, int, int, str]:
    tensor_parallel_size = 2 if cell.identity.topology == "tp2_dp1" else 1
    data_parallel_size = 2 if cell.identity.topology == "two_replica_tp1_dp2" else 1
    world_size = len(cell.resources.gpu_uuids)
    digest = _sha(
        {
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "topology": cell.identity.topology,
            "gpu_uuids": list(cell.resources.gpu_uuids),
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_size": data_parallel_size,
            "world_size": world_size,
        }
    )
    return tensor_parallel_size, data_parallel_size, world_size, digest


def _locked_cell(cell: ExperimentCell) -> dict[str, object]:
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
    }
    if cell.identity.experiment != "preflight":
        contract["rank_config_sha256s"] = [
            _sha({"kind": "rank_config", "cell_id": cell.cell_id, "rank": rank})
            for rank in range(len(cell.resources.gpu_uuids))
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
        output_sha256=output_sha256,
        ttft_ms=ttft_ms,
        finished=finished,
        stop_reason="length" if finished else "cancelled_before_first_token",
        outcome_status="completed" if finished else "cancelled",
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
                "--gpu-uuid",
                "GPU-industrial-a",
                "GPU-industrial-b",
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
        gpu_uuids=("GPU-industrial-a", "GPU-industrial-b"),
        cache_root=cache_root,
        evidence_root=evidence_root,
    )
    return registry_path, registry


def _preflight_bundle(
    tmp_path: Path,
    *,
    invalid_attestation: bool = False,
    mismatched_rank_output: bool = False,
    reused_nonce: bool = False,
) -> dict[str, object]:
    registry_path, registry = _build_registry(tmp_path)
    runtime = {"schema_version": 1, "kind": "industrial_runtime_test"}
    runtime_path = tmp_path / "runtime.json"
    _write_bound(runtime_path, runtime)
    runtime_sha256 = _sha(runtime)

    cells = tuple(cell for cell in registry.cells_for("preflight") if cell.runnable)
    contracts = {cell.cell_id: _locked_cell(cell) for cell in cells}
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

    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(cells):
        contract = contracts[cell.cell_id]
        request_id = str(contract["request_ids"][0])
        run_id = f"preflight-{cell.cell_id[:16]}"
        run_nonce_sha256 = _sha(
            {
                "kind": "run_nonce",
                "cell_id": None if reused_nonce else cell.cell_id,
            }
        )
        tp_size, dp_size, world_size, topology_sha256 = _topology(cell)
        for rank, gpu_uuid in enumerate(cell.resources.gpu_uuids):
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
                    runtime_sha256=runtime_sha256,
                    split_sha256=split_sha256,
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
                    "preflight_attestation_path": str(attestation_path),
                    "preflight_attestation_sha256": attestation_sha256,
                    "status": "MEASURED",
                }
            )

    completed = {
        "schema_version": 2,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry.sha256,
        "experiment": "preflight",
        "runtime_sha256": runtime_sha256,
        "split_sha256": split_sha256,
        "split_contract": split,
        "rows": rows,
    }
    completed_path = tmp_path / "completed.json"
    _write_bound(completed_path, completed)
    locked_output = tmp_path / "runtime-envelope.json"
    _write_bound(locked_output, {"schema_version": 1, "status": "PASS"})
    return {
        "registry": registry,
        "registry_path": registry_path,
        "runtime": runtime,
        "runtime_path": runtime_path,
        "split": split,
        "split_path": split_path,
        "completed": completed,
        "completed_path": completed_path,
        "locked_output": locked_output,
    }


def _validate_bundle(
    bundle: dict[str, object], path: Path
) -> tuple[tuple[str, ...], str]:
    registry = bundle["registry"]
    runtime = bundle["runtime"]
    split = bundle["split"]
    assert isinstance(registry, ExperimentRegistry)
    assert isinstance(split, dict)
    completed, digest = _completed_industrial_cells(
        str(path),
        registry,
        experiment="preflight",
        runtime_sha256=_sha(runtime),
        split_sha256=_sha(split),
        split_contract=split,
        require_industrial_contract=True,
    )
    assert digest is not None
    return completed, digest


def test_preflight_stage_cannot_seal_without_trusted_attester(tmp_path: Path) -> None:
    bundle = _preflight_bundle(tmp_path)
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    completed, digest = _validate_bundle(bundle, completed_path)
    registry = bundle["registry"]
    assert isinstance(registry, ExperimentRegistry)
    assert set(completed) == {
        cell.cell_id for cell in registry.cells_for("preflight") if cell.runnable
    }
    assert digest == _sha(bundle["completed"])

    receipt_path = tmp_path / "preflight-receipt.json"
    assert (
        main(
            [
                "seal-industrial-stage",
                "--registry",
                str(bundle["registry_path"]),
                "--experiment",
                "preflight",
                "--runtime-artifact",
                str(bundle["runtime_path"]),
                "--split-artifact",
                str(bundle["split_path"]),
                "--completed-cells",
                str(completed_path),
                "--locked-output",
                f"runtime_envelope={bundle['locked_output']}",
                "--output",
                str(receipt_path),
            ]
        )
        == 42
    )
    decision = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert decision == {
        "schema_version": 1,
        "kind": "industrial_stage_seal_decision",
        "status": "BLOCKED",
        "gpu_evidence": "UNMEASURED",
        "reason_code": "trusted_hardware_attester_unavailable",
        "registry_sha256": registry.sha256,
        "experiment": "preflight",
        "trusted_attester_id": None,
    }

    wrong_root = copy.deepcopy(bundle["completed"])
    wrong_root["rows"][0]["evidence_root"] = str(tmp_path / "wrong-root")
    wrong_root_path = tmp_path / "wrong-root-completed.json"
    _write_bound(wrong_root_path, wrong_root)
    with pytest.raises(ValueError, match="resource claim"):
        _validate_bundle(bundle, wrong_root_path)

    missing_rank = copy.deepcopy(bundle["completed"])
    missing_rank["rows"].pop()
    missing_rank_path = tmp_path / "missing-rank-completed.json"
    _write_bound(missing_rank_path, missing_rank)
    with pytest.raises(ValueError, match="one measured outcome per claimed GPU"):
        _validate_bundle(bundle, missing_rank_path)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("attestation", "preflight attestation contract is incomplete"),
        ("consensus", "cell ranks do not agree"),
        ("nonce", "run nonce is reused"),
    ],
)
def test_preflight_rejects_task_or_rank_contract_tamper(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    bundle = _preflight_bundle(
        tmp_path,
        invalid_attestation=failure == "attestation",
        mismatched_rank_output=failure == "consensus",
        reused_nonce=failure == "nonce",
    )
    completed_path = bundle["completed_path"]
    assert isinstance(completed_path, Path)
    with pytest.raises(ValueError, match=message):
        _validate_bundle(bundle, completed_path)


def test_blocked_and_not_applicable_outcomes_are_explicit_and_exact(
    tmp_path: Path,
) -> None:
    registry = build_industrial_registry(
        gpu_uuids=("GPU-outcome-a", "GPU-outcome-b"),
        cache_root=str(tmp_path / "cache"),
        evidence_root=str(tmp_path / "evidence"),
    )
    blocked = next(
        cell for cell in registry.cells_for("E1a") if cell.status is CellStatus.BLOCKED
    )
    not_applicable = blocked.with_status(
        CellStatus.NOT_APPLICABLE,
        reason_code="test_not_applicable",
        reason="The dedicated contract test marks this declaration not applicable.",
    )
    registry = replace(
        registry,
        cells=tuple(
            not_applicable if cell.cell_id == blocked.cell_id else cell
            for cell in registry.cells
        ),
    )
    stage_cells = registry.cells_for("E1a")
    contracts = [_locked_cell(cell) for cell in stage_cells if cell.runnable]
    split = {
        "schema_version": 1,
        "kind": "industrial_locked_split",
        "registry_sha256": registry.sha256,
        "experiment": "E1a",
        "cells": contracts,
    }
    runtime_sha256 = _sha({"kind": "runtime", "stage": "E1a"})

    exact_outcomes = [
        {
            "cell_id": cell.cell_id,
            "status": cell.status.value,
            "reason_code": cell.reason_code,
            "reason": cell.reason,
        }
        for cell in stage_cells
        if not cell.runnable
    ]
    measured_placeholders = [
        {"cell_id": cell.cell_id, "status": "MEASURED"}
        for cell in stage_cells
        if cell.runnable
    ]
    bad_na = next(
        row
        for row in exact_outcomes
        if row["status"] == CellStatus.NOT_APPLICABLE.value
    )
    bad_na = {**bad_na, "reason_code": "edited_reason"}
    rows = [bad_na, *measured_placeholders]
    rows.extend(
        row for row in exact_outcomes if row["cell_id"] != not_applicable.cell_id
    )
    invalid_na = {
        "schema_version": 2,
        "kind": "industrial_completed_cells",
        "registry_sha256": registry.sha256,
        "experiment": "E1a",
        "runtime_sha256": runtime_sha256,
        "split_sha256": _sha(split),
        "split_contract": split,
        "rows": rows,
    }
    invalid_na_path = tmp_path / "invalid-na.json"
    _write_bound(invalid_na_path, invalid_na)
    with pytest.raises(ValueError, match="BLOCKED/N/A"):
        _completed_industrial_cells(
            str(invalid_na_path),
            registry,
            experiment="E1a",
            runtime_sha256=runtime_sha256,
            split_sha256=_sha(split),
            split_contract=split,
            require_industrial_contract=True,
        )

    omitted = next(
        row for row in exact_outcomes if row["status"] == CellStatus.BLOCKED.value
    )
    missing_blocked = {
        **invalid_na,
        "rows": [
            *measured_placeholders,
            *(row for row in exact_outcomes if row["cell_id"] != omitted["cell_id"]),
        ],
    }
    missing_blocked_path = tmp_path / "missing-blocked.json"
    _write_bound(missing_blocked_path, missing_blocked)
    with pytest.raises(ValueError, match="do not cover every declared stage cell"):
        _completed_industrial_cells(
            str(missing_blocked_path),
            registry,
            experiment="E1a",
            runtime_sha256=runtime_sha256,
            split_sha256=_sha(split),
            split_contract=split,
            require_industrial_contract=True,
        )


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
