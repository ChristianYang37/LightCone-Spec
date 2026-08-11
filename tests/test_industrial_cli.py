from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lightcone_spec.cli.main import main
from lightcone_spec.experiments.evidence import evidence_files_sha256
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
                output_sha256="d" * 64,
                ttft_ms=1.0,
                finished=True,
                stop_reason="length",
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
                "--gpu-uuid",
                "GPU-aaaaaaaa",
                "GPU-bbbbbbbb",
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


def _interference_condition(
    registry: dict,
    *,
    block: int,
    condition: str,
) -> dict[str, object]:
    cell = next(
        row
        for row in registry["registry"]["cells"]
        if row["identity"]["experiment"] == "preflight"
        and row["identity"]["task"] == "simultaneous_single_gpu_interference"
    )
    cell_id = _sha(cell["identity"])
    root = Path(cell["resources"]["evidence_root"])
    run_id = f"interference-{condition}-block-{block}"
    bindings: list[dict[str, str]] = []
    for rank, gpu_uuid in enumerate(registry["registry"]["gpu_uuids"]):
        writer = EvidenceWriter(
            root,
            run_id=run_id,
            rank=rank,
            process_id=20_000
            + block * 10
            + rank
            + (0 if condition == "isolated" else 2),
            checkpoint_interval_s=None,
        )
        writer.write(
            RunRecord(
                run_id=run_id,
                manifest_sha256=registry["registry_sha256"],
                config_sha256=cell_id,
                method="static",
                model_pair=cell["identity"]["model"],
                repetition_block=block,
                started_ns=1,
                completed_ns=2,
                status="complete",
            )
        )
        arrival_ns = 1_000_000_000
        completed_ns = 2_000_000_000
        writer.write(
            RequestRecord(
                run_id=run_id,
                request_id=f"request-{condition}-{block}-{rank}",
                prompt_id="interference",
                method="static",
                repetition_block=block,
                concurrency=1,
                input_tokens=8,
                output_tokens=100,
                output_hash_format=OUTPUT_HASH_FORMAT,
                output_sha256=_sha(
                    {"condition": condition, "block": block, "rank": rank}
                ),
                ttft_ms=10.0,
                finished=True,
                stop_reason="length",
                outcome_status="completed",
                arrival_ns=arrival_ns,
                queue_enter_ns=arrival_ns,
                admitted_ns=arrival_ns,
                first_token_ns=arrival_ns + 10_000_000,
                completed_ns=completed_ns,
                token_timestamps_ns=json.dumps(
                    [arrival_ns + 10_000_000 * index for index in range(1, 101)]
                ),
                inter_token_ms=json.dumps([10.0] * 99),
                token_timing_coverage=1.0,
                coalesced_intervals=0,
                admission_code="admitted",
            )
        )
        writer.write(_performance(run_id, "static", repetition_block=block))
        writer.close()
        terminal = root / f"{run_id}.rank{rank}.complete.json"
        bindings.append(
            {
                "path": str(terminal),
                "sha256": _file_sha(terminal),
                "gpu_uuid": gpu_uuid,
            }
        )
    return {"terminal_receipts": bindings}


def test_registry_planner_is_serial_until_interference_is_attested(
    tmp_path: Path,
) -> None:
    registry_path, registry = _build_registry(tmp_path)
    first_plan = tmp_path / "preflight-plan.json"
    assert (
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--output",
                str(first_plan),
            ]
        )
        == 0
    )
    preflight = json.loads(first_plan.read_text(encoding="utf-8"))
    assert preflight["experiment"] == "preflight"
    assert preflight["interference_gate"] == "NOT_PASSED"
    assert preflight["waves"]
    assert all(len(wave["cells"]) == 1 for wave in preflight["waves"])
    assert all(len(wave["cells"][0]["gpu_uuids"]) == 2 for wave in preflight["waves"])

    paired_blocks = [
        {
            "block_id": f"block-{index}",
            "isolated": _interference_condition(
                registry,
                block=index,
                condition="isolated",
            ),
            "simultaneous": _interference_condition(
                registry,
                block=index,
                condition="simultaneous",
            ),
        }
        for index in range(4)
    ]
    interference_source = tmp_path / "interference-calibration.json"
    _write_bound(
        interference_source,
        {
            "schema_version": 2,
            "kind": "two_gpu_interference_calibration",
            "registry_sha256": registry["registry_sha256"],
            "gpu_uuids": ["GPU-aaaaaaaa", "GPU-bbbbbbbb"],
            "paired_blocks": paired_blocks,
        },
    )
    interference = {
        "schema_version": 1,
        "kind": "two_gpu_interference_gate",
        "status": "PASS",
        "registry_sha256": registry["registry_sha256"],
        "gpu_uuids": ["GPU-aaaaaaaa", "GPU-bbbbbbbb"],
        "source_evidence_files": [str(interference_source)],
        "source_evidence_sha256": evidence_files_sha256((interference_source,)),
        "metrics": {
            name: {
                "mean_relative_difference": 0.0,
                "ci95_relative_difference": [0.0, 0.0],
                "paired_blocks": [f"block-{index}" for index in range(4)],
            }
            for name in ("throughput", "itl")
        },
    }
    interference_path = tmp_path / "interference.json"
    _write_bound(interference_path, interference)
    parallel_path = tmp_path / "e3a-parallel.json"
    assert (
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--interference-receipt",
                str(interference_path),
                "--output",
                str(parallel_path),
            ]
        )
        == 0
    )
    parallel = json.loads(parallel_path.read_text(encoding="utf-8"))
    assert parallel["interference_gate"] == "PASS"
    assert all(len(wave["cells"]) == 1 for wave in parallel["waves"])


def test_stage_sealing_is_blocked_before_cpu_evidence_can_mint_a_receipt(
    tmp_path: Path,
) -> None:
    registry_path, registry = _build_registry(tmp_path)
    runtime = tmp_path / "runtime.json"
    split = tmp_path / "split.json"
    locked = tmp_path / "runtime-envelope.json"
    _write_bound(runtime, {"runtime": "test"})
    _write_bound(split, {"split": "preflight"})
    _write_bound(locked, {"runtime_envelope": "passed"})
    completed = _completed_stage(tmp_path, registry, "preflight")
    output = tmp_path / "forged-receipt.json"
    assert (
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
                "--locked-output",
                f"runtime_envelope={locked}",
                "--output",
                str(output),
            ]
        )
        == 42
    )
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["status"] == "BLOCKED"
    assert decision["gpu_evidence"] == "UNMEASURED"
    assert decision["reason_code"] == "trusted_hardware_attester_unavailable"


def test_registry_and_receipt_edits_fail_closed(tmp_path: Path) -> None:
    registry_path, registry = _build_registry(tmp_path)
    registry["registry"]["name"] = "edited-after-generation"
    _write_bound(registry_path, registry)
    with pytest.raises(ValueError, match="edited after generation"):
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )


def test_parallel_dispatch_rejects_unattested_boolean_substitutes(
    tmp_path: Path,
) -> None:
    registry_path, registry = _build_registry(tmp_path)
    forged = {
        "schema_version": 1,
        "kind": "two_gpu_interference_gate",
        "status": "PASS",
        "registry_sha256": registry["registry_sha256"],
        "gpu_uuids": ["GPU-aaaaaaaa", "GPU-bbbbbbbb"],
    }
    forged_path = tmp_path / "forged.json"
    _write_bound(forged_path, forged)
    with pytest.raises(ValueError, match="matching PASS"):
        main(
            [
                "plan-industrial-dispatch",
                "--registry",
                str(registry_path),
                "--interference-receipt",
                str(forged_path),
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )
