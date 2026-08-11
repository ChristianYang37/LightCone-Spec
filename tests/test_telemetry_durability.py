from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import lightcone_spec.telemetry.writer as writer_module
from lightcone_spec.telemetry import (
    OUTPUT_HASH_FORMAT,
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RoundRecord,
    RunRecord,
    UpdateRecord,
    load_completed_evidence,
)


def run_record(run_id: str, method: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        method=method,
        model_pair="pair",
        repetition_block=0,
        started_ns=1,
        completed_ns=2,
        status="complete",
    )


def request_record(
    run_id: str,
    method: str,
    index: int = 0,
    **changes: object,
) -> RequestRecord:
    row = RequestRecord(
        run_id=run_id,
        request_id=f"request-{index}",
        prompt_id=f"prompt-{index}",
        method=method,
        repetition_block=0,
        concurrency=1,
        input_tokens=2,
        output_tokens=4,
        output_hash_format=OUTPUT_HASH_FORMAT,
        output_sha256="c" * 64,
        ttft_ms=1.0,
        finished=True,
        stop_reason="length",
    )
    return replace(row, **changes)


def performance_record(
    run_id: str,
    method: str,
    index: int = 0,
    **changes: object,
) -> PerformanceRecord:
    row = PerformanceRecord(
        run_id=run_id,
        prompt_id=f"prompt-{index}",
        method=method,
        repetition_block=0,
        region=f"region-{index}",
        concurrency=1,
        generated_bucket_start=index,
        generated_bucket_end=index + 2,
        at_risk_requests=1,
        output_tokens=2,
        elapsed_s=0.1,
        decode_goodput_tps=20.0,
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
        peak_hbm_bytes=100,
        kv_bytes=50,
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
        graph_replay_hit_rate=1.0,
        updates_launched=0,
        updates_published=0,
        exactness_violations=0,
        version_mismatches=0,
        fallbacks=0,
        nonfinite_updates=0,
        oom_events=0,
        retractions=0,
    )
    return replace(row, **changes)


def round_record(run_id: str) -> RoundRecord:
    return RoundRecord(
        run_id=run_id,
        request_id="request-0",
        round_index=0,
        generated_tokens_before=0,
        prefix_len_before=2,
        verify_len=4,
        accepted_drafts=2,
        committed_tokens=3,
        target_calls=1,
        proposal_source_version=0,
        kv_source_versions='[{"start":0,"end":2,"source_version":0}]',
    )


def update_record(run_id: str) -> UpdateRecord:
    return UpdateRecord(
        run_id=run_id,
        cohort_sha256="d" * 64,
        parameter_layout_sha256="e" * 64,
        update_index=0,
        request_ids='["request-0"]',
        prefix_len_before="[2]",
        prefix_len_min=2,
        prefix_len_max=2,
        prefix_len_mean=2.0,
        source_round=0,
        source_version=0,
        optimizer_step=1,
        published_version=1,
        candidate_status="published",
        loss=0.1,
        gradient_norm=0.2,
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


def test_bounded_writer_builds_multiple_row_groups_and_preserves_optional_schema(
    tmp_path,
) -> None:
    writer = EvidenceWriter(
        tmp_path,
        run_id="bounded",
        rank=0,
        process_id=7,
        max_queued_rows=3,
        row_group_rows=2,
        checkpoint_interval_s=None,
    )
    writer.write(run_record("bounded", "static"))
    writer.write(
        request_record(
            "bounded",
            "static",
            token_timing_coverage=None,
            coalesced_intervals=None,
        )
    )
    for index in range(5):
        writer.write(
            performance_record(
                "bounded",
                "static",
                index,
                power_watts=None if index < 2 else 250.0 + index,
            )
        )
        assert writer.queued_rows <= 3
    paths = writer.close()

    parquet = pq.ParquetFile(paths["performance"])
    assert parquet.metadata.num_row_groups == 3
    assert parquet.read(columns=["power_watts"])["power_watts"].to_pylist() == [
        None,
        None,
        252.0,
        253.0,
        254.0,
    ]
    assert writer.counters["max_observed_queued_rows"] <= 3


def test_flush_and_abort_leave_inspectable_wal_but_no_completion(tmp_path) -> None:
    writer = EvidenceWriter(
        tmp_path,
        run_id="interrupted",
        rank=0,
        process_id=8,
        row_group_rows=100,
        checkpoint_interval_s=None,
    )
    writer.write(run_record("interrupted", "static"))
    writer.write(request_record("interrupted", "static"))
    writer.flush()

    wal = sorted(tmp_path.glob(f"{writer.prefix}.*.wal.*.parquet"))
    assert len(wal) == 2
    assert all(pq.ParquetFile(path).metadata.num_rows == 1 for path in wal)
    checkpoint = json.loads(
        (tmp_path / f"{writer.prefix}.checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["durable_rows"]["run"] == 1
    assert load_completed_evidence(tmp_path, run_id="interrupted", rank=0) is None

    writer.abort("injected interruption")
    assert (tmp_path / f"{writer.prefix}.aborted.json").is_file()
    assert load_completed_evidence(tmp_path, run_id="interrupted", rank=0) is None


def test_backpressure_is_lossless_and_drop_blocks_completion(tmp_path) -> None:
    writer = EvidenceWriter(
        tmp_path,
        run_id="backpressure",
        rank=0,
        max_queued_rows=2,
        row_group_rows=100,
        checkpoint_interval_s=None,
        overflow_policy="backpressure",
    )
    writer.write(run_record("backpressure", "static"))
    writer.write(request_record("backpressure", "static"))
    assert writer.write(performance_record("backpressure", "static")) is True
    assert writer.backpressure_events == 1
    assert writer.dropped_rows == 0
    writer.close()

    dropping = EvidenceWriter(
        tmp_path,
        run_id="dropping",
        rank=0,
        max_queued_rows=1,
        row_group_rows=100,
        checkpoint_interval_s=None,
        overflow_policy="drop",
    )
    dropping.write(run_record("dropping", "static"))
    assert dropping.write(request_record("dropping", "static")) is False
    assert dropping.dropped_rows == 1
    assert dropping.counters["dropped_by_table"]["request"] == 1
    with pytest.raises(RuntimeError, match="completion contract"):
        dropping.close()
    assert load_completed_evidence(tmp_path, run_id="dropping", rank=0) is None
    dropping.abort("required row dropped")


@pytest.mark.parametrize("method", ["target_only", "static"])
def test_allocation_free_methods_require_only_allocation_free_tables(
    tmp_path, method: str
) -> None:
    run_id = f"disabled-{method}"
    writer = EvidenceWriter(tmp_path, run_id=run_id, rank=0, row_group_rows=1)
    writer.write(run_record(run_id, method))
    writer.write(request_record(run_id, method))
    writer.write(performance_record(run_id, method))
    paths = writer.close()
    assert set(paths) == {"run", "request", "performance"}


@pytest.mark.parametrize("method", ["tts", "l0"])
def test_adapted_methods_require_nonempty_round_and_update_tables(
    tmp_path, method: str
) -> None:
    run_id = f"adapted-{method}"
    incomplete = EvidenceWriter(tmp_path, run_id=run_id, rank=0)
    incomplete.write(run_record(run_id, method))
    incomplete.write(request_record(run_id, method))
    incomplete.write(performance_record(run_id, method))
    with pytest.raises(RuntimeError, match="completion contract"):
        incomplete.close()
    incomplete.abort("missing adapted evidence")

    complete_id = f"complete-{method}"
    complete = EvidenceWriter(tmp_path, run_id=complete_id, rank=0)
    complete.write(run_record(complete_id, method))
    complete.write(request_record(complete_id, method))
    complete.write(round_record(complete_id))
    complete.write(update_record(complete_id))
    complete.write(performance_record(complete_id, method))
    assert set(complete.close()) == {
        "run",
        "request",
        "round",
        "update",
        "performance",
    }


def test_receipt_is_published_after_final_tables_and_rejects_tamper_and_duplicate(
    tmp_path, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    original = writer_module._publish_receipt_exclusive

    def publish(path, value) -> None:
        observed["tables_exist"] = all(
            (path.parent / entry["name"]).is_file()
            and pq.ParquetFile(path.parent / entry["name"]).metadata.num_rows > 0
            for entry in value["files"].values()
        )
        observed["receipt_absent"] = not path.exists()
        original(path, value)

    monkeypatch.setattr(writer_module, "_publish_receipt_exclusive", publish)
    writer = EvidenceWriter(tmp_path, run_id="terminal", rank=0)
    writer.write(run_record("terminal", "static"))
    writer.write(request_record("terminal", "static"))
    writer.write(performance_record("terminal", "static"))
    paths = writer.close()
    assert observed == {"tables_exist": True, "receipt_absent": True}
    assert load_completed_evidence(tmp_path, run_id="terminal", rank=0) == paths
    with pytest.raises(RuntimeError, match="already exists"):
        EvidenceWriter(tmp_path, run_id="terminal", rank=0)

    paths["performance"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not bind"):
        load_completed_evidence(tmp_path, run_id="terminal", rank=0)


def test_receipt_publication_leaves_no_writable_hardlink_alias(tmp_path) -> None:
    writer = EvidenceWriter(tmp_path, run_id="single-inode", rank=0)
    writer.write(run_record("single-inode", "static"))
    writer.write(request_record("single-inode", "static"))
    writer.write(performance_record("single-inode", "static"))
    writer.close()

    prepared = tmp_path / "single-inode.rank0.prepared.json"
    canonical = tmp_path / "single-inode.rank0.complete.json"
    assert prepared.stat().st_nlink == 1
    assert canonical.stat().st_nlink == 1
    assert not tuple(tmp_path.glob("*.candidate.*"))


def test_receipt_publication_rejects_a_symlinked_lock(tmp_path) -> None:
    target = tmp_path / "locked.rank0.complete.json"
    lock = tmp_path / f".{target.name}.publish.lock"
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("do not follow\n", encoding="utf-8")
    lock.symlink_to(unrelated.name)

    with pytest.raises(RuntimeError, match="invalid receipt publication lock"):
        writer_module._publish_receipt_exclusive(
            target,
            {"schema_version": 3, "run_id": "locked", "rank": 0},
        )
    assert not target.exists()
    assert unrelated.read_text(encoding="utf-8") == "do not follow\n"
    assert not tuple(tmp_path.glob("*.candidate.*"))


def test_receipt_publication_race_has_one_unaliased_winner(tmp_path) -> None:
    target = tmp_path / "race.rank0.complete.json"

    def publish(index: int) -> str:
        try:
            writer_module._publish_receipt_exclusive(
                target,
                {
                    "schema_version": 3,
                    "run_id": "race",
                    "rank": 0,
                    "publisher": index,
                },
            )
        except RuntimeError:
            return "blocked"
        return "published"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = tuple(pool.map(publish, range(8)))
    assert outcomes.count("published") == 1
    assert outcomes.count("blocked") == 7
    assert target.stat().st_nlink == 1
    assert not tuple(tmp_path.glob("*.candidate.*"))


def test_prepare_and_publish_close_preserve_generic_writer_contract(tmp_path) -> None:
    writer = EvidenceWriter(tmp_path, run_id="prepared", rank=0)
    writer.write(run_record("prepared", "static"))
    writer.write(request_record("prepared", "static"))
    writer.write(performance_record("prepared", "static"))

    prepared_files, prepared_receipt = writer.prepare_close()
    canonical = tmp_path / "prepared.rank0.complete.json"
    assert prepared_receipt.is_file() and not prepared_receipt.is_symlink()
    prepared_body = prepared_receipt.read_bytes()
    assert not canonical.exists()
    assert load_completed_evidence(tmp_path, run_id="prepared", rank=0) is None

    published_files = writer.publish_close()
    assert published_files == prepared_files
    assert (tmp_path / "prepared.rank0.complete.json").read_bytes() == prepared_body
    assert (
        load_completed_evidence(tmp_path, run_id="prepared", rank=0) == published_files
    )


def test_schema_v2_receipt_remains_readable_without_schema_v3_bindings(
    tmp_path,
) -> None:
    run_id = "legacy-v2"
    tables = {
        "run": pa.table(
            {"run_id": [run_id], "method": ["static"], "status": ["complete"]}
        ),
        "request": pa.table({"run_id": [run_id], "method": ["static"]}),
        "performance": pa.table({"run_id": [run_id], "method": ["static"]}),
    }
    files: dict[str, dict[str, object]] = {}
    for table, data in tables.items():
        path = tmp_path / f"{run_id}.rank0.pid1.{table}.parquet"
        pq.write_table(data, path)
        files[table] = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "rank": 0,
        "files": files,
    }
    (tmp_path / f"{run_id}.rank0.complete.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    loaded = load_completed_evidence(tmp_path, run_id=run_id, rank=0)
    assert loaded is not None and set(loaded) == {"run", "request", "performance"}


def test_schema_v2_receipt_cannot_downgrade_industrial_shards(tmp_path) -> None:
    run_id = "industrial-v2-downgrade"
    tables = {
        "run": pa.table(
            {
                "run_id": [run_id],
                "method": ["target_only"],
                "status": ["complete"],
                "workload_contract": ["industrial_target_only"],
                "experiment_budget_sha256": ["a" * 64],
            }
        ),
        "request": pa.table(
            {
                "run_id": [run_id],
                "method": ["target_only"],
                "output_hash_format": [OUTPUT_HASH_FORMAT],
                "output_tokens": [1],
                "output_sha256": ["b" * 64],
                "output_token_ids": ["[1]"],
                "output_token_ids_sha256": ["b" * 64],
            }
        ),
        "performance": pa.table({"run_id": [run_id], "method": ["target_only"]}),
    }
    files: dict[str, dict[str, object]] = {}
    for table, data in tables.items():
        path = tmp_path / f"{run_id}.rank0.pid1.{table}.parquet"
        pq.write_table(data, path)
        files[table] = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "schema_version": 2,
        "run_id": run_id,
        "rank": 0,
        "files": files,
    }
    (tmp_path / f"{run_id}.rank0.complete.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="schema-v2 receipt cannot wrap schema-v3 industrial evidence",
    ):
        load_completed_evidence(tmp_path, run_id=run_id, rank=0)


@pytest.mark.parametrize("tamper", ["shard", "prepared_symlink"])
def test_publish_close_revalidates_prepared_evidence_before_completion(
    tmp_path,
    tamper: str,
) -> None:
    run_id = f"publish-tamper-{tamper}"
    writer = EvidenceWriter(tmp_path, run_id=run_id, rank=0)
    writer.write(run_record(run_id, "static"))
    writer.write(request_record(run_id, "static"))
    writer.write(performance_record(run_id, "static"))
    prepared_files, prepared_receipt = writer.prepare_close()

    if tamper == "shard":
        prepared_files["performance"].write_bytes(b"tampered")
        message = "does not bind"
    else:
        target = prepared_receipt.with_suffix(".real.json")
        prepared_receipt.rename(target)
        prepared_receipt.symlink_to(target.name)
        message = "invalid prepared receipt"

    with pytest.raises(RuntimeError, match=message):
        writer.publish_close()
    assert not (tmp_path / f"{run_id}.rank0.complete.json").exists()
    assert load_completed_evidence(tmp_path, run_id=run_id, rank=0) is None


def test_duplicate_row_and_duplicate_terminal_receipt_fail_closed(tmp_path) -> None:
    writer = EvidenceWriter(tmp_path, run_id="duplicates", rank=0)
    writer.write(run_record("duplicates", "static"))
    request = request_record("duplicates", "static")
    writer.write(request)
    with pytest.raises(ValueError, match="duplicate request"):
        writer.write(request)
    writer.write(performance_record("duplicates", "static"))
    writer.close()

    canonical = tmp_path / "duplicates.rank0.complete.json"
    legacy = tmp_path / "duplicates.rank0.pid999.complete.json"
    legacy.write_bytes(canonical.read_bytes())
    with pytest.raises(RuntimeError, match="multiple completed attempts"):
        load_completed_evidence(tmp_path, run_id="duplicates", rank=0)
