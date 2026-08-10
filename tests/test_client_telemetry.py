from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

from lightcone_spec.experiments.protocol import confirmation_blocks
from lightcone_spec.experiments.runner import (
    _adaptation_fields,
    _earlier_slices,
    _payloads,
    _prompt_budgets,
    _region,
    _round_records,
    _write_updates,
)
from lightcone_spec.experiments.sampling import SamplingProfile
from lightcone_spec.sglang_bridge.client import (
    GenerationResult,
    MethodRun,
    ServerSnapshot,
    SGLangHTTPClient,
    independent_method_run,
)
from lightcone_spec.telemetry import (
    EvidenceWriter,
    PerformanceRecord,
    RequestRecord,
    RunRecord,
    load_completed_evidence,
)


class StreamingResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b"".join(self.lines)


def stream_line(completion: int, *, finish: bool = False) -> bytes:
    value = {
        "text": "tokens",
        "meta_info": {
            "prompt_tokens": 7,
            "completion_tokens": completion,
            "finish_reason": {"type": "length"} if finish else None,
        },
    }
    return f"data: {json.dumps(value)}\n".encode()


def test_streaming_itl_uses_chunk_arrival_semantics() -> None:
    response = StreamingResponse(
        [stream_line(2), stream_line(4, finish=True), b"data: [DONE]\n"]
    )
    payload = {
        "rid": "r0",
        "text": "prompt",
        "sampling_params": {"max_new_tokens": 4},
    }
    with (
        patch("urllib.request.urlopen", return_value=response),
        patch("time.perf_counter", side_effect=[0.0, 0.1, 0.3, 0.31]),
    ):
        result = SGLangHTTPClient("http://server").stream_generate(payload)
    assert result.input_tokens == 7
    assert result.completion_tokens == 4
    assert result.ttft_ms == pytest.approx(100.0)
    assert result.inter_token_ms == pytest.approx((0.0, 200.0, 0.0))
    assert result.token_arrival_ms == pytest.approx((100.0, 100.0, 300.0, 300.0))
    assert result.stop_reason == "length"


def test_stream_arrivals_share_one_process_monotonic_clock() -> None:
    response = StreamingResponse(
        [stream_line(2), stream_line(4, finish=True), b"data: [DONE]\n"]
    )
    with (
        patch("urllib.request.urlopen", return_value=response),
        patch("time.perf_counter", side_effect=[10.0, 10.1, 10.3, 10.31]),
    ):
        result = SGLangHTTPClient("http://server").stream_generate(
            {"rid": "r0", "text": "x", "sampling_params": {}}
        )
    assert result.token_arrival_ms == pytest.approx(
        (10100.0, 10100.0, 10300.0, 10300.0)
    )


def test_streaming_rejects_missing_prompt_count() -> None:
    value = {"meta_info": {"completion_tokens": 1, "finish_reason": {"type": "length"}}}
    response = StreamingResponse(
        [f"data: {json.dumps(value)}\n".encode(), b"data: [DONE]\n"]
    )
    with (
        patch("urllib.request.urlopen", return_value=response),
        patch("time.perf_counter", side_effect=[0.0, 0.1, 0.2]),
        pytest.raises(RuntimeError, match="prompt token"),
    ):
        SGLangHTTPClient("http://server").stream_generate(
            {"rid": "r0", "text": "x", "sampling_params": {}}
        )


def test_tokenize_prompts_returns_real_counts_and_limit() -> None:
    response = StreamingResponse(
        [json.dumps({"count": [3, 4], "max_model_len": 40960}).encode()]
    )
    with patch("urllib.request.urlopen", return_value=response):
        counts, limit = SGLangHTTPClient("http://server").tokenize_prompts(
            ("first", "second")
        )
    assert counts == (3, 4)
    assert limit == 40960


def test_engine_reset_waits_for_scheduler_idle() -> None:
    response = StreamingResponse([b"Cache flushed.\n"])
    with patch("urllib.request.urlopen", return_value=response) as opened:
        SGLangHTTPClient("http://server").reset_engine()
    request = opened.call_args.args[0]
    assert request.full_url == "http://server/flush_cache?timeout=30"
    assert request.method == "POST"


def test_prompt_budget_uses_total_prompt_plus_generation_limit() -> None:
    class Tokenizer:
        def tokenize_prompts(self, prompts):
            assert tuple(prompts) == ("a", "b")
            return (100, 200), 40960

    samples = (
        type("Sample", (), {"sample_id": "a", "prompt": "a"})(),
        type("Sample", (), {"sample_id": "b", "prompt": "b"})(),
    )
    assert _prompt_budgets(Tokenizer(), samples, safe_context_limit=40960) == {
        "a": (100, 40860),
        "b": (200, 40760),
    }


def snapshot_payload(*, adapted: bool = False, target_calls: int = 4) -> dict:
    state = {
        "speed_study_metrics": {
            "target_calls": target_calls,
            "accepted_drafts": target_calls * 2,
            "committed_tokens": target_calls * 3,
            "verified_drafts": target_calls * 15,
            "verification_waste": target_calls * 13,
            "oom_events": 0,
            "retractions": 0,
            "peak_hbm_bytes": 100,
            "kv_bytes": 50,
            "kv_token_capacity": 409600,
            "batch_fill": 3.0,
            "queue_occupancy": 1.0,
            "graph_replay_hit_rate": 0.9,
        }
    }
    if adapted:
        state["speculative_adaptation_info_record"] = {
            "online_adaptation": adaptation_payload(target_calls=target_calls)
        }
    return {"internal_state": state}


def adaptation_payload(*, target_calls: int = 4) -> dict:
    return {
        "schema_version": 2,
        "adaptation_config_sha256": "c" * 64,
        "enabled": True,
        "disabled_reason": None,
        "cohort_sha256": "a" * 64,
        "parameter_layout_sha256": "b" * 64,
        "optimizer_bytes": 60,
        "trainable_parameters": 20,
        "resident_bytes": 80,
        "peak_bytes": 140,
        "peak_hbm_bytes": 200,
        "memory_ledger": {
            "active_or_base_bytes": 10,
            "master_fp32_bytes": 20,
            "gradient_bytes": 20,
            "first_moment_bytes": 20,
            "second_moment_bytes": 20,
            "online_state_bytes": 0,
            "optimizer_metadata_bytes": 0,
            "staging_bytes": 10,
            "training_activation_bytes": 20,
            "kv_gather_scratch_bytes": 10,
            "candidate_scratch_bytes": 10,
            "graph_buffer_bytes": 0,
            "telemetry_bytes": 0,
            "resident_bytes": 80,
            "optimizer_bytes": 60,
            "peak_bytes": 140,
        },
        "exposed_update_ms": 0.5,
        "main_side_overlap_ratio": 0.8,
        "counters": {
            "updates_launched": 1,
            "updates_published": 1,
            "target_calls": target_calls,
            "exactness_violations": 0,
            "version_mismatches": 0,
            "fallbacks": 0,
            "nonfinite_updates": 0,
            "oom_events": 0,
            "retractions": 0,
        },
        "timings_ms": {
            "training": 2.0,
            "optimizer": 1.0,
            "merge": 1.0,
            "publish": 0.25,
            "barrier": 0.25,
        },
        "updates": [
            {
                "source_round": 10,
                "source_version": 0,
                "optimizer_step": 1,
                "request_ids": ["r0"],
                "prefix_len_before": [4096],
                "published_version": 1,
                "status": "published",
                "loss": 0.5,
                "gradient_norm": 0.25,
                "reconstruction_ok": True,
                "reconstruction_max_abs": 0.01,
                "reconstruction_relative_rms": 0.001,
                "reconstruction_top1_match": 1.0,
                "reconstruction_mean_kl": 0.0001,
                "supervision_nonempty": True,
            }
        ],
        "rounds": [
            {
                "round_index": 1,
                "source_version": 0,
                "request_ids": ["r0"],
                "prefix_len_before": [7],
                "verify_len": [3],
                "accepted_drafts": [3],
                "committed_tokens": [4],
            }
        ],
        "kv_segments": {"r0": [{"start": 0, "end": 11, "source_version": 0}]},
    }


def test_server_snapshot_requires_complete_consistent_metrics() -> None:
    snapshot = ServerSnapshot.parse(snapshot_payload())
    assert snapshot.target_calls == 4
    assert snapshot.verification_waste == 52
    broken = snapshot_payload()
    broken["internal_state"]["speed_study_metrics"]["future_metric"] = 1
    del broken["internal_state"]["speed_study_metrics"]["kv_bytes"]
    with pytest.raises(RuntimeError, match="incomplete"):
        ServerSnapshot.parse(broken)
    inconsistent = snapshot_payload()
    inconsistent["internal_state"]["speed_study_metrics"]["committed_tokens"] = 1
    with pytest.raises(RuntimeError, match="inconsistent"):
        ServerSnapshot.parse(inconsistent)


def test_snapshot_accepts_current_sglang_internal_states_envelope() -> None:
    state = snapshot_payload()["internal_state"]
    snapshot = ServerSnapshot.parse({"internal_states": [state]})
    assert snapshot.target_calls == 4


@pytest.mark.parametrize("states", [[], [{}, {}], "not-a-list"])
def test_snapshot_rejects_ambiguous_dp_state_aggregation(states: object) -> None:
    with pytest.raises(RuntimeError, match="exactly one SGLang DP state"):
        ServerSnapshot.parse({"internal_states": states})


def result(request_id: str = "r0", tokens: int = 4) -> GenerationResult:
    return GenerationResult(
        request_id=request_id,
        input_tokens=7,
        completion_tokens=tokens,
        ttft_ms=10.0,
        inter_token_ms=tuple(1.0 for _ in range(tokens - 1)),
        token_arrival_ms=tuple(10.0 + index for index in range(tokens)),
        elapsed_s=0.02,
        stop_reason="length",
        response={
            "text": "x" * tokens,
            "meta_info": {"prompt_tokens": 7, "completion_tokens": tokens},
        },
    )


class FakeClient:
    def __init__(self, *, adapted: bool) -> None:
        self.adapted = adapted
        self.reset_count = 0

    def reset_engine(self) -> None:
        self.reset_count += 1

    def server_info(self) -> dict:
        calls = 0 if self.reset_count == 1 and not hasattr(self, "ran") else 4
        return snapshot_payload(adapted=self.adapted, target_calls=calls)

    def run_loaded_batch(self, payloads, *, concurrency):
        self.ran = True
        rows = tuple(result(str(payload["rid"])) for payload in payloads)
        return rows, 1.0


@pytest.mark.parametrize(
    "method",
    [
        "static",
        "tts",
        "naive_async",
        "onlinespec_ogd",
        "onlinespec_opt",
        "onlinespec_ens",
    ],
)
def test_independent_run_resets_and_collects_post_run_snapshot(method: str) -> None:
    client = FakeClient(adapted=method != "static")
    run = independent_method_run(
        client,
        method=method,
        payloads=({"rid": "r0"},),
        concurrency=1,
        adaptation_group_id=None if method == "static" else "group-a",
    )
    assert isinstance(run, MethodRun)
    assert client.reset_count == 1
    assert run.before.target_calls == 0
    assert run.after.target_calls == 4


def test_payloads_use_native_sglang_rid_and_paired_seed() -> None:
    sample = type("Sample", (), {"sample_id": "p", "prompt": "x", "seed": 7})()
    profile = SamplingProfile()
    tts = _payloads(
        sample,
        method="tts",
        block=1,
        concurrency=2,
        max_new_tokens=8,
        sampling_profile=profile,
    )
    l0 = _payloads(
        sample,
        method="naive_async",
        block=1,
        concurrency=2,
        max_new_tokens=8,
        sampling_profile=profile,
    )
    assert all("rid" in row and "request_id" not in row for row in tts + l0)
    assert [row["sampling_params"]["sampling_seed"] for row in tts] == [7, 8]
    assert [row["sampling_params"]["sampling_seed"] for row in l0] == [7, 8]
    assert all("seed" not in row["sampling_params"] for row in tts + l0)


def test_warmup_request_namespace_cannot_reuse_measured_rids() -> None:
    sample = type("Sample", (), {"sample_id": "p", "prompt": "x", "seed": 7})()
    measured = _payloads(
        sample,
        method="static",
        block=-1,
        concurrency=2,
        max_new_tokens=8,
        sampling_profile=SamplingProfile(),
    )
    warmup = _payloads(
        sample,
        method="static",
        block=-1,
        concurrency=2,
        max_new_tokens=8,
        sampling_profile=SamplingProfile(),
        request_namespace="warmup",
    )
    assert {row["rid"] for row in measured}.isdisjoint(
        row["rid"] for row in warmup
    )
    with pytest.raises(ValueError, match="namespace"):
        _payloads(
            sample,
            method="static",
            block=-1,
            concurrency=1,
            max_new_tokens=8,
            sampling_profile=SamplingProfile(),
            request_namespace="",
        )


def test_confirmation_slice_order_is_manifest_seeded() -> None:
    jobs = [
        (block.block, method)
        for block in confirmation_blocks(20260809)
        for method in block.method_order
    ]
    assert _earlier_slices(
        method=jobs[5][1], block=jobs[5][0], schedule_seed=20260809
    ) == tuple(jobs[:5])


def test_region_goodput_excludes_ttft_and_tracks_at_risk_requests() -> None:
    first = replace(result("a", 4), token_arrival_ms=(100.0, 110.0, 120.0, 130.0))
    second = replace(result("b", 2), token_arrival_ms=(200.0, 210.0))
    measured = _region((first, second), start=0, end=4)
    assert measured is not None
    at_risk, output_tokens, elapsed_s, intervals = measured
    assert at_risk == 2
    assert output_tokens == 6
    assert elapsed_s == pytest.approx(0.11)
    assert len(intervals) == 4
    tail = _region((first, second), start=2, end=4)
    assert tail is not None
    assert tail[0] == 1


def test_adaptation_evidence_never_uses_missing_defaults() -> None:
    snapshot = ServerSnapshot.parse(snapshot_payload(adapted=True))
    fields, updates, rounds = _adaptation_fields("tts", snapshot, "c" * 64)
    assert fields["optimizer_bytes"] == 60
    assert fields["updates_published"] == 1
    assert len(updates) == 1
    records = _round_records(
        run_id="run-a",
        diagnostics=snapshot.adaptation,
        results=(result(),),
        rounds=rounds,
    )
    assert len(records) == 1
    assert records[0].prefix_len_before == 7
    assert records[0].generated_tokens_before == 0
    broken = snapshot_payload(adapted=True)
    del broken["internal_state"]["speculative_adaptation_info_record"][
        "online_adaptation"
    ]["counters"]["fallbacks"]
    with pytest.raises(RuntimeError, match="incomplete"):
        _adaptation_fields("tts", ServerSnapshot.parse(broken), "c" * 64)

    malformed_root = snapshot_payload(adapted=True)
    diagnostics = malformed_root["internal_state"][
        "speculative_adaptation_info_record"
    ]["online_adaptation"]
    diagnostics["future_field"] = "ignored"
    del diagnostics["kv_segments"]
    with pytest.raises(RuntimeError, match="incomplete"):
        _adaptation_fields("tts", ServerSnapshot.parse(malformed_root), "c" * 64)


def test_update_trace_required_fields_cannot_be_hidden_by_extras() -> None:
    diagnostics = adaptation_payload()
    update = dict(diagnostics["updates"][0])
    del update["loss"]
    update["future_field"] = "ignored"

    class Writer:
        def write(self, _record) -> None:
            raise AssertionError("malformed updates must fail before writing")

    with pytest.raises(RuntimeError, match="incomplete"):
        _write_updates(
            Writer(),
            run_id="run-a",
            method="tts",
            diagnostics=diagnostics,
            updates=(update,),
        )


def test_update_trace_requires_method_specific_online_state() -> None:
    diagnostics = adaptation_payload()
    plain = diagnostics["updates"][0]
    optimistic = {**plain, "online_hint_error": 0.2}

    class Writer:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    writer = Writer()
    _write_updates(
        writer,
        run_id="run-a",
        method="onlinespec_opt",
        diagnostics=diagnostics,
        updates=(optimistic,),
    )
    assert writer.records[0].optimizer_step == 1
    assert writer.records[0].online_hint_error == pytest.approx(0.2)

    with pytest.raises(RuntimeError, match="foreign OnlineSPEC"):
        _write_updates(
            Writer(),
            run_id="run-a",
            method="onlinespec_ogd",
            diagnostics=diagnostics,
            updates=(optimistic,),
        )
    with pytest.raises(RuntimeError, match="non-empty update evidence"):
        _write_updates(
            Writer(),
            run_id="run-a",
            method="onlinespec_ogd",
            diagnostics=diagnostics,
            updates=(),
        )
    with pytest.raises(RuntimeError, match="Static"):
        _write_updates(
            Writer(),
            run_id="run-a",
            method="static",
            diagnostics=diagnostics,
            updates=(plain,),
        )


def test_onlinespec_update_diagnostics_are_preserved_and_validated() -> None:
    diagnostics = adaptation_payload()
    update = {
        **diagnostics["updates"][0],
        "online_hint_error": None,
        "online_ensemble_entropy": 0.5,
        "online_effective_experts": 1.648721,
        "online_expert_probabilities": [0.75, 0.25],
        "online_cumulative_losses": [0.1, 1.2],
        "online_expert_gradient_norms": [0.3, 0.8],
    }

    class Writer:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    writer = Writer()
    _write_updates(
        writer,
        run_id="run-a",
        method="onlinespec_ens",
        diagnostics=diagnostics,
        updates=(update,),
    )
    assert writer.records[0].online_ensemble_entropy == pytest.approx(0.5)
    assert writer.records[0].online_expert_probabilities == "[0.75,0.25]"
    assert writer.records[0].gradient_norm == pytest.approx(0.25)
    assert writer.records[0].online_expert_gradient_norms == "[0.3,0.8]"
    invalid = {**update, "online_expert_probabilities": [0.8, 0.3]}
    with pytest.raises(RuntimeError, match="ensemble evidence"):
        _write_updates(
            Writer(),
            run_id="run-a",
            method="onlinespec_ens",
            diagnostics=diagnostics,
            updates=(invalid,),
        )


def test_adaptation_evidence_rejects_nonfinite_and_bad_memory_totals() -> None:
    nonfinite = snapshot_payload(adapted=True)
    nonfinite["internal_state"]["speculative_adaptation_info_record"][
        "online_adaptation"
    ]["timings_ms"]["optimizer"] = float("nan")
    with pytest.raises(RuntimeError, match="finite and non-negative"):
        _adaptation_fields("tts", ServerSnapshot.parse(nonfinite), "c" * 64)

    bad_ledger = snapshot_payload(adapted=True)
    bad_ledger["internal_state"]["speculative_adaptation_info_record"][
        "online_adaptation"
    ]["memory_ledger"]["candidate_scratch_bytes"] += 1
    with pytest.raises(RuntimeError, match="categories do not sum"):
        _adaptation_fields("tts", ServerSnapshot.parse(bad_ledger), "c" * 64)


def test_adaptation_evidence_rejects_wrong_runtime_config() -> None:
    snapshot = ServerSnapshot.parse(snapshot_payload(adapted=True))
    with pytest.raises(RuntimeError, match="config identity mismatch"):
        _adaptation_fields("tts", snapshot, "d" * 64)


def performance_record(run_id: str) -> PerformanceRecord:
    return PerformanceRecord(
        run_id=run_id,
        prompt_id="p0",
        method="static",
        repetition_block=0,
        region="full_trajectory",
        concurrency=1,
        generated_bucket_start=0,
        generated_bucket_end=4,
        at_risk_requests=1,
        output_tokens=4,
        elapsed_s=0.1,
        decode_goodput_tps=40.0,
        itl_p50_ms=1.0,
        itl_p95_ms=1.0,
        itl_p99_ms=1.0,
        survival_weighted_accepted_prefix=2.0,
        accepted_drafts_per_verify=2.0,
        committed_tokens_per_verify=3.0,
        verified_drafts_per_verify=15.0,
        verification_waste=13.0,
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
        graph_replay_hit_rate=0.9,
        updates_launched=0,
        updates_published=0,
        exactness_violations=0,
        version_mismatches=0,
        fallbacks=0,
        nonfinite_updates=0,
        oom_events=0,
        retractions=0,
    )


def request_record(run_id: str) -> RequestRecord:
    return RequestRecord(
        run_id=run_id,
        request_id=f"{run_id}-request",
        prompt_id="p0",
        method="static",
        repetition_block=0,
        concurrency=1,
        input_tokens=2,
        output_tokens=4,
        output_sha256="c" * 64,
        ttft_ms=1.0,
        finished=True,
        stop_reason="length",
    )


def test_evidence_writer_is_process_unique_atomic_and_nonempty(tmp_path) -> None:
    writer = EvidenceWriter(tmp_path, run_id="run", rank=0, process_id=7)
    writer.write(
        RunRecord("run", "a" * 64, "b" * 64, "static", "pair", 0, 1, 2, "complete")
    )
    writer.write(request_record("run"))
    writer.write(performance_record("run"))
    paths = writer.close()
    assert set(paths) == {"run", "request", "performance"}
    assert pq.read_table(paths["performance"]).num_rows == 1
    completed = load_completed_evidence(tmp_path, run_id="run", rank=0)
    assert completed == paths
    with pytest.raises(RuntimeError, match="already exists"):
        EvidenceWriter(tmp_path, run_id="run", rank=0, process_id=7)


def test_completion_receipt_rejects_tampering_and_ignores_partial_attempt(
    tmp_path,
) -> None:
    partial = tmp_path / "partial.rank0.pid8.performance.parquet"
    partial.write_bytes(b"interrupted")
    assert load_completed_evidence(tmp_path, run_id="partial", rank=0) is None
    retry = EvidenceWriter(tmp_path, run_id="partial", rank=0, process_id=8)
    assert ".attempt" in retry.prefix
    retry.write(
        RunRecord(
            "partial",
            "a" * 64,
            "b" * 64,
            "static",
            "pair",
            0,
            1,
            2,
            "complete",
        )
    )
    retry.write(request_record("partial"))
    retry.write(performance_record("partial"))
    assert load_completed_evidence(tmp_path, run_id="partial", rank=0) is None
    retry.close()
    assert load_completed_evidence(tmp_path, run_id="partial", rank=0) is not None

    writer = EvidenceWriter(tmp_path, run_id="complete", rank=0, process_id=9)
    writer.write(
        RunRecord(
            "complete",
            "a" * 64,
            "b" * 64,
            "static",
            "pair",
            0,
            1,
            2,
            "complete",
        )
    )
    writer.write(request_record("complete"))
    writer.write(performance_record("complete"))
    paths = writer.close()
    paths["performance"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not bind"):
        load_completed_evidence(tmp_path, run_id="complete", rank=0)


def test_evidence_writer_rejects_cross_run_and_incomplete_adaptation(
    tmp_path,
) -> None:
    writer = EvidenceWriter(tmp_path, run_id="bound", rank=0, process_id=10)
    with pytest.raises(ValueError, match="another run"):
        writer.write(request_record("other"))

    adapted = EvidenceWriter(tmp_path, run_id="adapted", rank=0, process_id=11)
    adapted.write(
        RunRecord(
            "adapted",
            "a" * 64,
            "b" * 64,
            "tts",
            "pair",
            0,
            1,
            2,
            "complete",
        )
    )
    adapted.write(request_record("adapted"))
    adapted.write(performance_record("adapted"))
    with pytest.raises(RuntimeError, match="completion contract"):
        adapted.close()
