from __future__ import annotations

import json

import torch

from lightcone_spec.adapters.adapter_params import AdapterShapes
from lightcone_spec.config.loader import validate_adaptation_config_dict
from lightcone_spec.methods.registry import build_method
from lightcone_spec.methods.optim import AdamWDeltaState, adamw_delta
from lightcone_spec.sglang_bridge.hooks import (
    DraftInputsReady,
    RequestLifecycle,
    RoundCommitted,
    UpdatePollPoint,
)
from lightcone_spec.sglang_bridge.runtime import (
    AdaptationRuntime,
    _evaluation_pair_from_request_id,
    _l2_delta_from_arrival_state,
)
from lightcone_spec.sglang_bridge.telemetry import TelemetrySink
from conftest import make_signal


def _config(
    tmp_path,
    *,
    delay: int = 0,
    max_in_flight: int = 1,
    method: str = "naive_async",
):
    return validate_adaptation_config_dict(
        {
            "schema_version": 1,
            "method": method,
            "optimizer": "adamw",
            "update_stride": 4,
            "async": {
                "enabled": True,
                "logical_delay_rounds": delay,
                "max_in_flight": max_in_flight,
            },
            "trace": {"artifact_root": str(tmp_path)},
            "model": {"pair_id": "toy_markov4"},
            "dataset": {"adapter": "toy_markov4"},
        }
    )


def _runtime(
    tmp_path,
    *,
    delay: int = 0,
    max_in_flight: int = 1,
    constant_controller_delay: int | None = None,
    trace_capture_bytes: int = 0,
    enable_replay_writer: bool = True,
    method: str = "naive_async",
    gradient_consensus_fn=None,
):
    shapes = AdapterShapes(rank=8, markov_dim=6, vocab_size=32)
    basis, _ = torch.linalg.qr(torch.randn(32, 8))
    # Multi-flight is production-gated to lc_transport.  This fixture raises
    # the already-validated limit only to exercise the generic version-race
    # machinery without requiring controller artifacts.
    cfg = _config(tmp_path, delay=delay, max_in_flight=1, method=method)
    cfg.async_.max_in_flight = max_in_flight
    cfg.trace.trace_capture_max_bytes = trace_capture_bytes
    sink = TelemetrySink(tmp_path / "telemetry.jsonl")
    runtime = AdaptationRuntime(
        config=cfg,
        method_factory=lambda: build_method(cfg, shapes, basis),
        shapes=shapes,
        basis=basis,
        telemetry=sink,
        num_slots=1,
        device="cpu",
        constant_controller_delay=constant_controller_delay,
        enable_replay_writer=enable_replay_writer,
        gradient_consensus_fn=gradient_consensus_fn,
    )
    runtime.on_request_lifecycle(
        RequestLifecycle(
            request_id="r0",
            event="begin",
            request_epoch=0,
            slot_index=-1,
            tenant_id_hash="t0",
        )
    )
    ctx = runtime.requests["r0"]
    runtime.on_draft_inputs_ready(
        DraftInputsReady(
            request_id="r0",
            round_id=4,
            request_epoch=ctx.request_epoch,
            slot_index=ctx.slot_index,
            active_version=0,
            prefix_len=10,
        )
    )
    return runtime, sink, ctx


def _poll(runtime, ctx, round_id: int, active_version: int = 0):
    return runtime.on_update_poll(
        UpdatePollPoint(
            request_id="r0",
            round_id=round_id,
            request_epoch=ctx.request_epoch,
            slot_index=ctx.slot_index,
            active_version=active_version,
        )
    )


def _set_canvas(runtime, ctx, round_id: int, active_version: int = 0):
    runtime.on_draft_inputs_ready(
        DraftInputsReady(
            request_id="r0",
            round_id=round_id,
            request_epoch=ctx.request_epoch,
            slot_index=ctx.slot_index,
            active_version=active_version,
            prefix_len=10,
        )
    )


def test_evaluation_pair_id_ignores_method_run_but_binds_exact_sample_seed():
    checkpoint = "a" * 64
    prompt = "b" * 64
    left = f"lightcone-g{checkpoint}-p{prompt}-tts-run-s0000000000000042-3-17"
    right = (
        f"lightcone-g{checkpoint}-p{prompt}-"
        "lc-transport-run-s0000000000000042-3-17"
    )

    assert _evaluation_pair_from_request_id(left) == (
        _evaluation_pair_from_request_id(right)
    )
    assert _evaluation_pair_from_request_id(left) != (
        _evaluation_pair_from_request_id(
            f"lightcone-g{checkpoint}-p{prompt}-lc-transport-run-4-17"
        )
    )
    assert _evaluation_pair_from_request_id(left) != (
        _evaluation_pair_from_request_id(
            f"lightcone-g{checkpoint}-p{prompt}-"
            "lc-transport-run-s0000000000000043-3-17"
        )
    )
    assert _evaluation_pair_from_request_id("custom-rid") == "custom-rid"


def test_paired_l2_preview_uses_arrival_not_source_adam_state():
    gradient = torch.tensor([1.0, -2.0, 0.5])
    source_state = AdamWDeltaState(num_params=3)
    stale_source_preview = adamw_delta(
        gradient, source_state.clone(), lr=0.1
    )
    arrival_state = source_state.clone()
    adamw_delta(
        torch.tensor([-4.0, -1.0, 3.0]), arrival_state, lr=0.1
    )
    expected_state = arrival_state.clone()
    expected = adamw_delta(gradient, expected_state, lr=0.1)

    paired = _l2_delta_from_arrival_state(
        gradient, arrival_state.clone(), lr=0.1
    )

    torch.testing.assert_close(paired, expected)
    assert not torch.allclose(paired, stale_source_preview)


def _assert_candidate_source_evidence(row: dict, expected_prefix: int = 10) -> None:
    assert row["source_training_loss"] is not None
    assert row["source_expected_accepted_prefix"] is not None
    assert row["source_prefix_len"] == expected_prefix


def test_logical_delay_is_additional_to_pipeline_minimum(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=0)
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    assert _poll(runtime, ctx, 4) is None
    assert _poll(runtime, ctx, 5) == 1
    sink.close()
    update = next(
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
        if json.loads(line).get("kind") == "update"
    )
    _assert_candidate_source_evidence(update)


def test_update_launch_rejects_mixed_source_before_lane_or_state_mutation(
    tmp_path, monkeypatch
):
    runtime, sink, ctx = _runtime(tmp_path, delay=0)
    method = runtime.methods[ctx.slot_index]
    acquire = runtime._acquire_update_telemetry_lane
    acquire_calls = 0

    def counted_acquire(*args, **kwargs):
        nonlocal acquire_calls
        acquire_calls += 1
        return acquire(*args, **kwargs)

    monkeypatch.setattr(
        runtime, "_acquire_update_telemetry_lane", counted_acquire
    )

    assert runtime.launch_update(
        "r0", make_signal(round_id=3, version=0)
    ) is None
    assert runtime.launch_update(
        "r0", make_signal(round_id=4, version=1)
    ) is None
    ctx.canvas_version = 1
    assert runtime.launch_update(
        "r0", make_signal(round_id=4, version=1)
    ) is None

    assert acquire_calls == 0
    assert ctx.pending == []
    assert runtime.bank.slots[ctx.slot_index].active_version == 0
    assert method.generator.state.step == 0

    ctx.canvas_version = 0
    assert runtime.launch_update(
        "r0", make_signal(round_id=4, version=0)
    )
    assert acquire_calls == 1
    assert method.generator.state.step == 1

    runtime.cancel_pending("r0")
    sink.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
        if json.loads(line).get("kind") == "update"
    ]
    reasons = [record.get("failure_reason") for record in records]
    assert any(
        reason.startswith("source_round_mismatch:")
        for reason in reasons
        if reason is not None
    )
    assert any(
        reason.startswith("source_canvas_version_mismatch:")
        for reason in reasons
        if reason is not None
    )
    assert any(
        reason.startswith("canvas_active_version_mismatch:")
        for reason in reasons
        if reason is not None
    )


def test_runtime_binds_gradient_consensus_to_request_method(tmp_path):
    callback = lambda grad, finite: (grad, finite)
    runtime, sink, ctx = _runtime(
        tmp_path, gradient_consensus_fn=callback
    )
    method = runtime.methods[ctx.slot_index]
    assert method.generator.gradient_consensus_fn is callback
    sink.close()


def test_constant_controller_missed_boundary_drops_without_blocking(
    tmp_path, monkeypatch
):
    class _LateEvent:
        def query(self):
            return False

        def synchronize(self):
            return None

    runtime, sink, ctx = _runtime(tmp_path, constant_controller_delay=1)
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    ctx.pending[0]["event"] = _LateEvent()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: object())
    assert _poll(runtime, ctx, 5) is None
    assert ctx.pending == []
    assert runtime.bank.slots[ctx.slot_index].active_version == 0
    sink.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    discarded = next(
        r
        for r in records
        if r.get("failure_reason") == "constant_controller_deadline_miss"
    )
    assert discarded["decision"] == "discard"
    assert discarded["published_version"] is None
    _assert_candidate_source_evidence(discarded)


def test_max_in_flight_fails_closed_and_is_telemetried(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=3)
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    _set_canvas(runtime, ctx, 5)
    assert runtime.launch_update("r0", make_signal(round_id=5, version=0)) is None
    sink.close()
    records = [json.loads(x) for x in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    rejected = next(r for r in records if r.get("failure_reason") == "max_in_flight")
    assert rejected["snapshot_ts_us"] > 0
    assert rejected["teacher_ts_us"] is None
    assert rejected["launch_ts_us"] is None
    assert rejected["done_ts_us"] is None
    assert rejected["source_training_loss"] is None
    assert rejected["source_expected_accepted_prefix"] is None
    assert rejected["source_prefix_len"] is None


def test_update_telemetry_lane_exhaustion_freezes_adaptation(tmp_path):
    class _ExhaustedPool:
        def acquire(self):
            raise RuntimeError("CUDA telemetry lanes exhausted")

    runtime, sink, ctx = _runtime(tmp_path)
    # CPU/mock coverage of the production fail-closed branch.  No CUDA object
    # is constructed and the original speculative service can keep decoding.
    runtime._update_telemetry_pool = _ExhaustedPool()
    assert runtime.launch_update(
        "r0", make_signal(round_id=4, version=0)
    ) is None
    assert runtime._adaptation_frozen_reason == "cuda_update_lane_exhausted"
    _set_canvas(runtime, ctx, 5)
    assert runtime.launch_update(
        "r0", make_signal(round_id=5, version=0)
    ) is None
    sink.close()
    rows = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    reasons = [row["failure_reason"] for row in rows]
    assert reasons == [
        "cuda_update_lane_exhausted",
        "adaptation_frozen:cuda_update_lane_exhausted",
    ]
    assert all(row["decision"] == "discard" for row in rows)


def test_prefix_feature_exactness_is_sticky_and_telemetried(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.on_round_committed(
        RoundCommitted(
            request_id="r0",
            round_id=4,
            prefix_len_after=None,
            active_version=0,
            prefix_feature_exact=False,
        )
    )
    runtime.on_round_committed(
        RoundCommitted(
            request_id="r0",
            round_id=5,
            prefix_len_after=11,
            active_version=0,
            prefix_feature_exact=True,
        )
    )
    assert ctx.prefix_feature_exact is False
    sink.close()
    rounds = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
        if json.loads(line).get("kind") == "round"
    ]
    assert [row["prefix_feature_exact"] for row in rounds] == [False, False]


def test_stale_direct_add_is_discarded(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=0)
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    runtime.bank.slots[ctx.slot_index].active_version = 1
    assert _poll(runtime, ctx, 5, active_version=1) is None
    assert runtime.bank.slots[ctx.slot_index].active_version == 1
    sink.close()
    records = [json.loads(x) for x in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    conflict = next(r for r in records if r.get("failure_reason") == "version_conflict")
    _assert_candidate_source_evidence(conflict)


def test_deferred_nonfinite_health_drops_without_version_publish(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=0)
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    item = ctx.pending[0]
    item["candidate"].numerical_ok = torch.tensor(False)
    item["health_ok_direct"] = False

    assert _poll(runtime, ctx, 5, active_version=0) is None
    assert runtime.bank.slots[ctx.slot_index].active_version == 0
    assert ctx.pending == []
    sink.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    failed = next(
        row for row in records if row.get("failure_reason") == "non_finite_candidate"
    )
    assert failed["published_version"] is None
    _assert_candidate_source_evidence(failed)


def test_tts_waits_at_original_barrier_and_records_cost(tmp_path, monkeypatch):
    class _LateEvent:
        def __init__(self):
            self.sync_calls = 0

        def query(self):
            return False

        def synchronize(self):
            self.sync_calls += 1

    class _RecordedEvent:
        def __init__(self, *args, **kwargs):
            self.recorded = False

        def record(self, stream=None):
            self.recorded = True

    class _Stream:
        def __init__(self):
            self.waited = []

        def wait_event(self, event):
            self.waited.append(event)

    runtime, sink, ctx = _runtime(tmp_path, method="tts")
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    item = ctx.pending[0]
    assert item["ready_round"] == 8
    late = _LateEvent()
    stream = _Stream()
    item["event"] = late
    ticks = iter(range(100, 10_000, 100))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "Event", _RecordedEvent)
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.runtime.monotonic_us",
        lambda: float(next(ticks)),
    )

    assert _poll(runtime, ctx, 8, active_version=0) == 1
    assert late.sync_calls == 1
    assert stream.waited == [late]
    assert item["ready_round"] == 8
    assert runtime.bank.slots[ctx.slot_index].active_version == 1
    sink.close()
    record = next(
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
        if json.loads(line).get("kind") == "update"
    )
    assert record["barrier_wait_cpu_us"] == 100.0
    assert record["effective_delay_rounds"] == 4
    _assert_candidate_source_evidence(record)


def test_same_boundary_candidates_recheck_newly_published_version(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=0, max_in_flight=2)
    # Exercise the poll loop's version race independently of the normal
    # single-flight policy: only the first direct-add candidate may publish.
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    _set_canvas(runtime, ctx, 5)
    assert runtime.launch_update("r0", make_signal(round_id=5, version=0))
    assert _poll(runtime, ctx, 6, active_version=0) == 1
    assert runtime.bank.slots[ctx.slot_index].active_version == 1
    sink.close()
    records = [
        json.loads(x)
        for x in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    updates = [r for r in records if r.get("kind") == "update"]
    assert sum(r.get("published_version") is not None for r in updates) == 1
    assert any(r.get("failure_reason") == "version_conflict" for r in updates)
    for update in updates:
        _assert_candidate_source_evidence(update)


def test_request_end_retires_pending_without_publish(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=2)
    runtime.launch_update("r0", make_signal(round_id=4, version=0))
    runtime.on_request_lifecycle(
        RequestLifecycle(
            request_id="r0",
            event="end",
            request_epoch=ctx.request_epoch,
            slot_index=ctx.slot_index,
            tenant_id_hash="t0",
        )
    )
    assert "r0" not in runtime.requests
    assert not runtime.bank.slots[ctx.slot_index].in_use
    assert ctx.pending == []
    assert ctx.pending_states == []
    assert ctx.signals_by_round == {}
    sink.close()
    records = [
        json.loads(line)
        for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    retired = next(
        r for r in records if r.get("failure_reason") == "request_ended"
    )
    assert retired["done_ts_us"] >= retired["launch_ts_us"]
    _assert_candidate_source_evidence(retired)


def test_stream_end_destroys_request_local_method_state(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.config.lifecycle = "stream"
    slot_index = ctx.slot_index
    assert slot_index in runtime.methods
    runtime.on_request_lifecycle(
        RequestLifecycle(
            request_id="r0",
            event="stream_end",
            request_epoch=ctx.request_epoch,
            slot_index=slot_index,
            stream_id="stream-0",
            tenant_id_hash="t0",
        )
    )
    assert slot_index not in runtime.methods
    sink.close()


def test_pre_draft_publish_defers_replay_label_until_teacher_arrives(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=0)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    assert _poll(runtime, ctx, 5, active_version=0) == 1
    assert len(ctx.pending_replay_starts) == 1
    assert ctx.replay_labels == []
    runtime.observe_signal("r0", make_signal(round_id=5, version=1))
    assert ctx.pending_replay_starts == []
    assert len(ctx.replay_labels) == 1
    assert len(ctx.replay_labels[0]["utility_terms"]) == 1
    sink.close()


def test_verify_then_poll_order_starts_replay_label_at_arrival(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path, delay=0)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))

    runtime.observe_signal("r0", make_signal(round_id=5, version=0))
    assert 5 in ctx.signals_by_round
    assert _poll(runtime, ctx, 5, active_version=0) == 1

    assert ctx.pending_replay_starts == []
    assert len(ctx.replay_labels) == 1
    assert len(ctx.replay_labels[0]["utility_terms"]) == 1
    assert ctx.signals_by_round == {}
    sink.close()


def test_tts_trace_pairs_first_ready_boundary_with_later_barrier(tmp_path):
    runtime, sink, ctx = _runtime(
        tmp_path,
        method="tts",
        trace_capture_bytes=1 << 20,
    )
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))

    runtime.observe_signal("r0", make_signal(round_id=5, version=0))
    assert _poll(runtime, ctx, 5, active_version=0) is None
    assert runtime.bank.slots[ctx.slot_index].active_version == 0
    assert len(ctx.replay_labels) == 1
    label = ctx.replay_labels[0]
    assert label["candidate_arrival_round"] == 5
    assert label["actual_arrival_round"] is None
    assert all(len(terms) == 1 for terms in label["oracle_utility_terms"].values())
    assert label["utility_terms"] == []

    runtime.observe_signal("r0", make_signal(round_id=8, version=0))
    assert _poll(runtime, ctx, 8, active_version=0) == 1
    assert len(ctx.replay_labels) == 1
    label = ctx.replay_labels[0]
    assert label["candidate_arrival_round"] == 5
    assert label["actual_arrival_round"] == 8
    assert label["paired_tts_barrier"] is True
    assert len(label["utility_terms"]) == 1
    sink.close()


def test_tts_trace_candidate_and_barrier_can_share_one_boundary(tmp_path):
    runtime, sink, ctx = _runtime(
        tmp_path,
        method="tts",
        trace_capture_bytes=1 << 20,
    )
    _set_canvas(runtime, ctx, 7)
    assert runtime.launch_update("r0", make_signal(round_id=7, version=0))
    runtime.observe_signal("r0", make_signal(round_id=8, version=0))

    assert _poll(runtime, ctx, 8, active_version=0) == 1
    label = ctx.replay_labels[0]
    assert label["candidate_arrival_round"] == 8
    assert label["actual_arrival_round"] == 8
    assert len(label["utility_terms"]) == 1
    assert all(len(terms) == 1 for terms in label["oracle_utility_terms"].values())
    sink.close()


def test_incomplete_tts_pair_is_persisted_on_request_end(tmp_path):
    runtime, sink, ctx = _runtime(
        tmp_path,
        method="tts",
        trace_capture_bytes=1 << 20,
    )
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    runtime.observe_signal("r0", make_signal(round_id=5, version=0))
    assert _poll(runtime, ctx, 5, active_version=0) is None

    runtime.on_request_lifecycle(
        RequestLifecycle(
            request_id="r0",
            event="end",
            request_epoch=ctx.request_epoch,
            slot_index=ctx.slot_index,
            tenant_id_hash="t0",
        )
    )
    sink.close()
    marker = next((tmp_path / "real-replay").glob("incomplete-paired-tts-*.jsonl"))
    payload = json.loads(marker.read_text().splitlines()[0])
    assert payload["candidate_arrival_round"] == 5
    assert payload["actual_arrival_round"] is None
    assert payload["paired_tts_barrier"] is True


def test_tts_pair_writes_when_barrier_signal_completes_both_horizons(tmp_path):
    runtime, sink, ctx = _runtime(
        tmp_path,
        method="tts",
        trace_capture_bytes=1 << 20,
    )
    assert runtime.launch_update("r0", make_signal(round_id=4, version=0))
    runtime.observe_signal("r0", make_signal(round_id=5, version=0))
    assert _poll(runtime, ctx, 5, active_version=0) is None
    ctx.replay_labels[0]["horizon"] = 1

    runtime.observe_signal("r0", make_signal(round_id=8, version=0))
    assert _poll(runtime, ctx, 8, active_version=0) == 1
    assert ctx.replay_labels == []
    sink.close()

    index = next((tmp_path / "real-replay").glob("index-*.jsonl"))
    shard_name = json.loads(index.read_text())["path"]
    payload = torch.load(index.parent / shard_name, weights_only=True)
    assert payload["candidate_arrival_round"] == 5
    assert payload["actual_arrival_round"] == 8
    assert payload["paired_tts_barrier"] is True
    assert payload["trace_clock"] == "synchronous_cpu_v1"
    assert payload["fresh_gradient_scope"] == "writer_rank_local_v1"


def test_live_replay_label_reserves_request_quota_before_cloning(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    runtime.config.trace.trace_capture_max_records_per_request = 1

    assert runtime._reserve_trace_label(ctx) is not None
    assert runtime._reserve_trace_label(ctx) is None
    sink.close()


def test_trace_signal_retention_stops_at_quota_but_finishes_live_label(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    runtime.config.trace.trace_capture_max_records_per_request = 1

    assert runtime.wants_trace_signal("r0")
    assert runtime._reserve_trace_label(ctx) is not None
    # The reserved label still needs future windows to complete its horizon.
    assert runtime.wants_trace_signal("r0")
    runtime._release_trace_label(ctx)
    # Its record reservation consumed the request quota; heavy logits stop.
    assert not runtime.wants_trace_signal("r0")
    assert not runtime.wants_trace_signal("unknown")
    sink.close()


def test_staged_trace_capture_spreads_quota_across_request_progress(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    runtime.config.trace.trace_capture_max_records_per_request = 3
    runtime.config.trace.trace_capture_sampling = "staged"
    runtime.config.sampling.max_new_tokens = 100

    # The initial proposal at prefix 10 is the early phase.
    assert runtime.wants_trace_signal("r0")
    assert runtime._reserve_trace_label(ctx, source_prefix_len=10) is not None
    runtime._release_trace_label(ctx)

    # The middle phase is measured from the candidate's source prefix, not
    # from a later arrival/poll prefix.
    ctx.prefix_len = 60
    assert runtime._reserve_trace_label(ctx, source_prefix_len=59) is None
    assert runtime._reserve_trace_label(ctx, source_prefix_len=60) is not None
    runtime._release_trace_label(ctx)

    ctx.prefix_len = 89
    assert not runtime.wants_trace_signal("r0")
    ctx.prefix_len = 90
    assert runtime.wants_trace_signal("r0")
    assert runtime._reserve_trace_label(ctx, source_prefix_len=90) is not None
    runtime._release_trace_label(ctx)
    assert not runtime.wants_trace_signal("r0")
    sink.close()


def test_trace_signal_retention_respects_global_byte_budget(tmp_path):
    runtime, sink, _ctx = _runtime(tmp_path)
    runtime.config.trace.trace_capture_max_bytes = (
        runtime._default_trace_reservation_bytes() - 1
    )
    assert not runtime.wants_trace_signal("r0")
    sink.close()


def test_replay_capture_quota_is_per_request(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    runtime.config.trace.trace_capture_max_records_per_request = 1
    vector = torch.zeros(runtime.shapes.num_params())
    state = torch.zeros(387)

    def label(update_id):
        return {
            "update_id": update_id,
            "source_round": 1,
            "arrival_round": 2,
            "round_delay": 1,
            "token_delay": 2,
            "wall_us": 3,
            "endpoint_distance": 0.1,
            "rho_path": 0.2,
            "parameter_displacement": 0.3,
            "source_prefix_len": 128,
            "source_acceptance": 1.5,
            "source_training_loss": 0.4,
            "source_grad_norm": 0.7,
            "utility_terms": [torch.tensor(1.0)],
            # Emulate a gated/damped policy whose actual publication helped even
            # though the undamped raw candidate would have been harmful.
            "oracle_utility_terms": {
                0.0: [torch.tensor(0.0)],
                1.0: [torch.tensor(-2.0)],
            },
            "g_stale": vector,
            "g_fresh": vector + 1,
            "delta_z": state,
            "source_z_raw": state,
            "arrival_z_raw": state,
        }

    runtime._write_replay_label(ctx, label("first"))
    runtime._write_replay_label(ctx, label("quota-rejected"))
    sink.close()
    index = tmp_path / "real-replay" / f"index-p{__import__('os').getpid()}.jsonl"
    assert len(index.read_text().splitlines()) == 1
    shard_name = __import__("json").loads(index.read_text())["path"]
    shard = torch.load(index.parent / shard_name, weights_only=True)
    assert shard["schema_version"] == 3
    assert shard["evaluation_pair_id"] == "r0"
    assert shard["utility_metric"] == "survival_weighted_accepted_prefix_v1"
    assert shard["controller_label_source"] == "full_candidate_utility"
    assert shard["actual_published_utility"] == 1.0
    assert shard["full_candidate_utility"] == -2.0
    assert shard["harmful"] == 1
    assert "utility" not in shard
    assert shard["training_loss_gain"] == 0.0


def test_schema_v3_writer_marks_only_real_joint_transport_utility(tmp_path):
    runtime, sink, ctx = _runtime(tmp_path)
    runtime.config.trace.trace_capture_max_bytes = 1 << 20
    # The writer is under test in isolation; construction of a real L3 method
    # and controller artifact is covered by registry tests.
    object.__setattr__(runtime.config, "method", "lc_transport")
    vector = torch.zeros(runtime.shapes.num_params())
    state = torch.zeros(387)
    runtime._write_replay_label(
        ctx,
        {
            "update_id": "joint-l3",
            "source_round": 1,
            "arrival_round": 2,
            "round_delay": 1,
            "token_delay": 2,
            "wall_us": 3,
            "endpoint_distance": 0.1,
            "rho_path": 0.2,
            "parameter_displacement": 0.3,
            "source_prefix_len": 128,
            "source_acceptance": 1.5,
            "source_training_loss": 0.4,
            "source_grad_norm": 0.7,
            "utility_terms": [torch.tensor(1.25)],
            "l2_utility_terms": [torch.tensor(0.5)],
            "oracle_utility_terms": {
                0.0: [torch.tensor(0.0)],
                1.0: [torch.tensor(0.75)],
            },
            "transport_evaluation_contract": (
                "joint_fisher_transport_adamw_damping_v1"
            ),
            "transport_variant": "joint",
            "transport_map_sha256": "a" * 64,
            "g_stale": vector,
            "g_fresh": vector + 1,
            "delta_z": state,
            "source_z_raw": state,
            "arrival_z_raw": state,
        },
    )
    sink.close()

    index = next((tmp_path / "real-replay").glob("index-*.jsonl"))
    shard_name = json.loads(index.read_text())["path"]
    shard = torch.load(index.parent / shard_name, weights_only=True)
    assert shard["transported_candidate_utility"] == 1.25
    assert shard["paired_l2_utility"] == 0.5
    assert shard["transport_evaluation_contract"] == (
        "joint_fisher_transport_adamw_damping_v1"
    )
    assert shard["transport_map_sha256"] == "a" * 64


def test_non_writer_rank_never_captures_or_writes_replay(tmp_path):
    runtime, sink, ctx = _runtime(
        tmp_path,
        trace_capture_bytes=1 << 20,
        enable_replay_writer=False,
    )

    assert not runtime.wants_trace_signal("r0")
    assert runtime._reserve_trace_label(ctx) is None
    assert not runtime.needs_trajectory
    sink.close()
    assert not (tmp_path / "real-replay").exists()
