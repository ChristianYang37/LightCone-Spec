from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.sglang_bridge.static_observer import (
    StaticSpeculativeObserver,
    evidence_only_static_bypass,
)
from lightcone_spec.sglang_bridge.hooks import rng_substream_identity
from lightcone_spec.sglang_bridge.telemetry import (
    CudaTelemetryLanePool,
    TelemetrySink,
)


def _observer(tmp_path, **kwargs):
    sink = TelemetrySink(tmp_path / "static.jsonl")
    observer = StaticSpeculativeObserver(
        telemetry=sink,
        device="cpu",
        offered_concurrency=4,
        max_batch_size=4,
        **kwargs,
    )
    return observer, sink


def _bypass_config(*, method="lc_gate", delay=0, trace_bytes=0):
    return SimpleNamespace(
        method=method,
        model=SimpleNamespace(pair_id="toy_dspark"),
        weight_update_mode="output_residual",
        trace=SimpleNamespace(trace_capture_max_bytes=trace_bytes),
        controller=SimpleNamespace(artifact_path="controller.json"),
        async_=SimpleNamespace(logical_delay_rounds=delay),
    )


def test_evidence_only_bypass_requires_a_proven_constant_discard(monkeypatch):
    artifact = SimpleNamespace(
        gate_discard_all=False,
        extra={"gate_constant_discard_delays": [2]},
    )
    monkeypatch.setattr(
        "lightcone_spec.controller.artifact.load_bound_controller_artifact",
        lambda *_args, **_kwargs: artifact,
    )
    monkeypatch.setattr(
        "lightcone_spec.methods.registry.validate_controller_artifact",
        lambda *_args, **_kwargs: None,
    )

    assert evidence_only_static_bypass(_bypass_config(method="static")) == (
        True,
        "static",
    )
    # A planned delay bucket does not bound the actual CUDA arrival delay, so
    # the whole-run bypass is unsafe unless the artifact discards globally.
    assert evidence_only_static_bypass(_bypass_config(delay=1)) == (False, None)
    assert evidence_only_static_bypass(_bypass_config(delay=0)) == (False, None)

    artifact.gate_discard_all = True
    assert evidence_only_static_bypass(_bypass_config())[0] is True
    # A trace needs the real candidate even when publication is predictably a
    # no-op, so it may not use the evidence-only native path.
    assert evidence_only_static_bypass(_bypass_config(trace_bytes=1)) == (
        False,
        None,
    )
    assert evidence_only_static_bypass(_bypass_config(method="lc_damp")) == (
        False,
        None,
    )
    with pytest.raises(ConfigError, match="cannot capture candidate traces"):
        evidence_only_static_bypass(
            _bypass_config(method="static", trace_bytes=1)
        )


def test_evidence_only_bypass_fails_closed_on_controller_binding_error(monkeypatch):
    def reject(*_args, **_kwargs):
        raise ConfigError("controller model pair mismatch")

    monkeypatch.setattr(
        "lightcone_spec.controller.artifact.load_bound_controller_artifact",
        reject,
    )
    with pytest.raises(ConfigError, match="model pair mismatch"):
        evidence_only_static_bypass(_bypass_config())


def test_static_observer_emits_exact_native_round_without_adaptation(
    tmp_path, monkeypatch
):
    ticks = iter(float(value) for value in range(1, 100))
    monkeypatch.setattr(
        "lightcone_spec.sglang_bridge.static_observer.time.monotonic",
        lambda: next(ticks),
    )
    observer, sink = _observer(tmp_path)
    commit_lens = torch.tensor([4, 2], dtype=torch.int32)
    new_seq_lens = torch.tensor([4100, 8194], dtype=torch.int64)
    commit_before = commit_lens.clone()
    seq_before = new_seq_lens.clone()

    observer.begin_round(
        request_ids=("r0", "r1"),
        prefix_lens=(4096, 8192),
        draft_tokens=7,
    )
    observer.record_stage("draft_end")
    observer.record_stage("verify_end")
    observer.after_accept(
        commit_lens=commit_lens,
        algorithmic_censored=torch.tensor([False, True]),
    )
    observer.commit_round(new_seq_lens=new_seq_lens)

    # The observer has no proposal/tail protocol and cannot change native
    # sampling tensors or create an adapter version domain.
    assert not hasattr(observer, "tail_logits_offset")
    assert not hasattr(observer, "runtime")
    assert not hasattr(observer, "bank")
    torch.testing.assert_close(commit_lens, commit_before)
    torch.testing.assert_close(new_seq_lens, seq_before)

    observer.close()
    rows = [json.loads(line) for line in (tmp_path / "static.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    first, second = rows
    assert first["draft_tokens"] == 7
    assert first["accepted_drafts"] == 3
    assert first["committed_per_verify"] == 4
    assert first["target_calls"] == 1
    assert first["prefix_len_before"] == 4096
    assert first["prefix_pos_after"] == 4100
    assert first["verify_len"] == 8
    assert first["active_version"] == first["proposal_version"] == 0
    assert first["version_canary_ok"] is True
    assert first["prefix_feature_exact"] is True
    assert first["algorithmic_censored"] is False
    assert first["draft_cpu_us"] > 0
    assert first["verify_cpu_us"] > 0
    assert first["round_wall_us"] > 0
    assert first["rng_substream_id"].startswith("deterministic-greedy-v1:")
    assert second["accepted_drafts"] == 1
    assert second["algorithmic_censored"] is True


def test_rng_identity_binds_seed_round_prefix_and_rejects_unseeded_sampling():
    left = rng_substream_identity(
        request_id="method-a-rid",
        sampling_seed=123,
        is_greedy=False,
        round_id=4,
        prefix_len=16384,
    )
    right = rng_substream_identity(
        request_id="method-b-rid",
        sampling_seed=123,
        is_greedy=False,
        round_id=4,
        prefix_len=16384,
    )
    assert left == right
    assert "seeded-stochastic-v1:seed-123:round-4:prefix-16384" == left
    assert left != rng_substream_identity(
        request_id="method-b-rid",
        sampling_seed=123,
        is_greedy=False,
        round_id=5,
        prefix_len=16384,
    )
    with pytest.raises(ValueError, match="requires a request-level sampling_seed"):
        rng_substream_identity(
            request_id="rid",
            sampling_seed=None,
            is_greedy=False,
            round_id=0,
            prefix_len=1,
        )


def test_static_observer_uses_seeded_stochastic_identity(tmp_path):
    observer, _sink = _observer(tmp_path)
    observer.begin_round(
        request_ids=("method-specific-rid",),
        prefix_lens=(4096,),
        draft_tokens=3,
        sampling_seeds=(77,),
        greedy_flags=(False,),
    )
    observer.record_stage("draft_end")
    observer.record_stage("verify_end")
    observer.after_accept(commit_lens=torch.tensor([1], dtype=torch.int32))
    observer.commit_round(new_seq_lens=torch.tensor([4097], dtype=torch.int64))
    observer.close()
    row = json.loads((tmp_path / "static.jsonl").read_text())
    assert row["rng_substream_id"] == (
        "seeded-stochastic-v1:seed-77:round-0:prefix-4096"
    )


def test_static_observer_tracks_rounds_and_request_lifecycle(tmp_path):
    observer, _sink = _observer(tmp_path)
    for expected_round in (0, 1):
        observer.begin_round(
            request_ids=("r0",), prefix_lens=(512 + expected_round,), draft_tokens=3
        )
        observer.record_stage("draft_end")
        observer.record_stage("verify_end")
        observer.after_accept(commit_lens=torch.tensor([1], dtype=torch.int32))
        observer.commit_round(
            new_seq_lens=torch.tensor([513 + expected_round], dtype=torch.int64)
        )
    observer.finish_request("r0")
    assert observer.diagnostics()["active_requests"] == 0
    observer.close()
    rows = [json.loads(line) for line in (tmp_path / "static.jsonl").read_text().splitlines()]
    assert [row["round_id"] for row in rows] == [0, 1]


def test_static_observer_fails_closed_on_inexact_or_partial_round(tmp_path):
    observer, sink = _observer(tmp_path)
    with pytest.raises(ValueError, match="prefix length count"):
        observer.begin_round(
            request_ids=("r0", "r1"), prefix_lens=(1,), draft_tokens=3
        )

    observer.begin_round(request_ids=("r0",), prefix_lens=(1,), draft_tokens=3)
    with pytest.raises(RuntimeError, match="commit precedes acceptance"):
        observer.commit_round(new_seq_lens=torch.tensor([2], dtype=torch.int64))
    with pytest.raises(RuntimeError, match="active batch"):
        observer.close()
    # Complete the batch so the test can close the evidence sink cleanly.
    observer.record_stage("draft_end")
    observer.record_stage("verify_end")
    observer.after_accept(commit_lens=torch.tensor([1], dtype=torch.int32))
    observer.commit_round(new_seq_lens=torch.tensor([2], dtype=torch.int64))
    sink.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_observer_captures_device_prefix_without_host_mirror(tmp_path):
    sink = TelemetrySink(tmp_path / "static-cuda.jsonl")
    observer = StaticSpeculativeObserver(
        telemetry=sink,
        device="cuda:0",
        offered_concurrency=1,
        max_batch_size=1,
        lane_count=2,
    )
    # This is the real needs_cpu_seq_lens=False contract: only the device
    # sequence length exists at the proposal boundary.
    observer.begin_round(
        request_ids=("r0",),
        prefix_lens=torch.tensor([4096], device="cuda", dtype=torch.int64),
        draft_tokens=7,
    )
    observer.record_stage("draft_end")
    observer.record_stage("verify_end")
    observer.after_accept(
        commit_lens=torch.tensor([4], device="cuda", dtype=torch.int32)
    )
    observer.commit_round(
        new_seq_lens=torch.tensor([4100], device="cuda", dtype=torch.int64)
    )
    observer.close()

    row = json.loads((tmp_path / "static-cuda.jsonl").read_text())
    assert row["prefix_len_before"] == 4096
    assert row["prefix_pos_after"] == 4100
    assert row["committed_per_verify"] == 4
    assert row["accepted_drafts"] == 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_telemetry_lane_pool_is_bounded_and_reuses_addresses():
    pool = CudaTelemetryLanePool(
        device="cuda:0",
        max_batch_size=2,
        lane_count=2,
        event_names=("draft_start", "telemetry_ready"),
        host_names=("commit",),
        device_scalars={
            "delta_norm": torch.float32,
            "optimizer_step": torch.int64,
        },
    )
    first = pool.acquire()
    second = pool.acquire()
    with pytest.raises(RuntimeError, match="lanes exhausted"):
        pool.acquire()
    first_event = pool.event(first, "draft_start")
    first_ptr = pool.host(first, "commit", torch.int32, 2).data_ptr()
    norm_ptr = pool.device_scalar(first, "delta_norm").data_ptr()
    step_ptr = pool.device_scalar(first, "optimizer_step").data_ptr()

    pool.release(first)
    reused = pool.acquire()
    assert reused == first
    assert pool.event(reused, "draft_start") is first_event
    assert pool.host(reused, "commit", torch.int32, 2).data_ptr() == first_ptr
    assert pool.device_scalar(reused, "delta_norm").data_ptr() == norm_ptr
    assert pool.device_scalar(reused, "optimizer_step").data_ptr() == step_ptr
    assert pool.device_scalar_bytes == 2 * (4 + 8)
    pool.release(second)
    pool.release(reused)
    assert pool.leased_count == 0
