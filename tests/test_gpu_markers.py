"""GPU / integration / system tests (spec 15.9-15.11).

These run only on declared CUDA hardware with the pinned SGLang fork
installed; locally they are excluded by the default marker filter and
validated by `pytest --collect-only`. They never fake results on
unsupported hosts (fail closed via the executor guards).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.fail("gpu-marked test executed without CUDA (marker misuse)")


def _cuda_candidate_timing_bundle(device):
    """Test-only external lane equivalent for direct generator calls."""

    return {
        "backward_start": torch.cuda.Event(enable_timing=True),
        "backward_end": torch.cuda.Event(enable_timing=True),
        "optimizer_end": torch.cuda.Event(enable_timing=True),
        "optimizer_step_out": torch.empty(
            (), dtype=torch.int64, device=device
        ),
    }


def _tail_forward_for_graph(mode, phi, shapes, u, hidden, base, projection):
    """Small graph-safe equivalent of the online tail head."""
    from lightcone_spec.adapters.adapter_params import parameter_views

    views = parameter_views(phi, shapes)
    if mode == "output_residual":
        coordinates = torch.bmm(
            u.to(phi.dtype).unsqueeze(0),
            views["a_h"].T.unsqueeze(0),
        ).squeeze(0)
        delta = coordinates @ projection.T
    elif mode == "tail_lora":
        latent = torch.bmm(
            hidden.to(phi.dtype).unsqueeze(0),
            views["a_h"].unsqueeze(0),
        )
        delta_hidden = torch.bmm(
            latent, views["b_h"].unsqueeze(0)
        ).squeeze(0)
        delta = delta_hidden.to(projection.dtype) @ projection.T
    else:
        delta_hidden = torch.bmm(
            hidden.to(phi.dtype).unsqueeze(0),
            views["d_h"].unsqueeze(0),
        ).squeeze(0)
        delta = delta_hidden.to(projection.dtype) @ projection.T
    return (base + delta.to(base.dtype)).float()


def _tp_gradient_consensus_worker(rank, world_size, rendezvous):
    """Two-rank CUDA worker used by the system-marked TP contract below."""
    import torch.distributed as dist

    from lightcone_spec.methods.base import consensus_gradient
    from lightcone_spec.sglang_bridge.bank import AdapterBank

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    def average_and_health(local, finite):
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        local.div_(world_size)
        health = finite.to(dtype=torch.uint8)
        dist.all_reduce(health, op=dist.ReduceOp.MIN)
        return local, health.bool()

    try:
        local = torch.full((257,), float(rank + 1), device=device)
        consensus, ok = consensus_gradient(
            local, torch.ones((), dtype=torch.bool, device=device), average_and_health
        )
        torch.testing.assert_close(consensus, torch.full_like(consensus, 1.5))
        assert bool(ok)

        checksum = torch.stack((consensus.sum(), consensus.square().sum()))
        checksums = [torch.empty_like(checksum) for _ in range(world_size)]
        dist.all_gather(checksums, checksum)
        for other in checksums[1:]:
            torch.testing.assert_close(other, checksums[0], rtol=0, atol=0)

        # Every TP rank owns one replicated canonical bank.  Publishing the
        # consensus candidate must produce identical Q-DQ masters and serving
        # rows while retaining fixed graph-visible addresses.
        bank = AdapterBank(
            num_slots=1,
            num_params=consensus.numel(),
            device=str(device),
            forward_dtype=torch.bfloat16,
            with_optimizer=False,
        )
        slot = bank.allocate("tp-q-dq", "tenant")
        pointers = (
            bank.active.data_ptr(),
            bank.staging.data_ptr(),
            bank.forward_active.data_ptr(),
        )
        bank.write_staging(slot.slot_index, slot.request_epoch, consensus)
        bank.publish(slot.slot_index, slot.request_epoch)
        torch.testing.assert_close(
            bank.read_active(slot.slot_index),
            bank.read_forward_active(slot.slot_index).float(),
            rtol=0,
            atol=0,
        )
        bank_checksum = torch.stack(
            (
                bank.read_active(slot.slot_index).sum(),
                bank.read_active(slot.slot_index).square().sum(),
                bank.read_forward_active(slot.slot_index).float().sum(),
            )
        )
        bank_checksums = [
            torch.empty_like(bank_checksum) for _ in range(world_size)
        ]
        dist.all_gather(bank_checksums, bank_checksum)
        for other in bank_checksums[1:]:
            torch.testing.assert_close(other, bank_checksums[0], rtol=0, atol=0)
        assert pointers == (
            bank.active.data_ptr(),
            bank.staging.data_ptr(),
            bank.forward_active.data_ptr(),
        )

        # One unhealthy rank must fail closed identically before an optimizer or
        # active-bank publication can diverge.
        bad_ok = torch.tensor(rank == 0, dtype=torch.bool, device=device)
        failed, global_ok = consensus_gradient(local, bad_ok, average_and_health)
        assert not bool(global_ok)
        assert torch.count_nonzero(failed) == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
def test_adapter_bank_publish_on_device():
    _require_cuda()
    from lightcone_spec.sglang_bridge.bank import AdapterBank

    bank = AdapterBank(
        num_slots=4,
        num_params=1024,
        device="cuda",
        max_in_flight=2,
        with_optimizer_preview=True,
        with_fisher=True,
    )
    slot = bank.allocate("req-0", "t0")
    params = torch.ones(1024, device="cuda")
    ptrs = {
        name: getattr(bank, name).data_ptr()
        for name in (
            "active",
            "staging",
            "exp_avg",
            "exp_avg_sq",
            "preview_exp_avg",
            "preview_exp_avg_sq",
            "phi_source",
            "candidate_grad",
            "candidate_delta",
            "fisher",
            "candidate_fisher",
            "candidate_health_host",
        )
    }
    bank.write_staging(slot.slot_index, slot.request_epoch, params)
    version = bank.publish(slot.slot_index, slot.request_epoch)
    assert version == 1
    assert {name: getattr(bank, name).data_ptr() for name in ptrs} == ptrs
    assert torch.equal(bank.read_active(slot.slot_index), params)


@pytest.mark.gpu
def test_static_observer_uses_only_native_lengths_and_pinned_host_lanes(tmp_path):
    _require_cuda()
    import json

    from lightcone_spec.sglang_bridge.static_observer import (
        StaticSpeculativeObserver,
    )
    from lightcone_spec.sglang_bridge.telemetry import TelemetrySink

    sink = TelemetrySink(tmp_path / "static-cuda.jsonl")
    observer = StaticSpeculativeObserver(
        telemetry=sink,
        device="cuda",
        offered_concurrency=4,
        max_batch_size=4,
        lane_count=2,
    )
    commit_lens = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    new_seq_lens = torch.tensor([4099, 8193], dtype=torch.int64, device="cuda")
    commit_before = commit_lens.clone()
    seq_before = new_seq_lens.clone()
    allocated_before = torch.cuda.memory_allocated()

    observer.begin_round(
        request_ids=("r0", "r1"),
        prefix_lens=torch.tensor([4096, 8192], dtype=torch.int64),
        draft_tokens=7,
    )
    observer.record_stage("draft_end")
    observer.record_stage("verify_end")
    observer.after_accept(commit_lens=commit_lens)
    observer.commit_round(new_seq_lens=new_seq_lens)
    observer.close()

    torch.testing.assert_close(commit_lens, commit_before)
    torch.testing.assert_close(new_seq_lens, seq_before)
    assert torch.cuda.memory_allocated() == allocated_before
    rows = [
        json.loads(line)
        for line in (tmp_path / "static-cuda.jsonl").read_text().splitlines()
    ]
    assert [row["accepted_drafts"] for row in rows] == [2, 0]
    assert [row["prefix_pos_after"] for row in rows] == [4099, 8193]


@pytest.mark.gpu
def test_side_stream_update_never_blocks_boundary():
    _require_cuda()
    side = torch.cuda.Stream(priority=0)
    done = torch.cuda.Event()
    with torch.cuda.stream(side):
        a = torch.randn(4096, 4096, device="cuda")
        (a @ a).sum()
    done.record(side)
    # The poll must be non-blocking: query() only, never synchronize().
    _ = done.query()
    done.synchronize()
    assert done.query()


@pytest.mark.gpu
def test_all_adapter_components_train_on_cuda_without_host_copy(shapes, basis, signal):
    _require_cuda()
    from lightcone_spec.methods.base import evaluate_loss_and_grad

    device = torch.device("cuda")
    phi = torch.zeros(shapes.num_params(), device=device)
    cuda_signal = type(signal)(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in vars(signal).items()
        }
    )
    with torch.inference_mode():
        inference_signal = type(signal)(
            **{
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in vars(cuda_signal).items()
            }
        )
        inference_phi = phi.clone()
        inference_basis = basis.to(device).clone()
        loss, grad = evaluate_loss_and_grad(
            inference_phi, inference_signal, shapes, inference_basis
        )
    assert loss.total.is_cuda and grad is not None and grad.is_cuda
    n_d = shapes.rank * 128
    n_m = shapes.rank * shapes.markov_dim
    assert torch.count_nonzero(grad[:n_d]) > 0
    assert torch.count_nonzero(grad[n_d : n_d + n_m]) > 0
    assert torch.count_nonzero(grad[n_d + n_m :]) > 0


@pytest.mark.gpu
def test_candidate_hot_window_has_no_device_to_host_copy(shapes, basis, signal):
    _require_cuda()
    from lightcone_spec.methods.base import CandidateGeneratorConfig
    from lightcone_spec.methods.simple import NaiveAsyncMethod

    device = torch.device("cuda")
    cuda_signal = type(signal)(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in vars(signal).items()
        }
    )
    method = NaiveAsyncMethod(
        shapes,
        basis.to(device),
        CandidateGeneratorConfig(
            lr=1e-4,
            grad_clip=1.0,
            trust_region_radius=1.0,
            confidence_loss_weight=1.0,
            lambda_prox=0.0,
        ),
    )
    phi = torch.zeros(shapes.num_params(), device=device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        candidate = method.make_candidate(
            phi,
            cuda_signal,
            cuda_timing_ref=_cuda_candidate_timing_bundle(device),
        )
    assert candidate is not None and candidate.candidate_delta.is_cuda
    assert candidate.cuda_timing_ref is not None
    assert set(candidate.cuda_timing_ref) == {
        "backward_start",
        "backward_end",
        "optimizer_end",
        "optimizer_step_out",
    }
    keys = "\n".join(event.key.lower() for event in prof.key_averages())
    assert "dtoh" not in keys
    assert "device synchronize" not in keys


@pytest.mark.gpu
def test_global_nonfinite_candidate_preserves_gpu_adam_state(shapes, basis, signal):
    _require_cuda()
    from lightcone_spec.methods.base import (
        CandidateGeneratorConfig,
        CommonCandidateGenerator,
    )

    device = torch.device("cuda")
    cuda_signal = type(signal)(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in vars(signal).items()
        }
    )
    generator = CommonCandidateGenerator(
        shapes,
        basis.to(device),
        CandidateGeneratorConfig(
            lr=1e-4,
            grad_clip=1.0,
            trust_region_radius=1.0,
            confidence_loss_weight=1.0,
            lambda_prox=0.0,
        ),
    )
    generator.state.ensure_device(device)
    before_avg = generator.state.exp_avg.clone()
    before_sq = generator.state.exp_avg_sq.clone()
    before_step = generator.state.step.clone()
    generator.bind_gradient_consensus(
        lambda grad, _finite: (
            torch.zeros_like(grad),
            torch.zeros((), dtype=torch.bool, device=grad.device),
        )
    )

    candidate = generator.candidate(
        torch.zeros(shapes.num_params(), device=device),
        cuda_signal,
        cuda_timing_ref=_cuda_candidate_timing_bundle(device),
    )

    assert not bool(candidate.numerical_ok)
    assert torch.count_nonzero(candidate.candidate_delta) == 0
    torch.testing.assert_close(generator.state.exp_avg, before_avg)
    torch.testing.assert_close(generator.state.exp_avg_sq, before_sq)
    torch.testing.assert_close(generator.state.step, before_step)


@pytest.mark.gpu
def test_batched_source_candidates_have_no_hot_path_host_sync_or_state_rebind():
    _require_cuda()
    from lightcone_spec.adapters.adapter_params import (
        AdapterShapes,
        initial_parameter_vector,
        parameter_views,
    )
    from lightcone_spec.methods.base import (
        CandidateGeneratorConfig,
        CommonCandidateGenerator,
        SourceBoundCandidateBatch,
    )

    device = torch.device("cuda")
    batch, depth, hidden, vocab, rank = 4, 4, 32, 257, 4
    shapes = AdapterShapes(
        rank=rank,
        markov_dim=0,
        vocab_size=vocab,
        weight_update_mode="tail_lora",
        hidden_size=hidden,
        draft_depth=depth,
        has_markov=False,
        has_confidence=False,
        algorithm="DFLASH",
    )
    basis = torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16)
    phi = torch.stack(
        [initial_parameter_vector(shapes, device=device) for _ in range(batch)]
    )
    forward = phi.to(torch.bfloat16)
    hidden_values = torch.randn(
        batch, depth, hidden, device=device, dtype=torch.bfloat16
    )
    base = torch.randn(batch, depth, vocab, device=device, dtype=torch.bfloat16)
    corrected = []
    for row in range(batch):
        views = parameter_views(forward[row], shapes)
        delta_hidden = torch.bmm(
            torch.bmm(hidden_values[row : row + 1], views["a_h"].unsqueeze(0)),
            views["b_h"].unsqueeze(0),
        ).squeeze(0)
        corrected.append(base[row] + (delta_hidden @ basis.T).to(base.dtype))
    compact = SourceBoundCandidateBatch(
        source_rounds=tuple(range(10, 10 + batch)),
        source_versions=(0,) * batch,
        u=torch.randn(batch, depth, 128, device=device, dtype=torch.bfloat16),
        m_prev=torch.empty(batch, depth, 0, device=device, dtype=torch.bfloat16),
        proposal_logits=torch.stack(corrected),
        target_logits=torch.randn(
            batch, depth, vocab, device=device, dtype=torch.bfloat16
        ),
        valid_mask=torch.ones(batch, depth, dtype=torch.bool, device=device),
        tail_hidden=hidden_values,
        proposal_distribution_kind="deterministic_argmax",
    )
    cfg = CandidateGeneratorConfig(
        lr=1e-4,
        grad_clip=1.0,
        trust_region_radius=1.0,
        confidence_loss_weight=0.0,
        lambda_prox=1.0,
        weight_decay=1e-2,
    )
    generators = [CommonCandidateGenerator(shapes, basis, cfg) for _ in range(batch)]
    for generator in generators:
        generator.state.ensure_device(device)
    exp_avg = torch.stack([generator.state.exp_avg for generator in generators])
    exp_avg_sq = torch.stack([generator.state.exp_avg_sq for generator in generators])
    steps = torch.stack([generator.state.step for generator in generators])
    timing = [_cuda_candidate_timing_bundle(device) for _ in range(batch)]
    pointers = (
        phi.data_ptr(),
        forward.data_ptr(),
        compact.proposal_logits.data_ptr(),
        compact.target_logits.data_ptr(),
        tuple(generator.state.exp_avg.data_ptr() for generator in generators),
        tuple(generator.state.exp_avg_sq.data_ptr() for generator in generators),
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        candidates = CommonCandidateGenerator.candidate_batch(
            generators,
            phi,
            forward,
            compact,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            steps=steps,
            defer_state_advance=False,
            cuda_timing_refs=timing,
        )

    keys = "\n".join(event.key.lower() for event in prof.key_averages())
    assert "dtoh" not in keys
    assert "device synchronize" not in keys
    assert len(candidates) == batch
    assert all(item["candidate_batch_size"] == batch for item in timing)
    assert pointers == (
        phi.data_ptr(),
        forward.data_ptr(),
        compact.proposal_logits.data_ptr(),
        compact.target_logits.data_ptr(),
        tuple(generator.state.exp_avg.data_ptr() for generator in generators),
        tuple(generator.state.exp_avg_sq.data_ptr() for generator in generators),
    )


@pytest.mark.gpu
def test_runtime_publish_window_has_no_device_to_host_copy(
    tmp_path, shapes, basis, signal
):
    _require_cuda()
    from lightcone_spec.config.loader import validate_adaptation_config_dict
    from lightcone_spec.methods.registry import build_method
    from lightcone_spec.sglang_bridge.hooks import (
        DraftInputsReady,
        RequestLifecycle,
        UpdatePollPoint,
    )
    from lightcone_spec.sglang_bridge.runtime import AdaptationRuntime

    class NullTelemetry:
        def emit(self, *_args, **_kwargs):
            return None

        def emit_update_deferred(self, *_args, **_kwargs):
            return None

    config = validate_adaptation_config_dict(
        {
            "schema_version": 1,
            "method": "naive_async",
            "optimizer": "adamw",
            "update_stride": 4,
            "async": {
                "enabled": True,
                "logical_delay_rounds": 0,
                "max_in_flight": 1,
            },
            "trace": {"artifact_root": str(tmp_path)},
            "model": {"pair_id": "toy_markov4"},
            "dataset": {"adapter": "toy_markov4"},
        }
    )
    device_basis = basis.to("cuda")
    runtime = AdaptationRuntime(
        config=config,
        method_factory=lambda: build_method(config, shapes, device_basis),
        shapes=shapes,
        basis=device_basis,
        telemetry=NullTelemetry(),
        num_slots=1,
        device="cuda",
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
    cuda_signal = type(signal)(
        **{
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value
            for key, value in vars(signal).items()
        }
    )
    assert runtime.launch_update("r0", cuda_signal)
    # Synchronization belongs to test setup only. The profiled legal publish
    # boundary must enqueue its dependency and fixed-address bank copy without
    # a DtoH transfer or a device-wide host synchronization.
    ctx.pending[0]["event"].synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        version = runtime.on_update_poll(
            UpdatePollPoint(
                request_id="r0",
                round_id=5,
                request_epoch=ctx.request_epoch,
                slot_index=ctx.slot_index,
                active_version=0,
            )
        )
    assert version == 1
    keys = "\n".join(event.key.lower() for event in prof.key_averages())
    assert "dtoh" not in keys
    assert "device synchronize" not in keys


@pytest.mark.gpu
def test_publish_waits_for_every_inflight_qdq_source_snapshot(
    tmp_path, shapes, basis, signal
):
    """A ready update cannot overwrite a later lane's source bank copy."""

    _require_cuda()
    from lightcone_spec.config.loader import validate_adaptation_config_dict
    from lightcone_spec.methods.registry import build_method
    from lightcone_spec.sglang_bridge.hooks import (
        DraftInputsReady,
        RequestLifecycle,
        UpdatePollPoint,
    )
    from lightcone_spec.sglang_bridge.runtime import AdaptationRuntime

    class NullTelemetry:
        def emit(self, *_args, **_kwargs):
            return None

        def emit_update_deferred(self, *_args, **_kwargs):
            return None

    config = validate_adaptation_config_dict(
        {
            "schema_version": 1,
            "method": "naive_async",
            "optimizer": "adamw",
            "update_stride": 4,
            "async": {
                "enabled": True,
                "logical_delay_rounds": 0,
                "max_in_flight": 1,
            },
            "trace": {"artifact_root": str(tmp_path)},
            "model": {"pair_id": "toy_markov4"},
            "dataset": {"adapter": "toy_markov4"},
        }
    )
    # Production limits two lanes to L3.  This test isolates the shared runtime
    # ownership protocol without requiring a controller artifact.
    config.async_.max_in_flight = 2
    device_basis = basis.to(device="cuda", dtype=torch.bfloat16)
    runtime = AdaptationRuntime(
        config=config,
        method_factory=lambda: build_method(config, shapes, device_basis),
        shapes=shapes,
        basis=device_basis,
        telemetry=NullTelemetry(),
        num_slots=1,
        device="cuda",
        forward_dtype=torch.bfloat16,
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
    cuda_signal = type(signal)(
        **{
            key: value.to("cuda") if isinstance(value, torch.Tensor) else value
            for key, value in vars(signal).items()
        }
    )
    assert runtime.launch_update("r0", cuda_signal)
    with torch.cuda.stream(runtime.side_stream):
        torch.cuda._sleep(20_000_000)
    assert runtime.launch_update("r0", cuda_signal)
    assert all(item["preview_event"] is not None for item in ctx.pending)
    telemetry_pool = runtime._update_telemetry_pool
    assert telemetry_pool is not None
    assert len({item["telemetry_lane"] for item in ctx.pending}) == 2
    for item in ctx.pending:
        telemetry_lane = item["telemetry_lane"]
        assert item["ready_event"] is telemetry_pool.event(
            telemetry_lane, "ready_event"
        )
        assert item["preview_event"] is telemetry_pool.event(
            telemetry_lane, "preview_ready"
        )
        assert item["event"] is telemetry_pool.event(
            telemetry_lane, "candidate_end"
        )
        assert item["candidate_delta_norm"].data_ptr() == (
            telemetry_pool.device_scalar(
                telemetry_lane, "candidate_delta_norm"
            ).data_ptr()
        )

    # Candidate one is ready before the artificial delay and candidate two's
    # source copy.  Publication must enqueue a wait for lane two's snapshot,
    # not wait for its backward, and must leave that copied row at version 0.
    ctx.pending[0]["event"].synchronize()
    version = runtime.on_update_poll(
        UpdatePollPoint(
            request_id="r0",
            round_id=5,
            request_epoch=ctx.request_epoch,
            slot_index=ctx.slot_index,
            active_version=0,
        )
    )
    assert version == 1
    torch.cuda.current_stream().synchronize()
    assert torch.count_nonzero(runtime.bank.candidate_forward_buffer()) == 0
    assert torch.count_nonzero(
        runtime.bank.read_forward_active(ctx.slot_index)
    ) > 0
    runtime.cancel_pending("r0")
    torch.cuda.synchronize()


@pytest.mark.gpu
def test_l3_transport_stays_on_cuda_without_dtoh():
    _require_cuda()
    import numpy as np

    from lightcone_spec.transport.apply import transport_gradient
    from lightcone_spec.transport.fisher import FisherEMA
    from lightcone_spec.transport.fit import TransportMap

    p, k, zdim = 64, 4, 8
    transport = TransportMap(
        rank=k,
        basis=np.eye(p, k, dtype=np.float32),
        grad_mean=np.zeros(p, dtype=np.float32),
        a_matrix=np.ones((k, zdim), dtype=np.float32),
        ridge_intercept=np.zeros(k, dtype=np.float32),
    )
    raw = torch.randn(p, device="cuda")
    fisher = FisherEMA(p)
    fisher.update(raw)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        result = transport_gradient(
            raw,
            fisher,
            torch.randn(p, device="cuda"),
            transport,
            torch.randn(zdim, device="cuda"),
        )
    assert result.transported_grad.is_cuda
    assert isinstance(result.parameter_comp_norm, torch.Tensor)
    assert isinstance(result.state_transport_norm, torch.Tensor)
    keys = "\n".join(event.key.lower() for event in prof.key_averages())
    assert "dtoh" not in keys
    assert "device synchronize" not in keys


@pytest.mark.gpu
def test_gpu_trajectory_distance_matches_frozen_cpu_definition():
    _require_cuda()
    import numpy as np

    from lightcone_spec.sglang_bridge.runtime import (
        AdaptationRuntime,
        _GpuTrajectoryState,
    )
    from lightcone_spec.trajectory.distance import DistanceWeights, d_z
    from lightcone_spec.trajectory.state import TrajectoryState

    weights = DistanceWeights(a_p=0.3, a_h=0.4, a_e=0.3)
    cpu_a = TrajectoryState(
        round_id=0,
        topk_token_ids=np.array([1, 3, 5], dtype=np.int32),
        topk_probs=np.array([0.4, 0.3, 0.2], dtype=np.float32),
        other_mass=0.1,
        hidden_proj=np.linspace(0, 1, 128, dtype=np.float32),
        event_sketch=np.array([1.0, 0.4, 0.1], dtype=np.float32),
    )
    cpu_b = TrajectoryState(
        round_id=1,
        topk_token_ids=np.array([1, 2, 5], dtype=np.int32),
        topk_probs=np.array([0.2, 0.5, 0.1], dtype=np.float32),
        other_mass=0.2,
        hidden_proj=np.linspace(0.1, 1.1, 128, dtype=np.float32),
        event_sketch=np.array([1.2, 0.5, 0.2], dtype=np.float32),
    )

    def gpu(state):
        return _GpuTrajectoryState(
            round_id=state.round_id,
            topk_token_ids=torch.tensor(state.topk_token_ids, device="cuda"),
            topk_probs=torch.tensor(state.topk_probs, device="cuda"),
            other_mass=torch.tensor(state.other_mass, device="cuda"),
            hidden_proj=torch.tensor(state.hidden_proj, device="cuda"),
            event_sketch=torch.tensor(state.event_sketch, device="cuda"),
        )

    runtime = AdaptationRuntime.__new__(AdaptationRuntime)
    runtime.weights = weights
    runtime._distance_tensor_cache = {
        "hidden_mean": None,
        "hidden_std": None,
        "event_mean": None,
        "event_std": None,
    }
    actual = runtime._gpu_distance(gpu(cpu_a), gpu(cpu_b))
    assert float(actual) == pytest.approx(
        d_z(cpu_a, cpu_b, weights), rel=1e-5, abs=1e-6
    )


@pytest.mark.gpu
def test_adapter_bank_updates_are_visible_to_cuda_graph_replay():
    _require_cuda()
    from lightcone_spec.sglang_bridge.bank import AdapterBank

    bank = AdapterBank(num_slots=1, num_params=8, device="cuda")
    slot = bank.allocate("r", "t")
    out = torch.zeros((), device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out.copy_(bank.read_active(slot.slot_index).sum())
    graph.replay()
    assert out.item() == 0.0
    ptr = bank.active.data_ptr()
    bank.write_staging(
        slot.slot_index, slot.request_epoch, torch.ones(8, device="cuda")
    )
    bank.publish(slot.slot_index, slot.request_epoch)
    graph.replay()
    assert out.item() == 8.0
    assert bank.active.data_ptr() == ptr


@pytest.mark.gpu
def test_batched_adapter_row_gather_keeps_graph_buffer_addresses_fixed():
    _require_cuda()
    from lightcone_spec.sglang_bridge.bank import AdapterBank

    bank = AdapterBank(num_slots=4, num_params=8, device="cuda")
    for index in range(4):
        bank.active[index].fill_(float(index + 1))
    indices = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    rows = torch.empty(2, 8, dtype=torch.float32, device="cuda")
    torch.index_select(bank.active, 0, indices, out=rows)
    output = torch.zeros((), device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output.copy_(rows.sum())
    graph.replay()
    assert output.item() == 48.0
    pointers = (indices.data_ptr(), rows.data_ptr())
    indices.copy_(torch.tensor([0, 2], dtype=torch.int64, device="cuda"))
    torch.index_select(bank.active, 0, indices, out=rows)
    graph.replay()
    assert output.item() == 32.0
    assert (indices.data_ptr(), rows.data_ptr()) == pointers


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode", ["output_residual", "tail_lora", "full_rank_tail"]
)
def test_all_tail_modes_match_training_reconstruction_after_graph_publish(mode):
    """Toy CUDA contract; real-model graph parity remains a remote gate."""
    _require_cuda()
    from lightcone_spec.adapters.adapter_params import (
        AdapterShapes,
        initial_parameter_vector,
    )
    from lightcone_spec.methods.base import (
        TeacherSignal,
        _reconstruct_online_outputs,
        evaluate_loss_and_grad,
    )
    from lightcone_spec.sglang_bridge.bank import AdapterBank

    torch.manual_seed(91)
    device = torch.device("cuda")
    depth, hidden_size, vocab_size, rank = 3, 8, 16, 4
    shapes = AdapterShapes(
        rank=rank,
        markov_dim=0,
        vocab_size=vocab_size,
        weight_update_mode=mode,
        hidden_size=hidden_size,
        draft_depth=depth,
        has_markov=False,
        has_confidence=False,
        algorithm="EAGLE3",
    )
    projection = torch.randn(
        vocab_size,
        rank if mode == "output_residual" else hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    u = torch.randn(depth, 128, device=device, dtype=torch.bfloat16)
    hidden = torch.randn(
        depth, hidden_size, device=device, dtype=torch.bfloat16
    )
    base = torch.randn(depth, vocab_size, device=device, dtype=torch.bfloat16)
    target = torch.randn(depth, vocab_size, device=device)
    initial = initial_parameter_vector(shapes, device=device)
    published = torch.randn(shapes.num_params(), device=device).mul_(0.02)

    bank = AdapterBank(
        num_slots=1,
        num_params=shapes.num_params(),
        device="cuda",
        forward_dtype=torch.bfloat16,
        with_optimizer=False,
    )
    slot = bank.allocate("tail-graph", "tenant")
    bank.initialize_slot(slot.slot_index, initial)
    graph_out = torch.empty(depth, vocab_size, device=device)
    pointers = (
        bank.active.data_ptr(),
        bank.staging.data_ptr(),
        bank.forward_active.data_ptr(),
        bank.canonical_forward_scratch.data_ptr(),
        graph_out.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    # Initialize cuBLAS/allocator state before entering capture.
    for _ in range(3):
        graph_out.copy_(
            _tail_forward_for_graph(
                mode,
                bank.read_forward_active(slot.slot_index),
                shapes,
                u,
                hidden,
                base,
                projection,
            )
        )
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_out.copy_(
            _tail_forward_for_graph(
                mode,
                bank.read_forward_active(slot.slot_index),
                shapes,
                u,
                hidden,
                base,
                projection,
            )
        )
    graph.replay()
    before = graph_out.clone()

    bank.write_staging(slot.slot_index, slot.request_epoch, published)
    bank.publish(slot.slot_index, slot.request_epoch)
    graph.replay()

    signal = TeacherSignal(
        source_round=1,
        source_version=1,
        u=u,
        m_prev=torch.empty(depth, 0, device=device),
        base_proposal_logits=base,
        base_confidence_logits=torch.zeros(depth, device=device),
        target_logits=target,
        valid_mask=torch.ones(depth, dtype=torch.bool, device=device),
        source_proposal_logits=base.clone(),
        confidence_targets=None,
        tail_hidden=hidden if mode != "output_residual" else None,
    )
    reconstructed = _reconstruct_online_outputs(
        bank.read_active(slot.slot_index),
        signal,
        shapes,
        projection,
        forward_phi=bank.read_forward_active(slot.slot_index),
    )[0]
    torch.testing.assert_close(graph_out, reconstructed, rtol=0, atol=0)
    loss, grad = evaluate_loss_and_grad(
        bank.read_active(slot.slot_index),
        signal,
        shapes,
        projection,
        forward_phi=bank.read_forward_active(slot.slot_index),
    )
    assert torch.isfinite(loss.total) and grad is not None
    assert torch.isfinite(grad).all()
    assert not torch.equal(before, graph_out)
    assert pointers == (
        bank.active.data_ptr(),
        bank.staging.data_ptr(),
        bank.forward_active.data_ptr(),
        bank.canonical_forward_scratch.data_ptr(),
        graph_out.data_ptr(),
    )


@pytest.mark.gpu
def test_high_batch_fixed_allocator_has_no_steady_state_growth():
    """Stress only fixed tail state; this is not a real-model HBM claim."""
    _require_cuda()
    from lightcone_spec.sglang_bridge.bank import AdapterBank

    slots, num_params = 128, 4096
    bank = AdapterBank(
        num_slots=slots,
        num_params=num_params,
        device="cuda",
        forward_dtype=torch.bfloat16,
        with_optimizer=False,
    )
    states = [bank.allocate(f"r-{index}", "tenant") for index in range(slots)]
    indices = torch.arange(slots, dtype=torch.int64, device="cuda")
    rows = torch.empty(
        slots, num_params, device="cuda", dtype=torch.bfloat16
    )
    total = torch.empty((), device="cuda")
    candidate = torch.empty(num_params, device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.index_select(bank.forward_active, 0, indices, out=rows)
    total.copy_(rows.sum())
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        torch.index_select(bank.forward_active, 0, indices, out=rows)
        total.copy_(rows.sum())
    graph.replay()
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    pointers = (
        bank.active.data_ptr(),
        bank.staging.data_ptr(),
        bank.forward_active.data_ptr(),
        bank.canonical_forward_scratch.data_ptr(),
        indices.data_ptr(),
        rows.data_ptr(),
    )

    for iteration in range(256):
        state = states[iteration % slots]
        candidate.fill_(float(iteration % 11))
        bank.write_staging(state.slot_index, state.request_epoch, candidate)
        bank.publish(state.slot_index, state.request_epoch)
        graph.replay()
    torch.cuda.synchronize()

    assert torch.cuda.memory_reserved() == reserved
    assert pointers == (
        bank.active.data_ptr(),
        bank.staging.data_ptr(),
        bank.forward_active.data_ptr(),
        bank.canonical_forward_scratch.data_ptr(),
        indices.data_ptr(),
        rows.data_ptr(),
    )
    assert torch.isfinite(total)


@pytest.mark.gpu
def test_cuda_greedy_and_coupled_stochastic_one_step_exactness():
    """CUDA toy law using the same q for proposal and rejection denominator."""
    _require_cuda()
    device = torch.device("cuda")
    p = torch.tensor([0.05, 0.20, 0.15, 0.30, 0.30], device=device)
    q = torch.tensor([0.25, 0.05, 0.30, 0.10, 0.30], device=device)
    overlap = torch.minimum(p, q)
    rejection_mass = 1.0 - overlap.sum()
    residual = torch.clamp(p - q, min=0)
    residual.div_(residual.sum())
    exact_law = overlap + rejection_mass * residual
    torch.testing.assert_close(exact_law, p, rtol=1e-6, atol=1e-7)

    samples = 200_000
    generator = torch.Generator(device=device).manual_seed(1234)
    proposal_coin = torch.rand(samples, generator=generator, device=device)
    acceptance_coin = torch.rand(samples, generator=generator, device=device)
    residual_coin = torch.rand(samples, generator=generator, device=device)
    proposed = torch.searchsorted(torch.cumsum(q, 0), proposal_coin)
    accept_prob = torch.minimum(
        torch.ones(samples, device=device), p[proposed] / q[proposed]
    )
    fallback = torch.searchsorted(torch.cumsum(residual, 0), residual_coin)
    committed = torch.where(acceptance_coin <= accept_prob, proposed, fallback)
    observed = torch.bincount(committed, minlength=p.numel()).float() / samples
    torch.testing.assert_close(observed, p, rtol=0, atol=0.006)

    # Greedy target exactness holds even when the corrected proposal argmax is
    # different: rejection falls back to the target point mass.
    target_token, proposal_token = 2, 4
    p_greedy = torch.nn.functional.one_hot(
        torch.tensor(target_token, device=device), p.numel()
    ).float()
    q_greedy = torch.nn.functional.one_hot(
        torch.tensor(proposal_token, device=device), p.numel()
    ).float()
    greedy_accept = torch.minimum(p_greedy, q_greedy)
    greedy_residual = torch.clamp(p_greedy - q_greedy, min=0)
    greedy_law = greedy_accept + (1.0 - greedy_accept.sum()) * (
        greedy_residual / greedy_residual.sum()
    )
    assert int(greedy_law.argmax()) == target_token
    torch.testing.assert_close(greedy_law, p_greedy, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.system
def test_two_rank_cuda_gradient_consensus_has_identical_checksum(tmp_path):
    _require_cuda()
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("TP consensus system test requires two visible CUDA devices")
    if not dist.is_nccl_available():
        pytest.skip("TP consensus system test requires NCCL")
    rendezvous = tmp_path / "tp-consensus-init"
    mp.spawn(
        _tp_gradient_consensus_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )


@pytest.mark.integration
def test_fork_importable_and_flag_present():
    sglang = pytest.importorskip("sglang")
    from sglang.srt.server_args import ServerArgs

    assert hasattr(ServerArgs, "dspark_adaptation_config") or (
        "dspark_adaptation_config" in getattr(ServerArgs, "__annotations__", {})
    )


@pytest.mark.integration
def test_worker_parity_when_unconfigured():
    """DSparkWorkerV2 with the flag unset must not import lightcone_spec."""
    import importlib
    import sys

    pytest.importorskip("sglang")
    mod = importlib.import_module(
        "sglang.srt.speculative.dspark_components.dspark_adaptation"
    )

    class _SA:
        dspark_adaptation_config = None

    before = set(sys.modules)
    assert mod.maybe_build_adaptation_manager(_SA(), worker=None) is None
    imported = {
        name
        for name in set(sys.modules) - before
        if name == "lightcone_spec" or name.startswith("lightcone_spec.")
    }
    assert imported == set()


@pytest.mark.integration
def test_dspark_target_hidden_reshape_accepts_graph_packed_rows():
    pytest.importorskip("sglang")
    from sglang.srt.speculative.dspark_components.dspark_adaptation import (
        _reshape_target_hidden,
    )

    hidden = torch.arange(2 * 5 * 7).reshape(2, 5 * 7)
    unpacked = _reshape_target_hidden(hidden, batch_size=2, hidden_size=7)
    assert unpacked.shape == (2, 5, 7)
    assert unpacked.data_ptr() == hidden.data_ptr()
    with pytest.raises(RuntimeError, match="incompatible with the locked model"):
        _reshape_target_hidden(hidden[:, :-1], batch_size=2, hidden_size=7)


@pytest.mark.integration
def test_eager_dspark_retains_raw_and_actual_markov_proposal_logits():
    pytest.importorskip("sglang")
    from sglang.srt.speculative.dspark_components.dspark_draft import (
        sample_draft_block,
    )

    class Markov:
        def sample_block(
            self, base_logits, *, first_prev_tokens, hidden_states, sampler
        ):
            del hidden_states
            prev = first_prev_tokens
            tokens, logits = [], []
            for step in range(base_logits.shape[1]):
                native = base_logits[:, step] + prev.float()[:, None] * 0.1
                token = sampler(native, step)
                tokens.append(token)
                logits.append(native[:, None])
                prev = token
            return torch.stack(tokens, 1), torch.cat(logits, 1)

    base = torch.tensor(
        [[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]], dtype=torch.float32
    )
    result = sample_draft_block(
        base_logits=base,
        anchor_tokens=torch.tensor([1]),
        draft_hidden=torch.zeros(1, 2, 1),
        sampling_info=None,
        markov_head=Markov(),
        device=torch.device("cpu"),
        step_logits_offset_fn=lambda **_kwargs: torch.full((1, 3), 0.5),
    )
    assert result.raw_logits is not None
    assert result.proposal_logits is not None
    assert torch.allclose(result.corrected_logits, result.raw_logits + 0.5)
    assert torch.equal(
        result.draft_tokens, result.corrected_logits.argmax(dim=-1)
    )
    unconfigured = sample_draft_block(
        base_logits=base,
        anchor_tokens=torch.tensor([1]),
        draft_hidden=torch.zeros(1, 2, 1),
        sampling_info=None,
        markov_head=Markov(),
        device=torch.device("cpu"),
    )
    assert unconfigured.raw_logits is None
    assert unconfigured.proposal_logits is None
    assert unconfigured.valid_mask is None


@pytest.mark.gpu
@pytest.mark.system
def test_multi_gpu_stream_contention_profile():
    _require_cuda()
    if torch.cuda.device_count() < 8:
        pytest.skip("system profile requires 8 visible GPUs")
    # Full-system smoke is driven by the P2 manifest via
    # `lightcone-spec run-manifest`; this placeholder asserts hardware.
