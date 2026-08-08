from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    initial_parameter_vector,
    parameter_views,
)
from lightcone_spec.methods.base import (
    CandidateGeneratorConfig,
    CommonCandidateGenerator,
    SourceBoundCandidateBatch,
    TeacherSignal,
    evaluate_loss_and_grad,
    evaluate_source_bound_loss_and_grad_batch,
)
from lightcone_spec.methods.optim import AdamWDeltaState, adamw_delta, adamw_delta_batched
from lightcone_spec.config.loader import validate_adaptation_config_dict
from lightcone_spec.methods.registry import build_method
from lightcone_spec.sglang_bridge.hooks import (
    DraftInputsReady,
    RequestLifecycle,
    UpdatePollPoint,
)
from lightcone_spec.sglang_bridge.runtime import AdaptationRuntime
from lightcone_spec.sglang_bridge.telemetry import TelemetrySink, UpdateTelemetry


def test_batched_adamw_is_request_independent_and_matches_rows():
    generator = torch.Generator().manual_seed(91)
    batch, params = 4, 17
    grad = torch.randn(batch, params, generator=generator)
    parameter = torch.randn(batch, params, generator=generator)
    exp_avg = torch.randn(batch, params, generator=generator) * 0.1
    exp_avg_sq = torch.rand(batch, params, generator=generator) + 0.1
    steps = torch.tensor([0, 2, 7, 11], dtype=torch.int64)
    valid = torch.tensor([True, False, True, True])
    expected_avg = exp_avg.clone()
    expected_sq = exp_avg_sq.clone()
    expected_steps = steps.clone()
    expected_delta = []
    for row in range(batch):
        state = AdamWDeltaState(
            num_params=params,
            step=int(expected_steps[row]),
            exp_avg=expected_avg[row],
            exp_avg_sq=expected_sq[row],
        )
        expected_delta.append(
            adamw_delta(
                grad[row],
                state,
                3e-4,
                valid=bool(valid[row]),
                parameter=parameter[row],
                weight_decay=1e-2,
            )
        )
        expected_steps[row] = int(state.step)

    actual_delta = adamw_delta_batched(
        grad,
        exp_avg,
        exp_avg_sq,
        steps,
        3e-4,
        valid,
        parameter=parameter,
        weight_decay=1e-2,
    )

    torch.testing.assert_close(actual_delta, torch.stack(expected_delta))
    torch.testing.assert_close(exp_avg, expected_avg)
    torch.testing.assert_close(exp_avg_sq, expected_sq)
    assert torch.equal(steps, expected_steps)


def _source_fixture(mode: str, *, batch: int = 3):
    torch.manual_seed(37)
    rank, hidden, vocab, depth = 3, 7, 19, 4
    shapes = AdapterShapes(
        rank=rank,
        markov_dim=0,
        vocab_size=vocab,
        weight_update_mode=mode,
        hidden_size=hidden,
        draft_depth=depth,
        has_markov=False,
        has_confidence=False,
        algorithm="DFLASH",
    )
    basis = torch.randn(vocab, rank if mode == "output_residual" else hidden)
    phi = torch.stack(
        [
            initial_parameter_vector(shapes)
            + 0.01 * torch.randn(shapes.num_params())
            for _ in range(batch)
        ]
    )
    hidden_values = torch.randn(batch, depth, hidden)
    u = torch.randn(batch, depth, 128)
    base = torch.randn(batch, depth, vocab)
    target = torch.randn(batch, depth, vocab)
    mask = torch.ones(batch, depth, dtype=torch.bool)
    mask[1, -1] = False
    corrected = []
    signals = []
    for row in range(batch):
        views = parameter_views(phi[row], shapes)
        if mode == "output_residual":
            delta = (u[row] @ views["a_h"].T) @ basis.T
        elif mode == "tail_lora":
            delta = ((hidden_values[row] @ views["a_h"]) @ views["b_h"]) @ basis.T
        else:
            delta = (hidden_values[row] @ views["d_h"]) @ basis.T
        score = base[row] + delta
        corrected.append(score)
        signals.append(
            TeacherSignal(
                source_round=10 + row,
                source_version=20 + row,
                u=u[row],
                m_prev=torch.empty(depth, 0),
                base_proposal_logits=base[row],
                base_confidence_logits=torch.zeros(depth),
                target_logits=target[row],
                valid_mask=mask[row],
                source_proposal_logits=score.detach(),
                confidence_targets=None,
                tail_hidden=(hidden_values[row] if mode != "output_residual" else None),
                proposal_distribution_kind="deterministic_argmax",
            )
        )
    compact = SourceBoundCandidateBatch(
        source_rounds=tuple(signal.source_round for signal in signals),
        source_versions=tuple(signal.source_version for signal in signals),
        u=u,
        m_prev=torch.empty(batch, depth, 0),
        proposal_logits=torch.stack(corrected),
        target_logits=target,
        valid_mask=mask,
        tail_hidden=(hidden_values if mode != "output_residual" else None),
        proposal_distribution_kind="deterministic_argmax",
    )
    return shapes, basis, phi, signals, compact


@pytest.mark.parametrize("mode", ["output_residual", "tail_lora", "full_rank_tail"])
def test_source_bound_batched_loss_and_gradient_matches_row_reconstruction(mode):
    shapes, basis, phi, signals, compact = _source_fixture(mode)
    expected = [
        evaluate_loss_and_grad(
            phi[row],
            signal,
            shapes,
            basis,
            lambda_prox=3.0,
            forward_phi=phi[row],
        )
        for row, signal in enumerate(signals)
    ]
    losses, gradients = evaluate_source_bound_loss_and_grad_batch(
        phi,
        compact,
        shapes,
        basis,
        lambda_prox=3.0,
    )
    for row, ((expected_loss, expected_grad), actual_loss) in enumerate(
        zip(expected, losses)
    ):
        torch.testing.assert_close(actual_loss.total, expected_loss.total, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            actual_loss.expected_accepted_prefix,
            expected_loss.expected_accepted_prefix,
            atol=2e-6,
            rtol=2e-6,
        )
        assert actual_loss.proximal == 0
        assert expected_grad is not None
        torch.testing.assert_close(gradients[row], expected_grad, atol=2e-6, rtol=2e-5)


def test_source_bound_batch_matches_temperature_scaled_online_scores():
    shapes, basis, phi, signals, compact = _source_fixture("tail_lora")
    scale = torch.stack(
        [
            torch.linspace(
                0.5 + row * 0.1, 1.5, compact.valid_mask.shape[1]
            )
            for row in range(compact.batch_size)
        ]
    )
    signals = [
        replace(
            signal,
            proposal_logit_scale=scale[row],
            source_proposal_logits=signal.source_proposal_logits * scale[row, :, None],
        )
        for row, signal in enumerate(signals)
    ]
    compact = replace(compact, proposal_logit_scale=scale)
    expected = [
        evaluate_loss_and_grad(
            phi[row],
            signal,
            shapes,
            basis,
            lambda_prox=2.0,
            forward_phi=phi[row],
        )
        for row, signal in enumerate(signals)
    ]
    losses, gradients = evaluate_source_bound_loss_and_grad_batch(
        phi, compact, shapes, basis, lambda_prox=2.0
    )
    for row, (loss, gradient) in enumerate(expected):
        torch.testing.assert_close(losses[row].total, loss.total, atol=2e-6, rtol=2e-6)
        assert gradient is not None
        torch.testing.assert_close(gradients[row], gradient, atol=2e-6, rtol=2e-5)


def test_common_candidate_batch_matches_independent_generators():
    shapes, basis, phi, signals, compact = _source_fixture("tail_lora")
    cfg = CandidateGeneratorConfig(
        lr=1e-4,
        grad_clip=0.7,
        trust_region_radius=1.0,
        confidence_loss_weight=0.0,
        lambda_prox=4.0,
        weight_decay=1e-2,
    )
    expected_generators = [CommonCandidateGenerator(shapes, basis, cfg) for _ in signals]
    actual_generators = [CommonCandidateGenerator(shapes, basis, cfg) for _ in signals]
    for row, (expected, actual) in enumerate(zip(expected_generators, actual_generators)):
        avg = torch.randn(shapes.num_params()) * 0.01
        sq = torch.rand(shapes.num_params()) + 0.1
        for item in (expected, actual):
            item.state.exp_avg.copy_(avg)
            item.state.exp_avg_sq.copy_(sq)
            item.state.step = row + 2
    expected = [
        generator.candidate(phi[row], signals[row], forward_phi_source=phi[row])
        for row, generator in enumerate(expected_generators)
    ]
    exp_avg = torch.stack([item.state.exp_avg for item in actual_generators])
    exp_avg_sq = torch.stack([item.state.exp_avg_sq for item in actual_generators])
    steps = torch.tensor([int(item.state.step) for item in actual_generators], dtype=torch.int64)
    actual = CommonCandidateGenerator.candidate_batch(
        actual_generators,
        phi,
        phi,
        compact,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        steps=steps,
        defer_state_advance=False,
    )
    for row in range(len(signals)):
        torch.testing.assert_close(
            actual[row].raw_gradient,
            expected[row].raw_gradient,
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            actual[row].candidate_delta,
            expected[row].candidate_delta,
            atol=2e-7,
            rtol=2e-5,
        )
        torch.testing.assert_close(exp_avg[row], expected_generators[row].state.exp_avg)
        torch.testing.assert_close(exp_avg_sq[row], expected_generators[row].state.exp_avg_sq)
        assert int(steps[row]) == int(expected_generators[row].state.step)


def test_runtime_launch_update_batch_is_the_used_candidate_path(tmp_path):
    shapes, basis, phi, _signals, compact = _source_fixture("tail_lora", batch=2)
    compact = replace(compact, source_versions=(0, 0))
    cfg = validate_adaptation_config_dict(
        {
            "schema_version": 1,
            "method": "naive_async",
            "optimizer": "adamw",
            "update_stride": 1,
            "async": {"enabled": True, "logical_delay_rounds": 0, "max_in_flight": 1},
            "trace": {"artifact_root": str(tmp_path)},
            "model": {"pair_id": "toy_markov4"},
            "dataset": {"adapter": "toy_markov4"},
        }
    )
    sink = TelemetrySink(tmp_path / "batch-runtime.jsonl")
    runtime = AdaptationRuntime(
        config=cfg,
        method_factory=lambda: build_method(cfg, shapes, basis),
        shapes=shapes,
        basis=basis,
        telemetry=sink,
        num_slots=2,
        device="cpu",
    )
    request_ids = ("batch-r0", "batch-r1")
    for row, rid in enumerate(request_ids):
        runtime.on_request_lifecycle(
            RequestLifecycle(
                request_id=rid,
                event="begin",
                request_epoch=0,
                slot_index=-1,
                tenant_id_hash=f"tenant-{row}",
            )
        )
        ctx = runtime.requests[rid]
        runtime.bank.initialize_slot(ctx.slot_index, phi[row])
        runtime.on_draft_inputs_ready(
            DraftInputsReady(
                request_id=rid,
                round_id=compact.source_rounds[row],
                request_epoch=ctx.request_epoch,
                slot_index=ctx.slot_index,
                active_version=0,
                prefix_len=4096,
            )
        )

    update_ids = runtime.launch_update_batch(request_ids, compact)

    assert len(update_ids) == 2
    for rid in request_ids:
        ctx = runtime.requests[rid]
        assert len(ctx.pending) == 1
        pending = ctx.pending[0]
        phi_buf, grad_buf, delta_buf = runtime.bank.candidate_buffers(
            ctx.slot_index, pending["lane"]
        )
        candidate = pending["candidate"]
        assert candidate.phi_source.data_ptr() == phi_buf.data_ptr()
        assert candidate.raw_gradient.data_ptr() == grad_buf.data_ptr()
        assert candidate.candidate_delta.data_ptr() == delta_buf.data_ptr()
        assert runtime.methods[ctx.slot_index].generator.state.step == 1
        published = runtime.on_update_poll(
            UpdatePollPoint(
                request_id=rid,
                round_id=compact.source_rounds[request_ids.index(rid)] + 1,
                request_epoch=ctx.request_epoch,
                slot_index=ctx.slot_index,
                active_version=0,
            )
        )
        assert published == 1
    sink.close()


def test_batched_update_telemetry_is_amortized_per_request(tmp_path):
    class Event:
        def __init__(self, milliseconds):
            self.milliseconds = milliseconds

        @staticmethod
        def synchronize():
            return None

        def elapsed_time(self, end):
            return end.milliseconds - self.milliseconds

    sink = TelemetrySink(tmp_path / "amortized.jsonl")
    sink.emit_update_deferred(
        UpdateTelemetry(
            request_id="r0",
            update_id="u0",
            source_round=1,
            source_version=0,
            snapshot_ts_us=1.0,
        ),
        {
            "event": Event(8.0),
            "numerical_ok": True,
            "grad_norm": 2.0,
            "candidate_delta_norm": 3.0,
            "source_training_loss": 4.0,
            "source_expected_accepted_prefix": 5.0,
            "optimizer_step": 1,
            "candidate_batch_size": 4,
            "side_start": Event(0.0),
            "backward_start": Event(1.0),
            "backward_end": Event(5.0),
            "optimizer_end": Event(7.0),
            "candidate_end": Event(8.0),
        },
    )
    sink.close()
    row = json.loads((tmp_path / "amortized.jsonl").read_text())

    assert row["candidate_batch_size"] == 4
    assert row["candidate_cuda_us"] == 2000.0
    assert row["backward_cuda_us"] == 1000.0
    assert row["optimizer_cuda_us"] == 500.0
