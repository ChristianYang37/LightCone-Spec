from __future__ import annotations

import pytest
import torch

from lightcone_spec.adapters.adapter_params import (
    ANALYTIC_GRADIENT_CONTRACT,
    FORWARD_QUANTIZATION_CONTRACT,
    AdapterShapes,
    canonicalize_master_vector,
    initial_parameter_vector,
    parameter_layout_identity,
    parameter_layout_sha256,
    parameter_views,
    rmsnorm,
)
from lightcone_spec.adapters.losses import common_loss
from lightcone_spec.methods.base import TeacherSignal, evaluate_loss_and_grad
from lightcone_spec.methods.base import _reconstruct_online_outputs


def _tail_fixture(mode: str, *, dspark: bool):
    generator = torch.Generator().manual_seed(31)
    k, hidden, markov, vocab, depth = 3, 6, 4, 11, 4
    shapes = AdapterShapes(
        rank=3,
        markov_dim=markov,
        vocab_size=vocab,
        weight_update_mode=mode,
        hidden_size=hidden,
        draft_depth=depth,
        has_markov=dspark,
        has_confidence=dspark,
        algorithm="DSPARK" if dspark else "DFLASH",
    )
    projection = torch.randn(vocab, hidden, generator=generator)
    hidden_rows = torch.randn(k, hidden, generator=generator)
    m_prev = torch.randn(k, markov, generator=generator)
    signal = TeacherSignal(
        source_round=2,
        source_version=1,
        u=torch.randn(k, 128, generator=generator),
        m_prev=m_prev,
        base_proposal_logits=torch.randn(k, vocab, generator=generator),
        base_confidence_logits=torch.randn(k, generator=generator),
        target_logits=torch.randn(k, vocab, generator=generator),
        valid_mask=torch.tensor([True, True, False]),
        source_proposal_logits=torch.randn(k, vocab, generator=generator),
        confidence_targets=torch.rand(k, generator=generator),
        proposal_logit_scale=torch.tensor([0.75, 1.0, 1.25]),
        tail_hidden=hidden_rows,
    )
    return shapes, projection, signal


@pytest.mark.parametrize("mode", ["tail_lora", "full_rank_tail"])
@pytest.mark.parametrize("dspark", [False, True])
def test_tail_zero_initialization_is_exact_noop(mode, dspark):
    shapes, projection, signal = _tail_fixture(mode, dspark=dspark)
    phi0 = initial_parameter_vector(shapes)
    views = parameter_views(phi0, shapes)
    if mode == "tail_lora":
        assert torch.count_nonzero(views["a_h"]) > 0
        assert torch.count_nonzero(views["b_h"]) == 0
    else:
        assert torch.count_nonzero(phi0) == 0

    actual, _ = evaluate_loss_and_grad(
        phi0, signal, shapes, projection, need_grad=False
    )
    expected = common_loss(
        signal.target_logits * signal.proposal_logit_scale[:, None],
        signal.base_proposal_logits * signal.proposal_logit_scale[:, None],
        signal.base_confidence_logits,
        signal.confidence_targets,
        signal.valid_mask,
        confidence_loss_weight=1.0 if dspark else 0.0,
        source_proposal_logits=signal.source_proposal_logits,
    )
    assert torch.equal(actual.total, expected.total)


@pytest.mark.parametrize("mode", ["tail_lora", "full_rank_tail"])
@pytest.mark.parametrize("dspark", [False, True])
def test_tail_analytic_gradient_matches_pytorch_reference(mode, dspark):
    shapes, projection, signal = _tail_fixture(mode, dspark=dspark)
    phi = initial_parameter_vector(shapes)
    # Move away from the exact no-op so every factor receives a useful oracle
    # gradient; request admission itself still uses the zero-effect vector.
    phi = phi + 0.02 * torch.randn_like(phi)
    actual_loss, actual_grad = evaluate_loss_and_grad(
        phi,
        signal,
        shapes,
        projection,
        confidence_loss_weight=0.6,
        lambda_prox=0.3,
    )

    oracle_phi = phi.detach().clone().requires_grad_(True)
    views = parameter_views(oracle_phi, shapes)
    h = signal.tail_hidden
    assert h is not None
    if mode == "tail_lora":
        latent = h @ views["a_h"]
        if dspark:
            latent = latent + signal.m_prev @ views["a_m"]
        delta_hidden = latent @ views["b_h"]
    else:
        delta_hidden = h @ views["d_h"]
        if dspark:
            delta_hidden = delta_hidden + signal.m_prev @ views["d_m"]
    q_logits = (signal.base_proposal_logits + delta_hidden @ projection.T) * (
        signal.proposal_logit_scale[:, None]
    )
    if dspark:
        confidence_features = torch.cat(
            [
                h,
                signal.m_prev,
                torch.ones(h.shape[0], 1),
            ],
            dim=1,
        )
        confidence_logits = signal.base_confidence_logits + (
            views["a_c"][: h.shape[0]] * confidence_features
        ).sum(dim=1)
    else:
        confidence_logits = signal.base_confidence_logits
    expected_loss = common_loss(
        signal.target_logits * signal.proposal_logit_scale[:, None],
        q_logits,
        confidence_logits,
        signal.confidence_targets,
        signal.valid_mask,
        confidence_loss_weight=0.6 if dspark else 0.0,
        source_proposal_logits=signal.source_proposal_logits,
        lambda_prox=0.3,
    )
    expected_loss.total.backward()

    assert actual_grad is not None
    assert torch.allclose(actual_loss.total, expected_loss.total.detach(), atol=1e-7)
    assert torch.allclose(actual_grad, oracle_phi.grad, atol=2e-6, rtol=2e-5)


def test_output_residual_confidence_keeps_compressed_markov_features():
    shapes = AdapterShapes(rank=2, markov_dim=3, vocab_size=5)
    phi = torch.zeros(shapes.num_params(), requires_grad=True)
    views = parameter_views(phi, shapes)
    assert views["a_c"].shape[-1] == 128 + 3 + 1
    # This guards the intentional difference from Tail LoRA/full confidence,
    # whose feature is the complete unnormalised [h, m, 1].
    m = torch.tensor([[3.0, 4.0, 0.0]])
    assert not torch.equal(rmsnorm(m), m)


def test_lora_initial_coordinates_are_locked_in_layout_identity():
    shapes, _projection, _signal = _tail_fixture("tail_lora", dspark=True)

    assert torch.equal(
        initial_parameter_vector(shapes), initial_parameter_vector(shapes)
    )
    assert shapes.layout_dict()["initialization"] == {
        "scheme": "normal_fan_in_input_zero_output",
        "seed": 0,
        "alpha_over_rank": 1.0,
    }


@pytest.mark.parametrize(
    "mode", ["output_residual", "tail_lora", "full_rank_tail"]
)
def test_layout_identity_records_parameter_shape_and_storage_dtypes(mode):
    shapes, _projection, _signal = _tail_fixture(mode, dspark=True)

    layout = shapes.layout_dict(forward_dtype="torch.bfloat16")

    slices = shapes.parameter_slices()
    assert [row["name"] for row in layout["parameters"]] == list(slices)
    for row in layout["parameters"]:
        count = 1
        for dim in row["shape"]:
            count *= dim
        assert count == row["stop"] - row["start"]
        assert row["master_dtype"] == "torch.float32"
        assert row["forward_dtype"] == "torch.bfloat16"


def test_replicated_tail_layout_hash_is_canonical_across_tp_ranks():
    shapes, _projection, _signal = _tail_fixture("tail_lora", dspark=False)

    def identity(rank: int):
        return parameter_layout_identity(
            shapes,
            forward_dtype="torch.bfloat16",
            tensor_parallel_rank=rank,
            tensor_parallel_world_size=2,
            target_revision="target-rev",
            drafter_revision="draft-rev",
            projection_identity={
                "kind": "frozen_head",
                "tensor_parallel_rank": rank,
            },
            head_identity={
                "module": "ParallelLMHead",
                "tensor_parallel_rank": rank,
                "local_shape": [64, shapes.hidden_size],
            },
        )

    assert identity(0) == identity(1)
    assert identity(0)["tensor_parallel"] == {
        "bank_replication": "replicated",
        "head_sharding": "vocab_sharded_v1",
        "world_size": 2,
    }
    assert (
        identity(0)["forward_quantization_contract"]
        == FORWARD_QUANTIZATION_CONTRACT
    )
    assert identity(0)["analytic_gradient_contract"] == ANALYTIC_GRADIENT_CONTRACT


def test_tail_layout_identity_binds_optimizer_weight_decay():
    shapes, _projection, _signal = _tail_fixture("tail_lora", dspark=False)

    def identity(weight_decay: float):
        return parameter_layout_identity(
            shapes,
            forward_dtype="torch.bfloat16",
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
            target_revision="target-rev",
            drafter_revision="draft-rev",
            projection_identity={"kind": "frozen_head"},
            head_identity={"module": "ParallelLMHead", "local_shape": [64, 8]},
            optimizer_identity={
                "name": "adamw",
                "update_contract": "decoupled_adamw_delta_v1",
                "weight_decay": weight_decay,
            },
        )

    zero = identity(0.0)
    decayed = identity(1e-2)
    assert zero["schema_version"] == 3
    assert zero["optimizer_identity"]["weight_decay"] == 0.0
    assert parameter_layout_sha256(zero) != parameter_layout_sha256(decayed)


@pytest.mark.parametrize(
    "mode", ["output_residual", "tail_lora", "full_rank_tail"]
)
def test_bf16_native_forward_and_ste_gradient_match_pytorch(mode):
    """The source reconstruction and analytic STE share one native graph."""

    generator = torch.Generator().manual_seed(73)
    depth, hidden, markov, vocab, rank = 3, 6, 4, 11, 3
    shapes = AdapterShapes(
        rank=rank,
        markov_dim=markov,
        vocab_size=vocab,
        weight_update_mode=mode,
        hidden_size=hidden,
        draft_depth=depth,
        has_markov=True,
        has_confidence=True,
        algorithm="DSPARK",
    )
    projection_width = rank if mode == "output_residual" else hidden
    projection = torch.randn(
        vocab, projection_width, generator=generator
    ).to(torch.bfloat16)
    master = canonicalize_master_vector(
        initial_parameter_vector(shapes)
        + 0.02 * torch.randn(shapes.num_params(), generator=generator),
        torch.bfloat16,
    )
    u = torch.randn(depth, 128, generator=generator).to(torch.bfloat16)
    m_prev = torch.randn(depth, markov, generator=generator).to(torch.bfloat16)
    hidden_rows = torch.randn(depth, hidden, generator=generator).to(torch.bfloat16)
    base = torch.randn(depth, vocab, generator=generator).to(torch.bfloat16)
    base_conf = torch.randn(depth, generator=generator)
    target = torch.randn(depth, vocab, generator=generator)
    valid = torch.tensor([True, True, False])
    confidence_targets = torch.rand(depth, generator=generator)

    def native_outputs(phi_forward):
        views = parameter_views(phi_forward, shapes)
        if mode == "output_residual":
            residuals = [
                torch.bmm(
                    u.unsqueeze(0), views["a_h"].T.unsqueeze(0)
                ).squeeze(0)
                @ projection.T,
                torch.bmm(
                    m_prev.unsqueeze(0), views["a_m"].T.unsqueeze(0)
                ).squeeze(0)
                @ projection.T,
            ]
            features = torch.cat(
                [
                    u,
                    rmsnorm(m_prev),
                    torch.ones(depth, 1, dtype=torch.bfloat16),
                ],
                dim=-1,
            )
        elif mode == "tail_lora":
            residuals = [
                torch.bmm(
                    torch.bmm(
                        hidden_rows.unsqueeze(0),
                        views["a_h"].unsqueeze(0),
                    ),
                    views["b_h"].unsqueeze(0),
                ).squeeze(0)
                @ projection.T,
                torch.bmm(
                    torch.bmm(
                        m_prev.unsqueeze(0),
                        views["a_m"].unsqueeze(0),
                    ),
                    views["b_h"].unsqueeze(0),
                ).squeeze(0)
                @ projection.T,
            ]
            features = torch.cat(
                [
                    hidden_rows,
                    m_prev,
                    torch.ones(depth, 1, dtype=torch.bfloat16),
                ],
                dim=-1,
            )
        else:
            residuals = [
                torch.bmm(
                    hidden_rows.unsqueeze(0),
                    views["d_h"].unsqueeze(0),
                ).squeeze(0)
                @ projection.T,
                torch.bmm(
                    m_prev.unsqueeze(0),
                    views["d_m"].unsqueeze(0),
                ).squeeze(0)
                @ projection.T,
            ]
            features = torch.cat(
                [
                    hidden_rows,
                    m_prev,
                    torch.ones(depth, 1, dtype=torch.bfloat16),
                ],
                dim=-1,
            )
        q_native = base
        for residual in residuals:
            q_native = q_native + residual.to(base.dtype)
        confidence = base_conf + (
            views["a_c"][:depth] * features
        ).sum(-1).to(torch.float32)
        return q_native, confidence

    source_q, _source_conf = native_outputs(master.to(torch.bfloat16))
    signal = TeacherSignal(
        source_round=2,
        source_version=1,
        u=u,
        m_prev=m_prev,
        base_proposal_logits=base,
        base_confidence_logits=base_conf,
        target_logits=target,
        valid_mask=valid,
        source_proposal_logits=source_q.float().detach(),
        confidence_targets=confidence_targets,
        tail_hidden=hidden_rows if mode != "output_residual" else None,
    )

    reconstructed_q, reconstructed_conf, *_ = _reconstruct_online_outputs(
        master,
        signal,
        shapes,
        projection,
        forward_phi=master.to(torch.bfloat16),
    )
    torch.testing.assert_close(reconstructed_q, source_q.float(), rtol=0, atol=0)

    oracle_master = master.detach().clone().requires_grad_(True)
    oracle_q, oracle_conf = native_outputs(oracle_master.to(torch.bfloat16))
    expected_loss = common_loss(
        target,
        oracle_q.float(),
        oracle_conf,
        confidence_targets,
        valid,
        confidence_loss_weight=0.6,
        source_proposal_logits=source_q.float(),
        lambda_prox=0.3,
    )
    expected_loss.total.backward()
    actual_loss, actual_grad = evaluate_loss_and_grad(
        master,
        signal,
        shapes,
        projection,
        confidence_loss_weight=0.6,
        lambda_prox=0.3,
        forward_phi=master.to(torch.bfloat16),
    )
    assert actual_grad is not None
    torch.testing.assert_close(reconstructed_conf, oracle_conf.detach(), rtol=0, atol=0)
    torch.testing.assert_close(actual_loss.total, expected_loss.total.detach(), rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, oracle_master.grad, rtol=0, atol=0)
