from __future__ import annotations

import pytest
import torch

from lightcone_spec.adapters.adapter_params import (
    MAX_SUPPORTED_DRAFT_DEPTH,
    AdapterShapes,
)
from lightcone_spec.adapters.losses import common_loss, position_weights
from lightcone_spec.methods.base import TeacherSignal, evaluate_loss_and_grad


def _loss_inputs(depth: int) -> dict[str, torch.Tensor]:
    return {
        "target_logits": torch.randn(depth, 5),
        "proposal_logits": torch.randn(depth, 5),
        "confidence_logits": torch.zeros(depth),
        "confidence_targets": torch.full((depth,), 0.5),
        "valid_mask": torch.ones(depth, dtype=torch.bool),
    }


def test_dspark7_declared_depth_accepts_seven_and_rejects_eight() -> None:
    weights = position_weights(
        torch.ones(7, dtype=torch.bool), max_draft_depth=7
    )
    assert weights.shape == (7,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))

    with pytest.raises(ValueError, match="8 exceeds declared depth 7"):
        common_loss(**_loss_inputs(8), max_draft_depth=7)


@pytest.mark.parametrize("observed_depth", [15, 16])
def test_dflash16_declared_depth_accepts_full_window(
    observed_depth: int,
) -> None:
    result = common_loss(
        **_loss_inputs(observed_depth), max_draft_depth=16
    )
    assert torch.isfinite(result.total)


def test_unsupported_or_inconsistent_depth_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the supported range"):
        AdapterShapes(
            rank=2,
            markov_dim=0,
            vocab_size=5,
            draft_depth=MAX_SUPPORTED_DRAFT_DEPTH + 1,
            has_markov=False,
            has_confidence=False,
            algorithm="DFLASH",
        )

    with pytest.raises(ValueError, match="17 exceeds declared depth 16"):
        position_weights(
            torch.ones(17, dtype=torch.bool), max_draft_depth=16
        )


def test_loss_rejects_inconsistent_position_shapes() -> None:
    inputs = _loss_inputs(16)
    inputs["proposal_logits"] = torch.randn(15, 5)
    with pytest.raises(ValueError, match="share shape"):
        common_loss(**inputs, max_draft_depth=16)


def test_online_loss_uses_model_pair_depth_from_adapter_shapes() -> None:
    depth, hidden, vocab = 15, 4, 5
    shapes = AdapterShapes(
        rank=2,
        markov_dim=0,
        vocab_size=vocab,
        weight_update_mode="lora",
        hidden_size=hidden,
        draft_depth=16,
        has_markov=False,
        has_confidence=False,
        algorithm="DFLASH",
    )
    signal = TeacherSignal(
        source_round=0,
        source_version=0,
        u=torch.randn(depth, 128),
        m_prev=torch.empty(depth, 0),
        base_proposal_logits=torch.randn(depth, vocab),
        base_confidence_logits=torch.zeros(depth),
        target_logits=torch.randn(depth, vocab),
        valid_mask=torch.ones(depth, dtype=torch.bool),
        source_proposal_logits=torch.randn(depth, vocab),
        confidence_targets=torch.full((depth,), 0.5),
        tail_hidden=torch.randn(depth, hidden),
    )
    phi = torch.zeros(shapes.num_params())
    loss, _ = evaluate_loss_and_grad(
        phi,
        signal,
        shapes,
        torch.randn(vocab, hidden),
        need_grad=False,
    )
    assert torch.isfinite(loss.total)
