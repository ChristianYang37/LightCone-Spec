"""Shared fixtures for the CPU test suite (spec 15/16.2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lightcone_spec.adapters.adapter_params import AdapterShapes
from lightcone_spec.adapters.losses import confidence_soft_targets
from lightcone_spec.methods.base import TeacherSignal

VOCAB = 32
RANK = 8
MARKOV_DIM = 6
K = 4


@pytest.fixture
def shapes() -> AdapterShapes:
    return AdapterShapes(rank=RANK, markov_dim=MARKOV_DIM, vocab_size=VOCAB)


@pytest.fixture
def basis(shapes) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(
        torch.randn(shapes.vocab_size, shapes.rank, generator=g, dtype=torch.float64)
    )
    return q.float()


def make_signal(seed: int = 0, k: int = K, vocab: int = VOCAB,
                markov_dim: int = MARKOV_DIM, round_id: int = 4,
                version: int = 0) -> TeacherSignal:
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(k, vocab, generator=g)
    target = torch.randn(k, vocab, generator=g)
    return TeacherSignal(
        source_round=round_id,
        source_version=version,
        u=torch.randn(k, 128, generator=g),
        m_prev=torch.randn(k, markov_dim, generator=g),
        base_proposal_logits=base,
        base_confidence_logits=torch.zeros(k),
        target_logits=target,
        valid_mask=torch.ones(k, dtype=torch.bool),
        source_proposal_logits=base.clone(),
        confidence_targets=confidence_soft_targets(target, base),
    )


@pytest.fixture
def signal() -> TeacherSignal:
    return make_signal()


def random_probs(rng: np.random.Generator, n: int) -> np.ndarray:
    p = rng.random(n)
    return p / p.sum()
