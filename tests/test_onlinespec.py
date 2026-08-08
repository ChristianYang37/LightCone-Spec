"""Closed-form conformance of the clean-room OnlineSpec ports
(spec 6.9-6.11, 15.3): projected OGD, optimistic two-step, Hedge."""

from __future__ import annotations

import numpy as np
import torch

from lightcone_spec.adapters.adapter_params import trust_region_project
from lightcone_spec.methods.base import ArrivalContext
from lightcone_spec.methods.onlinespec import (
    ENSEMBLE_LR_MULTIPLIERS,
    HEDGE_EPSILON,
    OnlineSpecEnsemble,
    OnlineSpecOGD,
    OnlineSpecOptimistic,
)

from conftest import make_signal

LR = 5e-3


def _ctx(phi_active: torch.Tensor, version: int = 0) -> ArrivalContext:
    return ArrivalContext(
        arrival_round=5,
        active_version=version,
        phi_active=phi_active,
        delay_rounds=1,
        delay_tokens=4,
        delay_wall_us=100.0,
        delay_versions=0,
        rho_path=0.0,
        endpoint_distance=0.0,
        parameter_displacement=0.0,
    )


def _method(cls, shapes, basis):
    return cls(
        shapes, basis, lr=LR, grad_clip=1.0,
        trust_region_radius=1.0, confidence_loss_weight=1.0,
    )


def test_ogd_closed_form(shapes, basis):
    m = _method(OnlineSpecOGD, shapes, basis)
    phi = torch.zeros(shapes.num_params())
    for seed in range(3):
        sig = make_signal(seed=seed)
        cand = m.make_candidate(phi, sig)
        dec = m.decide(cand, _ctx(phi))
        expected = trust_region_project(
            phi - LR * cand.raw_gradient, torch.zeros_like(phi), 1.0
        )
        new_phi = phi + dec.published_delta
        assert torch.allclose(new_phi, expected, atol=1e-6)
        phi = new_phi


def test_optimistic_two_step_closed_form(shapes, basis):
    m = _method(OnlineSpecOptimistic, shapes, basis)
    phi = torch.zeros(shapes.num_params())
    hat = torch.zeros_like(phi)
    hint = torch.zeros_like(phi)
    for seed in range(3):
        sig = make_signal(seed=10 + seed)
        cand = m.make_candidate(phi, sig)
        dec = m.decide(cand, _ctx(phi))
        g = cand.raw_gradient
        hat = trust_region_project(hat - LR * g, torch.zeros_like(phi), 1.0)
        hint = g.clone()
        expected = trust_region_project(
            hat - LR * hint, torch.zeros_like(phi), 1.0
        )
        phi = phi + dec.published_delta
        assert torch.allclose(phi, expected, atol=1e-6)


def test_hedge_weights_closed_form(shapes, basis):
    m = _method(OnlineSpecEnsemble, shapes, basis)
    phi = torch.zeros(shapes.num_params())
    sig = make_signal(seed=20)
    cand = m.make_candidate(phi, sig)
    m.decide(cand, _ctx(phi))
    losses = np.asarray(m.history[-1]["losses"])
    lw = np.log(np.full(3, 1.0 / 3.0)) - HEDGE_EPSILON * losses
    lw = lw - (lw.max() + np.log(np.exp(lw - lw.max()).sum()))
    expected = np.exp(lw)
    got = np.asarray(m.history[-1]["weights"])
    assert np.allclose(got, expected, atol=1e-9)
    assert abs(got.sum() - 1.0) < 1e-9
    # mixture uses the updated weights over the three learners
    mults = np.asarray(ENSEMBLE_LR_MULTIPLIERS)
    eff = float((got * LR * mults).sum())
    assert abs(eff - m.history[-1]["effective_lr"]) < 1e-12


def test_sgd_methods_never_use_candidate_delta(shapes, basis):
    """OnlineSpec candidates carry a zero delta: the SGD step happens at
    arrival on phi_active, never pre-baked on the side stream."""
    m = _method(OnlineSpecOGD, shapes, basis)
    cand = m.make_candidate(torch.zeros(shapes.num_params()), make_signal())
    assert torch.count_nonzero(cand.candidate_delta) == 0
    assert torch.count_nonzero(cand.raw_gradient) > 0


def test_onlinespec_uses_bound_gradient_consensus_before_clip(shapes, basis):
    baseline = _method(OnlineSpecOGD, shapes, basis)
    method = _method(OnlineSpecOGD, shapes, basis)
    signal = make_signal(seed=42)
    phi = torch.zeros(shapes.num_params())
    expected = baseline.make_candidate(phi, signal)
    calls = []

    def quarter_consensus(grad, finite_t):
        calls.append(finite_t)
        return grad * 0.25, finite_t

    method.bind_gradient_consensus(quarter_consensus)
    actual = method.make_candidate(phi, signal)

    assert len(calls) == 1
    assert torch.allclose(actual.raw_gradient, expected.raw_gradient * 0.25)
    assert actual.numerical_ok is True
