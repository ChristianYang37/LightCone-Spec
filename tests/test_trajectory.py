"""Trajectory-clock invariants (spec 7.1-7.2, 15.6): idle insertion adds
zero path length; A->B->A shows large rho with small endpoint distance."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lightcone_spec.trajectory.distance import (
    DistanceWeights,
    d_z,
    endpoint_distance,
    rho_path,
)
from lightcone_spec.trajectory.features import UpdateFeatureRow, design_matrix
from lightcone_spec.trajectory.state import make_state

from conftest import random_probs

W = DistanceWeights(a_p=1 / 3, a_h=1 / 3, a_e=1 / 3)


def test_controller_design_matrix_rejects_offline_only_missing_feature():
    row = UpdateFeatureRow(
        sequence_id="prompt",
        update_id="u0",
        round_delay=1,
        token_delay=4,
        wall_us=10.0,
        endpoint_distance=0.1,
        rho_path=float("nan"),
        parameter_displacement=0.0,
        utility=0.2,
        relative_gradient_mismatch=0.1,
        harmful=0,
    )

    with pytest.raises(ValueError, match="rho_path.*u0"):
        design_matrix([row], "path_length")


def test_frozen_predictors_have_tensor_runtime_parity():
    from lightcone_spec.trajectory.predictors import HarmfulClassifier, RidgePredictor

    ridge = RidgePredictor(
        mean=np.array([1.0, -1.0]),
        std=np.array([2.0, 4.0]),
        coef=np.array([0.25, -0.5]),
        intercept=0.3,
        log1p_target=True,
    )
    harmful = HarmfulClassifier(
        mean=np.array([1.0, -1.0]),
        std=np.array([2.0, 4.0]),
        coef=np.array([0.25, -0.5]),
        intercept=0.3,
        iso_x=np.linspace(0.0, 1.0, 5),
        iso_y=np.array([0.0, 0.1, 0.4, 0.8, 1.0]),
    )
    x = np.array([2.0, 3.0])
    xt = torch.tensor(x, dtype=torch.float32)
    assert float(ridge.predict_tensor(xt)) == pytest.approx(float(ridge.predict(x)[0]))
    assert float(harmful.probability_tensor(xt)) == pytest.approx(
        float(harmful.probability(x)[0])
    )


def _state(round_id: int, rng: np.random.Generator, probs=None, hidden=None):
    p = probs if probs is not None else random_probs(rng, 32)
    h = hidden if hidden is not None else rng.standard_normal(128)
    return make_state(round_id=round_id, target_probs=p, hidden_projected=h)


def test_dz_identity_and_symmetry():
    rng = np.random.Generator(np.random.PCG64(0))
    a = _state(0, rng)
    b = _state(1, rng)
    assert d_z(a, a, W) == 0.0
    assert abs(d_z(a, b, W) - d_z(b, a, W)) < 1e-12
    assert d_z(a, b, W) > 0.0


def test_idle_insertion_adds_zero_rho():
    """Repeating the identical state (idle round) leaves rho unchanged."""
    rng = np.random.Generator(np.random.PCG64(1))
    s0 = _state(0, rng)
    s1 = _state(1, rng)
    s2 = _state(2, rng)
    rho_direct = rho_path([s0, s1, s2], 0, 2, W)

    def _probs_of(src):
        p = np.zeros(32)
        p[src.topk_token_ids] = src.topk_probs
        # put residual mass on the least-probable slot to keep sum 1
        p[p.argmin()] += src.other_mass
        return p

    # Insert two idle rounds duplicating s1's content.
    def dup(src, rid):
        return make_state(
            round_id=rid,
            target_probs=_probs_of(src),
            hidden_projected=src.hidden_proj,
        )

    dilated = [
        s0,
        s1,
        dup(s1, 2),
        dup(s1, 3),
        make_state(4, _probs_of(s2), s2.hidden_proj),
    ]
    rho_dilated = rho_path(dilated, 0, 4, W)
    assert abs(rho_dilated - rho_direct) < 1e-6


def test_aba_large_path_small_endpoint():
    rng = np.random.Generator(np.random.PCG64(2))
    p_a = random_probs(rng, 32)
    h_a = rng.standard_normal(128)
    p_b = random_probs(rng, 32)
    h_b = rng.standard_normal(128)
    states = [
        _state(0, rng, probs=p_a, hidden=h_a),
        _state(1, rng, probs=p_b, hidden=h_b),
        _state(2, rng, probs=p_a, hidden=h_a),
    ]
    rho = rho_path(states, 0, 2, W)
    endp = endpoint_distance(states, 0, 2, W)
    assert endp < 1e-9, "A->B->A endpoint must be ~0"
    assert rho > 0.1, "A->B->A path length must be large"


def test_rho_missing_round_fails_closed():
    rng = np.random.Generator(np.random.PCG64(3))
    states = [_state(0, rng), _state(2, rng)]
    import pytest

    with pytest.raises(KeyError):
        rho_path(states, 0, 2, W)


def test_state_mass_validates():
    rng = np.random.Generator(np.random.PCG64(4))
    s = _state(0, rng)
    s.validate()  # top-k + other sums to 1
