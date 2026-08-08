"""Transport map fit conformance (spec 7.6, 15.4): PCA basis recovery
and ridge consistency on synthetic linear data."""

from __future__ import annotations

import numpy as np

from lightcone_spec.transport.fit import TransportMap, fit_transport_map


def _linear_dataset(n=200, p=40, z_dim=6, rank=4, seed=0, noise=1e-4):
    rng = np.random.Generator(np.random.PCG64(seed))
    basis, _ = np.linalg.qr(rng.standard_normal((p, rank)))
    a_true = rng.standard_normal((rank, z_dim))
    delta_z = rng.standard_normal((n, z_dim))
    delta_g = (delta_z @ a_true.T) @ basis.T
    delta_g += noise * rng.standard_normal(delta_g.shape)
    groups = [f"g{i % 8}" for i in range(n)]
    return delta_g, delta_z, groups, basis


def test_pca_basis_spans_true_subspace():
    delta_g, delta_z, groups, basis_true = _linear_dataset()
    tm = fit_transport_map(delta_g, delta_z, groups, rank=4)
    # subspace alignment: projector distance ~ 0
    p_est = tm.basis @ tm.basis.T
    p_true = basis_true @ basis_true.T
    err = np.linalg.norm(p_est - p_true) / np.linalg.norm(p_true)
    assert err < 0.05, f"PCA subspace error {err}"


def test_ridge_predicts_state_correction():
    delta_g, delta_z, groups, _ = _linear_dataset(seed=1)
    tm = fit_transport_map(delta_g, delta_z, groups, rank=4)
    errs = []
    for i in range(0, 200, 17):
        pred = tm.state_correction(delta_z[i])
        errs.append(
            np.linalg.norm(pred - delta_g[i]) / max(np.linalg.norm(delta_g[i]), 1e-12)
        )
    assert float(np.mean(errs)) < 0.1, f"ridge relative error {np.mean(errs)}"


def test_ridge_online_map_keeps_mean_and_intercept_for_shifted_state():
    """The serialized online kernel must equal sklearn's fitted affine map."""
    rng = np.random.Generator(np.random.PCG64(17))
    n, p, z_dim, rank = 80, 24, 5, 3
    basis, _ = np.linalg.qr(rng.standard_normal((p, rank)))
    a_true = rng.standard_normal((rank, z_dim))
    z = 3.0 + rng.standard_normal((n, z_dim))
    offset = rng.standard_normal(p)
    drift = offset + (z @ a_true.T) @ basis.T
    groups = [f"g{i % 8}" for i in range(n)]

    tm = fit_transport_map(drift, z, groups, rank=rank)
    predicted = np.stack([tm.state_correction(row) for row in z])

    rel = np.linalg.norm(predicted - drift) / np.linalg.norm(drift)
    assert rel < 1e-3
    assert np.linalg.norm(tm.grad_mean) > 0.0
    assert np.linalg.norm(tm.ridge_intercept) > 0.0


def test_transport_map_roundtrip_serialization():
    delta_g, delta_z, groups, _ = _linear_dataset(seed=2, n=60)
    tm = fit_transport_map(delta_g, delta_z, groups, rank=4)
    tm2 = TransportMap.from_dict(tm.to_dict())
    z = delta_z[0]
    assert np.allclose(tm.state_correction(z), tm2.state_correction(z))
    assert tm2.train_group_hash == tm.train_group_hash


def test_transport_map_numpy_and_tensor_paths_are_affine_parity():
    import torch

    delta_g, delta_z, groups, _ = _linear_dataset(seed=23, n=60)
    # Shift both arrays so dropping the learned affine terms is observable.
    delta_g = delta_g + 2.0
    delta_z = delta_z + 3.0
    tm = fit_transport_map(delta_g, delta_z, groups, rank=4)
    z = delta_z[7]

    assert np.allclose(
        tm.state_correction(z),
        tm.state_correction_tensor(torch.tensor(z, dtype=torch.float32)).numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_transport_map_rejects_pre_affine_schema():
    delta_g, delta_z, groups, _ = _linear_dataset(seed=9, n=40)
    payload = fit_transport_map(delta_g, delta_z, groups, rank=4).to_dict()
    payload.pop("schema_version")

    import pytest

    with pytest.raises(ValueError, match="refit"):
        TransportMap.from_dict(payload)


def test_transport_map_rejects_incomplete_affine_state_at_construction():
    import pytest

    with pytest.raises(ValueError, match="grad_mean"):
        TransportMap(
            rank=2,
            basis=np.eye(4, 2),
            grad_mean=np.zeros(0),
            a_matrix=np.ones((2, 3)),
            ridge_intercept=np.zeros(2),
        )


def test_rank_deficient_padding():
    """Fewer samples than requested rank: zero-padded basis, no crash."""
    delta_g, delta_z, groups, _ = _linear_dataset(n=3, rank=2, seed=3)
    tm = fit_transport_map(delta_g, delta_z, groups[:3], rank=16)
    assert tm.basis.shape[1] == 16
    assert np.linalg.norm(tm.basis[:, 8:]) == 0.0


def test_fisher_ema_matches_closed_form():
    import torch

    from lightcone_spec.transport.fisher import FisherEMA

    ema = FisherEMA(num_params=5, decay=0.9)
    g1 = torch.ones(5)
    g2 = 2 * torch.ones(5)
    ema.update(g1)
    ema.update(g2)
    expected = 0.9 * (0.1 * g1**2) + 0.1 * g2**2
    assert torch.allclose(ema.value, expected)
    comp = ema.parameter_compensation(torch.full((5,), 2.0))
    assert torch.allclose(comp, expected * 2.0)


def test_parameter_transport_uses_positive_local_gradient_compensation():
    import torch

    from lightcone_spec.transport.apply import transport_gradient

    # For L(phi)=0.5*H*phi^2, g(phi+s)=g(phi)+H*s.  A diagonal Fisher/Hessian
    # proxy must therefore be added with a positive sign.
    raw = torch.tensor([2.0, -1.0])
    hessian_proxy = torch.tensor([3.0, 4.0])
    displacement = torch.tensor([0.5, -0.25])

    result = transport_gradient(
        raw,
        hessian_proxy,
        displacement,
        transport_map=None,
        delta_z=None,
        variant="parameter_only",
    )

    assert torch.allclose(
        result.transported_grad,
        raw + hessian_proxy * displacement,
    )
