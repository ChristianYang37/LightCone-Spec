"""P0-A delayed-update optimization toy and P0-B exactness enumeration
(spec 15.7-15.8, 16.2 local acceptance)."""

from __future__ import annotations

import numpy as np

from lightcone_spec.toys import run_p0a


def test_p0a_regret_grows_with_delay_under_drift():
    res = run_p0a(delays=(1, 6), drift_rates=(0.15,), seed=0)
    assert res.regret[(6, 0.15)] > res.regret[(1, 0.15)]


def test_p0a_rho_tracks_drift_not_delay_alone():
    res = run_p0a(delays=(6,), drift_rates=(0.0, 0.15), seed=0)
    assert res.rho[(6, 0.15)] > res.rho[(6, 0.0)] + 1e-9


def test_p0a_damping_helps_at_high_drift():
    res = run_p0a(delays=(10,), drift_rates=(0.15,), seed=0)
    assert res.damped_regret[(10, 0.15)] <= res.regret[(10, 0.15)] * 1.05


def test_p0a_idle_insertion_invariance():
    res = run_p0a(seed=0)
    assert res.idle_regret_delta < 0.35, (
        f"idle dilation changed regret by {res.idle_regret_delta}"
    )


def test_p0b_exactness_suite():
    from lightcone_spec.runtime.exactness_harness import (
        TV_CANARY_MIN,
        TV_EXACT_LIMIT,
        run_exactness_suite,
    )

    report = run_exactness_suite(horizon=5, depth=3, mc_samples=4000, seed=0)
    assert report.tv_correct_static <= TV_EXACT_LIMIT
    assert report.tv_correct_version_change <= TV_EXACT_LIMIT
    assert report.tv_race_canary >= TV_CANARY_MIN
    assert report.race_flagged
    assert report.mc_holm_pass
    assert report.engine_canary_caught
    assert report.passed


def test_exact_sampler_residual_distribution():
    from lightcone_spec.runtime.exact_sampler import (
        acceptance_probability,
        residual_distribution,
    )

    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.5, 0.3])
    r = residual_distribution(p, q)
    assert abs(r.sum() - 1.0) < 1e-12
    # residual proportional to max(p - q, 0)
    raw = np.maximum(p - q, 0)
    assert np.allclose(r, raw / raw.sum())
    assert acceptance_probability(0.5, 0.2) == 1.0
    assert abs(acceptance_probability(0.2, 0.5) - 0.4) < 1e-12


def test_philox_substreams_deterministic_and_disjoint():
    from lightcone_spec.runtime.rng import DrawKind, request_id_hash, uniform_draw

    ha = request_id_hash("req-a")
    hb = request_id_hash("req-b")
    a = uniform_draw(0, ha, 3, 2, DrawKind.ACCEPTANCE)
    b = uniform_draw(0, ha, 3, 2, DrawKind.ACCEPTANCE)
    c = uniform_draw(0, ha, 3, 2, DrawKind.RESIDUAL)
    d = uniform_draw(0, hb, 3, 2, DrawKind.ACCEPTANCE)
    e = uniform_draw(1, ha, 3, 2, DrawKind.ACCEPTANCE)
    assert a == b
    assert len({a, c, d, e}) == 4, "substreams must be disjoint"
    assert 0.0 <= a < 1.0
