"""Cluster-paired BCa bootstrap (spec 14.6).

The statistical unit is the request/sequence, never the round. Clusters:
base prompt_id in P2/P4 (all seeds of one prompt stay together, and all
trajectories of one base prompt stay together), full stream_id for
streaming. B = 5000, 95% interval, bootstrap seed 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

B_DEFAULT = 5000
CI_LEVEL = 0.95


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    b: int
    n_clusters: int
    method: str = "bca"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _cluster_indices(clusters: np.ndarray) -> list[np.ndarray]:
    uniq = np.unique(clusters)
    return [np.where(clusters == c)[0] for c in uniq]


def cluster_bca(
    values: np.ndarray,
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    b: int = B_DEFAULT,
    seed: int = 0,
) -> BootstrapResult:
    """BCa interval for `statistic` of values, resampling whole clusters.

    values may be 1-D (plain statistic) or 2-D with the statistic taking
    rows (e.g. paired differences already computed per unit).
    """
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.ndim == 0 or len(values) == 0:
        raise ValueError("cluster BCa requires at least one observation")
    if len(values) != len(clusters):
        raise ValueError("cluster BCa values and clusters differ in length")
    if not isinstance(b, (int, np.integer)) or int(b) <= 0:
        raise ValueError("cluster BCa bootstrap count must be positive")
    if not np.isfinite(values).all():
        raise ValueError("cluster BCa values must be finite")
    idx_by_cluster = _cluster_indices(clusters)
    n_c = len(idx_by_cluster)
    if n_c < 2:
        est = float(statistic(values))
        return BootstrapResult(est, est, est, 0, n_c, method="degenerate")
    theta_hat = float(statistic(values))
    rng = np.random.Generator(np.random.PCG64(seed))
    boot = np.empty(b, dtype=np.float64)
    for i in range(b):
        take = rng.integers(0, n_c, size=n_c)
        idx = np.concatenate([idx_by_cluster[t] for t in take])
        boot[i] = statistic(values[idx])
    # Bias correction.
    prop = np.clip(np.mean(boot < theta_hat), 1e-9, 1 - 1e-9)
    z0 = _norm_ppf(prop)
    # Acceleration via cluster jackknife.
    jack = np.empty(n_c, dtype=np.float64)
    for i in range(n_c):
        keep = np.concatenate(
            [idx_by_cluster[j] for j in range(n_c) if j != i]
        )
        jack[i] = statistic(values[keep])
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    alpha = (1.0 - CI_LEVEL) / 2.0
    lo = _bca_quantile(boot, z0, a, alpha)
    hi = _bca_quantile(boot, z0, a, 1.0 - alpha)
    return BootstrapResult(theta_hat, lo, hi, b, n_c)


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(x))


def _bca_quantile(boot: np.ndarray, z0: float, a: float, alpha: float) -> float:
    z_alpha = _norm_ppf(alpha)
    denominator = 1.0 - a * (z0 + z_alpha)
    if abs(denominator) < 1e-9:
        denominator = 1e-9 if denominator >= 0.0 else -1e-9
    adj = _norm_cdf(z0 + (z0 + z_alpha) / denominator)
    adj = min(max(adj, 0.0), 1.0)
    return float(np.quantile(boot, adj))


def paired_difference_by_cluster(
    a_values: dict[str, float], b_values: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Pair per-unit metrics of two methods over the same prompt+seed
    units; returns (differences a-b, cluster labels). Units present in
    only one method are dropped (pairing is strict)."""
    keys = sorted(set(a_values) & set(b_values))
    diffs = np.asarray([a_values[k] - b_values[k] for k in keys])
    clusters = np.asarray([k.split("::", 1)[0] for k in keys])
    return diffs, clusters
