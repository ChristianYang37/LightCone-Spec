"""Benjamini-Hochberg FDR at 0.05 for all non-confirmatory ablations
(spec 14.6). H1/H2/H3 are confirmatory and never enter the FDR pool."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Returns rejection decisions aligned with the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    k = int(np.max(np.where(passed)[0])) + 1 if passed.any() else 0
    reject = np.zeros(n, dtype=bool)
    reject[order[:k]] = True
    return reject.tolist()
