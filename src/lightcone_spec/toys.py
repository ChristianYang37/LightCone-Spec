"""P0-A delayed-update optimization toy (spec 13.1).

A quadratic tracking problem with a drifting optimum: the learner
receives gradients with delay d and step size eta. Regret grows with
the *state movement during the delay window* (the toy analogue of rho),
not with wall-clock time itself: idle periods with a frozen optimum add
delay but no regret. Damping the stale step by exp(-rho/R) mitigates
the loss under drift. Curves for different (d, drift-rate) settings
collapse when plotted against rho.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class P0AResult:
    delays: list[int]
    drift_rates: list[float]
    regret: dict  # (d, rate) -> mean regret
    rho: dict  # (d, rate) -> mean rho over the delay window
    damped_regret: dict
    idle_regret_delta: float
    collapse_spread: float

    def to_dict(self) -> dict:
        def keyed(d):
            return {f"d{k[0]}_rate{k[1]}": v for k, v in d.items()}

        return {
            "delays": self.delays,
            "drift_rates": self.drift_rates,
            "regret": keyed(self.regret),
            "rho": keyed(self.rho),
            "damped_regret": keyed(self.damped_regret),
            "idle_regret_delta": self.idle_regret_delta,
            "collapse_spread": self.collapse_spread,
        }


def _run_chain(
    delay: int,
    drift_rate: float,
    steps: int = 400,
    eta: float = 0.2,
    dim: int = 8,
    seed: int = 0,
    damping_radius: float | None = None,
    idle_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Returns (mean regret, mean rho over delay windows)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    theta = np.zeros(dim)
    opt = np.zeros(dim)
    drift_dir = rng.standard_normal((steps, dim))
    drift_dir /= np.linalg.norm(drift_dir, axis=1, keepdims=True)
    grads: list[tuple[int, np.ndarray, np.ndarray]] = []
    opts = [opt.copy()]
    regret = 0.0
    rhos = []
    # Event time: idle wall steps host no round, so nothing is measured,
    # applied, or scored there (time-dilation invariance: wall delay grows
    # under idle insertion while rho and regret are untouched).
    et = 0
    for t in range(steps):
        if idle_mask is not None and idle_mask[t]:
            continue
        opt = opt + drift_rate * drift_dir[et]
        opts.append(opt.copy())
        # gradient of 0.5||theta-opt||^2 measured now, applied after delay
        grads.append((et, theta - opts[et], opts[et].copy()))
        if grads and grads[0][0] <= et - delay:
            t_src, g, opt_src = grads.pop(0)
            # rho: path length of the optimum during the delay window
            rho = sum(
                np.linalg.norm(opts[j + 1] - opts[j])
                for j in range(t_src, et)
            )
            rhos.append(rho)
            scale = 1.0
            if damping_radius is not None:
                scale = float(np.exp(-rho / damping_radius))
            theta = theta - eta * scale * g
        regret += 0.5 * float(np.linalg.norm(theta - opt) ** 2)
        et += 1
    return regret / max(et, 1), float(np.mean(rhos)) if rhos else 0.0


def run_p0a(
    delays: tuple[int, ...] = (1, 3, 6, 10),
    drift_rates: tuple[float, ...] = (0.0, 0.05, 0.15),
    seed: int = 0,
) -> P0AResult:
    regret, rho, damped = {}, {}, {}
    for d in delays:
        for rate in drift_rates:
            r, p = _run_chain(d, rate, seed=seed)
            regret[(d, rate)] = r
            rho[(d, rate)] = p
            rd, _ = _run_chain(d, rate, seed=seed, damping_radius=max(p, 1e-3))
            damped[(d, rate)] = rd

    # Idle insertion: identical event sequence spread over twice the wall
    # steps; wall delay doubles but rho and regret must not change
    # (time-dilation invariance).
    steps = 400
    idle = np.zeros(2 * steps, dtype=bool)
    idle[::2] = True  # every other wall step hosts no round
    r_busy, _ = _run_chain(4, 0.1, steps=steps, seed=seed)
    r_idle, _ = _run_chain(4, 0.1, steps=2 * steps, seed=seed, idle_mask=idle)
    idle_delta = abs(r_idle - r_busy) / max(r_busy, 1e-9)

    # Collapse check: regret as a function of rho across (d, rate) settings
    # should be near-monotone with small spread (binned dispersion).
    pts = sorted(
        ((rho[k], regret[k]) for k in regret if rho[k] > 0), key=lambda x: x[0]
    )
    if len(pts) >= 4:
        xs = np.asarray([p[0] for p in pts])
        ys = np.asarray([p[1] for p in pts])
        bins = np.array_split(np.arange(len(xs)), min(4, len(xs)))
        spread = float(
            np.mean([np.std(ys[b]) / (np.mean(ys[b]) + 1e-9) for b in bins if len(b)])
        )
    else:
        spread = 0.0
    return P0AResult(
        delays=list(delays),
        drift_rates=list(drift_rates),
        regret=regret,
        rho=rho,
        damped_regret=damped,
        idle_regret_delta=idle_delta,
        collapse_spread=spread,
    )
