"""Clock predictors (spec 7.5).

One shared model class per task so feature sets never differ in
capacity:

  utility regression  : ridge on standardized features
  mismatch regression : ridge on log1p(relative_gradient_mismatch)
  harmful classifier  : L2 logistic + isotonic calibration

Regularization grids {1e-4 ... 100}; inner splits grouped by sequence;
probabilities calibrated on the calibration split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-8)
    return mean, std


@dataclass
class RidgePredictor:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    alpha: float = 1.0
    log1p_target: bool = False
    _tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        alphas: tuple[float, ...] = RIDGE_GRID,
    ) -> "RidgePredictor":
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold

        yy = np.log1p(np.maximum(y, 0.0)) if self.log1p_target else y
        self.mean, self.std = _standardize_fit(x)
        xs = (x - self.mean) / self.std
        n_groups = len(np.unique(groups))
        n_splits = min(5, max(2, n_groups))
        best_alpha, best_err = alphas[0], np.inf
        if n_groups >= 2:
            gkf = GroupKFold(n_splits=n_splits)
            for alpha in alphas:
                errs = []
                for tr, va in gkf.split(xs, yy, groups):
                    model = Ridge(alpha=alpha).fit(xs[tr], yy[tr])
                    errs.append(np.abs(model.predict(xs[va]) - yy[va]).mean())
                err = float(np.mean(errs))
                if err < best_err:
                    best_alpha, best_err = alpha, err
        model = Ridge(alpha=best_alpha).fit(xs, yy)
        self.coef = model.coef_.astype(np.float64)
        self.intercept = float(model.intercept_)
        self.alpha = best_alpha
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.atleast_2d(x) - self.mean) / self.std
        out = xs @ self.coef + self.intercept
        if self.log1p_target:
            out = np.expm1(out)
            out = np.maximum(out, 0.0)
        return out

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        key = str(x.device)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = tuple(
                torch.as_tensor(value, device=x.device, dtype=torch.float32)
                for value in (self.mean, self.std, self.coef)
            )
            self._tensor_cache[key] = cached
        mean, std, coef = cached
        out = ((x.float() - mean) / std.clamp_min(1e-8)) @ coef
        out = out + float(self.intercept)
        if self.log1p_target:
            out = torch.expm1(out).clamp_min(0.0)
        return out

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "alpha": self.alpha,
            "log1p_target": self.log1p_target,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RidgePredictor":
        p = cls(log1p_target=d["log1p_target"])
        p.mean = np.asarray(d["mean"])
        p.std = np.asarray(d["std"])
        p.coef = np.asarray(d["coef"])
        p.intercept = d["intercept"]
        p.alpha = d["alpha"]
        return p


@dataclass
class HarmfulClassifier:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    c_inv: float = 1.0  # regularization strength (1/C in sklearn terms)
    iso_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    iso_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _tensor_cache: dict[str, tuple[torch.Tensor, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        alphas: tuple[float, ...] = RIDGE_GRID,
    ) -> "HarmfulClassifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        self.mean, self.std = _standardize_fit(x)
        xs = (x - self.mean) / self.std
        if len(np.unique(y)) < 2:
            # Degenerate labels: constant classifier at the empirical rate.
            self.coef = np.zeros(x.shape[1])
            self.intercept = float(
                np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
            )
            return self
        n_groups = len(np.unique(groups))
        best_a, best_ll = alphas[0], -np.inf
        if n_groups >= 2:
            gkf = GroupKFold(n_splits=min(5, max(2, n_groups)))
            for a in alphas:
                lls = []
                for tr, va in gkf.split(xs, y, groups):
                    if len(np.unique(y[tr])) < 2:
                        continue
                    model = LogisticRegression(C=1.0 / a, max_iter=2000).fit(
                        xs[tr], y[tr]
                    )
                    p = np.clip(model.predict_proba(xs[va])[:, 1], 1e-9, 1 - 1e-9)
                    lls.append(float((y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)).mean()))
                if lls and np.mean(lls) > best_ll:
                    best_a, best_ll = a, float(np.mean(lls))
        model = LogisticRegression(C=1.0 / best_a, max_iter=2000).fit(xs, y)
        self.coef = model.coef_[0].astype(np.float64)
        self.intercept = float(model.intercept_[0])
        self.c_inv = best_a
        return self

    def raw_probability(self, x: np.ndarray) -> np.ndarray:
        xs = (np.atleast_2d(x) - self.mean) / self.std
        z = xs @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def calibrate(self, x_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Isotonic calibration on the calibration split (spec 7.5)."""
        from sklearn.isotonic import IsotonicRegression

        if len(y_cal) == 0:
            # No calibration data: keep raw probabilities.
            self.iso_x = np.zeros(0)
            self.iso_y = np.zeros(0)
            return
        p = self.raw_probability(x_cal)
        if len(np.unique(y_cal)) < 2 or len(y_cal) < 4:
            self.iso_x = np.array([0.0, 1.0])
            self.iso_y = np.array([float(np.mean(y_cal))] * 2)
            return
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            p, y_cal
        )
        grid = np.linspace(0.0, 1.0, 101)
        self.iso_x = grid
        self.iso_y = iso.predict(grid)

    def probability(self, x: np.ndarray) -> np.ndarray:
        p = self.raw_probability(x)
        if self.iso_x.size == 0:
            return p
        return np.interp(p, self.iso_x, self.iso_y)

    def probability_tensor(self, x: torch.Tensor) -> torch.Tensor:
        key = str(x.device)
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = tuple(
                torch.as_tensor(value, device=x.device, dtype=torch.float32)
                for value in (self.mean, self.std, self.coef, self.iso_x, self.iso_y)
            )
            self._tensor_cache[key] = cached
        mean, std, coef, iso_x, iso_y = cached
        z = ((x.float() - mean) / std.clamp_min(1e-8)) @ coef
        p = torch.sigmoid(z + float(self.intercept))
        if iso_x.numel() == 0:
            return p
        hi = torch.searchsorted(iso_x, p).clamp(1, iso_x.numel() - 1)
        lo = hi - 1
        x0, x1 = iso_x[lo], iso_x[hi]
        y0, y1 = iso_y[lo], iso_y[hi]
        frac = (p - x0) / (x1 - x0).clamp_min(1e-8)
        return torch.where(
            p <= iso_x[0],
            iso_y[0],
            torch.where(p >= iso_x[-1], iso_y[-1], y0 + frac * (y1 - y0)),
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "c_inv": self.c_inv,
            "iso_x": self.iso_x.tolist(),
            "iso_y": self.iso_y.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HarmfulClassifier":
        c = cls()
        c.mean = np.asarray(d["mean"])
        c.std = np.asarray(d["std"])
        c.coef = np.asarray(d["coef"])
        c.intercept = d["intercept"]
        c.c_inv = d["c_inv"]
        c.iso_x = np.asarray(d["iso_x"])
        c.iso_y = np.asarray(d["iso_y"])
        return c
