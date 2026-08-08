"""Transport fitting from replay data (spec 7.9).

1. PCA (rank-k basis P_g) of fresh-minus-stale gradients delta_g_r
   collected on the replay train split;
2. ridge fit of P_g^T delta_g_r ~ A * delta_z_r, alpha from the same
   grid as the predictors;
3. basis, full-space gradient mean, ridge intercept, A and the train-group
   hash frozen into the artifact. Online correction executes the identical
   affine reconstruction used by the fitted Ridge model.

The random-transport control uses a seed-0 Gaussian orthonormal basis of
the same shape and the same GEMM path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lightcone_spec.locking.hashing import sha256_json
from lightcone_spec.trajectory.predictors import RIDGE_GRID

TRANSPORT_MAP_SCHEMA_VERSION = 2


@dataclass
class TransportMap:
    rank: int
    basis: np.ndarray  # (P, k)
    grad_mean: np.ndarray  # (P,)
    a_matrix: np.ndarray  # (k, Z)
    ridge_intercept: np.ndarray  # (k,)
    ridge_alpha: float = 1.0
    train_group_hash: str = ""
    random_basis: bool = False
    _tensor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=np.float64)
        self.grad_mean = np.asarray(self.grad_mean, dtype=np.float64)
        self.a_matrix = np.asarray(self.a_matrix, dtype=np.float64)
        self.ridge_intercept = np.asarray(
            self.ridge_intercept, dtype=np.float64
        )
        p = int(self.basis.shape[0]) if self.basis.ndim == 2 else -1
        if (
            int(self.rank) < 1
            or p < 1
            or self.basis.shape != (p, int(self.rank))
        ):
            raise ValueError(
                "transport basis must have shape (P, rank) with positive dimensions"
            )
        if self.grad_mean.shape != (p,):
            raise ValueError("transport grad_mean must have shape (P,)")
        if self.a_matrix.ndim != 2 or self.a_matrix.shape[0] != int(self.rank):
            raise ValueError("transport a_matrix must have shape (rank, Z)")
        if self.ridge_intercept.shape != (int(self.rank),):
            raise ValueError("transport ridge_intercept must have shape (rank,)")
        if not all(
            np.isfinite(value).all()
            for value in (
                self.basis,
                self.grad_mean,
                self.a_matrix,
                self.ridge_intercept,
            )
        ):
            raise ValueError("transport map contains non-finite values")

    def state_correction(self, delta_z: np.ndarray) -> np.ndarray:
        """Predict ``fresh_gradient - stale_gradient`` in parameter space.

        The ridge target is PCA-projected *centered* gradient drift.  Its
        fitted intercept and the full-space gradient mean are therefore part
        of the learned map, not optional diagnostics.  Omitting either makes
        offline Ridge predictions differ from the online transport kernel
        whenever ``delta_z`` has non-zero mean.
        """
        coeff = (
            self.ridge_intercept
            + self.a_matrix @ np.asarray(delta_z, dtype=np.float64)
        )
        return self.grad_mean + self.basis @ coeff

    def state_correction_tensor(self, delta_z: torch.Tensor) -> torch.Tensor:
        """Apply the frozen transport map without leaving the accelerator.

        Artifact arrays are copied once per device and then reused at every
        arrival boundary.  This keeps L3 from turning a CUDA ``delta_z`` into
        NumPy and avoids repeated host-to-device transfers in the hot path.
        """
        device_key = str(delta_z.device)
        cached = self._tensor_cache.get(device_key)
        if cached is None:
            cached = (
                torch.as_tensor(self.basis, device=delta_z.device, dtype=torch.float32),
                torch.as_tensor(
                    self.a_matrix, device=delta_z.device, dtype=torch.float32
                ),
                torch.as_tensor(
                    self.grad_mean + self.basis @ self.ridge_intercept,
                    device=delta_z.device,
                    dtype=torch.float32,
                ),
            )
            self._tensor_cache[device_key] = cached
        basis, a_matrix, bias = cached
        return bias + basis @ (a_matrix @ delta_z.to(dtype=torch.float32))

    def to_dict(self) -> dict:
        return {
            "schema_version": TRANSPORT_MAP_SCHEMA_VERSION,
            "rank": self.rank,
            "basis": self.basis.tolist(),
            "grad_mean": self.grad_mean.tolist(),
            "a_matrix": self.a_matrix.tolist(),
            "ridge_intercept": self.ridge_intercept.tolist(),
            "ridge_alpha": self.ridge_alpha,
            "train_group_hash": self.train_group_hash,
            "random_basis": self.random_basis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransportMap":
        if d.get("schema_version") != TRANSPORT_MAP_SCHEMA_VERSION:
            raise ValueError(
                "transport map predates the centered-Ridge reconstruction "
                "contract; refit it from its bound replay data"
            )
        return cls(
            rank=d["rank"],
            basis=np.asarray(d["basis"], dtype=np.float64),
            grad_mean=np.asarray(d["grad_mean"], dtype=np.float64),
            a_matrix=np.asarray(d["a_matrix"], dtype=np.float64),
            ridge_intercept=np.asarray(d["ridge_intercept"], dtype=np.float64),
            ridge_alpha=d["ridge_alpha"],
            train_group_hash=d["train_group_hash"],
            random_basis=d.get("random_basis", False),
        )


def fit_transport_map(
    delta_g: np.ndarray,  # (N, P) fresh - stale gradients
    delta_z: np.ndarray,  # (N, Z) transport state diffs
    groups: list[str],
    rank: int,
    alphas: tuple[float, ...] = RIDGE_GRID,
) -> TransportMap:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    delta_g = np.asarray(delta_g, dtype=np.float64)
    delta_z = np.asarray(delta_z, dtype=np.float64)
    n, p = delta_g.shape
    mean = delta_g.mean(axis=0)
    centered = delta_g - mean
    # PCA via SVD of the centered gradient-difference matrix.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(rank, vt.shape[0])
    basis = vt[:k].T  # (P, k)
    if k < rank:
        # Pad with zero directions; the artifact records the true rank.
        basis = np.concatenate([basis, np.zeros((p, rank - k))], axis=1)
    targets = centered @ basis  # (N, rank)

    group_arr = np.asarray([hash_group(g) for g in groups])
    n_groups = len(np.unique(group_arr))
    best_alpha, best_err = alphas[0], np.inf
    if n_groups >= 2 and n >= 4:
        gkf = GroupKFold(n_splits=min(5, n_groups))
        for alpha in alphas:
            errs = []
            for tr, va in gkf.split(delta_z, targets, group_arr):
                model = Ridge(alpha=alpha).fit(delta_z[tr], targets[tr])
                errs.append(
                    float(np.abs(model.predict(delta_z[va]) - targets[va]).mean())
                )
            err = float(np.mean(errs))
            if err < best_err:
                best_alpha, best_err = alpha, err
    model = Ridge(alpha=best_alpha).fit(delta_z, targets)
    a_matrix = model.coef_.astype(np.float64)  # (rank, Z)
    ridge_intercept = np.asarray(model.intercept_, dtype=np.float64)
    return TransportMap(
        rank=rank,
        basis=basis,
        grad_mean=mean,
        a_matrix=a_matrix,
        ridge_intercept=ridge_intercept,
        ridge_alpha=best_alpha,
        train_group_hash=sha256_json(sorted(set(groups))),
    )


def hash_group(g: str) -> int:
    from lightcone_spec.locking.hashing import stable_hash_int

    return stable_hash_int(g) % (2**31)


def random_orthonormal_basis(p: int, rank: int, seed: int = 0) -> np.ndarray:
    """Seed-0 Gaussian orthonormal basis with the same shape/GEMM path as
    P_g (spec 6.14, 7.9)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    g = rng.standard_normal((p, rank))
    q, _ = np.linalg.qr(g)
    return q[:, :rank]
