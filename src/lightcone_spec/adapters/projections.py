"""Fixed projection artifacts (spec 5.1.1, 5.1.2, 7.1, 7.4).

All artifacts are constructed once per model pair with fully specified
deterministic recipes (NumPy PCG64 seed 0, CPU float64 linear algebra,
sign fixing), stored as float32 with SHA-256 sidecars, and shared by
every method.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lightcone_spec.exit_codes import ConfigError
from lightcone_spec.locking.hashing import sha256_bytes


def _fix_column_signs(mat: np.ndarray) -> np.ndarray:
    """Make the largest-|entry| element of each column positive."""
    out = mat.copy()
    idx = np.argmax(np.abs(out), axis=0)
    signs = np.sign(out[idx, np.arange(out.shape[1])])
    signs[signs == 0] = 1.0
    return out * signs[None, :]


def _array_sha256(arr: np.ndarray) -> str:
    arr32 = np.ascontiguousarray(arr, dtype=np.float32)
    return sha256_bytes(arr32.tobytes())


# ---------------------------------------------------------------------------
# 5.1.1 Fixed hidden projection R_h in R^{d x m}
# ---------------------------------------------------------------------------


def build_hidden_projection(d: int, m: int = 128, seed: int = 0) -> np.ndarray:
    """PCG64(seed) standard Gaussian d x m -> reduced QR in float64 ->
    column sign fixing -> float32 artifact."""
    if d < m:
        raise ConfigError(f"hidden size d={d} must be >= projection dim m={m}")
    rng = np.random.Generator(np.random.PCG64(seed))
    g = rng.standard_normal((d, m)).astype(np.float64)
    q, _ = np.linalg.qr(g, mode="reduced")
    q = _fix_column_signs(q)
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# 5.1.2 Fixed output basis B in R^{V x r}
# ---------------------------------------------------------------------------


def _colnorm(mat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Normalize columns to unit L2 norm; drop (and report) zero columns."""
    norms = np.linalg.norm(mat, axis=0)
    keep = np.where(norms > 0)[0]
    dropped = [int(i) for i in np.where(norms == 0)[0]]
    return mat[:, keep] / norms[keep][None, :], dropped


@dataclass
class OutputBasis:
    basis: np.ndarray  # float32, V x r
    singular_values: np.ndarray  # float64
    dropped_columns: list[int]
    input_sha256: str
    rank: int

    def sha256(self) -> str:
        return _array_sha256(self.basis)


def build_output_basis(
    w_lm: np.ndarray,
    w_m2: np.ndarray,
    r_h: np.ndarray,
    rank: int,
) -> OutputBasis:
    """A = [colnorm(W_lm @ R_h), colnorm(W_M2)] -> thin SVD (float64, CPU)
    -> first `rank` left singular vectors -> sign fixing.

    Each rank gets its own construction and hash record (spec 5.4): no
    silent truncation of a larger-rank basis.
    """
    w_lm = np.asarray(w_lm, dtype=np.float64)
    w_m2 = np.asarray(w_m2, dtype=np.float64)
    r_h64 = np.asarray(r_h, dtype=np.float64)
    if w_lm.shape[0] != w_m2.shape[0]:
        raise ConfigError(
            "target lm_head and DSpark markov_w2 vocabulary sizes differ "
            f"({w_lm.shape[0]} vs {w_m2.shape[0]}); pair is invalid (fail closed)"
        )
    a1 = w_lm @ r_h64
    a1n, dropped1 = _colnorm(a1)
    a2n, dropped2 = _colnorm(w_m2)
    a = np.concatenate([a1n, a2n], axis=1)
    input_sha = sha256_bytes(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
    )
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    if rank > u.shape[1]:
        raise ConfigError(f"requested rank {rank} exceeds available {u.shape[1]}")
    b = _fix_column_signs(u[:, :rank])
    dropped = dropped1 + [a1.shape[1] + i for i in dropped2]
    return OutputBasis(
        basis=b.astype(np.float32),
        singular_values=s[:rank].astype(np.float64),
        dropped_columns=dropped,
        input_sha256=input_sha,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# 7.4 Signed count-sketch for the transport z-vector
# ---------------------------------------------------------------------------


@dataclass
class CountSketch:
    dim: int
    seed: int

    def _hashes(self, token_id: int) -> tuple[int, float]:
        h = sha256_bytes(f"cs/{self.seed}/{token_id}".encode("utf-8"))
        bucket = int(h[:8], 16) % self.dim
        sign = 1.0 if int(h[8:9], 16) % 2 == 0 else -1.0
        return bucket, sign

    def project(self, token_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float64)
        for tid, val in zip(np.asarray(token_ids).ravel(), np.asarray(values).ravel()):
            bucket, sign = self._hashes(int(tid))
            out[bucket] += sign * float(val)
        return out

    def artifact_dict(self) -> dict:
        return {
            "kind": "signed_count_sketch",
            "dim": self.dim,
            "seed": self.seed,
            "token_hash": "sha256(cs/{seed}/{token_id})[:8] mod dim",
            "sign_hash": "sha256(cs/{seed}/{token_id})[8:9] parity",
        }


# ---------------------------------------------------------------------------
# Artifact IO with hash sidecars
# ---------------------------------------------------------------------------


def save_projection_artifact(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> str:
    """Save arrays (float32) + metadata + sha256 sidecar. Returns hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays32 = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in arrays.items()}
    payload = Path(str(path) if str(path).endswith(".npz") else str(path) + ".npz")
    tmp_payload = payload.with_name(payload.name + f".p{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp_payload, **arrays32)
    digest = sha256_bytes(tmp_payload.read_bytes())
    meta = dict(metadata)
    meta["arrays_sha256"] = {k: _array_sha256(v) for k, v in arrays32.items()}
    meta["file_sha256"] = digest
    meta_path = Path(str(payload) + ".meta.json")
    tmp_meta = meta_path.with_name(meta_path.name + f".p{os.getpid()}.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp_payload, payload)
    os.replace(tmp_meta, meta_path)
    return digest


def load_projection_artifact(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    payload = Path(str(path) if str(path).endswith(".npz") else str(path) + ".npz")
    meta_path = Path(str(payload) + ".meta.json")
    if not payload.is_file() or not meta_path.is_file():
        raise ConfigError(f"projection artifact missing: {payload}")
    meta = json.loads(meta_path.read_text())
    digest = sha256_bytes(payload.read_bytes())
    if digest != meta.get("file_sha256"):
        raise ConfigError(f"projection artifact hash drift: {payload}")
    with np.load(payload) as z:
        arrays = {k: z[k].copy() for k in z.files}
    expected_arrays = meta.get("arrays_sha256", {})
    for name, value in arrays.items():
        expected = expected_arrays.get(name)
        if expected is None or _array_sha256(value) != expected:
            raise ConfigError(f"projection array hash drift for {name}: {payload}")
    return arrays, meta
