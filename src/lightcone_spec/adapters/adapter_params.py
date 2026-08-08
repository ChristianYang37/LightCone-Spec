"""Request-local trainable tail updates (spec 5.1.3, 5.3).

Trainable parameters per request:
  A_d in R^{r x 128}   draft-logit residual  delta_l = B A_d u_k
  A_m in R^{r x r_M}   Markov residual       delta_l = B A_m m_{k-1}
  A_c in R^{K_max x (128+r_M+1)} confidence residual

The historical ``A_d/A_m/A_c`` output-residual layout remains the default.
Two cache-safe layouts share the same flat-bank/runtime protocol:

``tail_lora``
  ``delta_h = (h A_h + m A_m) B_h`` with a deterministic input-factor
  initialization and a zero output factor.

``full_rank_tail``
  ``delta_h = h D_h + m D_m`` with zero dense matrices.

Neither layout mutates the drafter backbone or the (possibly target-shared)
LM head, so publishing an update cannot invalidate historical draft KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

# Seven is the historical DSpark default, not a cross-backend limit.  Keep the
# name as a compatibility alias for callers which construct the default
# DSpark layout, while bounding every explicitly declared layout separately.
MAX_DRAFT_DEPTH = 7
MAX_SUPPORTED_DRAFT_DEPTH = 16
HIDDEN_PROJ_DIM = 128
TAIL_INIT_SEED = 0

# The serving bank keeps a model-dtype forward row and a canonical FP32
# dequantized mirror.  Naming these contracts in the layout identity prevents
# controllers/replay artifacts trained against the historical FP32-forward
# implementation from being reused silently.
FORWARD_QUANTIZATION_CONTRACT = "torch_cast_qdq_canonical_master_v1"
ANALYTIC_GRADIENT_CONTRACT = "native_forward_ste_fp32_master_v1"

TailLayoutMode = Literal[
    "output_residual",
    "tail_lora",
    "full_rank_tail",
]
WeightUpdateMode = Literal["residual", "lora", "full"]


def canonical_weight_update_mode(value: str) -> WeightUpdateMode:
    """Re-export the public schema normalization for low-level callers."""
    from typing import cast

    from lightcone_spec.config.schema import canonical_weight_update_mode as canonical

    return cast(WeightUpdateMode, canonical(value))


def canonical_tail_layout_mode(value: str) -> TailLayoutMode:
    """Resolve a public/legacy mode to the frozen bank-layout spelling."""
    from typing import cast

    from lightcone_spec.config.schema import canonical_tail_layout_mode as canonical

    return cast(TailLayoutMode, canonical(value))


def initial_parameter_vector(
    shapes: "AdapterShapes",
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the immutable request-start center ``phi0``.

    Output-residual and full-rank tail updates start at exact zero.  A LoRA
    update needs one non-zero factor or its zero-effect point has a zero
    gradient for *both* factors.  We therefore use the standard deterministic
    input-factor initialization and a zero output factor.  The effective tail
    delta is still exactly zero.
    """
    out = torch.zeros(shapes.num_params(), dtype=torch.float32, device=device)
    if shapes.mode != "tail_lora":
        return out
    # Use the CPU RNG so a locked seed defines identical coordinates on every
    # CUDA device and TP rank. Sampling/repetition seeds must not rotate the
    # controller's parameter space.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TAIL_INIT_SEED)
    slices = shapes.parameter_slices()
    a_h = out[slices["a_h"]].view(shapes.hidden_size, shapes.rank)
    a_h.copy_(
        torch.randn(a_h.shape, generator=generator, dtype=torch.float32).mul_(
            shapes.hidden_size**-0.5
        )
    )
    if shapes.has_markov:
        a_m = out[slices["a_m"]].view(shapes.markov_dim, shapes.rank)
        a_m.copy_(
            torch.randn(a_m.shape, generator=generator, dtype=torch.float32).mul_(
                max(shapes.markov_dim, 1) ** -0.5
            )
        )
    # b_h and confidence rows remain zero: exact no-op at request start.
    return out


def canonicalize_master_vector(
    phi: torch.Tensor, forward_dtype: torch.dtype
) -> torch.Tensor:
    """Return the FP32 Q-DQ representative used by the serving bank.

    Active masters are deliberately canonical: casting one to the forward
    dtype and back must be an identity.  This makes a source snapshot denote
    exactly the same parameter point as the graph-visible row while Adam,
    trust-region and transport arithmetic remain FP32.
    """

    if not torch.is_floating_point(phi):
        raise TypeError("tail parameters must be floating point")
    return phi.to(dtype=forward_dtype).to(dtype=torch.float32)


def parameter_views(phi: torch.Tensor, shapes: "AdapterShapes") -> dict[str, torch.Tensor]:
    """Typed views over a flat bank row without allocations or copies."""
    slices = shapes.parameter_slices()
    views: dict[str, torch.Tensor] = {}
    if shapes.mode == "output_residual":
        views["a_h"] = phi[slices["a_h"]].view(shapes.rank, HIDDEN_PROJ_DIM)
        if shapes.has_markov:
            views["a_m"] = phi[slices["a_m"]].view(shapes.rank, shapes.markov_dim)
    elif shapes.mode == "tail_lora":
        views["a_h"] = phi[slices["a_h"]].view(shapes.hidden_size, shapes.rank)
        if shapes.has_markov:
            views["a_m"] = phi[slices["a_m"]].view(shapes.markov_dim, shapes.rank)
        views["b_h"] = phi[slices["b_h"]].view(shapes.rank, shapes.hidden_size)
    else:
        views["d_h"] = phi[slices["d_h"]].view(shapes.hidden_size, shapes.hidden_size)
        if shapes.has_markov:
            views["d_m"] = phi[slices["d_m"]].view(shapes.markov_dim, shapes.hidden_size)
    if shapes.has_confidence:
        views["a_c"] = phi[slices["a_c"]].view(
            shapes.draft_depth, shapes.conf_feature_dim
        )
    return views


def rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


@dataclass
class AdapterShapes:
    rank: int
    markov_dim: int  # r_M
    vocab_size: int
    weight_update_mode: str = "output_residual"
    hidden_size: int = HIDDEN_PROJ_DIM
    draft_depth: int = MAX_DRAFT_DEPTH
    has_markov: bool = True
    has_confidence: bool = True
    algorithm: str = "DSPARK"

    def __post_init__(self) -> None:
        if isinstance(self.draft_depth, bool) or not isinstance(
            self.draft_depth, int
        ):
            raise TypeError("draft_depth must be an integer")
        if not 1 <= self.draft_depth <= MAX_SUPPORTED_DRAFT_DEPTH:
            raise ValueError(
                f"draft_depth {self.draft_depth} is outside the supported "
                f"range [1, {MAX_SUPPORTED_DRAFT_DEPTH}]"
            )

    @property
    def mode(self) -> TailLayoutMode:
        return canonical_tail_layout_mode(self.weight_update_mode)

    @property
    def public_weight_update_mode(self) -> WeightUpdateMode:
        return canonical_weight_update_mode(self.weight_update_mode)

    @property
    def conf_feature_dim(self) -> int:
        hidden_dim = HIDDEN_PROJ_DIM if self.mode == "output_residual" else self.hidden_size
        markov_dim = self.markov_dim if self.has_markov else 0
        return hidden_dim + markov_dim + 1

    @property
    def confidence_num_params(self) -> int:
        if not self.has_confidence:
            return 0
        return self.draft_depth * self.conf_feature_dim

    def parameter_slices(self) -> dict[str, slice]:
        """Canonical flat layout used by banks, artifacts and online kernels."""
        offset = 0
        out: dict[str, slice] = {}

        def take(name: str, count: int) -> None:
            nonlocal offset
            out[name] = slice(offset, offset + int(count))
            offset += int(count)

        if self.mode == "output_residual":
            take("a_h", self.rank * HIDDEN_PROJ_DIM)
            if self.has_markov:
                take("a_m", self.rank * self.markov_dim)
        elif self.mode == "tail_lora":
            take("a_h", self.hidden_size * self.rank)
            if self.has_markov:
                take("a_m", self.markov_dim * self.rank)
            take("b_h", self.rank * self.hidden_size)
        else:
            take("d_h", self.hidden_size * self.hidden_size)
            if self.has_markov:
                take("d_m", self.markov_dim * self.hidden_size)
        if self.has_confidence:
            take("a_c", self.confidence_num_params)
        return out

    def num_params(self) -> int:
        slices = self.parameter_slices()
        return max((value.stop for value in slices.values()), default=0)

    def parameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Canonical tensor shapes corresponding to :meth:`parameter_slices`."""

        if self.mode == "output_residual":
            out = {"a_h": (self.rank, HIDDEN_PROJ_DIM)}
            if self.has_markov:
                out["a_m"] = (self.rank, self.markov_dim)
        elif self.mode == "tail_lora":
            out = {"a_h": (self.hidden_size, self.rank)}
            if self.has_markov:
                out["a_m"] = (self.markov_dim, self.rank)
            out["b_h"] = (self.rank, self.hidden_size)
        else:
            out = {"d_h": (self.hidden_size, self.hidden_size)}
            if self.has_markov:
                out["d_m"] = (self.markov_dim, self.hidden_size)
        if self.has_confidence:
            out["a_c"] = (self.draft_depth, self.conf_feature_dim)
        return out

    def layout_dict(self, *, forward_dtype: str | None = None) -> dict:
        """Stable semantic identity; tensor addresses and request ids are excluded."""

        parameter_shapes = self.parameter_shapes()
        return {
            "schema_version": 2,
            "algorithm": self.algorithm.upper(),
            "weight_update_mode": self.mode,
            "hidden_size": self.hidden_size,
            "markov_dim": self.markov_dim if self.has_markov else 0,
            "vocab_size": self.vocab_size,
            "rank": self.rank if self.mode != "full_rank_tail" else None,
            "draft_depth": self.draft_depth,
            "has_markov": self.has_markov,
            "has_confidence": self.has_confidence,
            "initialization": (
                {
                    "scheme": "normal_fan_in_input_zero_output",
                    "seed": TAIL_INIT_SEED,
                    "alpha_over_rank": 1.0,
                }
                if self.mode == "tail_lora"
                else {"scheme": "zero"}
            ),
            "parameters": [
                {
                    "name": name,
                    "start": value.start,
                    "stop": value.stop,
                    "shape": list(parameter_shapes[name]),
                    "master_dtype": "torch.float32",
                    "forward_dtype": (
                        str(forward_dtype) if forward_dtype is not None else None
                    ),
                }
                for name, value in self.parameter_slices().items()
            ],
        }


def parameter_layout_identity(
    shapes: AdapterShapes,
    *,
    forward_dtype: str,
    tensor_parallel_rank: int | str,
    tensor_parallel_world_size: int,
    target_revision: str | None,
    drafter_revision: str | None,
    projection_identity: dict | None,
    head_identity: dict,
    optimizer_identity: dict | None = None,
) -> dict:
    """Canonical identity of the replicated tail bank and sharded head.

    The trainable bank is replicated across tensor-parallel workers.  A process
    rank is therefore execution metadata, not parameter semantics: including it
    would make identical per-rank replay shards impossible to combine into one
    controller artifact.
    """
    del tensor_parallel_rank
    projection_identity = (
        None
        if projection_identity is None
        else {
            key: value
            for key, value in projection_identity.items()
            if key not in {"tensor_parallel_rank", "process_rank"}
        }
    )
    head_identity = {
        key: value
        for key, value in head_identity.items()
        if key not in {"tensor_parallel_rank", "process_rank"}
    }
    optimizer_identity = dict(
        optimizer_identity
        or {
            "name": "adamw",
            "update_contract": "decoupled_adamw_delta_v1",
            "weight_decay": 0.0,
        }
    )
    return {
        "schema_version": 3,
        "layout": shapes.layout_dict(forward_dtype=str(forward_dtype)),
        "forward_dtype": str(forward_dtype),
        "forward_quantization_contract": FORWARD_QUANTIZATION_CONTRACT,
        "analytic_gradient_contract": ANALYTIC_GRADIENT_CONTRACT,
        "tensor_parallel": {
            "bank_replication": "replicated",
            "head_sharding": "vocab_sharded_v1",
            "world_size": int(tensor_parallel_world_size),
        },
        "target_revision": target_revision,
        "drafter_revision": drafter_revision,
        "projection_identity": projection_identity,
        "head_identity": head_identity,
        # Optimizer hyperparameters do not change tensor shapes, but they do
        # change the meaning of every candidate represented by this bank.
        # Binding them here makes the layout hash—and therefore controller
        # artifact filename—unambiguous.
        "optimizer_identity": optimizer_identity,
    }


def parameter_layout_sha256(identity: dict) -> str:
    from lightcone_spec.locking.hashing import sha256_json

    return sha256_json(identity)


class AdapterParams(torch.nn.Module):
    """FP32 master adapter parameters for one request."""

    def __init__(self, shapes: AdapterShapes, output_basis: torch.Tensor):
        super().__init__()
        self.shapes = shapes
        if output_basis.shape != (shapes.vocab_size, shapes.rank):
            raise ValueError(
                f"output basis shape {tuple(output_basis.shape)} != "
                f"({shapes.vocab_size}, {shapes.rank})"
            )
        # Frozen basis; FP32 buffer.
        self.register_buffer("basis", output_basis.to(torch.float32))
        device = self.basis.device
        self.a_d = torch.nn.Parameter(
            torch.zeros(
                shapes.rank, HIDDEN_PROJ_DIM, dtype=torch.float32, device=device
            )
        )
        self.a_m = torch.nn.Parameter(
            torch.zeros(
                shapes.rank, shapes.markov_dim, dtype=torch.float32, device=device
            )
        )
        self.a_c = torch.nn.Parameter(
            torch.zeros(
                shapes.draft_depth,
                shapes.conf_feature_dim,
                dtype=torch.float32,
                device=device,
            )
        )

    # ---- forward residuals (spec 5.1.3) -------------------------------

    def draft_logit_residual(self, u: torch.Tensor) -> torch.Tensor:
        """u: (K, 128) projected hiddens -> (K, V) logit residual."""
        return (u.to(torch.float32) @ self.a_d.T) @ self.basis.T

    def markov_logit_residual(self, m_prev: torch.Tensor) -> torch.Tensor:
        """m_prev: (K, r_M) frozen markov_w1 embeddings -> (K, V)."""
        return (m_prev.to(torch.float32) @ self.a_m.T) @ self.basis.T

    def confidence_residual(
        self, u: torch.Tensor, m_prev: torch.Tensor
    ) -> torch.Tensor:
        """Per-position scalar residual: delta_a_k = A_c[k] . feat_k, for
        positions k = 1..K (K <= the declared draft depth)."""
        k = u.shape[0]
        if k > self.shapes.draft_depth:
            raise ValueError(
                f"signal draft depth {k} exceeds declared depth "
                f"{self.shapes.draft_depth}"
            )
        feat = torch.cat(
            [
                u.to(torch.float32),
                rmsnorm(m_prev.to(torch.float32)),
                torch.ones(k, 1, dtype=torch.float32, device=u.device),
            ],
            dim=1,
        )
        return (self.a_c[:k] * feat).sum(dim=1)

    # ---- parameter vector utilities ------------------------------------

    def flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def load_flat(self, vec: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(
                    vec[offset : offset + n]
                    .reshape(p.shape)
                    .to(device=p.device, dtype=torch.float32)
                )
                offset += n
        if offset != vec.numel():
            raise ValueError("flat vector size mismatch")

    def grad_flat(self) -> torch.Tensor:
        parts = []
        for p in self.parameters():
            if p.grad is None:
                parts.append(
                    torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                )
            else:
                parts.append(p.grad.detach().reshape(-1).to(torch.float32))
        return torch.cat(parts)


def trust_region_project(
    phi: torch.Tensor, phi0: torch.Tensor, radius: float
) -> torch.Tensor:
    """Euclidean projection onto {phi: ||phi - phi0||_2 <= radius},
    performed on FP32 master parameters (spec 5.3)."""
    phi32 = phi.to(torch.float32)
    phi0_32 = phi0.to(device=phi.device, dtype=torch.float32)
    diff = phi32 - phi0_32
    norm = torch.linalg.vector_norm(diff)
    # Tensor-only projection keeps the CUDA launch asynchronous.  The old
    # Python ``if norm <= radius`` forced a device synchronization on every
    # candidate update.
    scale = torch.clamp(
        torch.as_tensor(radius, device=norm.device, dtype=norm.dtype)
        / norm.clamp_min(torch.finfo(norm.dtype).tiny),
        max=1.0,
    )
    return phi0_32 + diff * scale


def clip_gradient_global_norm(
    grad: torch.Tensor, max_norm: float
) -> tuple[torch.Tensor, torch.Tensor | float]:
    """Global-norm clipping (spec 6.3). Returns (clipped, scale)."""
    norm = torch.linalg.vector_norm(grad.to(torch.float32))
    if not grad.is_cuda and not torch.isfinite(norm):
        from lightcone_spec.exit_codes import NumericalFailure

        raise NumericalFailure(f"non-finite gradient norm: {norm.item()}")
    scale = torch.clamp(
        torch.as_tensor(max_norm, device=norm.device, dtype=norm.dtype)
        / norm.clamp_min(torch.finfo(norm.dtype).tiny),
        max=1.0,
    )
    clipped = grad.to(torch.float32) * scale
    if grad.is_cuda:
        return clipped, scale
    return clipped, float(scale)
