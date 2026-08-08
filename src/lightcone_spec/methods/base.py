"""Common method machinery (spec 6.0, 6.2, 6.8).

Fair-comparison invariants: every method consumes the same TeacherSignal
(same trigger rounds, teacher logits, full-window mask, confidence
targets), the same gradient clipping and trust region, and differs only
in loss extras, optimizer/update rule, publish timing, controller action
and method-private state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import torch

from lightcone_spec.adapters.adapter_params import (
    AdapterShapes,
    clip_gradient_global_norm,
    parameter_views,
    rmsnorm,
    trust_region_project,
)
from lightcone_spec.adapters.losses import (
    LossBreakdown,
    common_loss,
    common_loss_with_logit_grads,
)
from lightcone_spec.methods.optim import (
    AdamWDeltaState,
    adamw_delta,
    adamw_delta_batched,
)


GradientConsensusFn = Callable[
    [torch.Tensor, bool | torch.Tensor],
    tuple[torch.Tensor, bool | torch.Tensor],
]


def consensus_gradient(
    grad: torch.Tensor,
    finite_t: bool | torch.Tensor,
    callback: Optional[GradientConsensusFn],
) -> tuple[torch.Tensor, bool | torch.Tensor]:
    """Return one rank-consistent gradient without a CUDA scalar readback.

    A locally invalid rank contributes an exact zero to the collective.  The
    callback must return the averaged gradient and a global finite/health AND.
    We defensively recheck the reduced tensor and zero it on every rank before
    clipping or optimizer-state mutation.
    """

    finite_device = torch.as_tensor(
        finite_t, device=grad.device, dtype=torch.bool
    )
    safe_local = torch.where(finite_device, grad, torch.zeros_like(grad))
    if callback is None:
        consensus, global_ok = safe_local, finite_device
    else:
        consensus, global_ok = callback(safe_local, finite_device)
        if (
            consensus.shape != grad.shape
            or consensus.device != grad.device
            or consensus.dtype != grad.dtype
        ):
            raise ValueError(
                "gradient consensus must preserve gradient shape, device and dtype"
            )
        global_ok = torch.as_tensor(
            global_ok, device=grad.device, dtype=torch.bool
        )
        if global_ok.numel() != 1:
            raise ValueError("gradient consensus health flag must be scalar")
    ok = finite_device & global_ok & torch.isfinite(consensus).all()
    consensus = torch.where(ok, consensus, torch.zeros_like(consensus))
    if grad.is_cuda:
        return consensus, ok
    return consensus, bool(ok)


def consensus_gradient_batch(
    grad: torch.Tensor,
    finite_rows: torch.Tensor,
    callback: Optional[GradientConsensusFn],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Request-wise gradient consensus for a batch of independent updates.

    The TP callback may reduce the complete ``(B, P)`` tensor in one
    collective, but validity remains one predicate per request.  A legacy
    scalar-only callback is deliberately rejected rather than accidentally
    coupling the health of unrelated requests.
    """

    if grad.ndim != 2:
        raise ValueError("batched consensus gradient must have shape (B, P)")
    b = int(grad.shape[0])
    finite = torch.as_tensor(
        finite_rows, device=grad.device, dtype=torch.bool
    )
    if finite.shape != (b,):
        raise ValueError("batched consensus validity must have shape (B,)")
    safe_local = torch.where(finite[:, None], grad, torch.zeros_like(grad))
    if callback is None:
        consensus, global_ok = safe_local, finite
    else:
        consensus, global_ok = callback(safe_local, finite)
        if (
            consensus.shape != grad.shape
            or consensus.device != grad.device
            or consensus.dtype != grad.dtype
        ):
            raise ValueError(
                "batched gradient consensus must preserve shape, device and dtype"
            )
        global_ok = torch.as_tensor(
            global_ok, device=grad.device, dtype=torch.bool
        )
        if global_ok.shape != (b,):
            raise ValueError(
                "batched gradient consensus health must have one row per request"
            )
    ok = finite & global_ok & torch.isfinite(consensus).all(dim=1)
    return torch.where(ok[:, None], consensus, torch.zeros_like(consensus)), ok


@dataclass
class TeacherSignal:
    """Everything needed to evaluate the common loss / gradient at any
    adapter point, bound immutably to its source proposal version."""

    source_round: int
    source_version: int
    u: torch.Tensor  # (K, 128) projected hiddens
    m_prev: torch.Tensor  # (K, r_M)
    base_proposal_logits: torch.Tensor  # (K, V), retains online compute dtype
    base_confidence_logits: torch.Tensor  # (K,)
    target_logits: torch.Tensor  # (K, V) teacher, stop-gradient
    valid_mask: torch.Tensor  # (K,) bool
    source_proposal_logits: torch.Tensor  # (K, V) immutable source proposal
    # Optional on the real GPU path: when absent it is derived on the update
    # stream from the frozen target/source proposal logits.
    confidence_targets: Optional[torch.Tensor]  # (K,) soft targets 1 - TV
    # Per-position inverse temperature.  Adapter residuals are applied before
    # sampling temperature online, so reconstruction must scale the combined
    # raw+residual logits to represent the exact stochastic proposal q.
    proposal_logit_scale: Optional[torch.Tensor] = None  # (K,) or None
    # Full proposal-head input used by the cache-safe Tail LoRA/full-rank
    # layouts.  Output-residual keeps using the compact ``u`` projection.
    tail_hidden: Optional[torch.Tensor] = None  # (K, d)
    # DFlash proposes argmax(score) deterministically: its differentiable
    # training score is not the exact proposal measure q.  Other backends use
    # the full softmax score distribution for stochastic rejection sampling.
    proposal_distribution_kind: str = "softmax"


@dataclass
class SourceBoundCandidateBatch:
    """Compact supervision for candidates evaluated at their source point.

    DFlash/EAGLE do not train a confidence head.  Their single-step TTS and
    LightCone candidates are evaluated at exactly the proposal version that
    produced the draft, where the proximal-KL value and first derivative are
    identically zero.  Retaining raw logits solely to reconstruct that same
    point wastes one full-vocabulary tensor per request.  This batch therefore
    owns only the corrected proposal score, target score and tail sufficient
    features needed for the exact source-bound gradient.

    Tensor fields are batch-major and are normally one owned snapshot of the
    verify batch, not a Python list of per-request clones.  Source round/version
    remain request-local metadata and preserve the usual publication checks.
    """

    source_rounds: tuple[int, ...]
    source_versions: tuple[int, ...]
    u: torch.Tensor  # (B, K, 128)
    m_prev: torch.Tensor  # (B, K, r_M)
    proposal_logits: torch.Tensor  # (B, K, V), corrected source score
    target_logits: torch.Tensor  # (B, K, V), stop-gradient
    valid_mask: torch.Tensor  # (B, K) bool
    proposal_logit_scale: Optional[torch.Tensor] = None  # (B, K)
    tail_hidden: Optional[torch.Tensor] = None  # (B, K, d)
    proposal_distribution_kind: str = "softmax"

    @property
    def batch_size(self) -> int:
        return len(self.source_rounds)

    def validate(self, shapes: AdapterShapes) -> None:
        b = self.batch_size
        if b < 1 or len(self.source_versions) != b:
            raise ValueError("source-bound batch metadata has inconsistent rows")
        if self.valid_mask.ndim != 2 or self.valid_mask.dtype != torch.bool:
            raise ValueError("source-bound valid_mask must be bool (B, K)")
        if self.valid_mask.shape[0] != b:
            raise ValueError("source-bound valid_mask batch dimension mismatch")
        k = int(self.valid_mask.shape[1])
        if not 1 <= k <= shapes.draft_depth:
            raise ValueError("source-bound draft depth exceeds the layout")
        if self.proposal_logits.ndim != 3:
            raise ValueError("source-bound proposal logits must have shape (B, K, V)")
        expected_logits = (b, k, shapes.vocab_size)
        if tuple(self.proposal_logits.shape) != expected_logits:
            raise ValueError(
                "source-bound proposal logit shape mismatch: "
                f"{tuple(self.proposal_logits.shape)} != {expected_logits}"
            )
        if tuple(self.target_logits.shape) != expected_logits:
            raise ValueError("source-bound target/proposal logit shapes differ")
        if tuple(self.u.shape) != (b, k, 128):
            raise ValueError("source-bound projected hidden must have shape (B, K, 128)")
        markov_dim = shapes.markov_dim if shapes.has_markov else 0
        if tuple(self.m_prev.shape) != (b, k, markov_dim):
            raise ValueError("source-bound Markov feature shape mismatch")
        if self.proposal_logit_scale is not None and tuple(
            self.proposal_logit_scale.shape
        ) != (b, k):
            raise ValueError("source-bound proposal scale must have shape (B, K)")
        if shapes.mode != "output_residual":
            if self.tail_hidden is None or tuple(self.tail_hidden.shape) != (
                b,
                k,
                shapes.hidden_size,
            ):
                raise ValueError("source-bound tail hidden shape mismatch")
        if self.proposal_distribution_kind not in (
            "softmax",
            "deterministic_argmax",
        ):
            raise ValueError("unsupported source-bound proposal distribution")
        devices = {
            value.device
            for value in (
                self.u,
                self.m_prev,
                self.proposal_logits,
                self.target_logits,
                self.valid_mask,
                self.proposal_logit_scale,
                self.tail_hidden,
            )
            if value is not None
        }
        if len(devices) != 1:
            raise ValueError("source-bound tensors must share one device")

    def select(self, rows: Sequence[int]) -> "SourceBoundCandidateBatch":
        """Select request rows while preserving one owned batch allocation."""

        indices = tuple(int(row) for row in rows)
        if indices == tuple(range(self.batch_size)):
            return self
        if not indices:
            raise ValueError("cannot select an empty source-bound batch")
        if any(row < 0 or row >= self.batch_size for row in indices):
            raise IndexError("source-bound batch row is out of range")
        index = torch.tensor(indices, dtype=torch.int64, device=self.valid_mask.device)

        def take(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.index_select(0, index)

        return SourceBoundCandidateBatch(
            source_rounds=tuple(self.source_rounds[row] for row in indices),
            source_versions=tuple(self.source_versions[row] for row in indices),
            u=take(self.u),  # type: ignore[arg-type]
            m_prev=take(self.m_prev),  # type: ignore[arg-type]
            proposal_logits=take(self.proposal_logits),  # type: ignore[arg-type]
            target_logits=take(self.target_logits),  # type: ignore[arg-type]
            valid_mask=take(self.valid_mask),  # type: ignore[arg-type]
            proposal_logit_scale=take(self.proposal_logit_scale),
            tail_hidden=take(self.tail_hidden),
            proposal_distribution_kind=self.proposal_distribution_kind,
        )


class PublishPolicy(str, Enum):
    NONE = "none"  # static
    BLOCKING = "blocking"  # sync_fresh
    FIXED_STRIDE_BARRIER = "fixed_stride_barrier"  # tts
    ASYNC_BOUNDARY = "async_boundary"  # everything else


class DecisionKind(str, Enum):
    APPLY = "apply"
    DISCARD = "discard"
    DAMP = "damp"
    TRANSPORT = "transport"
    VERSION_CONFLICT = "version_conflict"


def _tail_forward_dtype(basis, signal: TeacherSignal) -> torch.dtype:
    """Resolve the dtype used by the real proposal-tail forward.

    Output residuals own a tensor basis; hidden tails own a lightweight
    projection wrapper around the resident LM head.  Falling back to the base
    logit dtype keeps the pure-CPU/reference path backwards compatible.
    """

    dtype = getattr(basis, "dtype", None)
    if dtype is None:
        weight = getattr(basis, "weight", None)
        dtype = getattr(weight, "dtype", None)
    if dtype is None:
        dtype = signal.base_proposal_logits.dtype
    if not dtype.is_floating_point:
        raise TypeError(f"tail forward dtype must be floating point, got {dtype}")
    return dtype


@dataclass
class CandidateUpdate:
    update_id: str
    source_round: int
    source_version: int
    raw_gradient: torch.Tensor  # clipped raw gradient g_r
    candidate_delta: torch.Tensor  # optimizer-transformed u_r (L0 delta)
    grad_norm: float | torch.Tensor
    grad_clip_scale: float | torch.Tensor
    loss: LossBreakdown
    optimizer_step: int | torch.Tensor
    signal: Optional[TeacherSignal]  # retained only when a method needs it at arrival
    phi_source: Optional[torch.Tensor] = None
    numerical_ok: bool | torch.Tensor = True
    failure_reason: Optional[str] = None
    fisher_snapshot: Optional[torch.Tensor] = None
    # CUDA events only; elapsed times are materialized by TelemetrySink after
    # the existing candidate completion event, never on the decode/side path.
    cuda_timing_ref: Optional[dict[str, object]] = None


@dataclass
class ArrivalContext:
    """Controller-visible information at the arrival graph boundary."""

    arrival_round: int
    active_version: int
    phi_active: torch.Tensor
    delay_rounds: int
    delay_tokens: int
    delay_wall_us: float
    delay_versions: int
    rho_path: float | torch.Tensor
    endpoint_distance: float | torch.Tensor
    parameter_displacement: float | torch.Tensor
    delta_z: Optional[torch.Tensor] = None  # transport state vector diff
    source_z_raw: Optional[torch.Tensor] = None
    arrival_z_raw: Optional[torch.Tensor] = None
    fresh_gradient: Optional[torch.Tensor] = None  # oracle-only
    source_prefix_len: int = 0
    source_acceptance: float | torch.Tensor = 0.0
    source_training_loss: float | torch.Tensor = 0.0
    source_grad_norm: float | torch.Tensor = 0.0


@dataclass
class Decision:
    kind: DecisionKind
    published_delta: torch.Tensor
    damping_factor: float | torch.Tensor = 1.0
    predicted_utility: Optional[float | torch.Tensor] = None
    predicted_mismatch: Optional[float | torch.Tensor] = None
    predicted_harm_probability: Optional[float | torch.Tensor] = None
    threshold: Optional[float] = None
    transport_rank: Optional[int] = None
    parameter_comp_norm: Optional[float | torch.Tensor] = None
    state_transport_norm: Optional[float | torch.Tensor] = None
    random_transport: bool = False
    gate_applied: Optional[bool | torch.Tensor] = None


def evaluate_source_bound_loss_and_grad_batch(
    forward_phi: torch.Tensor,
    signal: SourceBoundCandidateBatch,
    shapes: AdapterShapes,
    basis,
    *,
    lambda_prox: float = 0.0,
) -> tuple[list[LossBreakdown], torch.Tensor]:
    """Exact batched gradient for source-bound DFlash/EAGLE candidates.

    This is the vectorized counterpart of :func:`evaluate_loss_and_grad` at
    ``phi == phi_source``.  At that point TTS's proximal term is exactly zero,
    including its derivative, so it is represented explicitly as zero rather
    than reconstructing and retaining raw full-vocabulary logits.  The
    adapter Jacobian, native forward dtype and STE are otherwise identical to
    the row-wise path.
    """

    del lambda_prox  # source-point proximal KL and gradient are identically zero
    signal.validate(shapes)
    if shapes.has_confidence:
        raise ValueError(
            "compact source-bound batching does not support a confidence head"
        )
    b, k = signal.valid_mask.shape
    if forward_phi.shape != (b, shapes.num_params()):
        raise ValueError("source-bound forward parameter shape mismatch")
    forward_dtype = getattr(basis, "dtype", None)
    if forward_dtype is None:
        weight = getattr(basis, "weight", None)
        forward_dtype = getattr(weight, "dtype", None)
    if forward_dtype is None:
        forward_dtype = signal.proposal_logits.dtype
    if forward_phi.dtype != forward_dtype:
        raise ValueError("source-bound forward parameters have the wrong dtype")

    with torch.inference_mode(False), torch.no_grad(), torch.profiler.record_function(
        "lightcone::train_loss_batch"
    ):
        target_logits = signal.target_logits.to(torch.float32)
        proposal_logits = signal.proposal_logits.to(torch.float32)
        if signal.proposal_logit_scale is not None:
            scale = signal.proposal_logit_scale.to(torch.float32)[..., None]
            target_logits = target_logits * scale
            proposal_logits = proposal_logits * scale

        mask = signal.valid_mask
        idx = torch.arange(k, dtype=torch.float32, device=mask.device)
        weights = torch.exp(-idx / 4.0)[None, :] * mask.to(torch.float32)
        totals = weights.sum(dim=1, keepdim=True)
        if not mask.is_cuda and bool((totals <= 0).any()):
            from lightcone_spec.exit_codes import NumericalFailure

            raise NumericalFailure("empty supervision window (no valid positions)")
        weights = weights / totals.clamp_min(torch.finfo(torch.float32).tiny)

        target_log_z = torch.logsumexp(target_logits, dim=-1)
        proposal_log_z = torch.logsumexp(proposal_logits, dim=-1)
        target_prob = torch.softmax(target_logits, dim=-1)
        proposal_prob = torch.softmax(proposal_logits, dim=-1)
        # GEMV the KL dot product instead of materialising both a V-wide
        # difference and a second V-wide elementwise product.  The temporary
        # difference is released before the gradient phase.
        logit_difference = target_logits - proposal_logits
        per_position_kl = torch.bmm(
            target_prob.reshape(b * k, 1, -1),
            logit_difference.reshape(b * k, -1, 1),
        ).reshape(b, k)
        per_position_kl = per_position_kl + proposal_log_z - target_log_z
        del logit_difference
        distillation = (weights * per_position_kl).sum(dim=1)
        proximal = torch.zeros_like(distillation)
        confidence = torch.zeros_like(distillation)

        if signal.proposal_distribution_kind == "deterministic_argmax":
            conditional = target_prob.gather(
                -1, proposal_logits.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
        else:
            conditional = torch.minimum(target_prob, proposal_prob).sum(dim=-1)
        active = mask.to(torch.int64).cumprod(dim=1).to(torch.float32)
        survival = torch.cumprod(
            torch.where(active.bool(), conditional, torch.ones_like(conditional)),
            dim=1,
        )
        expected_prefix = (survival * active).sum(dim=1)
        total = distillation + confidence + proximal

        # Reuse the proposal-probability allocation as the full-vocabulary
        # logit gradient.  This avoids retaining logp/logq plus p/q for every
        # request in a high-concurrency batch.
        proposal_prob.sub_(target_prob).mul_(weights[..., None])
        if signal.proposal_logit_scale is not None:
            proposal_prob.mul_(
                signal.proposal_logit_scale.to(torch.float32)[..., None]
            )
        proposal_grad = proposal_prob
        # These full-vocabulary FP32 tensors are dead before the STE cast and
        # head backprojection. Explicit deletion lets the same-stream caching
        # allocator reuse them during a high-concurrency candidate batch.
        del target_prob, target_logits, proposal_logits

    with torch.profiler.record_function("lightcone::train_gradient_batch"):
        native_grad = proposal_grad.to(dtype=forward_dtype)
        grad = torch.zeros(
            b,
            shapes.num_params(),
            dtype=torch.float32,
            device=forward_phi.device,
        )
        slices = shapes.parameter_slices()
        if shapes.mode == "output_residual":
            basis_tensor = basis.to(
                device=forward_phi.device, dtype=forward_dtype
            )
            coordinates = native_grad @ basis_tensor
            grad[:, slices["a_h"]] = torch.bmm(
                coordinates.transpose(1, 2),
                signal.u.to(forward_dtype),
            ).reshape(b, -1).to(torch.float32)
            if shapes.has_markov:
                grad[:, slices["a_m"]] = torch.bmm(
                    coordinates.transpose(1, 2),
                    signal.m_prev.to(forward_dtype),
                ).reshape(b, -1).to(torch.float32)
        else:
            hidden = signal.tail_hidden
            assert hidden is not None
            hidden = hidden.to(forward_dtype)
            if hasattr(basis, "backproject_logits"):
                projection_grad = basis.backproject_logits(native_grad)
            else:
                projection_grad = native_grad @ basis
            if shapes.mode == "tail_lora":
                a_h = forward_phi[:, slices["a_h"]].view(
                    b, shapes.hidden_size, shapes.rank
                )
                b_h = forward_phi[:, slices["b_h"]].view(
                    b, shapes.rank, shapes.hidden_size
                )
                hidden_latent = torch.bmm(hidden, a_h)
                grad_b_h = torch.bmm(
                    hidden_latent.transpose(1, 2), projection_grad
                )
                if shapes.has_markov:
                    a_m = forward_phi[:, slices["a_m"]].view(
                        b, shapes.markov_dim, shapes.rank
                    )
                    markov_latent = torch.bmm(
                        signal.m_prev.to(forward_dtype), a_m
                    )
                    grad_b_h = grad_b_h + torch.bmm(
                        markov_latent.transpose(1, 2), projection_grad
                    )
                grad[:, slices["b_h"]] = grad_b_h.reshape(b, -1).to(
                    torch.float32
                )
                latent_grad = torch.bmm(projection_grad, b_h.transpose(1, 2))
                grad[:, slices["a_h"]] = torch.bmm(
                    hidden.transpose(1, 2), latent_grad
                ).reshape(b, -1).to(torch.float32)
                if shapes.has_markov:
                    grad[:, slices["a_m"]] = torch.bmm(
                        signal.m_prev.to(forward_dtype).transpose(1, 2),
                        latent_grad,
                    ).reshape(b, -1).to(torch.float32)
            else:
                grad[:, slices["d_h"]] = torch.bmm(
                    hidden.transpose(1, 2), projection_grad
                ).reshape(b, -1).to(torch.float32)
                if shapes.has_markov:
                    grad[:, slices["d_m"]] = torch.bmm(
                        signal.m_prev.to(forward_dtype).transpose(1, 2),
                        projection_grad,
                    ).reshape(b, -1).to(torch.float32)

    breakdowns = [
        LossBreakdown(
            total=total[row].detach(),
            distillation=distillation[row].detach(),
            confidence=confidence[row].detach(),
            proximal=proximal[row].detach(),
            expected_accepted_prefix=expected_prefix[row].detach(),
        )
        for row in range(b)
    ]
    return breakdowns, grad


def evaluate_loss_and_grad(
    phi: torch.Tensor,
    signal: TeacherSignal,
    shapes: AdapterShapes,
    basis: torch.Tensor,
    confidence_loss_weight: float = 1.0,
    lambda_prox: float = 0.0,
    need_grad: bool = True,
    forward_phi: Optional[torch.Tensor] = None,
) -> tuple[LossBreakdown, Optional[torch.Tensor]]:
    """Evaluate the common loss (optionally + prox term) and its gradient
    at adapter point phi, reconstructing proposal logits as base +
    tail correction.  The gradient is assembled analytically from the
    full-vocabulary logit gradient, so the update stream never constructs an
    inference-backbone autograd graph."""
    with torch.inference_mode(False), torch.no_grad():
        (
            q_logits,
            conf_logits,
            target_logits,
            valid_mask,
            source_proposal_logits,
            confidence_targets,
            basis,
            u,
            m_prev,
            proposal_scale,
            tail_hidden,
            confidence_features,
            views,
            forward_dtype,
        ) = _reconstruct_online_outputs(
            phi,
            signal,
            shapes,
            basis,
            forward_phi=forward_phi,
        )
        effective_confidence_weight = (
            confidence_loss_weight if shapes.has_confidence else 0.0
        )
        with torch.profiler.record_function("lightcone::train_loss"):
            if need_grad:
                breakdown, proposal_grad, confidence_grad = (
                    common_loss_with_logit_grads(
                        target_logits=target_logits,
                        proposal_logits=q_logits,
                        confidence_logits=conf_logits,
                        confidence_targets=confidence_targets,
                        valid_mask=valid_mask,
                        confidence_loss_weight=effective_confidence_weight,
                        source_proposal_logits=source_proposal_logits,
                        lambda_prox=lambda_prox,
                        proposal_distribution_kind=(
                            signal.proposal_distribution_kind
                        ),
                        max_draft_depth=shapes.draft_depth,
                    )
                )
            else:
                breakdown = common_loss(
                    target_logits=target_logits,
                    proposal_logits=q_logits,
                    confidence_logits=conf_logits,
                    confidence_targets=confidence_targets,
                    valid_mask=valid_mask,
                    confidence_loss_weight=effective_confidence_weight,
                    source_proposal_logits=source_proposal_logits,
                    lambda_prox=lambda_prox,
                    proposal_distribution_kind=(
                        signal.proposal_distribution_kind
                    ),
                    max_draft_depth=shapes.draft_depth,
                )
        grad = None
        if need_grad:
            with torch.profiler.record_function("lightcone::train_gradient"):
                if proposal_scale is not None:
                    proposal_grad = proposal_grad * proposal_scale
                # Match autograd through the online model-dtype residual cast.
                # Without this quantization a BF16 serving source produces a
                # different adapter gradient than the reconstructed forward.
                native_proposal_grad = proposal_grad.to(dtype=forward_dtype)
                grad = torch.zeros_like(phi, dtype=torch.float32)
                slices = shapes.parameter_slices()
                if shapes.mode == "output_residual":
                    # Preserve the exact forward orientation.  The serving
                    # path projects coordinates as ``C @ B.T``; its STE
                    # therefore backprojects as ``dL @ B``.  The algebraically
                    # equivalent transposed GEMM can round differently in
                    # BF16 and would break source-bound candidate parity.
                    basis_coordinates = native_proposal_grad @ basis
                    grad[slices["a_h"]] = (
                        torch.bmm(
                            basis_coordinates.unsqueeze(0).transpose(1, 2),
                            u.unsqueeze(0),
                        )
                        .squeeze(0)
                    ).reshape(-1).to(torch.float32)
                    if shapes.has_markov:
                        grad[slices["a_m"]] = (
                            torch.bmm(
                                basis_coordinates.unsqueeze(0).transpose(1, 2),
                                m_prev.unsqueeze(0),
                            )
                            .squeeze(0)
                        ).reshape(-1).to(torch.float32)
                else:
                    if tail_hidden is None:
                        raise ValueError(
                            f"{shapes.mode} requires TeacherSignal.tail_hidden"
                        )
                    # W remains in its model-compute dtype.  Casting the tiny
                    # KxV gradient is cheaper than materialising an FP32 copy
                    # of the vocabulary head on every request.
                    if hasattr(basis, "backproject_logits"):
                        projection_grad = basis.backproject_logits(
                            native_proposal_grad
                        )
                    else:
                        projection_grad = native_proposal_grad @ basis
                    if shapes.mode == "tail_lora":
                        hidden_latent = torch.bmm(
                            tail_hidden.unsqueeze(0),
                            views["a_h"].unsqueeze(0),
                        ).squeeze(0)
                        grad_b_h = torch.bmm(
                            hidden_latent.unsqueeze(0).transpose(1, 2),
                            projection_grad.unsqueeze(0),
                        ).squeeze(0)
                        if shapes.has_markov:
                            markov_latent = torch.bmm(
                                m_prev.unsqueeze(0),
                                views["a_m"].unsqueeze(0),
                            ).squeeze(0)
                            # Online applies the hidden and Markov residuals as
                            # two native-dtype matmuls/adds.  Preserve that
                            # order in the STE backward; combining latents
                            # first changes BF16 rounding and breaks the
                            # PyTorch reference gradient.
                            grad_b_h = grad_b_h + torch.bmm(
                                markov_latent.unsqueeze(0).transpose(1, 2),
                                projection_grad.unsqueeze(0),
                            ).squeeze(0)
                        grad[slices["b_h"]] = (
                            grad_b_h.reshape(-1).to(torch.float32)
                        )
                        latent_grad = torch.bmm(
                            projection_grad.unsqueeze(0),
                            views["b_h"].T.unsqueeze(0),
                        ).squeeze(0)
                        grad[slices["a_h"]] = (
                            torch.bmm(
                                tail_hidden.unsqueeze(0).transpose(1, 2),
                                latent_grad.unsqueeze(0),
                            )
                            .squeeze(0)
                        ).reshape(-1).to(torch.float32)
                        if shapes.has_markov:
                            grad[slices["a_m"]] = (
                                torch.bmm(
                                    m_prev.unsqueeze(0).transpose(1, 2),
                                    latent_grad.unsqueeze(0),
                                )
                                .squeeze(0)
                            ).reshape(-1).to(torch.float32)
                    else:
                        grad[slices["d_h"]] = (
                            torch.bmm(
                                tail_hidden.unsqueeze(0).transpose(1, 2),
                                projection_grad.unsqueeze(0),
                            )
                            .squeeze(0)
                        ).reshape(-1).to(torch.float32)
                        if shapes.has_markov:
                            grad[slices["d_m"]] = (
                                torch.bmm(
                                    m_prev.unsqueeze(0).transpose(1, 2),
                                    projection_grad.unsqueeze(0),
                                )
                                .squeeze(0)
                            ).reshape(-1).to(torch.float32)
                if shapes.has_confidence:
                    k = u.shape[0]
                    native_confidence_grad = confidence_grad.to(
                        dtype=forward_dtype
                    )
                    grad_a_c = torch.zeros(
                        shapes.draft_depth,
                        shapes.conf_feature_dim,
                        dtype=torch.float32,
                        device=u.device,
                    )
                    grad_a_c[:k] = (
                        native_confidence_grad[:, None]
                        * confidence_features
                    ).to(torch.float32)
                    grad[slices["a_c"]] = grad_a_c.reshape(-1)
        # Candidate updates keep the four scalar values for telemetry only.
        # Detaching here releases the temporary AdapterParams autograd graph
        # before a side-stream candidate can remain pending across rounds.
        breakdown = LossBreakdown(
            total=breakdown.total.detach(),
            distillation=breakdown.distillation.detach(),
            confidence=breakdown.confidence.detach(),
            proximal=breakdown.proximal.detach(),
            expected_accepted_prefix=(
                breakdown.expected_accepted_prefix.detach()
            ),
        )
        return breakdown, grad


def _reconstruct_online_outputs(
    phi: torch.Tensor,
    signal: TeacherSignal,
    shapes: AdapterShapes,
    basis: torch.Tensor,
    *,
    forward_phi: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    """Rebuild the exact native-dtype online outputs at ``phi``.

    ``phi`` is the canonical FP32 Q-DQ master. ``forward_phi`` may name the
    fixed model-dtype source scratch populated by the runtime; CPU/reference
    callers omit it and obtain the same quantized row with one local cast.
    Gradients use an explicit straight-through contract around this cast.
    """

    def regular(value: torch.Tensor) -> torch.Tensor:
        return value.clone() if torch.is_inference(value) else value

    forward_dtype = _tail_forward_dtype(basis, signal)
    if forward_phi is None:
        forward_phi = phi.to(dtype=forward_dtype)
    elif forward_phi.dtype != forward_dtype:
        raise ValueError(
            f"forward parameter dtype {forward_phi.dtype} != {forward_dtype}"
        )
    if forward_phi.shape != phi.shape or forward_phi.device != phi.device:
        raise ValueError("forward/master parameter sources must share shape/device")

    # Output residuals and hidden tails both execute in the proposal head's
    # model-compute dtype.  The projection wrapper keeps the multi-GB head
    # resident and sharded rather than constructing an FP32 duplicate.
    if shapes.mode == "output_residual":
        basis = basis.to(device=phi.device, dtype=forward_dtype)
    elif not hasattr(basis, "project_hidden"):
        basis = basis.to(device=phi.device, dtype=forward_dtype)
    u = regular(signal.u).to(device=phi.device, dtype=forward_dtype)
    m_prev = regular(signal.m_prev).to(
        device=phi.device, dtype=forward_dtype
    )
    base_proposal_logits = regular(signal.base_proposal_logits)
    base_confidence_logits = regular(
        signal.base_confidence_logits.to(torch.float32)
    )
    target_logits = regular(signal.target_logits).to(torch.float32)
    valid_mask = regular(signal.valid_mask)
    source_proposal_logits = regular(signal.source_proposal_logits)
    confidence_targets = (
        None
        if signal.confidence_targets is None
        else regular(signal.confidence_targets)
    )

    views = parameter_views(regular(forward_phi), shapes)
    tail_hidden = None
    proposal_residuals: list[torch.Tensor] = []
    if shapes.mode == "output_residual":
        with torch.profiler.record_function("lightcone::train_A_d"):
            coordinates = torch.bmm(
                u.unsqueeze(0),
                views["a_h"].T.unsqueeze(0),
            ).squeeze(0)
            proposal_residuals.append(coordinates @ basis.T)
        if shapes.has_markov:
            with torch.profiler.record_function("lightcone::train_A_m"):
                markov_coordinates = torch.bmm(
                    m_prev.unsqueeze(0),
                    views["a_m"].T.unsqueeze(0),
                ).squeeze(0)
                proposal_residuals.append(markov_coordinates @ basis.T)
    else:
        if signal.tail_hidden is None:
            raise ValueError(f"{shapes.mode} requires TeacherSignal.tail_hidden")
        tail_hidden = regular(signal.tail_hidden).to(
            device=phi.device, dtype=forward_dtype
        )
        if tail_hidden.shape != (u.shape[0], shapes.hidden_size):
            raise ValueError(
                "tail_hidden shape "
                f"{tuple(tail_hidden.shape)} != ({u.shape[0]}, {shapes.hidden_size})"
            )
        with torch.profiler.record_function("lightcone::train_tail_hidden"):
            if shapes.mode == "tail_lora":
                hidden_latent = torch.bmm(
                    tail_hidden.unsqueeze(0),
                    views["a_h"].unsqueeze(0),
                )
                hidden_delta = torch.bmm(
                    hidden_latent,
                    views["b_h"].unsqueeze(0),
                ).squeeze(0)
                markov_delta = (
                    torch.bmm(
                        torch.bmm(
                            m_prev.unsqueeze(0),
                            views["a_m"].unsqueeze(0),
                        ),
                        views["b_h"].unsqueeze(0),
                    ).squeeze(0)
                    if shapes.has_markov
                    else None
                )
            else:
                hidden_delta = torch.bmm(
                    tail_hidden.unsqueeze(0),
                    views["d_h"].unsqueeze(0),
                ).squeeze(0)
                markov_delta = (
                    torch.bmm(
                        m_prev.unsqueeze(0),
                        views["d_m"].unsqueeze(0),
                    ).squeeze(0)
                    if shapes.has_markov
                    else None
                )

            def project(delta_hidden: torch.Tensor) -> torch.Tensor:
                if hasattr(basis, "project_hidden"):
                    return basis.project_hidden(delta_hidden).to(torch.float32)
                return (
                    delta_hidden.to(dtype=basis.dtype) @ basis.T
                ).to(torch.float32)

            proposal_residuals.append(project(hidden_delta))
            if markov_delta is not None:
                proposal_residuals.append(project(markov_delta))
    # Reproduce online arithmetic exactly: proposal residuals are cast to the
    # model's proposal-head dtype before the add, then the rounded scores are
    # promoted for the stable loss/softmax.  Promoting the base first makes a
    # BF16 source point spuriously move and breaks TTS/L0 candidate parity.
    q_logits_native = base_proposal_logits
    for proposal_residual in proposal_residuals:
        q_logits_native = q_logits_native + proposal_residual.to(
            dtype=base_proposal_logits.dtype
        )
    q_logits = q_logits_native.to(torch.float32)
    proposal_scale = None
    if signal.proposal_logit_scale is not None:
        proposal_scale = regular(signal.proposal_logit_scale).to(
            device=q_logits.device, dtype=torch.float32
        ).view(-1, 1)
        q_logits = q_logits * proposal_scale
        # The target verifier samples from the same temperature-scaled
        # distribution.  Scaling q alone would train against the wrong p.
        target_logits = target_logits * proposal_scale
    k = u.shape[0]
    if shapes.mode == "output_residual":
        confidence_parts = [u]
        if shapes.has_markov:
            confidence_parts.append(rmsnorm(m_prev))
    else:
        assert tail_hidden is not None
        confidence_parts = [tail_hidden]
        if shapes.has_markov:
            confidence_parts.append(m_prev)
    confidence_parts.append(
        torch.ones(k, 1, dtype=forward_dtype, device=u.device)
    )
    confidence_features = torch.cat(confidence_parts, dim=1)
    if shapes.has_confidence:
        with torch.profiler.record_function("lightcone::train_A_c"):
            confidence_residual = (
                views["a_c"][:k] * confidence_features
            ).sum(dim=1)
    else:
        confidence_residual = torch.zeros_like(base_confidence_logits)
    conf_logits = base_confidence_logits + confidence_residual.to(torch.float32)
    return (
        q_logits,
        conf_logits,
        target_logits,
        valid_mask,
        source_proposal_logits,
        confidence_targets,
        basis,
        u,
        m_prev,
        proposal_scale,
        tail_hidden,
        confidence_features,
        views,
        forward_dtype,
    )


def survival_weighted_acceptance(
    phi: torch.Tensor,
    signal: TeacherSignal,
    shapes: AdapterShapes,
    basis: torch.Tensor,
    *,
    greedy: bool = False,
) -> torch.Tensor:
    """Expected accepted draft tokens on the observed proposal path.

    For stochastic sampling, each conditional acceptance is ``1-TV(p, q)``
    and the sum of its cumulative products is the expected accepted prefix.
    Greedy verification instead uses exact argmax agreement.  Invalid suffix
    positions terminate the prefix rather than inflating an acceptance rate.
    """
    with torch.inference_mode(False), torch.no_grad():
        q_logits, _, target_logits, valid_mask, *_ = (
            _reconstruct_online_outputs(phi, signal, shapes, basis)
        )
        if greedy:
            conditional = (
                target_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)
            ).to(torch.float32)
        elif signal.proposal_distribution_kind == "deterministic_argmax":
            p = torch.softmax(target_logits, dim=-1)
            conditional = p.gather(
                -1, q_logits.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
        else:
            p = torch.softmax(target_logits, dim=-1)
            q = torch.softmax(q_logits, dim=-1)
            conditional = (1.0 - 0.5 * (p - q).abs().sum(dim=-1)).clamp(
                0.0, 1.0
            )
        active = valid_mask.to(torch.int64).cumprod(dim=0).to(torch.float32)
        survival = torch.cumprod(
            torch.where(active.bool(), conditional, torch.ones_like(conditional)),
            dim=0,
        )
        return (survival * active).sum()


@dataclass
class CandidateGeneratorConfig:
    lr: float
    grad_clip: float
    trust_region_radius: float
    confidence_loss_weight: float
    lambda_prox: float
    weight_decay: float = 0.0


class CommonCandidateGenerator:
    """The shared single-step AdamW TTS candidate generator used by L0-L3,
    oracle_current and the diagnostic controls (spec 6.8): one shared
    request-local AdamW state so optimizer and controller effects never
    confound."""

    def __init__(
        self,
        shapes: AdapterShapes,
        basis: torch.Tensor,
        cfg: CandidateGeneratorConfig,
    ):
        self.shapes = shapes
        self.basis = basis
        self.cfg = cfg
        self.state = AdamWDeltaState(num_params=shapes.num_params())
        self.preview_state: Optional[AdamWDeltaState] = None
        self.gradient_consensus_fn: Optional[GradientConsensusFn] = None
        self._counter = 0

    def bind_gradient_consensus(
        self, callback: Optional[GradientConsensusFn]
    ) -> None:
        self.gradient_consensus_fn = callback

    def bind_preview_state(
        self, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor
    ) -> None:
        self.preview_state = AdamWDeltaState(
            num_params=self.shapes.num_params(),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
        )

    def prepare_preview_state(self) -> None:
        if self.preview_state is None:
            return
        self.preview_state.exp_avg.copy_(self.state.exp_avg)
        self.preview_state.exp_avg_sq.copy_(self.state.exp_avg_sq)
        if isinstance(self.state.step, torch.Tensor):
            if not isinstance(self.preview_state.step, torch.Tensor):
                self.preview_state.step = torch.zeros_like(self.state.step)
            self.preview_state.step.copy_(self.state.step)
        else:
            self.preview_state.step = self.state.step

    def raw_gradient(
        self,
        phi_source: torch.Tensor,
        signal: TeacherSignal,
        forward_phi_source: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        float | torch.Tensor,
        float | torch.Tensor,
        LossBreakdown,
        bool | torch.Tensor,
    ]:
        breakdown, grad = evaluate_loss_and_grad(
            phi_source,
            signal,
            self.shapes,
            self.basis,
            confidence_loss_weight=self.cfg.confidence_loss_weight,
            lambda_prox=self.cfg.lambda_prox,
            forward_phi=forward_phi_source,
        )
        assert grad is not None
        finite_t = torch.isfinite(grad).all() & torch.isfinite(breakdown.total)
        grad, numerical_ok = consensus_gradient(
            grad, finite_t, self.gradient_consensus_fn
        )
        if not grad.is_cuda:
            if not numerical_ok:
                from lightcone_spec.exit_codes import NumericalFailure

                raise NumericalFailure("non-finite candidate loss or gradient")
        norm_t = torch.linalg.vector_norm(grad)
        norm = norm_t if grad.is_cuda else float(norm_t)
        clipped, scale = clip_gradient_global_norm(grad, self.cfg.grad_clip)
        return clipped, norm, scale, breakdown, numerical_ok

    def candidate(
        self,
        phi_source: torch.Tensor,
        signal: TeacherSignal,
        defer_state_advance: bool = False,
        forward_phi_source: Optional[torch.Tensor] = None,
        cuda_timing_ref: Optional[dict[str, object]] = None,
    ) -> CandidateUpdate:
        """Exactly one optimizer step per trigger (spec 6.3).

        defer_state_advance=True (L3 only): the returned candidate_delta is
        the u_r preview computed in fixed scratch state; the shared state is
        advanced later, exactly once, when the transported gradient is fed
        through `delta_from_transported_gradient`. With a zero transport
        correction this reproduces u_r bitwise (d=0 parity, spec 15.5).
        """
        self._counter += 1
        cuda_timing: dict[str, object] | None = None
        if phi_source.is_cuda:
            if cuda_timing_ref is None:
                raise ValueError(
                    "CUDA candidates require an externally leased timing bundle"
                )
            required = {
                "backward_start",
                "backward_end",
                "optimizer_end",
                "optimizer_step_out",
            }
            missing = required.difference(cuda_timing_ref)
            if missing:
                raise ValueError(
                    "CUDA candidate timing bundle is missing "
                    + ", ".join(sorted(missing))
                )
            cuda_timing = cuda_timing_ref
            backward_start = cuda_timing["backward_start"]
            backward_start.record(torch.cuda.current_stream(phi_source.device))
        clipped, norm, scale, breakdown, numerical_ok = self.raw_gradient(
            phi_source,
            signal,
            forward_phi_source=forward_phi_source,
        )
        if cuda_timing is not None:
            backward_end = cuda_timing["backward_end"]
            backward_end.record(torch.cuda.current_stream(phi_source.device))
        if defer_state_advance:
            preview = self.preview_state or self.state.clone()
            delta = adamw_delta(
                clipped,
                preview,
                self.cfg.lr,
                valid=numerical_ok,
                parameter=phi_source,
                weight_decay=self.cfg.weight_decay,
            )
            step_source = preview.step
        else:
            delta = adamw_delta(
                clipped,
                self.state,
                self.cfg.lr,
                valid=numerical_ok,
                parameter=phi_source,
                weight_decay=self.cfg.weight_decay,
            )
            step_source = self.state.step
        if cuda_timing is not None:
            optimizer_end = cuda_timing["optimizer_end"]
            optimizer_end.record(torch.cuda.current_stream(phi_source.device))
            step = cuda_timing["optimizer_step_out"]
            if not isinstance(step, torch.Tensor) or not step.is_cuda:
                raise TypeError("optimizer_step_out must be a CUDA tensor scalar")
            if step.numel() != 1 or step.dtype != torch.int64:
                raise TypeError("optimizer_step_out must be one int64 scalar")
            step.copy_(step_source)
        else:
            step = step_source
        return CandidateUpdate(
            update_id=f"u{signal.source_round}-{self._counter}",
            source_round=signal.source_round,
            source_version=signal.source_version,
            raw_gradient=clipped,
            candidate_delta=delta,
            grad_norm=norm,
            grad_clip_scale=scale,
            loss=breakdown,
            optimizer_step=step,
            signal=signal,
            phi_source=phi_source,
            numerical_ok=numerical_ok,
            cuda_timing_ref=cuda_timing,
        )

    @staticmethod
    def candidate_batch(
        generators: Sequence["CommonCandidateGenerator"],
        phi_sources: torch.Tensor,
        forward_phi_sources: torch.Tensor,
        signal: SourceBoundCandidateBatch,
        *,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        steps: torch.Tensor,
        defer_state_advance: bool,
        cuda_timing_refs: Optional[Sequence[dict[str, object]]] = None,
    ) -> list[CandidateUpdate]:
        """Build one independent candidate per row with batched GPU kernels."""

        b = signal.batch_size
        if len(generators) != b:
            raise ValueError("candidate generator and signal batch sizes differ")
        if phi_sources.shape != (b, generators[0].shapes.num_params()):
            raise ValueError("candidate source parameter batch shape mismatch")
        first = generators[0]
        for generator in generators:
            if generator.shapes != first.shapes or generator.cfg != first.cfg:
                raise ValueError(
                    "batched candidates require one layout and optimizer config"
                )
            if generator.basis is not first.basis:
                raise ValueError("batched candidates require one shared projection")
            if generator.gradient_consensus_fn is not first.gradient_consensus_fn:
                raise ValueError("batched candidates require one TP consensus hook")
        if exp_avg.shape != phi_sources.shape or exp_avg_sq.shape != phi_sources.shape:
            raise ValueError("batched optimizer moment shape mismatch")
        if steps.shape != (b,) or steps.dtype != torch.int64:
            raise ValueError("batched optimizer step shape mismatch")

        timing_refs = list(cuda_timing_refs or ())
        if phi_sources.is_cuda:
            if len(timing_refs) != b:
                raise ValueError("CUDA candidate batch requires one timing lane per row")
            for timing in timing_refs:
                missing = {
                    "backward_start",
                    "backward_end",
                    "optimizer_end",
                    "optimizer_step_out",
                }.difference(timing)
                if missing:
                    raise ValueError(
                        "CUDA candidate timing bundle is missing "
                        + ", ".join(sorted(missing))
                    )
                timing["candidate_batch_size"] = b
                timing["backward_start"].record(
                    torch.cuda.current_stream(phi_sources.device)
                )
        elif timing_refs:
            raise ValueError("CPU candidate batch cannot consume CUDA timing lanes")

        breakdowns, raw_grad = evaluate_source_bound_loss_and_grad_batch(
            forward_phi_sources,
            signal,
            first.shapes,
            first.basis,
            lambda_prox=first.cfg.lambda_prox,
        )
        finite = torch.isfinite(raw_grad).all(dim=1) & torch.stack(
            [torch.isfinite(item.total) for item in breakdowns]
        )
        consensus, numerical_ok = consensus_gradient_batch(
            raw_grad,
            finite,
            first.gradient_consensus_fn,
        )
        if not consensus.is_cuda and not bool(numerical_ok.all()):
            from lightcone_spec.exit_codes import NumericalFailure

            raise NumericalFailure("non-finite batched candidate loss or gradient")
        norms = torch.linalg.vector_norm(consensus, dim=1)
        scales = torch.clamp(
            torch.as_tensor(
                first.cfg.grad_clip,
                device=norms.device,
                dtype=norms.dtype,
            )
            / norms.clamp_min(torch.finfo(norms.dtype).tiny),
            max=1.0,
        )
        clipped = consensus.to(torch.float32) * scales[:, None]
        if phi_sources.is_cuda:
            for timing in timing_refs:
                timing["backward_end"].record(
                    torch.cuda.current_stream(phi_sources.device)
                )

        # For L3, ``exp_avg``/``exp_avg_sq``/``steps`` are caller-owned
        # previews.  For TTS/L0/L1/L2 they are gathered request-local states
        # which the runtime scatters back after this call.
        delta = adamw_delta_batched(
            clipped,
            exp_avg,
            exp_avg_sq,
            steps,
            first.cfg.lr,
            numerical_ok,
            parameter=phi_sources,
            weight_decay=first.cfg.weight_decay,
        )
        if phi_sources.is_cuda:
            for row, timing in enumerate(timing_refs):
                timing["optimizer_end"].record(
                    torch.cuda.current_stream(phi_sources.device)
                )
                step_out = timing["optimizer_step_out"]
                if (
                    not isinstance(step_out, torch.Tensor)
                    or not step_out.is_cuda
                    or step_out.numel() != 1
                    or step_out.dtype != torch.int64
                ):
                    raise TypeError("optimizer_step_out must be one CUDA int64 scalar")
                step_out.copy_(steps[row])

        out: list[CandidateUpdate] = []
        for row, generator in enumerate(generators):
            generator._counter += 1
            out.append(
                CandidateUpdate(
                    update_id=(
                        f"u{signal.source_rounds[row]}-{generator._counter}"
                    ),
                    source_round=signal.source_rounds[row],
                    source_version=signal.source_versions[row],
                    raw_gradient=clipped[row],
                    candidate_delta=delta[row],
                    grad_norm=norms[row] if clipped.is_cuda else float(norms[row]),
                    grad_clip_scale=(
                        scales[row] if clipped.is_cuda else float(scales[row])
                    ),
                    loss=breakdowns[row],
                    optimizer_step=(
                        timing_refs[row]["optimizer_step_out"]
                        if clipped.is_cuda
                        else int(steps[row])
                    ),
                    signal=None,
                    phi_source=phi_sources[row],
                    numerical_ok=(
                        numerical_ok[row]
                        if clipped.is_cuda
                        else bool(numerical_ok[row])
                    ),
                    cuda_timing_ref=(timing_refs[row] if clipped.is_cuda else None),
                )
            )
        return out

    def delta_from_transported_gradient(
        self,
        transported_grad: torch.Tensor,
        parameter: torch.Tensor | None = None,
        valid: bool | torch.Tensor = True,
    ) -> torch.Tensor:
        """L3: apply the same AdamW transform, on the shared request-local
        state, to the transported gradient. This is the single optimizer
        step the deferred candidate consumes."""
        return adamw_delta(
            transported_grad,
            self.state,
            self.cfg.lr,
            valid=valid,
            parameter=parameter,
            weight_decay=self.cfg.weight_decay,
        )


def apply_delta_with_trust_region(
    phi_active: torch.Tensor,
    delta: torch.Tensor,
    phi0: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    return trust_region_project(phi_active + delta, phi0, radius)


class MethodRuntime:
    """Abstract per-request method runtime.

    The engine calls `make_candidate` on the side stream at trigger rounds
    (with the immutable source snapshot) and `decide` at the arrival graph
    boundary. Method-private state must be request/stream-local and never
    shared across tenants.
    """

    key: str = "abstract"
    publish_policy: PublishPolicy = PublishPolicy.ASYNC_BOUNDARY
    needs_fresh_gradient_at_arrival: bool = False
    needs_delta_z: bool = False
    retain_source_signal: bool = False

    def bind_slot_state(
        self,
        exp_avg: Optional[torch.Tensor],
        exp_avg_sq: Optional[torch.Tensor],
        fisher: Optional[torch.Tensor] = None,
    ) -> None:
        """Bind optional fixed-address state; stateless methods ignore it."""
        del exp_avg, exp_avg_sq, fisher

    def bind_candidate_preview(
        self,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
    ) -> None:
        del exp_avg, exp_avg_sq

    def bind_gradient_consensus(
        self, callback: Optional[GradientConsensusFn]
    ) -> None:
        """Bind an optional TP all-reduce/finite-AND hook."""
        del callback

    def prepare_candidate_preview(self) -> None:
        """Snapshot optimizer state before queued L3 candidate work."""

    def common_candidate_generator(self) -> Optional[CommonCandidateGenerator]:
        """Return the common AdamW generator when source batching is legal."""

        return None

    def after_batched_candidate(self, candidate: CandidateUpdate) -> None:
        """Update method-private state after a shared batched candidate."""

        del candidate

    def make_candidate(
        self,
        phi_source: torch.Tensor,
        signal: TeacherSignal,
        forward_phi_source: Optional[torch.Tensor] = None,
        cuda_timing_ref: Optional[dict[str, object]] = None,
    ) -> Optional[CandidateUpdate]:
        raise NotImplementedError

    def decide(self, candidate: CandidateUpdate, ctx: ArrivalContext) -> Decision:
        raise NotImplementedError

    def telemetry(self) -> dict:
        return {}
