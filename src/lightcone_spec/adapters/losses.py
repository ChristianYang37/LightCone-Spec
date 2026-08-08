"""Common training signal shared by every method (spec 5.2, 6.1, 6.2).

- Position weights w_k = exp(-(k-1)/4), renormalized over valid positions.
- Distillation: sum_k wbar_k KL(p_k || q_phi_k) on FP32 log-softmax.
- Confidence: BCE-with-logits against soft targets c_k = 1 - TV(p_k, q_k),
  supervised on the full window (never accepted-only).
- TTS adds a proximal KL(q_source || q_phi) term with weight lambda.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lightcone_spec.adapters.adapter_params import MAX_SUPPORTED_DRAFT_DEPTH
from lightcone_spec.exit_codes import NumericalFailure


def position_weights(
    valid_mask: torch.Tensor,
    *,
    max_draft_depth: int = MAX_SUPPORTED_DRAFT_DEPTH,
) -> torch.Tensor:
    """Return normalized weights for a declared speculative window.

    ``max_draft_depth`` is normally the locked model pair's depth propagated
    through :class:`AdapterShapes`.  The default is only for standalone loss
    utilities and remains deliberately bounded by the largest backend depth
    currently supported by this runtime.
    """
    if valid_mask.ndim != 1:
        raise ValueError("valid_mask must have shape (K,)")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must have boolean dtype")
    if isinstance(max_draft_depth, bool) or not isinstance(max_draft_depth, int):
        raise TypeError("max_draft_depth must be an integer")
    if not 1 <= max_draft_depth <= MAX_SUPPORTED_DRAFT_DEPTH:
        raise ValueError(
            f"declared draft depth {max_draft_depth} is outside the supported "
            f"range [1, {MAX_SUPPORTED_DRAFT_DEPTH}]"
        )
    k = valid_mask.shape[0]
    if k > max_draft_depth:
        raise ValueError(
            f"draft depth {k} exceeds declared depth {max_draft_depth}"
        )
    idx = torch.arange(k, dtype=torch.float32, device=valid_mask.device)
    w = torch.exp(-idx / 4.0) * valid_mask.to(torch.float32)
    total = w.sum()
    if not valid_mask.is_cuda and total <= 0:
        raise NumericalFailure("empty supervision window (no valid positions)")
    # A CUDA-empty window becomes a non-finite candidate and is rejected at
    # the legal publish boundary, without synchronizing the update stream.
    return w / total.clamp_min(torch.finfo(w.dtype).tiny)


def _log_softmax_fp32(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.to(torch.float32), dim=-1)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, FP32. p side is treated as stop-gradient."""
    logp = _log_softmax_fp32(p_logits).detach()
    logq = _log_softmax_fp32(q_logits)
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1)


def total_variation(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = _log_softmax_fp32(p_logits).exp()
    q = _log_softmax_fp32(q_logits).exp()
    return 0.5 * (p - q).abs().sum(dim=-1)


def confidence_soft_targets(
    p_logits: torch.Tensor, q_logits: torch.Tensor
) -> torch.Tensor:
    """c_k = 1 - TV(p_k, q_k) = sum_x min(p, q); computed from the frozen
    teacher and the (source-bound) proposal, detached."""
    with torch.no_grad():
        p = _log_softmax_fp32(p_logits).exp()
        q = _log_softmax_fp32(q_logits).exp()
        return torch.minimum(p, q).sum(dim=-1)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    distillation: torch.Tensor
    confidence: torch.Tensor
    proximal: torch.Tensor
    expected_accepted_prefix: torch.Tensor


def _common_loss_impl(
    target_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    confidence_logits: torch.Tensor,
    confidence_targets: torch.Tensor | None,
    valid_mask: torch.Tensor,
    confidence_loss_weight: float = 1.0,
    source_proposal_logits: torch.Tensor | None = None,
    lambda_prox: float = 0.0,
    proposal_distribution_kind: str = "softmax",
    max_draft_depth: int = MAX_SUPPORTED_DRAFT_DEPTH,
    *,
    need_logit_grads: bool,
) -> tuple[LossBreakdown, torch.Tensor | None, torch.Tensor | None]:
    """L = L_dist + w_conf * L_conf (+ lambda * prox KL for TTS).

    target_logits, proposal_logits: (K, V); confidence_logits: (K,);
    confidence_targets: (K,) soft targets in [0, 1]; valid_mask: (K,).
    Teacher logits missing -> the caller must fail closed before here.
    """
    if target_logits is None:
        raise NumericalFailure("teacher logits missing: update must fail closed")
    k = valid_mask.shape[0] if valid_mask.ndim == 1 else -1
    if target_logits.ndim != 2 or proposal_logits.shape != target_logits.shape:
        raise ValueError(
            "target_logits and proposal_logits must share shape (K, V)"
        )
    if target_logits.shape[0] != k:
        raise ValueError("logit rows must match valid_mask length")
    if confidence_logits.shape != (k,):
        raise ValueError("confidence_logits must have shape (K,)")
    if confidence_targets is not None and confidence_targets.shape != (k,):
        raise ValueError("confidence_targets must have shape (K,)")
    if (
        source_proposal_logits is not None
        and source_proposal_logits.shape != target_logits.shape
    ):
        raise ValueError("source_proposal_logits must have shape (K, V)")
    wbar = position_weights(valid_mask, max_draft_depth=max_draft_depth)
    if proposal_distribution_kind not in ("softmax", "deterministic_argmax"):
        raise ValueError(
            "proposal_distribution_kind must be softmax or "
            "deterministic_argmax"
        )
    logp = _log_softmax_fp32(target_logits).detach()
    logq = _log_softmax_fp32(proposal_logits)
    p = logp.exp()
    q = logq.exp()
    l_dist = (wbar * (p * (logp - logq)).sum(dim=-1)).sum()
    source_logq = None
    source_q = None
    if source_proposal_logits is not None and (
        confidence_targets is None or lambda_prox > 0.0
    ):
        source_logq = _log_softmax_fp32(source_proposal_logits).detach()
        source_q = source_logq.exp()
    if confidence_targets is None:
        if source_q is None:
            raise NumericalFailure(
                "lazy confidence target requires frozen source proposal logits"
            )
        if proposal_distribution_kind == "deterministic_argmax":
            confidence_targets = p.gather(
                -1, source_proposal_logits.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
        else:
            confidence_targets = torch.minimum(p, source_q).sum(dim=-1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_logits.to(torch.float32),
        confidence_targets.to(torch.float32).detach(),
        reduction="none",
    )
    l_conf = confidence_loss_weight * (wbar * bce).sum()
    if source_proposal_logits is not None and lambda_prox > 0.0:
        assert source_logq is not None and source_q is not None
        l_prox = lambda_prox * (
            wbar * (source_q * (source_logq - logq)).sum(dim=-1)
        ).sum()
    else:
        l_prox = torch.zeros(
            (), dtype=torch.float32, device=proposal_logits.device
        )
    total = l_dist + l_conf + l_prox
    if not total.is_cuda and not torch.isfinite(total):
        raise NumericalFailure("non-finite loss: update fails and is discarded")
    if proposal_distribution_kind == "deterministic_argmax":
        conditional_acceptance = p.gather(
            -1, proposal_logits.argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
    else:
        conditional_acceptance = torch.minimum(p, q).sum(dim=-1)
    active_prefix = valid_mask.to(torch.int64).cumprod(dim=0).to(torch.float32)
    survival = torch.cumprod(
        torch.where(
            active_prefix.bool(),
            conditional_acceptance,
            torch.ones_like(conditional_acceptance),
        ),
        dim=0,
    )
    breakdown = LossBreakdown(
        total=total,
        distillation=l_dist,
        confidence=l_conf,
        proximal=l_prox,
        expected_accepted_prefix=(survival * active_prefix).sum(),
    )
    if not need_logit_grads:
        return breakdown, None, None
    proposal_grad = wbar[:, None] * (q - p)
    if source_q is not None and lambda_prox > 0.0:
        proposal_grad = proposal_grad + (
            lambda_prox * wbar[:, None] * (q - source_q)
        )
    confidence_grad = (
        confidence_loss_weight
        * wbar
        * (torch.sigmoid(confidence_logits.to(torch.float32)) - confidence_targets)
    )
    return breakdown, proposal_grad, confidence_grad


def common_loss(
    target_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    confidence_logits: torch.Tensor,
    confidence_targets: torch.Tensor | None,
    valid_mask: torch.Tensor,
    confidence_loss_weight: float = 1.0,
    source_proposal_logits: torch.Tensor | None = None,
    lambda_prox: float = 0.0,
    proposal_distribution_kind: str = "softmax",
    max_draft_depth: int = MAX_SUPPORTED_DRAFT_DEPTH,
) -> LossBreakdown:
    breakdown, _, _ = _common_loss_impl(
        target_logits,
        proposal_logits,
        confidence_logits,
        confidence_targets,
        valid_mask,
        confidence_loss_weight,
        source_proposal_logits,
        lambda_prox,
        proposal_distribution_kind,
        max_draft_depth,
        need_logit_grads=False,
    )
    return breakdown


def common_loss_with_logit_grads(
    target_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    confidence_logits: torch.Tensor,
    confidence_targets: torch.Tensor | None,
    valid_mask: torch.Tensor,
    confidence_loss_weight: float = 1.0,
    source_proposal_logits: torch.Tensor | None = None,
    lambda_prox: float = 0.0,
    proposal_distribution_kind: str = "softmax",
    max_draft_depth: int = MAX_SUPPORTED_DRAFT_DEPTH,
) -> tuple[LossBreakdown, torch.Tensor, torch.Tensor]:
    """Return the common loss and its closed-form logit gradients.

    The online adapters are linear in proposal and confidence logits, so this
    avoids constructing a full-vocabulary autograd graph on the side stream.
    """
    breakdown, proposal_grad, confidence_grad = _common_loss_impl(
        target_logits,
        proposal_logits,
        confidence_logits,
        confidence_targets,
        valid_mask,
        confidence_loss_weight,
        source_proposal_logits,
        lambda_prox,
        proposal_distribution_kind,
        max_draft_depth,
        need_logit_grads=True,
    )
    assert proposal_grad is not None and confidence_grad is not None
    return breakdown, proposal_grad, confidence_grad
