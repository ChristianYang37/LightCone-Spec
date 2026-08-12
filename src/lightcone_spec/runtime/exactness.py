"""Exact speculative rejection sampling from the proposal actually used."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_probability(name: str, value: Tensor) -> None:
    if not value.is_floating_point() or value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} must be a non-empty floating distribution")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite probability")
    if not bool((value >= 0).all()):
        raise ValueError(f"{name} contains a negative probability")
    mass = value.sum(dim=-1)
    tolerance = 32 * torch.finfo(value.dtype).eps
    if not bool(
        torch.allclose(mass, torch.ones_like(mass), atol=tolerance, rtol=tolerance)
    ):
        raise ValueError(f"{name} rows must sum to one")


def rejection_sample(
    target_probability: Tensor,
    proposal_probability: Tensor,
    proposal_token: Tensor,
    uniform: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return sampled tokens and acceptance flags for one proposal position.

    The rejected branch samples from normalized positive(target - proposal),
    preserving the target distribution. The proposal probability must be the
    exact distribution that generated the proposal token.
    """
    if target_probability.shape != proposal_probability.shape:
        raise ValueError("target and proposal distributions must match")
    if target_probability.device != proposal_probability.device:
        raise ValueError("target and proposal distributions must share a device")
    _validate_probability("target_probability", target_probability)
    _validate_probability("proposal_probability", proposal_probability)
    if proposal_token.shape != target_probability.shape[:-1]:
        raise ValueError("proposal_token has an incompatible shape")
    if proposal_token.device != target_probability.device:
        raise ValueError("proposal_token must share the distribution device")
    if proposal_token.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("proposal_token must use an integer dtype")
    if not bool(
        ((proposal_token >= 0) & (proposal_token < target_probability.shape[-1])).all()
    ):
        raise ValueError("proposal_token is outside the vocabulary")
    if uniform.shape != proposal_token.shape:
        raise ValueError("one coupled uniform is required per token")
    if uniform.device != target_probability.device or not uniform.is_floating_point():
        raise ValueError("uniforms must be floating values on the distribution device")
    if not bool(torch.isfinite(uniform).all()) or not bool(
        ((uniform >= 0) & (uniform < 1)).all()
    ):
        raise ValueError("uniforms must be finite values in [0, 1)")
    gathered_target = target_probability.gather(
        -1, proposal_token.unsqueeze(-1)
    ).squeeze(-1)
    gathered_proposal = proposal_probability.gather(
        -1, proposal_token.unsqueeze(-1)
    ).squeeze(-1)
    if not bool((gathered_proposal > 0).all()):
        raise ValueError(
            "proposal token must have positive mass under the recorded proposal"
        )
    threshold = torch.minimum(
        torch.ones_like(gathered_target),
        gathered_target
        / gathered_proposal.clamp_min(torch.finfo(gathered_proposal.dtype).tiny),
    )
    accepted = uniform < threshold
    residual = torch.relu(target_probability - proposal_probability)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    rejection_probability = torch.where(
        residual_mass > 0,
        residual / residual_mass.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probability,
    )
    replacement = torch.multinomial(
        rejection_probability.reshape(-1, rejection_probability.shape[-1]),
        num_samples=1,
        generator=generator,
    ).reshape(proposal_token.shape)
    return torch.where(accepted, proposal_token, replacement), accepted


def greedy_exact(target_logits: Tensor) -> Tensor:
    return torch.argmax(target_logits, dim=-1)
