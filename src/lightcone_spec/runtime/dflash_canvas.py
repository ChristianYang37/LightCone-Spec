"""Differentiable current-canvas contract for DFlash update rounds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from lightcone_spec.exit_codes import ExactnessError


@dataclass(frozen=True)
class CanvasReconstruction:
    raw_logits: Tensor
    differentiable_logits: Tensor
    historical_key: Tensor
    historical_value: Tensor
    current_key: Tensor
    current_value: Tensor

    def validate_history_contract(self) -> None:
        if self.historical_key.requires_grad or self.historical_value.requires_grad:
            raise ExactnessError("historical KV must be detached")
        if not self.current_key.requires_grad or not self.current_value.requires_grad:
            raise ExactnessError("current-canvas KV must remain differentiable")


class DifferentiableCanvasContract:
    """Build and verify an update-only differentiable canvas."""

    def __init__(
        self,
        inference_forward: Callable[..., Tensor],
        differentiable_forward: Callable[..., CanvasReconstruction],
        *,
        atol: float = 2e-3,
        rtol: float = 2e-3,
    ) -> None:
        self.inference_forward = inference_forward
        self.differentiable_forward = differentiable_forward
        self.atol = atol
        self.rtol = rtol

    def reconstruct(self, *args: object, **kwargs: object) -> CanvasReconstruction:
        with torch.no_grad():
            raw = self.inference_forward(*args, **kwargs).detach()
        reconstruction = self.differentiable_forward(*args, **kwargs)
        reconstruction.validate_history_contract()
        if raw.shape != reconstruction.differentiable_logits.shape:
            raise ExactnessError("differentiable logits shape mismatch")
        if not torch.allclose(
            raw,
            reconstruction.differentiable_logits.detach(),
            atol=self.atol,
            rtol=self.rtol,
        ):
            raise ExactnessError(
                "differentiable DFlash canvas does not reconstruct inference logits"
            )
        return CanvasReconstruction(
            raw_logits=raw,
            differentiable_logits=reconstruction.differentiable_logits,
            historical_key=reconstruction.historical_key,
            historical_value=reconstruction.historical_value,
            current_key=reconstruction.current_key,
            current_value=reconstruction.current_value,
        )


def rms_norm(hidden: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    variance = hidden.to(torch.float32).square().mean(dim=-1, keepdim=True)
    normalized = hidden * torch.rsqrt(variance + epsilon).to(hidden.dtype)
    return normalized * weight


def scaled_dot_product_canvas(
    query: Tensor,
    historical_key: Tensor,
    historical_value: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    key = torch.cat((historical_key.detach(), current_key), dim=-2)
    value = torch.cat((historical_value.detach(), current_value), dim=-2)
    return functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        # DFlash predicts the canvas as one block; its backend attention over
        # that block is deliberately non-causal.
        is_causal=False,
    )


def position_weighted_kl(
    target_logits: Tensor,
    draft_logits: Tensor,
    valid_mask: Tensor,
    *,
    decay: float,
) -> Tensor:
    """Target-to-draft KL, normalized over valid positions and requests."""
    if target_logits.shape != draft_logits.shape:
        raise ValueError("target and draft logits must have equal shape")
    if valid_mask.shape != target_logits.shape[:-1]:
        raise ValueError("valid_mask must index logit positions")
    positions = torch.arange(
        target_logits.shape[-2],
        device=target_logits.device,
        dtype=torch.float32,
    )
    weights = decay**positions
    while weights.ndim < valid_mask.ndim:
        weights = weights.unsqueeze(0)
    weights = weights * valid_mask.to(torch.float32)
    target_probability = torch.softmax(target_logits.to(torch.float32), dim=-1)
    target_log = torch.log_softmax(target_logits.to(torch.float32), dim=-1)
    draft_log = torch.log_softmax(draft_logits.to(torch.float32), dim=-1)
    per_position = (
        target_probability * (target_log - draft_log)
    ).sum(dim=-1)
    return (per_position * weights).sum() / weights.sum().clamp_min(1.0)
