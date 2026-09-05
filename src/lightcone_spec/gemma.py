"""Differentiable Gemma4 draft mathematics, shared by inference and replay.

These helpers do not reinterpret Gemma as Qwen and do not allocate optimizer
state. KV replication/sharding belongs to the SGLang linear-layer wrapper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gemma_rms(
    value: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(-1, keepdim=True) + epsilon
    )
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized.to(value.dtype)


def gemma_softcap(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    if cap is None:
        return logits
    if cap <= 0:
        raise ValueError("Gemma logit softcap must be positive")
    return torch.tanh(logits / cap) * cap


def gemma_rotary(value: torch.Tensor, positions: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """NeoX partial RoPE; preserve the non-rotary suffix exactly."""
    cosine, sine = cache[positions.long()].chunk(2, dim=-1)
    cosine, sine = cosine.unsqueeze(-2), sine.unsqueeze(-2)
    half = cosine.shape[-1]
    if 2 * half > value.shape[-1]:
        raise ValueError("rotary cache exceeds head dimension")
    left, right = value[..., :half], value[..., half : 2 * half]
    rotary = torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)
    return torch.cat((rotary.to(value.dtype), value[..., 2 * half :]), dim=-1)


def gemma_canvas_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    history_valid: torch.Tensor,
    *,
    causal_canvas: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    """Batched attention; Gemma uses one, Qwen passes head_dim**-0.5.

    Historical KV are constants; current K/V remain connected to one shared K
    projection when attention_k_eq_v is true. Padding history never participates.
    """
    if query.ndim != 4 or key.shape != value.shape or key.shape[:2] != query.shape[:2]:
        raise ValueError("Gemma canvas Q/K/V shapes are inconsistent")
    if query.shape[2] % key.shape[2]:
        raise ValueError("Gemma query heads must be a multiple of KV heads")
    if history_valid.shape != history_key.shape[:2] or history_value.shape != history_key.shape:
        raise ValueError("Gemma history mask does not match cached KV")
    repeated = query.shape[2] // key.shape[2]
    all_key = torch.cat((history_key.detach(), key), dim=1).repeat_interleave(repeated, dim=2)
    all_value = torch.cat((history_value.detach(), value), dim=1).repeat_interleave(repeated, dim=2)
    count = query.shape[1]
    canvas = torch.ones((count, count), dtype=torch.bool, device=query.device)
    if causal_canvas:
        canvas = canvas.tril()
    mask = torch.cat(
        (
            history_valid[:, None, :].expand(-1, count, -1),
            canvas[None].expand(query.shape[0], -1, -1),
        ),
        dim=-1,
    )
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        all_key.transpose(1, 2),
        all_value.transpose(1, 2),
        attn_mask=mask[:, None],
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return output.transpose(1, 2).flatten(-2)


def gemma_residual_mlp(
    residual: torch.Tensor,
    attention_output: torch.Tensor,
    post_attention_weight: torch.Tensor,
    pre_feedforward_weight: torch.Tensor,
    post_feedforward_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Unfused four-norm Gemma residual ordering; no Qwen fused-add rounding."""
    hidden = residual + gemma_rms(attention_output, post_attention_weight, epsilon)
    inputs = gemma_rms(hidden, pre_feedforward_weight, epsilon)
    gate, up = F.linear(inputs, gate_up_weight).chunk(2, dim=-1)
    activated = F.gelu(gate, approximate="tanh") * up
    mlp_output = F.linear(activated, down_weight)
    return (hidden + gemma_rms(mlp_output, post_feedforward_weight, epsilon)) * layer_scalar
