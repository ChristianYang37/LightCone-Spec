"""Logical-rank LoRA primitives for native Qwen/Gemma tensor parallel replay."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient):
        result = gradient.contiguous()
        dist.all_reduce(result, group=ctx.group)
        return result, None


class LowRankDelta(torch.nn.Module):
    """Column A / row B are shared; replicated KV is reduced by the weight op."""

    def __init__(
        self,
        weight,
        *,
        rank,
        seed,
        partition="replicated",
        tp_rank=0,
        tp_size=1,
        group=None,
        scale=1.0,
    ):
        super().__init__()
        if weight.ndim != 2 or rank > min(weight.shape):
            raise ValueError("native LoRA rank exceeds its backend matrix")
        if partition not in {"column", "row", "replicated"}:
            raise ValueError("unknown native LoRA TP partition")
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError("invalid TP rank/size")
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(seed)
        # Generate the global A even for row shards so TP1 and TP2 start from
        # the same logical adapter. Its r x H allocation is transient.
        width = weight.shape[1] * (tp_size if partition == "row" else 1)
        initial = torch.empty(rank, width, device=weight.device, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=generator)
        if partition == "row":
            initial = initial.chunk(tp_size, dim=1)[tp_rank].contiguous()
        self.a = torch.nn.Parameter(initial)
        self.b = torch.nn.Parameter(
            torch.zeros(weight.shape[0], rank, device=weight.device, dtype=torch.float32)
        )
        self.partition, self.tp_size, self.group, self.scale = partition, tp_size, group, scale

    def forward(self, original):
        a, b = self.a, self.b
        if self.tp_size > 1:
            if self.partition == "column":
                a = _Copy.apply(a, self.group)
            elif self.partition == "row":
                b = _Copy.apply(b, self.group)
        return original + (self.scale * (b @ a)).to(original.dtype)


def global_gradient_norm(gradients, replicated, group, tp_size):
    """Shared gradients are already summed. De-duplicate only their norm cost."""
    squares = torch.stack(
        [
            gradient.detach().float().square().sum() / (tp_size if shared else 1)
            for gradient, shared in zip(gradients, replicated, strict=True)
        ]
    ).sum()
    dist.all_reduce(squares, group=group)
    return squares.sqrt()


def strided_teacher_rows(logits, batch_size, gamma, microbatch):
    """Verifier returns gamma+bonus rows per RID even for a compact input."""
    if logits.ndim != 2 or logits.shape[0] != batch_size * (gamma + 1):
        raise ValueError("native verifier teacher rows do not match RID stride")
    if not 0 < microbatch <= batch_size:
        raise ValueError("invalid teacher microbatch")
    return logits.reshape(batch_size, gamma + 1, -1)[:microbatch, :gamma]
