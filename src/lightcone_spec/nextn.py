"""Small NEXTN replay primitives shared by the runner and patched SGLang."""

from __future__ import annotations

import math
import types
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_TP_PARTIAL_REPLICATED = frozenset(
    {
        "layers.0.q_norm.weight",
        "layers.0.k_norm.weight",
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.shared_expert_gate.weight",
    }
)


def needs_tp_gradient_sum(name: str) -> bool:
    return name in _TP_PARTIAL_REPLICATED


@contextmanager
def grad_enabled_forwards(model: torch.nn.Module):
    """Temporarily bypass model-local no-grad decorators during shadow replay."""
    restored = []
    for module in model.modules():
        forward = module.forward
        raw = getattr(forward, "__wrapped__", None)
        if raw is None:
            continue
        restored.append((module, forward))
        module.forward = types.MethodType(raw, module)
    try:
        yield
    finally:
        for module, forward in reversed(restored):
            module.forward = forward


def gradient_leaves(parameters: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Expose resident optimizer values as differentiable replay inputs."""
    with torch.inference_mode(False), torch.enable_grad():
        return tuple(value.detach().requires_grad_(True) for value in parameters)


def anchor_replay_logits(
    inference_logits: torch.Tensor, replay_logits: torch.Tensor
) -> torch.Tensor:
    """Use inference values with the differentiable replay Jacobian."""
    return inference_logits.detach() + (replay_logits - replay_logits.detach())


def flatten_attention_output(
    attended: torch.Tensor, q_size: int
) -> torch.Tensor:
    """Flatten query heads independently of the model hidden size."""
    return attended.reshape(attended.shape[0], q_size)


def ste_block_fp8_activation(
    hidden: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    """Match dynamic block-FP8 activation values with an identity backward."""
    rows = hidden.reshape(-1, hidden.shape[-1])
    padding = (-rows.shape[-1]) % group_size
    padded = F.pad(rows.float(), (0, padding))
    groups = padded.reshape(padded.shape[0], -1, group_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    quantized = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale).reshape(padded.shape)
    dequantized = dequantized[:, : rows.shape[-1]].reshape(hidden.shape).to(hidden.dtype)
    return hidden + (dequantized - hidden).detach()


def torch_native_shared_expert_add(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    shared: torch.Tensor,
    routed: torch.Tensor,
) -> None:
    """Differentiable equivalent of Qwen's fused shared-expert add."""
    gate = F.linear(hidden, gate_weight.unsqueeze(0))
    routed.add_(torch.sigmoid(gate) * shared)


@dataclass
class RequestRow:
    rid: str
    proposal_slot: int
    verify_slot: int | None
    prefix: int
    draft_start: int
    draft_end: int
    teacher_start: int | None = None
    teacher_end: int | None = None
    terminal: str | None = None


class RequestLedger:
    """RID-indexed joins for one speculative round."""

    def __init__(self) -> None:
        self.rows: dict[str, RequestRow] = {}
        self.order: tuple[str, ...] = ()

    def begin(
        self,
        request_ids: Sequence[str],
        prefix_lens: Sequence[int],
        draft_offsets: Sequence[int],
    ) -> bool:
        ids = tuple(request_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(prefix_lens) != len(ids)
            or len(draft_offsets) != len(ids) + 1
        ):
            return False
        self.order = ids
        self.rows = {
            rid: RequestRow(
                rid=rid,
                proposal_slot=index,
                verify_slot=None,
                prefix=int(prefix_lens[index]),
                draft_start=int(draft_offsets[index]),
                draft_end=int(draft_offsets[index + 1]),
            )
            for index, rid in enumerate(ids)
        }
        return True

    def bind_verify(
        self,
        request_ids: Sequence[str],
        teacher_starts: Sequence[int],
        teacher_ends: Sequence[int] | None = None,
    ) -> bool:
        ids = tuple(request_ids)
        starts = tuple(int(value) for value in teacher_starts)
        ends = (
            tuple(int(value) for value in teacher_ends)
            if teacher_ends is not None
            else starts[1:]
        )
        if teacher_ends is None:
            starts = starts[:-1]
        if (
            len(set(ids)) != len(ids)
            or set(ids) != set(self.order)
            or len(starts) != len(ids)
            or len(ends) != len(ids)
        ):
            return False
        for index, rid in enumerate(ids):
            row = self.rows[rid]
            row.verify_slot = index
            row.teacher_start = starts[index]
            row.teacher_end = ends[index]
        return True

    def join_accept_lens(
        self, request_ids: Sequence[str], accept_lens: Sequence[int]
    ) -> tuple[int, ...] | None:
        ids = tuple(request_ids)
        if (
            len(ids) != len(accept_lens)
            or len(set(ids)) != len(ids)
            or set(ids) != set(self.order)
        ):
            return None
        by_rid = dict(zip(ids, (int(value) for value in accept_lens), strict=True))
        return tuple(by_rid[rid] for rid in self.order)

    def terminal(self, rid: str, state: str) -> None:
        row = self.rows.get(rid)
        if row is not None:
            row.terminal = state


def torch_native_moe(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Differentiable top-k MoE over only the experts selected by this block."""
    probabilities = F.softmax(F.linear(hidden.float(), router_weight.float()), dim=-1)
    top_weights, top_ids = torch.topk(probabilities, top_k, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    return torch_native_selected_moe(
        hidden, top_weights, top_ids, w13_weight, w2_weight
    )


def _selected_weight(
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    selected = weight[expert_ids].float()
    if scale is None or not weight.dtype.is_floating_point or weight.dtype not in {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    }:
        return selected
    selected_scale = scale[expert_ids].float()
    while selected_scale.ndim < selected.ndim:
        selected_scale = selected_scale.unsqueeze(-1)
    for dimension in (-2, -1):
        repeats = math.ceil(selected.shape[dimension] / selected_scale.shape[dimension])
        selected_scale = selected_scale.repeat_interleave(repeats, dim=dimension)
    return selected * selected_scale[..., : selected.shape[-2], : selected.shape[-1]]


def torch_native_selected_moe(
    hidden: torch.Tensor,
    top_weights: torch.Tensor,
    top_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable current-block MoE using only routed experts."""
    if w13_weight.dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
        hidden = ste_block_fp8_activation(hidden)
    selected_w13 = _selected_weight(w13_weight, top_ids, w13_scale)
    gate, up = torch.chunk(
        torch.einsum("th,teih->tei", hidden.float(), selected_w13), 2, dim=-1
    )
    activated = F.silu(gate) * up
    if w2_weight.dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
        activated = ste_block_fp8_activation(activated)
    selected_w2 = _selected_weight(w2_weight, top_ids, w2_scale)
    expert = torch.einsum("tei,tehi->teh", activated, selected_w2)
    return torch.einsum("teh,te->th", expert, top_weights.float()).to(hidden.dtype)


def torch_native_ragged_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    history_valid: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Differentiable one-token attention over detached ragged KV history."""
    key = torch.cat((history_key, current_key[:, None]), dim=1)
    value = torch.cat((history_value, current_value[:, None]), dim=1)
    repeat = query.shape[1] // current_key.shape[1]
    key = key.repeat_interleave(repeat, dim=2).transpose(1, 2)
    value = value.repeat_interleave(repeat, dim=2).transpose(1, 2)
    visible = torch.cat(
        (
            history_valid,
            torch.ones(
                (query.shape[0], 1), device=query.device, dtype=torch.bool
            ),
        ),
        dim=1,
    )
    return F.scaled_dot_product_attention(
        query[:, :, None, :],
        key,
        value,
        attn_mask=visible[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )[:, :, 0, :]


def ragged_history_locations(
    indptr: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each draft row's KV prefix, excluding its current write slot."""
    starts = indptr[:-1].to(torch.int64)
    lengths = (indptr[1:].to(torch.int64) - starts - 1).clamp_min(0)
    width = max(1, int(lengths.max().item()))
    offsets = torch.arange(width, device=indices.device, dtype=torch.int64)[None, :]
    valid = offsets < lengths[:, None]
    flat = starts[:, None] + offsets
    locations = indices[flat.clamp(max=indices.numel() - 1)]
    return locations.masked_fill(~valid, 0), valid


@dataclass(frozen=True)
class PublicationSlot:
    name: str
    live_weight: torch.Tensor
    base_master: torch.Tensor
    parameter_indices: tuple[int, ...]
    lora_scale: float = 1.0
    live_scale: torch.Tensor | None = None
    scale_name: str | None = None
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None


class MergedPublicationBank:
    """Merge candidates once, then copy weight and scale at publication."""

    def __init__(self, slots: Sequence[PublicationSlot]) -> None:
        self.slots = tuple(slots)
        if not self.slots:
            raise ValueError("publication bank cannot be empty")
        self.weight_staging = tuple(
            torch.empty_like(slot.live_weight) for slot in self.slots
        )
        self.scale_staging = tuple(
            torch.empty_like(slot.live_scale) if slot.live_scale is not None else None
            for slot in self.slots
        )
        self.staging = self.weight_staging + tuple(
            value for value in self.scale_staging if value is not None
        )
        self.addresses = tuple(
            value.data_ptr()
            for slot in self.slots
            for value in (slot.live_weight, slot.live_scale)
            if value is not None
        )

    def stage(self, parameters: Sequence[torch.Tensor]) -> None:
        with torch.no_grad():
            for slot, weight_staging, scale_staging in zip(
                self.slots,
                self.weight_staging,
                self.scale_staging,
                strict=True,
            ):
                if len(slot.parameter_indices) == 1:
                    merged = parameters[slot.parameter_indices[0]]
                else:
                    a, b = (parameters[index] for index in slot.parameter_indices)
                    merged = slot.base_master + slot.lora_scale * (b @ a)
                if slot.quantize is None:
                    weight_staging.copy_(merged.to(weight_staging.dtype))
                else:
                    quantized, scale = slot.quantize(merged)
                    weight_staging.copy_(quantized)
                    if scale_staging is None:
                        raise RuntimeError("quantized publication has no scale tensor")
                    scale_staging.copy_(scale)

    def publish(self, *, valid: torch.Tensor | None = None) -> None:
        publish = valid is None or bool(valid.reshape(()).item())
        if publish:
            with torch.no_grad():
                for slot, weight, scale in zip(
                    self.slots,
                    self.weight_staging,
                    self.scale_staging,
                    strict=True,
                ):
                    slot.live_weight.copy_(weight)
                    if scale is not None:
                        slot.live_scale.copy_(scale)
        addresses = tuple(
            value.data_ptr()
            for slot in self.slots
            for value in (slot.live_weight, slot.live_scale)
            if value is not None
        )
        if addresses != self.addresses:
            raise RuntimeError("published NEXTN tensor address changed")

    def matches(self, parameters: Sequence[torch.Tensor]) -> bool:
        with torch.no_grad():
            for slot in self.slots:
                if len(slot.parameter_indices) == 1:
                    merged = parameters[slot.parameter_indices[0]]
                else:
                    a, b = (parameters[index] for index in slot.parameter_indices)
                    merged = slot.base_master + slot.lora_scale * (b @ a)
                if slot.quantize is None:
                    if not torch.equal(slot.live_weight, merged.to(slot.live_weight.dtype)):
                        return False
                else:
                    weight, scale = slot.quantize(merged)
                    if not torch.equal(slot.live_weight, weight) or not torch.equal(
                        slot.live_scale, scale
                    ):
                        return False
        return True


def ragged_kl_loss(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    ledger: RequestLedger,
) -> torch.Tensor:
    with torch.inference_mode(False), torch.enable_grad():
        if draft_logits.ndim == 3:
            draft_logits = draft_logits.flatten(0, 1)
        pairs = []
        for rid in ledger.order:
            row = ledger.rows[rid]
            if row.teacher_start is None or row.teacher_end is None:
                continue
            draft = draft_logits[row.draft_start : row.draft_end]
            teacher = teacher_logits[row.teacher_start : row.teacher_end]
            count = min(draft.shape[0], teacher.shape[0])
            if count:
                pairs.append((draft[:count], teacher[:count]))
        if not pairs:
            raise ValueError("NEXTN replay has no joined teacher rows")
        draft = torch.cat([pair[0] for pair in pairs])
        teacher = torch.cat([pair[1] for pair in pairs])
        return F.kl_div(
            F.log_softmax(draft.float(), dim=-1),
            F.softmax(teacher.detach().float(), dim=-1),
            reduction="batchmean",
        )
