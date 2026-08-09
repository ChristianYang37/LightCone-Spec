"""DFlash-owned parameter selection and merge-only LoRA state."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

_BORROWED_COMPONENTS = (
    "embed_tokens",
    "lm_head",
    "target_model",
    "target_embedding",
)
_LORA_LINEAR = re.compile(
    r"(?:^|\.)(?:fc|q_proj|k_proj|v_proj|qkv_proj|o_proj|"
    r"gate_proj|gate_up_proj|up_proj|down_proj)(?:\.|$)"
)


def _is_owned(name: str) -> bool:
    components = frozenset(name.split("."))
    return components.isdisjoint(_BORROWED_COMPONENTS)


@dataclass(frozen=True)
class ParameterEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class DFlashParameterPlan:
    mode: str
    scope: str
    rank: int | None
    entries: tuple[ParameterEntry, ...]

    @classmethod
    def build(
        cls,
        named_parameters: Mapping[str, Tensor] | Iterable[tuple[str, Tensor]],
        *,
        mode: str,
        scope: str,
        rank: int | None = None,
        tail_names: Iterable[str] = (),
    ) -> DFlashParameterPlan:
        if mode not in {"residual", "lora", "full"}:
            raise ValueError(f"unknown update mode {mode!r}")
        if scope not in {"tail", "drafter"}:
            raise ValueError(f"unknown parameter scope {scope!r}")
        if mode == "residual" and scope != "tail":
            raise ValueError("residual is a tail-only parameterization")
        if mode == "full" and rank is not None:
            raise ValueError("full parameterization requires rank=null")
        if mode in {"lora", "residual"} and (
            rank is None or isinstance(rank, bool) or rank < 1
        ):
            raise ValueError(f"{mode} parameterization requires a positive rank")
        items = (
            tuple(named_parameters.items())
            if isinstance(named_parameters, Mapping)
            else tuple(named_parameters)
        )
        requested_tail = frozenset(tail_names)
        if scope == "tail" and mode != "residual" and not requested_tail:
            raise ValueError("tail LoRA/full requires an explicit parameter allowlist")
        selected: list[ParameterEntry] = []
        for name, parameter in items:
            if not _is_owned(name) or not parameter.is_floating_point():
                continue
            if scope == "tail" and name not in requested_tail:
                continue
            if mode == "lora" and (
                parameter.ndim != 2 or _LORA_LINEAR.search(name) is None
            ):
                continue
            if mode == "residual":
                continue
            selected.append(
                ParameterEntry(
                    name=name,
                    shape=tuple(parameter.shape),
                    dtype=str(parameter.dtype),
                )
            )
        if mode != "residual" and not selected:
            raise ValueError("parameter selection is empty")
        return cls(
            mode=mode, scope=scope, rank=rank, entries=tuple(selected)
        )

    @property
    def sha256(self) -> str:
        body = {
            "mode": self.mode,
            "scope": self.scope,
            "rank": self.rank,
            "entries": [
                {
                    "name": entry.name,
                    "shape": entry.shape,
                    "dtype": entry.dtype,
                }
                for entry in self.entries
            ],
        }
        encoded = json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def trainable_parameter_count(self) -> int:
        count = 0
        for entry in self.entries:
            if self.mode == "lora":
                if self.rank is None or len(entry.shape) != 2:
                    raise AssertionError("LoRA layout is not a ranked matrix")
                count += self.rank * (entry.shape[0] + entry.shape[1])
                continue
            size = 1
            for dimension in entry.shape:
                size *= dimension
            count += size
        return count


@dataclass
class LoRAFactors:
    """Two trainable factors; inference consumes only their merged weight."""

    a: Tensor
    b: Tensor
    rank: int

    @classmethod
    def initialize(
        cls,
        weight: Tensor,
        rank: int,
        *,
        seed: int,
    ) -> LoRAFactors:
        if weight.ndim != 2:
            raise ValueError("LoRA requires a matrix")
        if rank < 1 or rank > min(weight.shape):
            raise ValueError("invalid LoRA rank")
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(seed)
        a = torch.empty(
            (rank, weight.shape[1]),
            device=weight.device,
            dtype=torch.float32,
        )
        torch.nn.init.kaiming_uniform_(a, a=5**0.5, generator=generator)
        b = torch.zeros(
            (weight.shape[0], rank),
            device=weight.device,
            dtype=torch.float32,
        )
        return cls(a=a, b=b, rank=rank)

    def merged(self, base: Tensor) -> Tensor:
        if base.shape != (self.b.shape[0], self.a.shape[1]):
            raise ValueError("LoRA factors do not match the base matrix")
        return base.to(torch.float32) + self.b @ self.a
