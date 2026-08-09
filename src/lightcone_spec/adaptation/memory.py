"""Explicit adaptation memory accounting performed before KV-pool sizing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from torch import Tensor


def tensor_bytes(tensors: Iterable[Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


@dataclass(frozen=True)
class AdaptationMemoryLedger:
    active_base: int = 0
    master_fp32: int = 0
    gradients: int = 0
    first_moments: int = 0
    second_moments: int = 0
    staging: int = 0
    training_activations: int = 0
    kv_gather_scratch: int = 0
    candidate_scratch: int = 0
    graph_buffers: int = 0
    telemetry: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("memory categories cannot be negative")

    @property
    def resident_bytes(self) -> int:
        return (
            self.active_base
            + self.master_fp32
            + self.first_moments
            + self.second_moments
            + self.staging
            + self.graph_buffers
            + self.telemetry
        )

    @property
    def peak_bytes(self) -> int:
        return (
            self.resident_bytes
            + self.gradients
            + self.training_activations
            + self.kv_gather_scratch
            + self.candidate_scratch
        )

    def kv_budget(self, available_bytes: int, reserve_bytes: int = 0) -> int:
        if available_bytes < 0 or reserve_bytes < 0:
            raise ValueError("memory budgets cannot be negative")
        remaining = available_bytes - reserve_bytes - self.peak_bytes
        if remaining < 0:
            raise MemoryError(
                "adaptation reserve exceeds available HBM; no silent offload "
                "or mode downgrade is allowed"
            )
        return remaining
