"""Gradient-square EMA F_r = 0.99 F_{r-1} + 0.01 g_r^2 (spec 7.9)."""

from __future__ import annotations

import torch


class FisherEMA:
    def __init__(self, num_params: int, decay: float = 0.99):
        self.decay = decay
        self.value = torch.zeros(num_params, dtype=torch.float32)
        self.steps = 0

    def bind(self, value: torch.Tensor) -> None:
        if tuple(value.shape) != tuple(self.value.shape):
            raise ValueError("Fisher slot state shape mismatch")
        self.value = value
        self.steps = (
            torch.zeros((), dtype=torch.int64, device=value.device)
            if value.is_cuda
            else 0
        )

    def update(
        self, grad: torch.Tensor, valid: bool | torch.Tensor = True
    ) -> None:
        g = grad.to(torch.float32)
        if self.value.device != g.device:
            self.value = self.value.to(g.device)
        candidate = self.value * self.decay + g.square() * (1.0 - self.decay)
        if g.is_cuda:
            valid_t = torch.as_tensor(valid, device=g.device, dtype=torch.bool)
            self.value.copy_(torch.where(valid_t, candidate, self.value))
            if not isinstance(self.steps, torch.Tensor):
                self.steps = torch.zeros((), dtype=torch.int64, device=g.device)
            self.steps.copy_(torch.where(valid_t, self.steps + 1, self.steps))
        else:
            if not bool(valid):
                return
            self.value.copy_(candidate)
            self.steps += 1

    def parameter_compensation(self, s_phi: torch.Tensor) -> torch.Tensor:
        """diag(F_r) s_r^phi where s_r^phi = phi_{c(r)} - phi_r."""
        if self.value.device != s_phi.device:
            self.value = self.value.to(s_phi.device)
        return self.value * s_phi.to(torch.float32)
