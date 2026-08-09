"""GPU-resident optimizer proposals and fixed-address inference publication."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from lightcone_spec.config.schema import OptimizerConfig


@dataclass(frozen=True)
class OptimizerProposal:
    parameters: tuple[Tensor, ...]
    first_moments: tuple[Tensor, ...]
    second_moments: tuple[Tensor, ...]
    step: int


class GPUOptimizer:
    """Functional Adam/AdamW/SGD whose active state changes only on commit."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        config: OptimizerConfig,
    ) -> None:
        if config.name == "none":
            raise ValueError("GPUOptimizer cannot be constructed for none")
        self.config = config
        self.master = tuple(
            parameter.detach().to(dtype=torch.float32).clone()
            for parameter in parameters
        )
        if not self.master:
            raise ValueError("optimizer needs at least one parameter")
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.master):
            raise ValueError("optimizer parameters must be finite")
        self.first_moments = tuple(torch.zeros_like(p) for p in self.master)
        self.second_moments = tuple(torch.zeros_like(p) for p in self.master)
        self.step_number = 0

    def propose(self, gradients: Sequence[Tensor]) -> OptimizerProposal:
        if len(gradients) != len(self.master):
            raise ValueError("one gradient is required per master parameter")
        if any(
            gradient.shape != parameter.shape
            or gradient.device != parameter.device
            for gradient, parameter in zip(
                gradients, self.master, strict=True
            )
        ):
            raise ValueError("gradient layout does not match optimizer state")
        grads = tuple(g.detach().to(dtype=torch.float32) for g in gradients)
        if any(not bool(torch.isfinite(gradient).all()) for gradient in grads):
            raise ValueError("optimizer gradients must be finite")
        total_norm = torch.stack(
            tuple(gradient.square().sum() for gradient in grads)
        ).sum().sqrt()
        clip = torch.clamp(
            self.config.grad_clip / (total_norm + 1e-12),
            max=1.0,
        )
        grads = tuple(gradient * clip for gradient in grads)
        step = self.step_number + 1

        if self.config.name == "sgd":
            parameters = tuple(
                parameter - self.config.learning_rate * gradient
                for parameter, gradient in zip(self.master, grads, strict=True)
            )
            return OptimizerProposal(
                parameters,
                self.first_moments,
                self.second_moments,
                step,
            )

        beta1 = self.config.beta1
        beta2 = self.config.beta2
        first = tuple(
            beta1 * old + (1.0 - beta1) * gradient
            for old, gradient in zip(
                self.first_moments, grads, strict=True
            )
        )
        second = tuple(
            beta2 * old + (1.0 - beta2) * gradient.square()
            for old, gradient in zip(
                self.second_moments, grads, strict=True
            )
        )
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        parameters: list[Tensor] = []
        for parameter, moment1, moment2 in zip(
            self.master, first, second, strict=True
        ):
            direction = (moment1 / correction1) / (
                (moment2 / correction2).sqrt() + self.config.epsilon
            )
            if self.config.name == "adamw":
                direction = (
                    direction + self.config.weight_decay * parameter
                )
            parameters.append(
                parameter - self.config.learning_rate * direction
            )
        return OptimizerProposal(
            tuple(parameters), first, second, step
        )

    def commit(self, proposal: OptimizerProposal) -> None:
        if proposal.step != self.step_number + 1:
            raise ValueError("optimizer proposal step conflict")
        with torch.no_grad():
            for active, candidate in zip(
                self.master, proposal.parameters, strict=True
            ):
                active.copy_(candidate)
            for active, candidate in zip(
                self.first_moments,
                proposal.first_moments,
                strict=True,
            ):
                active.copy_(candidate)
            for active, candidate in zip(
                self.second_moments,
                proposal.second_moments,
                strict=True,
            ):
                active.copy_(candidate)
        self.step_number = proposal.step


class FixedAddressBank:
    """Inference tensors whose storage address never changes after creation."""

    def __init__(self, inference_parameters: Iterable[Tensor]) -> None:
        self.active = tuple(inference_parameters)
        if not self.active:
            raise ValueError("bank needs at least one inference tensor")
        self.staging = tuple(
            torch.empty_like(parameter) for parameter in self.active
        )
        self._addresses = tuple(
            parameter.data_ptr() for parameter in self.active
        )

    @property
    def addresses(self) -> tuple[int, ...]:
        return self._addresses

    def stage(self, master_parameters: Sequence[Tensor]) -> None:
        if len(master_parameters) != len(self.active):
            raise ValueError("parameter layout changed")
        if any(
            source.shape != target.shape
            or source.device != target.device
            for source, target in zip(
                master_parameters, self.active, strict=True
            )
        ):
            raise ValueError("parameter shape or device changed")
        with torch.no_grad():
            for target, source in zip(
                self.staging, master_parameters, strict=True
            ):
                target.copy_(source.to(dtype=target.dtype))

    def publish(self) -> None:
        with torch.no_grad():
            for target, source in zip(
                self.active, self.staging, strict=True
            ):
                target.copy_(source)
        if tuple(parameter.data_ptr() for parameter in self.active) != self._addresses:
            raise RuntimeError("fixed-address publication invariant failed")
