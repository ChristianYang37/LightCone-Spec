"""GPU-resident optimizer proposals and fixed-address inference publication."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lightcone_spec.config.schema import OptimizerConfig

_MUON_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeroth_power_newton_schulz(
    matrix: Tensor,
    *,
    steps: int,
    epsilon: float,
) -> Tensor:
    """Muon quintic Newton--Schulz orthogonalization for one matrix."""
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization requires a matrix")
    if not 1 <= steps <= 20:
        raise ValueError("Muon Newton--Schulz steps must be in [1, 20]")
    work = matrix.to(torch.bfloat16)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    work = work / work.norm().clamp_min(epsilon)
    a, b, c = _MUON_COEFFICIENTS
    for _ in range(steps):
        gram = work @ work.T
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        work = torch.addmm(work, polynomial, work, beta=a)
    if transposed:
        work = work.T
    return work.to(torch.float32)


@dataclass(frozen=True)
class OptimizerProposal:
    parameters: tuple[Tensor, ...]
    first_moments: tuple[Tensor, ...]
    second_moments: tuple[Tensor, ...]
    step: int
    numerical_predicate: Tensor
    safe_boundary_age: int | None = None
    source_version: int | None = None
    safe_boundary_version: int | None = None
    _owner_token: object = field(default=None, repr=False, compare=False)


class GPUOptimizer:
    """Functional GPU optimizer whose active state changes only on commit."""

    def __init__(
        self,
        parameters: Iterable[Tensor],
        config: OptimizerConfig,
        *,
        initial_safe_boundary_version: int = 0,
    ) -> None:
        supported = {
            "adam",
            "adamw",
            "chronobelief",
            "lion",
            "muon",
            "nag",
            "sgd",
            "sgdm",
        }
        if config.name == "none":
            raise ValueError("GPUOptimizer cannot be constructed for none")
        if config.name not in supported:
            raise ValueError("GPUOptimizer received an unsupported optimizer")
        if (
            type(initial_safe_boundary_version) is not int
            or initial_safe_boundary_version < 0
        ):
            raise ValueError("initial safe-boundary version must be non-negative")
        if config.name != "chronobelief" and initial_safe_boundary_version != 0:
            raise ValueError(
                "safe-boundary version is only defined for optimizer=chronobelief"
            )
        self.config = config
        self.safe_boundary_version = initial_safe_boundary_version
        self._proposal_owner = object()
        self.master = tuple(
            parameter.detach().to(dtype=torch.float32).clone()
            for parameter in parameters
        )
        if not self.master:
            raise ValueError("optimizer needs at least one parameter")
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.master):
            raise ValueError("optimizer parameters must be finite")
        first_names = {
            "adam",
            "adamw",
            "chronobelief",
            "sgdm",
            "nag",
            "muon",
            "lion",
        }
        self.first_moments = tuple(
            torch.zeros_like(parameter)
            if config.name in first_names
            else torch.empty(0, device=parameter.device, dtype=torch.float32)
            for parameter in self.master
        )
        self.second_moments = tuple(
            torch.zeros_like(parameter)
            if config.name in {"adam", "adamw", "chronobelief"}
            or (config.name == "muon" and parameter.ndim != 2)
            else torch.empty(0, device=parameter.device, dtype=torch.float32)
            for parameter in self.master
        )
        self.step_number = 0

    def scheduled_learning_rate(self, base_learning_rate: float, step: int) -> float:
        """Evaluate the registered schedule on the next published update index."""

        if step < 1:
            raise ValueError("published update step must be positive")
        if self.config.schedule == "constant":
            scale = 1.0
        elif self.config.schedule == "inverse_sqrt_published_update":
            scale = 1.0 / math.sqrt(step)
        else:
            horizon = self.config.schedule_total_published_updates
            if horizon is None:
                raise AssertionError("validated cosine schedule has no horizon")
            if step > horizon:
                raise ValueError("published update exceeds the cosine schedule horizon")
            scale = 0.5 * (1.0 + math.cos(math.pi * (step - 1) / (horizon - 1)))
        return base_learning_rate * scale

    def _proposal(
        self,
        parameters: tuple[Tensor, ...],
        first_moments: tuple[Tensor, ...],
        second_moments: tuple[Tensor, ...],
        step: int,
        gradients: tuple[Tensor, ...],
        safe_boundary_age: int | None = None,
        source_version: int | None = None,
        safe_boundary_version: int | None = None,
    ) -> OptimizerProposal:
        numerical = torch.stack(
            tuple(
                torch.isfinite(tensor).all()
                for tensor in (
                    *gradients,
                    *parameters,
                    *first_moments,
                    *second_moments,
                )
                if tensor.numel() > 0
            )
        ).all()
        if numerical.device.type == "cpu" and not bool(numerical):
            raise ValueError("optimizer proposal is non-finite")
        return OptimizerProposal(
            parameters,
            first_moments,
            second_moments,
            step,
            numerical,
            safe_boundary_age,
            source_version,
            safe_boundary_version,
            self._proposal_owner,
        )

    def _adamw_update(
        self,
        parameter: Tensor,
        gradient: Tensor,
        first: Tensor,
        second: Tensor,
        *,
        step: int,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        next_first = beta1 * first + (1.0 - beta1) * gradient
        next_second = beta2 * second + (1.0 - beta2) * gradient.square()
        direction = (next_first / (1.0 - beta1**step)) / (
            (next_second / (1.0 - beta2**step)).sqrt() + self.config.epsilon
        )
        base_learning_rate = (
            self.config.learning_rate if learning_rate is None else learning_rate
        )
        learning_rate = self.scheduled_learning_rate(base_learning_rate, step)
        weight_decay = (
            self.config.weight_decay if weight_decay is None else weight_decay
        )
        updated = parameter * (1.0 - learning_rate * weight_decay) - (
            learning_rate * direction
        )
        return updated, next_first, next_second

    def propose(
        self,
        gradients: Sequence[Tensor],
        *,
        safe_boundary_age: int | None = None,
        source_version: int | None = None,
        safe_boundary_version: int | None = None,
    ) -> OptimizerProposal:
        if self.config.name == "chronobelief":
            if safe_boundary_age is not None:
                raise ValueError(
                    "ChronoBelief safe-boundary age is derived from source versions"
                )
            if (
                type(source_version) is not int
                or source_version < 0
                or type(safe_boundary_version) is not int
                or safe_boundary_version < source_version
            ):
                raise ValueError(
                    "ChronoBelief requires ordered non-negative source versions"
                )
            if safe_boundary_version != self.safe_boundary_version:
                raise ValueError(
                    "ChronoBelief proposal does not bind the current safe boundary"
                )
            safe_boundary_age = safe_boundary_version - source_version
        elif any(
            value is not None
            for value in (safe_boundary_age, source_version, safe_boundary_version)
        ):
            raise ValueError(
                "source/safe-boundary versions are only defined for "
                "optimizer=chronobelief"
            )
        if len(gradients) != len(self.master):
            raise ValueError("one gradient is required per master parameter")
        if any(
            gradient.shape != parameter.shape or gradient.device != parameter.device
            for gradient, parameter in zip(gradients, self.master, strict=True)
        ):
            raise ValueError("gradient layout does not match optimizer state")
        grads = tuple(g.detach().to(dtype=torch.float32) for g in gradients)
        gradient_predicate = torch.stack(
            tuple(torch.isfinite(gradient).all() for gradient in grads)
        ).all()
        if grads[0].device.type == "cpu" and not bool(gradient_predicate):
            raise ValueError("optimizer gradients must be finite")
        if self.config.grad_clip is not None:
            total_norm = (
                torch.stack(tuple(gradient.square().sum() for gradient in grads))
                .sum()
                .sqrt()
            )
            clip = torch.clamp(
                self.config.grad_clip / (total_norm + 1e-12),
                max=1.0,
            )
            grads = tuple(gradient * clip for gradient in grads)
        step = self.step_number + 1
        learning_rate = self.scheduled_learning_rate(
            self.config.learning_rate,
            step,
        )

        if self.config.name == "sgd":
            parameters = tuple(
                parameter - learning_rate * gradient
                for parameter, gradient in zip(self.master, grads, strict=True)
            )
            return self._proposal(
                parameters,
                self.first_moments,
                self.second_moments,
                step,
                grads,
            )

        if self.config.name in {"sgdm", "nag"}:
            momentum = self.config.momentum
            if momentum is None:
                raise AssertionError("validated momentum optimizer has no momentum")
            effective = tuple(
                gradient + self.config.weight_decay * parameter
                for parameter, gradient in zip(self.master, grads, strict=True)
            )
            first = tuple(
                momentum * old + gradient
                for old, gradient in zip(self.first_moments, effective, strict=True)
            )
            directions = (
                first
                if self.config.name == "sgdm"
                else tuple(
                    gradient + momentum * moment
                    for gradient, moment in zip(effective, first, strict=True)
                )
            )
            parameters = tuple(
                parameter - learning_rate * direction
                for parameter, direction in zip(self.master, directions, strict=True)
            )
            return self._proposal(parameters, first, self.second_moments, step, grads)

        if self.config.name == "lion":
            beta1 = self.config.beta1
            beta2 = self.config.beta2
            directions = tuple(
                (beta1 * moment + (1.0 - beta1) * gradient).sign()
                for moment, gradient in zip(self.first_moments, grads, strict=True)
            )
            first = tuple(
                beta2 * moment + (1.0 - beta2) * gradient
                for moment, gradient in zip(self.first_moments, grads, strict=True)
            )
            decay = 1.0 - (learning_rate * self.config.weight_decay)
            parameters = tuple(
                parameter * decay - learning_rate * direction
                for parameter, direction in zip(self.master, directions, strict=True)
            )
            return self._proposal(parameters, first, self.second_moments, step, grads)

        if self.config.name == "muon":
            momentum = self.config.momentum
            ns_steps = self.config.muon_ns_steps
            if momentum is None or ns_steps is None:
                raise AssertionError("validated Muon configuration is incomplete")
            auxiliary_lr = self.config.muon_auxiliary_learning_rate
            auxiliary_decay = self.config.muon_auxiliary_weight_decay
            if auxiliary_lr is None or auxiliary_decay is None:
                raise AssertionError("validated Muon fallback is incomplete")
            parameters: list[Tensor] = []
            first: list[Tensor] = []
            second: list[Tensor] = []
            for parameter, gradient, old_first, old_second in zip(
                self.master,
                grads,
                self.first_moments,
                self.second_moments,
                strict=True,
            ):
                if parameter.ndim != 2:
                    updated, next_first, next_second = self._adamw_update(
                        parameter,
                        gradient,
                        old_first,
                        old_second,
                        step=step,
                        learning_rate=auxiliary_lr,
                        weight_decay=auxiliary_decay,
                    )
                else:
                    next_first = momentum * old_first + (1.0 - momentum) * gradient
                    nesterov = (1.0 - momentum) * gradient + (momentum * next_first)
                    direction = zeroth_power_newton_schulz(
                        nesterov,
                        steps=ns_steps,
                        epsilon=max(self.config.epsilon, 1e-7),
                    )
                    adjusted_lr = learning_rate * math.sqrt(
                        max(1.0, parameter.shape[0] / parameter.shape[1])
                    )
                    updated = (
                        parameter * (1.0 - learning_rate * self.config.weight_decay)
                        - adjusted_lr * direction
                    )
                    next_second = old_second
                parameters.append(updated)
                first.append(next_first)
                second.append(next_second)
            return self._proposal(
                tuple(parameters), tuple(first), tuple(second), step, grads
            )

        if self.config.name == "chronobelief":
            assert safe_boundary_age is not None
            beta1 = self.config.beta1
            beta2 = self.config.beta2
            first = tuple(
                beta1 * old + (1.0 - beta1) * gradient
                for old, gradient in zip(self.first_moments, grads, strict=True)
            )
            second = tuple(
                beta2 * old + (1.0 - beta2) * (gradient - moment).square()
                for old, gradient, moment in zip(
                    self.second_moments, grads, first, strict=True
                )
            )
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            age_ratio = beta1 / math.sqrt(beta2)
            kappa = 1.0 if age_ratio >= 1.0 else age_ratio**safe_boundary_age
            decay = 1.0 - learning_rate * self.config.weight_decay
            parameters = tuple(
                parameter * decay
                - learning_rate
                * kappa
                * (moment1 / correction1)
                / ((moment2 / correction2).sqrt() + self.config.epsilon)
                for parameter, moment1, moment2 in zip(
                    self.master, first, second, strict=True
                )
            )
            return self._proposal(
                parameters,
                first,
                second,
                step,
                grads,
                safe_boundary_age,
                source_version,
                safe_boundary_version,
            )

        if self.config.name not in {"adam", "adamw"}:
            raise ValueError("GPUOptimizer optimizer branch is not implemented")
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        first = tuple(
            beta1 * old + (1.0 - beta1) * gradient
            for old, gradient in zip(self.first_moments, grads, strict=True)
        )
        second = tuple(
            beta2 * old + (1.0 - beta2) * gradient.square()
            for old, gradient in zip(self.second_moments, grads, strict=True)
        )
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        parameters: list[Tensor] = []
        for parameter, moment1, moment2 in zip(self.master, first, second, strict=True):
            direction = (moment1 / correction1) / (
                (moment2 / correction2).sqrt() + self.config.epsilon
            )
            if self.config.name == "adamw":
                direction = direction + self.config.weight_decay * parameter
            parameters.append(parameter - learning_rate * direction)
        return self._proposal(tuple(parameters), first, second, step, grads)

    def commit(
        self,
        proposal: OptimizerProposal,
        *,
        numerical_receipt: bool | None = None,
    ) -> None:
        if proposal._owner_token is not self._proposal_owner:
            raise ValueError("optimizer proposal belongs to another state owner")
        if proposal.step != self.step_number + 1:
            raise ValueError("optimizer proposal step conflict")
        if self.config.name == "chronobelief":
            if (
                type(proposal.safe_boundary_age) is not int
                or type(proposal.source_version) is not int
                or type(proposal.safe_boundary_version) is not int
                or proposal.safe_boundary_version != self.safe_boundary_version
                or proposal.source_version + proposal.safe_boundary_age
                != proposal.safe_boundary_version
            ):
                raise ValueError(
                    "ChronoBelief proposal lacks the current source boundary"
                )
        elif any(
            value is not None
            for value in (
                proposal.safe_boundary_age,
                proposal.source_version,
                proposal.safe_boundary_version,
            )
        ):
            raise ValueError("non-ChronoBelief proposal carries source-boundary state")
        if proposal.numerical_predicate.device.type == "cpu":
            valid = bool(proposal.numerical_predicate)
        elif numerical_receipt is None:
            raise RuntimeError(
                "CUDA commit requires an event-complete numerical receipt"
            )
        else:
            valid = numerical_receipt
        if not valid:
            raise ValueError("optimizer proposal is non-finite")
        with torch.no_grad():
            for active, candidate in zip(self.master, proposal.parameters, strict=True):
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
        if self.config.name == "chronobelief":
            self.safe_boundary_version += 1


class FixedAddressBank:
    """Inference tensors whose storage address never changes after creation."""

    def __init__(self, inference_parameters: Iterable[Tensor]) -> None:
        self.active = tuple(inference_parameters)
        if not self.active:
            raise ValueError("bank needs at least one inference tensor")
        self.staging = tuple(torch.empty_like(parameter) for parameter in self.active)
        self._addresses = tuple(parameter.data_ptr() for parameter in self.active)

    @property
    def addresses(self) -> tuple[int, ...]:
        return self._addresses

    def stage(self, master_parameters: Sequence[Tensor]) -> None:
        if len(master_parameters) != len(self.active):
            raise ValueError("parameter layout changed")
        if any(
            source.shape != target.shape or source.device != target.device
            for source, target in zip(master_parameters, self.active, strict=True)
        ):
            raise ValueError("parameter shape or device changed")
        with torch.no_grad():
            for target, source in zip(self.staging, master_parameters, strict=True):
                target.copy_(source.to(dtype=target.dtype))

    def publish(self) -> None:
        with torch.no_grad():
            for target, source in zip(self.active, self.staging, strict=True):
                target.copy_(source)
        if tuple(parameter.data_ptr() for parameter in self.active) != self._addresses:
            raise RuntimeError("fixed-address publication invariant failed")
