"""Clean-room OnlineSPEC online learners derived from the published equations.

The upstream research repository does not publish a software license. This
module therefore implements the paper's state transitions without copying its
source. Learners are functional: ``propose`` never mutates active state, so a
stale or invalid CUDA candidate can be discarded atomically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

Parameters = tuple[Tensor, ...]
Gradients = tuple[Tensor, ...]


def _as_parameters(values: Sequence[Tensor], *, name: str) -> Parameters:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(not value.is_floating_point() for value in result):
        raise ValueError(f"{name} must contain floating tensors")
    if any(not bool(torch.isfinite(value).all()) for value in result):
        raise ValueError(f"{name} must contain finite tensors")
    return result


def _validate_state(
    reference: Sequence[Tensor],
    values: Sequence[Tensor],
    *,
    name: str,
) -> Parameters:
    expected = tuple(reference)
    result = _as_parameters(values, name=name)
    if len(result) != len(expected) or any(
        value.shape != base.shape
        or value.device != base.device
        or value.dtype != base.dtype
        for value, base in zip(result, expected, strict=True)
    ):
        raise ValueError(f"{name} does not match the learner state")
    return result


def _validate_gradients(
    parameters: Parameters, gradients: Sequence[Tensor]
) -> Gradients:
    result = tuple(gradients)
    if len(result) != len(parameters) or any(
        gradient.shape != parameter.shape
        or gradient.device != parameter.device
        or not gradient.is_floating_point()
        for parameter, gradient in zip(parameters, result, strict=True)
    ):
        raise ValueError("gradient layout does not match the online learner")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in result):
        raise ValueError("online gradients must be finite")
    return tuple(gradient.detach().to(torch.float32) for gradient in result)


def _clip_global_norm(gradients: Gradients, limit: float) -> Gradients:
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("grad_clip must be positive and finite")
    norm = torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
    scale = torch.clamp(
        torch.as_tensor(limit, device=norm.device) / norm.clamp_min(1e-12),
        max=1.0,
    )
    return tuple(gradient * scale for gradient in gradients)


def project_l2_ball(
    parameters: Sequence[Tensor],
    origin: Sequence[Tensor],
    radius: float | None,
) -> Parameters:
    """Project a parameter tuple onto a Euclidean ball around its initial state."""
    values = _as_parameters(parameters, name="projection parameters")
    centre = _as_parameters(origin, name="projection origin")
    if len(values) != len(centre) or any(
        value.shape != base.shape or value.device != base.device
        for value, base in zip(values, centre, strict=True)
    ):
        raise ValueError("projection origin does not match the parameter layout")
    if radius is None:
        return values
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("projection radius must be positive and finite")
    distance = (
        torch.stack(
            [
                (value.to(torch.float32) - base.to(torch.float32)).square().sum()
                for value, base in zip(values, centre, strict=True)
            ]
        )
        .sum()
        .sqrt()
    )
    scale = torch.clamp(
        torch.as_tensor(radius, device=distance.device) / distance.clamp_min(1e-12),
        max=1.0,
    )
    return tuple(
        base.to(torch.float32)
        + (value.to(torch.float32) - base.to(torch.float32)) * scale
        for value, base in zip(values, centre, strict=True)
    )


def ogd_update(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    learning_rate: float,
    *,
    origin: Sequence[Tensor] | None = None,
    projection_radius: float | None = None,
    grad_clip: float = 1.0,
) -> Parameters:
    """One projected online-gradient-descent transition."""
    active = _as_parameters(parameters, name="parameters")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    grads = _clip_global_norm(
        _validate_gradients(active, gradients),
        grad_clip,
    )
    candidate = tuple(
        parameter.to(torch.float32) - learning_rate * gradient
        for parameter, gradient in zip(active, grads, strict=True)
    )
    if projection_radius is not None and origin is None:
        raise ValueError("projected OGD requires an explicit fixed origin")
    centre = active if origin is None else tuple(origin)
    return project_l2_ball(candidate, centre, projection_radius)


@dataclass(frozen=True)
class OnlineSpecProposal:
    parameters: Parameters
    auxiliary: tuple[Tensor, ...]
    step: int


class OnlineSpecOGD:
    """Projected OGD with transactional proposal/commit semantics."""

    def __init__(
        self,
        parameters: Sequence[Tensor],
        *,
        learning_rate: float,
        projection_radius: float | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        initial = _as_parameters(parameters, name="parameters")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.initial = tuple(
            value.detach().to(torch.float32).clone() for value in initial
        )
        self.parameters = tuple(value.clone() for value in self.initial)
        self.learning_rate = float(learning_rate)
        self.projection_radius = projection_radius
        if not math.isfinite(grad_clip) or grad_clip <= 0:
            raise ValueError("grad_clip must be positive and finite")
        self.grad_clip = float(grad_clip)
        self.step = 0

    def propose(self, gradients: Sequence[Tensor]) -> OnlineSpecProposal:
        candidate = ogd_update(
            self.parameters,
            gradients,
            self.learning_rate,
            origin=self.initial,
            projection_radius=self.projection_radius,
            grad_clip=self.grad_clip,
        )
        return OnlineSpecProposal(candidate, (), self.step + 1)

    def commit(self, proposal: OnlineSpecProposal) -> None:
        if (
            type(proposal.step) is not int
            or proposal.step != self.step + 1
            or proposal.auxiliary
        ):
            raise ValueError("OGD proposal does not extend the active state")
        parameters = _validate_state(
            self.parameters, proposal.parameters, name="OGD proposal parameters"
        )
        self.parameters = tuple(value.detach().clone() for value in parameters)
        self.step = proposal.step


class OnlineSpecOptimistic:
    r"""The paper's two-state optimistic update, not a momentum shortcut.

    At round ``t``, the published decision is
    ``w_t = projection(hat_w_t - eta * h_t)``. After observing the gradient
    at that decision, ``hat_w`` takes an OGD step and the new hint is ``g_t``.
    """

    def __init__(
        self,
        parameters: Sequence[Tensor],
        *,
        learning_rate: float,
        projection_radius: float | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        initial = _as_parameters(parameters, name="parameters")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.initial = tuple(
            value.detach().to(torch.float32).clone() for value in initial
        )
        self.anchor = tuple(value.clone() for value in self.initial)
        self.hint = tuple(torch.zeros_like(value) for value in self.initial)
        self.parameters = tuple(value.clone() for value in self.initial)
        self.learning_rate = float(learning_rate)
        self.projection_radius = projection_radius
        if not math.isfinite(grad_clip) or grad_clip <= 0:
            raise ValueError("grad_clip must be positive and finite")
        self.grad_clip = float(grad_clip)
        self.step = 0

    def propose(self, gradients: Sequence[Tensor]) -> OnlineSpecProposal:
        grads = _clip_global_norm(
            _validate_gradients(self.parameters, gradients),
            self.grad_clip,
        )
        next_anchor = project_l2_ball(
            tuple(
                anchor - self.learning_rate * gradient
                for anchor, gradient in zip(self.anchor, grads, strict=True)
            ),
            self.initial,
            self.projection_radius,
        )
        next_decision = project_l2_ball(
            tuple(
                anchor - self.learning_rate * hint
                for anchor, hint in zip(next_anchor, grads, strict=True)
            ),
            self.initial,
            self.projection_radius,
        )
        return OnlineSpecProposal(
            next_decision,
            next_anchor + grads,
            self.step + 1,
        )

    def commit(self, proposal: OnlineSpecProposal) -> None:
        width = len(self.parameters)
        if (
            type(proposal.step) is not int
            or proposal.step != self.step + 1
            or len(proposal.auxiliary) != 2 * width
        ):
            raise ValueError("optimistic proposal does not extend the active state")
        parameters = _validate_state(
            self.parameters,
            proposal.parameters,
            name="optimistic proposal parameters",
        )
        anchors = _validate_state(
            self.anchor,
            proposal.auxiliary[:width],
            name="optimistic proposal anchors",
        )
        hints = _validate_state(
            self.hint,
            proposal.auxiliary[width:],
            name="optimistic proposal hints",
        )
        self.parameters = tuple(value.detach().clone() for value in parameters)
        self.anchor = tuple(value.detach().clone() for value in anchors)
        self.hint = tuple(value.detach().clone() for value in hints)
        self.step = proposal.step


class OnlineSpecHedge:
    """Independent OGD experts combined by cumulative-loss Hedge."""

    def __init__(
        self,
        parameters: Sequence[Tensor],
        *,
        learning_rates: Sequence[float],
        hedge_learning_rate: float,
        projection_radius: float | None = None,
        grad_clip: float = 1.0,
    ) -> None:
        initial = _as_parameters(parameters, name="parameters")
        rates = tuple(float(value) for value in learning_rates)
        if len(rates) < 2 or any(
            not math.isfinite(value) or value <= 0 for value in rates
        ):
            raise ValueError("Hedge needs at least two positive finite learning rates")
        if len(set(rates)) != len(rates) or tuple(sorted(rates)) != rates:
            raise ValueError("Hedge learning rates must be unique and increasing")
        if not math.isfinite(hedge_learning_rate) or hedge_learning_rate <= 0:
            raise ValueError("hedge_learning_rate must be positive and finite")
        self.initial = tuple(
            value.detach().to(torch.float32).clone() for value in initial
        )
        self.experts = tuple(
            tuple(value.clone() for value in self.initial) for _ in rates
        )
        self.learning_rates = rates
        self.hedge_learning_rate = float(hedge_learning_rate)
        self.projection_radius = projection_radius
        if not math.isfinite(grad_clip) or grad_clip <= 0:
            raise ValueError("grad_clip must be positive and finite")
        self.grad_clip = float(grad_clip)
        self.cumulative_losses = torch.zeros(
            len(rates), device=self.initial[0].device, dtype=torch.float32
        )
        self.parameters = tuple(value.clone() for value in self.initial)
        self.step = 0

    @property
    def probabilities(self) -> Tensor:
        return torch.softmax(-self.hedge_learning_rate * self.cumulative_losses, dim=0)

    def propose(
        self,
        losses: Tensor,
        gradients: Sequence[Sequence[Tensor]],
    ) -> OnlineSpecProposal:
        if (
            losses.shape != self.cumulative_losses.shape
            or losses.device != self.cumulative_losses.device
            or not losses.is_floating_point()
            or not bool(torch.isfinite(losses).all())
        ):
            raise ValueError("one finite loss is required per Hedge expert")
        member_gradients = tuple(tuple(values) for values in gradients)
        if len(member_gradients) != len(self.experts):
            raise ValueError("one gradient tuple is required per Hedge expert")
        updated_experts = tuple(
            ogd_update(
                expert,
                member_gradient,
                learning_rate,
                origin=self.initial,
                projection_radius=self.projection_radius,
                grad_clip=self.grad_clip,
            )
            for expert, member_gradient, learning_rate in zip(
                self.experts,
                member_gradients,
                self.learning_rates,
                strict=True,
            )
        )
        cumulative = self.cumulative_losses + losses.detach().to(torch.float32)
        probabilities = torch.softmax(-self.hedge_learning_rate * cumulative, dim=0)
        decision = tuple(
            torch.stack([expert[index] for expert in updated_experts], dim=0)
            .mul(probabilities.view((-1,) + (1,) * parameter.ndim))
            .sum(0)
            for index, parameter in enumerate(self.initial)
        )
        flattened = tuple(value for expert in updated_experts for value in expert)
        return OnlineSpecProposal(decision, flattened + (cumulative,), self.step + 1)

    def commit(self, proposal: OnlineSpecProposal) -> None:
        width = len(self.initial)
        expected = len(self.experts) * width + 1
        if (
            type(proposal.step) is not int
            or proposal.step != self.step + 1
            or len(proposal.auxiliary) != expected
        ):
            raise ValueError("Hedge proposal does not extend the active state")
        parameters = _validate_state(
            self.parameters,
            proposal.parameters,
            name="Hedge proposal parameters",
        )
        values = proposal.auxiliary[:-1]
        experts = tuple(
            _validate_state(
                self.experts[expert_index],
                values[expert_index * width : (expert_index + 1) * width],
                name=f"Hedge expert {expert_index}",
            )
            for expert_index in range(len(self.experts))
        )
        cumulative = _validate_state(
            (self.cumulative_losses,),
            (proposal.auxiliary[-1],),
            name="Hedge cumulative loss",
        )[0]
        self.experts = tuple(
            tuple(value.detach().clone() for value in expert) for expert in experts
        )
        self.cumulative_losses = cumulative.detach().clone()
        self.parameters = tuple(value.detach().clone() for value in parameters)
        self.step = proposal.step
