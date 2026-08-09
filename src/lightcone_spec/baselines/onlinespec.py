"""Small clean-room implementation of the published OnlineSpec equations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def ogd_update(parameter: Tensor, gradient: Tensor, learning_rate: float) -> Tensor:
    return parameter - learning_rate * gradient


def optimistic_update(
    parameter: Tensor,
    gradient: Tensor,
    previous_gradient: Tensor,
    learning_rate: float,
) -> Tensor:
    return parameter - learning_rate * (2.0 * gradient - previous_gradient)


@dataclass
class OnlineSpecEnsemble:
    weights: Tensor
    learning_rate: float

    def combine(self, candidates: Tensor) -> Tensor:
        probabilities = torch.softmax(self.weights, dim=0)
        return torch.tensordot(probabilities, candidates, dims=([0], [0]))

    def update(self, losses: Tensor) -> None:
        if losses.shape != self.weights.shape:
            raise ValueError("one loss is required per ensemble member")
        self.weights.sub_(self.learning_rate * losses)
