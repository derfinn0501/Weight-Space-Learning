"""Configurable target-network models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("activation must be one of: relu, tanh, gelu")


class MLP(nn.Module):
    """Simple fully connected classifier with configurable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int] = (32, 32),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = [int(width) for width in hidden_layers]
        self.activation = activation

        layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_make_activation(activation))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of inputs."""
        return self.net(x)

