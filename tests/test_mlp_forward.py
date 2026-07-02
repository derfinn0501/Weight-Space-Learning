from __future__ import annotations

import torch

from src.network_learning.models import MLP


def test_mlp_forward_shape() -> None:
    model = MLP(input_dim=2, output_dim=2, hidden_layers=[8, 4], activation="relu")
    x = torch.randn(5, 2)
    logits = model(x)

    assert logits.shape == (5, 2)

