"""Minimal autoencoder training loop."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def _batch_to_image(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def train_autoencoder(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train an autoencoder with MSE reconstruction loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {"epoch": [], "loss": []}

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in data_loader:
            images = _batch_to_image(batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images)
            loss = criterion(reconstruction, images)
            loss.backward()
            optimizer.step()

            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        history["epoch"].append(float(epoch + 1))
        history["loss"].append(total_loss / max(total_count, 1))

    return model, history

