"""Metrics used for classification and reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return cross-entropy loss for multiclass classification logits."""
    return F.cross_entropy(logits, targets)


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute multiclass classification accuracy."""
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean squared error."""
    return float(F.mse_loss(prediction, target).item())


def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean absolute error."""
    return float(F.l1_loss(prediction, target).item())


def relative_l2_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Compute relative L2 reconstruction error."""
    numerator = torch.linalg.norm((prediction - target).reshape(-1))
    denominator = torch.linalg.norm(target.reshape(-1)).clamp_min(eps)
    return float((numerator / denominator).item())

