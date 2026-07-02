"""Per-image normalization methods for generated weight images."""

from __future__ import annotations

from typing import Any

import torch


def normalize_image(
    image: torch.Tensor,
    method: str = "none",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize a 2D image tensor and return normalization metadata."""
    method = method.lower()
    image = image.detach().cpu().float()

    if method == "none":
        return image.clone(), {"method": "none"}

    if method == "per_image_standardization":
        mean = image.mean()
        std = image.std(unbiased=False)
        if float(std.item()) < eps:
            normalized = image - mean
        else:
            normalized = (image - mean) / std
        return normalized, {
            "method": method,
            "mean": float(mean.item()),
            "std": float(std.item()),
            "eps": eps,
        }

    if method == "per_image_minmax":
        min_value = image.min()
        max_value = image.max()
        scale = max_value - min_value
        if float(scale.item()) < eps:
            normalized = torch.zeros_like(image)
        else:
            normalized = (image - min_value) / scale
        return normalized, {
            "method": method,
            "min": float(min_value.item()),
            "max": float(max_value.item()),
            "eps": eps,
        }

    raise ValueError(
        "Unknown normalization method. Expected one of: none, "
        "per_image_standardization, per_image_minmax."
    )

