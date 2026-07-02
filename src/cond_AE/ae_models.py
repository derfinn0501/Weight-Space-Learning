"""Autoencoder models for weight-image reconstruction."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _normalize_image_shape(image_shape: Sequence[int]) -> tuple[int, int, int]:
    """Return image shape as (C, H, W)."""
    shape = tuple(int(dim) for dim in image_shape)
    if len(shape) == 2:
        return (1, shape[0], shape[1])
    if len(shape) == 3:
        return shape
    raise ValueError("image_shape must be [H, W] or [C, H, W].")


class FullyConnectedAutoencoder(nn.Module):
    """Fully connected autoencoder that reconstructs flattened weight images."""

    def __init__(
        self,
        image_shape: Sequence[int],
        latent_dim: int = 32,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_shape = _normalize_image_shape(image_shape)
        self.input_dim = int(prod(self.image_shape))
        self.latent_dim = int(latent_dim)
        hidden = int(hidden_dim or min(512, max(64, self.input_dim // 2)))

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into latent vectors."""
        return self.encoder(x.reshape(x.shape[0], -1))

    def decode(self, z: torch.Tensor, output_shape: Sequence[int] | None = None) -> torch.Tensor:
        """Decode latent vectors into image tensors."""
        shape = tuple(output_shape) if output_shape is not None else (z.shape[0], *self.image_shape)
        return self.decoder(z).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input image batch with the same shape."""
        return self.decode(self.encode(x), output_shape=x.shape)


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for single-channel weight images."""

    def __init__(self, image_shape: Sequence[int], latent_dim: int = 32) -> None:
        super().__init__()
        channels, height, width = _normalize_image_shape(image_shape)
        if channels != 1:
            raise ValueError("ConvAutoencoder currently expects single-channel images.")

        self.image_shape = (channels, height, width)
        self.latent_dim = int(latent_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.to_latent = nn.Linear(32 * 4 * 4, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, 32 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single-channel image batch into latent vectors."""
        features = self.encoder_conv(x).reshape(x.shape[0], -1)
        return self.to_latent(features)

    def decode(self, z: torch.Tensor, spatial_shape: Sequence[int] | None = None) -> torch.Tensor:
        """Decode latent vectors and resize exactly to the requested H x W."""
        target_hw = tuple(int(dim) for dim in (spatial_shape or self.image_shape[-2:]))
        features = self.from_latent(z).reshape(z.shape[0], 32, 4, 4)
        features = F.interpolate(features, size=target_hw, mode="bilinear", align_corners=False)
        return self.decoder_conv(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input image batch with the same H x W shape."""
        return self.decode(self.encode(x), spatial_shape=x.shape[-2:])


def build_autoencoder(config: dict[str, Any], image_shape: Sequence[int]) -> nn.Module:
    """Build the configured autoencoder for a given image shape."""
    ae_config = config.get("autoencoder", config)
    model_type = str(ae_config.get("model_type", "cae")).lower()
    latent_dim = int(ae_config.get("latent_dim", 32))

    if model_type == "ae":
        return FullyConnectedAutoencoder(
            image_shape=image_shape,
            latent_dim=latent_dim,
            hidden_dim=ae_config.get("hidden_dim"),
        )
    if model_type == "cae":
        return ConvAutoencoder(image_shape=image_shape, latent_dim=latent_dim)

    raise ValueError("autoencoder.model_type must be 'ae' or 'cae'.")


# Backward-compatible aliases from the Milestone 1 placeholder.
FullyConnectedAE = FullyConnectedAutoencoder
SmallConvAE = ConvAutoencoder

