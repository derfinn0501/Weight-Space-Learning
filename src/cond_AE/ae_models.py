"""Small autoencoder models for weight-image experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FullyConnectedAE(nn.Module):
    """A simple fully connected autoencoder for flattened images."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input batch."""
        original_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1)
        reconstruction = self.decoder(self.encoder(x_flat))
        return reconstruction.reshape(original_shape)


class SmallConvAE(nn.Module):
    """A compact convolutional autoencoder for single-channel weight images."""

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.to_latent = nn.Linear(16 * 4 * 4, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, 16 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a single-channel image batch with the original spatial size."""
        spatial_shape = x.shape[-2:]
        features = self.encoder_conv(x).reshape(x.shape[0], -1)
        latent = self.to_latent(features)
        decoded = self.from_latent(latent).reshape(x.shape[0], 16, 4, 4)
        decoded = F.interpolate(decoded, size=spatial_shape, mode="bilinear", align_corners=False)
        return self.decoder_conv(decoded)

