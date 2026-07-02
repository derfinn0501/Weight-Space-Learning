from __future__ import annotations

import torch

from src.cond_AE.ae_models import ConvAutoencoder, FullyConnectedAutoencoder


def test_fully_connected_autoencoder_forward_shape() -> None:
    x = torch.randn(3, 1, 8, 10)
    model = FullyConnectedAutoencoder(image_shape=x.shape[1:], latent_dim=4)
    reconstruction = model(x)

    assert reconstruction.shape == x.shape


def test_conv_autoencoder_forward_shape() -> None:
    x = torch.randn(3, 1, 9, 17)
    model = ConvAutoencoder(image_shape=x.shape[1:], latent_dim=4)
    reconstruction = model(x)

    assert reconstruction.shape == x.shape

