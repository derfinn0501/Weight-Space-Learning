"""Dataset encoder models that predict trained autoencoder latents."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class DeepSetsDatasetEncoder(nn.Module):
    """Permutation-invariant encoder from tabular datasets to latent vectors."""

    def __init__(
        self,
        row_dim: int,
        latent_dim: int,
        point_hidden_dim: int = 64,
        point_output_dim: int = 128,
        encoder_hidden_dim: int = 128,
        num_layers: int = 2,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        self.row_dim = int(row_dim)
        self.latent_dim = int(latent_dim)
        self.aggregation = str(aggregation).lower()
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError("dataset_encoder.aggregation must be 'mean' or 'sum'.")

        self.point_mlp = _make_mlp(
            input_dim=self.row_dim,
            hidden_dim=int(point_hidden_dim),
            output_dim=int(point_output_dim),
            num_layers=int(num_layers),
        )
        self.dataset_mlp = _make_mlp(
            input_dim=int(point_output_dim),
            hidden_dim=int(encoder_hidden_dim),
            output_dim=self.latent_dim,
            num_layers=int(num_layers),
        )

    def forward(self, rows: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode padded row tensors of shape [batch, n_rows, row_dim]."""
        batch_size, n_rows, row_dim = rows.shape
        if row_dim != self.row_dim:
            raise ValueError(f"Expected row_dim={self.row_dim}, found {row_dim}.")

        point_features = self.point_mlp(rows.reshape(batch_size * n_rows, row_dim))
        point_features = point_features.reshape(batch_size, n_rows, -1)

        if mask is None:
            if self.aggregation == "sum":
                pooled = point_features.sum(dim=1)
            else:
                pooled = point_features.mean(dim=1)
        else:
            weights = mask.to(dtype=point_features.dtype).unsqueeze(-1)
            masked_features = point_features * weights
            if self.aggregation == "sum":
                pooled = masked_features.sum(dim=1)
            else:
                counts = weights.sum(dim=1).clamp_min(1.0)
                pooled = masked_features.sum(dim=1) / counts

        return self.dataset_mlp(pooled)


def build_dataset_encoder(
    config: dict[str, Any],
    row_dim: int,
    latent_dim: int,
) -> nn.Module:
    """Build the configured dataset-to-latent encoder."""
    encoder_config = config.get("dataset_encoder", config)
    model_type = str(encoder_config.get("model_type", "deepsets")).lower()

    if model_type == "deepsets":
        return DeepSetsDatasetEncoder(
            row_dim=row_dim,
            latent_dim=latent_dim,
            point_hidden_dim=int(encoder_config.get("point_hidden_dim", 64)),
            point_output_dim=int(encoder_config.get("point_output_dim", 128)),
            encoder_hidden_dim=int(encoder_config.get("encoder_hidden_dim", 128)),
            num_layers=int(encoder_config.get("num_layers", 2)),
            aggregation=str(encoder_config.get("aggregation", "mean")),
        )

    raise ValueError("dataset_encoder.model_type must be 'deepsets'.")
