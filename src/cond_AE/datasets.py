"""Dataset helpers for autoencoder training."""

from __future__ import annotations

from pathlib import Path

from src.image_gen.weight_image_dataset import WeightImageDataset


def load_weight_image_dataset(root_dir: str | Path = "data/weight_images") -> WeightImageDataset:
    """Return the generated weight-image dataset with a channel dimension."""
    return WeightImageDataset(root_dir=root_dir, add_channel_dim=True)

