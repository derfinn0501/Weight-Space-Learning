"""PyTorch Dataset for generated weight-image files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from src.utils.io import load_torch


class WeightImageDataset(Dataset):
    """Load generated weight images saved by script 03."""

    def __init__(
        self,
        root_dir: str | Path = "data/weight_images",
        add_channel_dim: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.image_paths = sorted(self.image_dir.glob("dataset_*.pt"))
        self.add_channel_dim = add_channel_dim
        self.transform = transform

        if not self.image_paths:
            raise FileNotFoundError(
                f"No weight images found in {self.image_dir}. "
                "Run scripts/03_generate_weight_images.py first."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = load_torch(self.image_paths[index], map_location="cpu")
        image = payload["image"] if isinstance(payload, dict) and "image" in payload else payload
        image = image.float()
        if self.add_channel_dim and image.ndim == 2:
            image = image.unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)

        dataset_id = payload.get("dataset_id") if isinstance(payload, dict) else self.image_paths[index].stem
        return {
            "image": image,
            "dataset_id": dataset_id,
            "image_id": payload.get("image_id", dataset_id) if isinstance(payload, dict) else dataset_id,
            "path": str(self.image_paths[index]),
        }
