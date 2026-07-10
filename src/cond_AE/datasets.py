"""Dataset helpers for autoencoder training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, random_split

from src.image_gen.weight_image_dataset import WeightImageDataset


class AutoencoderWeightImageDataset(Dataset):
    """Load generated weight images with a consistent channel-first shape."""

    def __init__(
        self,
        root_dir: str | Path = "data/weight_images",
        validate_same_shape: bool = True,
    ) -> None:
        self.base_dataset = WeightImageDataset(root_dir=root_dir, add_channel_dim=True)
        first = self.base_dataset[0]
        self.image_shape = tuple(int(dim) for dim in first["image"].shape)
        self.dataset_ids = [str(first["dataset_id"])]
        self.image_ids = [str(first.get("image_id", first["dataset_id"]))]

        if validate_same_shape:
            for index in range(1, len(self.base_dataset)):
                item = self.base_dataset[index]
                shape = tuple(int(dim) for dim in item["image"].shape)
                if shape != self.image_shape:
                    raise ValueError(
                        "All weight images must have the same shape for AE/CAE training. "
                        f"Expected {self.image_shape}, found {shape} at index {index}."
                    )
                self.dataset_ids.append(str(item["dataset_id"]))
                self.image_ids.append(str(item.get("image_id", item["dataset_id"])))
        else:
            for index in range(1, len(self.base_dataset)):
                item = self.base_dataset[index]
                self.dataset_ids.append(str(item["dataset_id"]))
                self.image_ids.append(str(item.get("image_id", item["dataset_id"])))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base_dataset[index]
        return {
            "image": item["image"].float(),
            "dataset_id": item["dataset_id"],
            "image_id": item.get("image_id", item["dataset_id"]),
            "path": item["path"],
        }


def load_weight_image_dataset(
    root_dir: str | Path = "data/weight_images",
) -> AutoencoderWeightImageDataset:
    """Return the generated weight-image dataset for AE/CAE training."""
    return AutoencoderWeightImageDataset(root_dir=root_dir)


def train_validation_split(
    dataset: Dataset,
    train_split: float = 0.8,
    seed: int = 42,
    split_by: str = "image",
) -> tuple[Subset, Subset]:
    """Split a dataset into train and validation subsets.

    split_by="image" randomly splits individual images. split_by="model"
    groups by dataset_id, so images from one trained target model never appear
    in both train and validation.
    """
    if not 0.0 < train_split <= 1.0:
        raise ValueError("train_split must be in (0, 1].")

    n_total = len(dataset)
    if n_total == 0:
        raise ValueError("Cannot split an empty dataset.")
    if n_total == 1:
        single = Subset(dataset, [0])
        return single, single

    split_by = split_by.lower()
    if split_by in {"model", "dataset", "dataset_id"}:
        dataset_ids = _dataset_ids(dataset)
        unique_ids = list(dict.fromkeys(dataset_ids))
        if len(unique_ids) == 1:
            single = Subset(dataset, list(range(n_total)))
            return single, single

        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(unique_ids), generator=generator).tolist()
        n_train_groups = int(round(len(unique_ids) * train_split))
        n_train_groups = min(max(n_train_groups, 1), len(unique_ids) - 1)
        train_ids = {unique_ids[index] for index in order[:n_train_groups]}
        train_indices = [index for index, dataset_id in enumerate(dataset_ids) if dataset_id in train_ids]
        val_indices = [index for index, dataset_id in enumerate(dataset_ids) if dataset_id not in train_ids]
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    if split_by not in {"image", "random"}:
        raise ValueError("split_by must be one of: image, random, model, dataset, dataset_id.")

    n_train = int(round(n_total * train_split))
    n_train = min(max(n_train, 1), n_total - 1)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(int(seed))
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)
    return train_dataset, val_dataset


def _dataset_ids(dataset: Dataset) -> list[str]:
    if hasattr(dataset, "dataset_ids"):
        return [str(dataset_id) for dataset_id in dataset.dataset_ids]
    return [str(dataset[index]["dataset_id"]) for index in range(len(dataset))]


def subset_dataset_ids(subset: Subset) -> list[str]:
    """Return sorted unique dataset IDs contained in a subset."""
    parent_ids = _dataset_ids(subset.dataset)
    ids = {parent_ids[int(index)] for index in subset.indices}
    return sorted(ids)
