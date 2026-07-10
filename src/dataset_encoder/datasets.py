"""Datasets for training tabular-dataset encoders against AE latents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.utils.io import load_torch


VALID_INPUT_SPLITS = {"train", "test", "all"}


def _labels_as_features(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    if labels.ndim == 1:
        return labels.unsqueeze(1)
    return labels.reshape(labels.shape[0], -1)


def _select_rows(
    payload: dict[str, Any],
    input_split: str,
    include_labels: bool,
) -> torch.Tensor:
    if input_split not in VALID_INPUT_SPLITS:
        raise ValueError(f"dataset_encoder.input_split must be one of {sorted(VALID_INPUT_SPLITS)}.")

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    if input_split in {"train", "all"}:
        features.append(payload["X_train"].float())
        labels.append(payload["y_train"])
    if input_split in {"test", "all"}:
        features.append(payload["X_test"].float())
        labels.append(payload["y_test"])

    x = torch.cat(features, dim=0)
    if not include_labels:
        return x

    y = _labels_as_features(torch.cat(labels, dim=0))
    if y.shape[0] != x.shape[0]:
        raise ValueError("Selected labels and features have different row counts.")
    return torch.cat([x, y], dim=1)


class DatasetLatentPairDataset(Dataset):
    """Pair generated tabular datasets with their generated weight images."""

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        weight_image_dir: str | Path = "data/weight_images",
        input_split: str = "all",
        include_labels: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.weight_image_dir = Path(weight_image_dir)
        self.input_split = str(input_split)
        self.include_labels = bool(include_labels)
        self.dataset_dir = self.processed_dir / "datasets"
        self.image_dir = self.weight_image_dir / "images"

        if self.input_split not in VALID_INPUT_SPLITS:
            raise ValueError(f"dataset_encoder.input_split must be one of {sorted(VALID_INPUT_SPLITS)}.")

        dataset_paths = sorted(self.dataset_dir.glob("dataset_*.pt"))
        if not dataset_paths:
            raise FileNotFoundError(
                f"No generated datasets found in {self.dataset_dir}. "
                "Run scripts/01_generate_datasets.py first."
            )

        self.records: list[dict[str, Any]] = []
        row_dims: set[int] = set()
        input_dims: set[int] = set()
        self.n_rows_per_dataset: list[int] = []
        for dataset_path in dataset_paths:
            dataset_id = dataset_path.stem
            image_path = self.image_dir / f"{dataset_id}.pt"
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing weight image for {dataset_id}: {image_path}. "
                    "Run scripts/03_generate_weight_images.py first."
                )

            payload = load_torch(dataset_path, map_location="cpu")
            rows = _select_rows(payload, self.input_split, self.include_labels)
            row_dims.add(int(rows.shape[1]))
            input_dims.add(int(payload["X_train"].shape[1]))
            self.n_rows_per_dataset.append(int(rows.shape[0]))
            self.records.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_path": dataset_path,
                    "image_path": image_path,
                }
            )

        if len(row_dims) != 1:
            raise ValueError(f"All dataset-encoder rows must have the same width, found {sorted(row_dims)}.")
        if len(input_dims) != 1:
            raise ValueError(f"All generated datasets must have the same input dimension, found {sorted(input_dims)}.")

        self.row_dim = row_dims.pop()
        self.input_dim = input_dims.pop()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        dataset_payload = load_torch(record["dataset_path"], map_location="cpu")
        image_payload = load_torch(record["image_path"], map_location="cpu")
        image = image_payload["image"] if isinstance(image_payload, dict) and "image" in image_payload else image_payload
        image = image.float()
        if image.ndim == 2:
            image = image.unsqueeze(0)

        return {
            "rows": _select_rows(dataset_payload, self.input_split, self.include_labels),
            "image": image,
            "dataset_id": record["dataset_id"],
            "dataset_path": str(record["dataset_path"]),
            "image_path": str(record["image_path"]),
        }


def collate_dataset_latent_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad dataset rows and return a mask for permutation-invariant encoders."""
    max_rows = max(int(item["rows"].shape[0]) for item in batch)
    row_dim = int(batch[0]["rows"].shape[1])
    rows = torch.zeros(len(batch), max_rows, row_dim, dtype=batch[0]["rows"].dtype)
    mask = torch.zeros(len(batch), max_rows, dtype=torch.bool)

    for index, item in enumerate(batch):
        n_rows = int(item["rows"].shape[0])
        rows[index, :n_rows] = item["rows"]
        mask[index, :n_rows] = True

    return {
        "rows": rows,
        "mask": mask,
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "dataset_id": [item["dataset_id"] for item in batch],
        "dataset_path": [item["dataset_path"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def load_dataset_latent_pair_dataset(
    processed_dir: str | Path = "data/processed",
    weight_image_dir: str | Path = "data/weight_images",
    input_split: str = "all",
    include_labels: bool = True,
) -> DatasetLatentPairDataset:
    """Return paired tabular datasets and weight images for dataset-encoder training."""
    return DatasetLatentPairDataset(
        processed_dir=processed_dir,
        weight_image_dir=weight_image_dir,
        input_split=input_split,
        include_labels=include_labels,
    )
