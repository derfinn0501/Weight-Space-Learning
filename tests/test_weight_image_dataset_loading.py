from __future__ import annotations

import torch

from src.cond_AE.datasets import load_weight_image_dataset, train_validation_split


def test_weight_image_dataset_loading(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    for index in range(3):
        torch.save(
            {
                "dataset_id": f"dataset_{index:05d}",
                "image": torch.randn(4, 5),
            },
            image_dir / f"dataset_{index:05d}.pt",
        )

    dataset = load_weight_image_dataset(tmp_path)
    item = dataset[0]
    train_dataset, val_dataset = train_validation_split(dataset, train_split=0.67, seed=0)

    assert len(dataset) == 3
    assert item["image"].shape == (1, 4, 5)
    assert item["dataset_id"] == "dataset_00000"
    assert len(train_dataset) == 2
    assert len(val_dataset) == 1

