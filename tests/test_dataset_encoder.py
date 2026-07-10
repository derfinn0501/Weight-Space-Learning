from __future__ import annotations

import torch

from src.cond_AE.ae_models import FullyConnectedAutoencoder
from src.dataset_encoder.datasets import collate_dataset_latent_pairs, load_dataset_latent_pair_dataset
from src.dataset_encoder.models import DeepSetsDatasetEncoder
from src.dataset_encoder.train_encoder import train_dataset_encoder


def _write_pair(root, index: int) -> None:
    dataset_id = f"dataset_{index:05d}"
    dataset_dir = root / "processed" / "datasets"
    image_dir = root / "weight_images" / "images"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "dataset_id": dataset_id,
            "X_train": torch.randn(3, 2),
            "y_train": torch.tensor([0, 1, 0]),
            "X_test": torch.randn(2, 2),
            "y_test": torch.tensor([1, 0]),
            "metadata": {"dataset_id": dataset_id},
        },
        dataset_dir / f"{dataset_id}.pt",
    )
    torch.save(
        {
            "dataset_id": dataset_id,
            "image": torch.randn(2, 3),
        },
        image_dir / f"{dataset_id}.pt",
    )


def test_deepsets_dataset_encoder_forward_shape() -> None:
    rows = torch.randn(4, 7, 3)
    mask = torch.ones(4, 7, dtype=torch.bool)
    model = DeepSetsDatasetEncoder(row_dim=3, latent_dim=5)

    latents = model(rows, mask)

    assert latents.shape == (4, 5)


def test_dataset_latent_pair_dataset_uses_all_rows(tmp_path) -> None:
    _write_pair(tmp_path, 0)
    dataset = load_dataset_latent_pair_dataset(
        processed_dir=tmp_path / "processed",
        weight_image_dir=tmp_path / "weight_images",
        input_split="all",
        include_labels=True,
    )
    item = dataset[0]
    batch = collate_dataset_latent_pairs([item])

    assert len(dataset) == 1
    assert dataset.input_dim == 2
    assert dataset.row_dim == 3
    assert item["rows"].shape == (5, 3)
    assert item["image"].shape == (1, 2, 3)
    assert batch["mask"].sum().item() == 5


def test_train_dataset_encoder_saves_outputs(tmp_path) -> None:
    for index in range(3):
        _write_pair(tmp_path, index)

    autoencoder_dir = tmp_path / "results" / "autoencoders"
    autoencoder_dir.mkdir(parents=True, exist_ok=True)
    autoencoder = FullyConnectedAutoencoder(image_shape=(1, 2, 3), latent_dim=4)
    torch.save(
        {
            "state_dict": autoencoder.state_dict(),
            "autoencoder_config": {"model_type": "ae", "latent_dim": 4},
            "image_shape": [1, 2, 3],
            "metadata": {"latent_dim": 4},
        },
        autoencoder_dir / "checkpoint.pt",
    )

    config = {
        "seed": 0,
        "device": "cpu",
        "paths": {
            "processed_dir": str(tmp_path / "processed"),
            "weight_image_dir": str(tmp_path / "weight_images"),
            "autoencoder_dir": str(autoencoder_dir),
            "dataset_encoder_dir": str(tmp_path / "results" / "dataset_encoders"),
            "figure_dir": str(tmp_path / "results" / "figures"),
        },
        "dataset_encoder": {
            "model_type": "deepsets",
            "input_split": "all",
            "include_labels": True,
            "point_hidden_dim": 8,
            "point_output_dim": 8,
            "encoder_hidden_dim": 8,
            "num_layers": 2,
            "aggregation": "mean",
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.001,
            "weight_decay": 0.0,
            "train_split": 0.67,
            "num_workers": 0,
        },
        "evaluation": {
            "save_reconstructions": False,
            "plot_format": "png",
        },
    }

    result = train_dataset_encoder(config)

    assert result["metadata"]["latent_dim"] == 4
    assert result["metadata"]["n_total"] == 3
    assert "latent_mse" in result["metrics"]["validation"]
    assert (tmp_path / "results" / "dataset_encoders" / "checkpoint.pt").exists()
    assert (tmp_path / "results" / "dataset_encoders" / "training_history.json").exists()
