"""Training and checkpoint helpers for AE/CAE weight-image reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cond_AE.ae_models import build_autoencoder
from src.cond_AE.datasets import load_weight_image_dataset, subset_dataset_ids, train_validation_split
from src.utils.io import load_torch, save_json, save_torch
from src.utils.paths import create_output_dirs


def _batch_to_images(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _run_train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in data_loader:
        images = _batch_to_images(batch).to(device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(images)
        loss = criterion(reconstruction, images)
        loss.backward()
        optimizer.step()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_autoencoder(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate reconstruction quality over a dataloader."""
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    total_elements = 0
    relative_l2_sum = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in data_loader:
            images = _batch_to_images(batch).to(device)
            reconstruction = model(images)
            diff = reconstruction - images

            squared_error += float(torch.sum(diff.square()).item())
            absolute_error += float(torch.sum(diff.abs()).item())
            total_elements += int(diff.numel())

            per_image_diff = diff.reshape(diff.shape[0], -1)
            per_image_target = images.reshape(images.shape[0], -1)
            numerator = torch.linalg.norm(per_image_diff, dim=1)
            denominator = torch.linalg.norm(per_image_target, dim=1).clamp_min(1e-12)
            relative_l2_sum += float((numerator / denominator).sum().item())
            total_images += int(images.shape[0])

    return {
        "mse": squared_error / max(total_elements, 1),
        "mae": absolute_error / max(total_elements, 1),
        "relative_l2_error": relative_l2_sum / max(total_images, 1),
    }


def train_autoencoder(config: dict[str, Any]) -> dict[str, Any]:
    """Train the configured AE/CAE on generated weight images and save outputs."""
    create_output_dirs(config)
    paths = config.get("paths", {})
    ae_config = config.get("autoencoder", {})
    seed = int(config.get("seed", 42))
    device = config.get("device", "cpu")

    dataset = load_weight_image_dataset(paths.get("weight_image_dir", "data/weight_images"))
    train_dataset, val_dataset = train_validation_split(
        dataset,
        train_split=float(ae_config.get("train_split", 0.8)),
        seed=seed,
        split_by=str(ae_config.get("split_by", "image")),
    )

    batch_size = int(ae_config.get("batch_size", 16))
    num_workers = int(ae_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_autoencoder(config, image_shape=dataset.image_shape).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(ae_config.get("lr", 1e-3)),
        weight_decay=float(ae_config.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()

    output_dir = Path(paths.get("autoencoder_dir", "data/results/autoencoders"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = int(ae_config.get("checkpoint_every", 0))
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    n_epochs = int(ae_config.get("epochs", 50))
    progress = tqdm(range(1, n_epochs + 1), desc="Training autoencoder")
    for epoch in progress:
        train_loss = _run_train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_autoencoder(model, val_loader, device=device)
        val_loss = val_metrics["mse"]

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        progress.set_postfix({"train": f"{train_loss:.4g}", "val": f"{val_loss:.4g}"})

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_torch(
                {
                    "state_dict": model.state_dict(),
                    "autoencoder_config": ae_config,
                    "image_shape": list(dataset.image_shape),
                    "epoch": epoch,
                },
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
            )

    train_metrics = evaluate_autoencoder(model, train_loader, device=device)
    val_metrics = evaluate_autoencoder(model, val_loader, device=device)
    final_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
    }
    train_dataset_ids = subset_dataset_ids(train_dataset)
    validation_dataset_ids = subset_dataset_ids(val_dataset)
    overlap_dataset_ids = sorted(set(train_dataset_ids) & set(validation_dataset_ids))
    metadata = {
        "model_type": ae_config.get("model_type", "cae"),
        "latent_dim": int(ae_config.get("latent_dim", 32)),
        "image_shape": list(dataset.image_shape),
        "n_images": len(dataset),
        "n_train": len(train_dataset),
        "n_validation": len(val_dataset),
        "split_by": str(ae_config.get("split_by", "image")),
        "n_train_models": len(train_dataset_ids),
        "n_validation_models": len(validation_dataset_ids),
        "train_dataset_ids": train_dataset_ids,
        "validation_dataset_ids": validation_dataset_ids,
        "overlap_dataset_ids": overlap_dataset_ids,
        "batch_size": batch_size,
        "epochs": n_epochs,
    }

    checkpoint = {
        "state_dict": model.state_dict(),
        "autoencoder_config": ae_config,
        "image_shape": list(dataset.image_shape),
        "metadata": metadata,
        "metrics": final_metrics,
    }
    save_torch(checkpoint, output_dir / "checkpoint.pt")
    save_json(history, output_dir / "training_history.json")
    save_json(final_metrics, output_dir / "metrics.json")
    save_json(metadata, output_dir / "metadata.json")

    return {
        "model": model,
        "history": history,
        "metrics": final_metrics,
        "metadata": metadata,
        "output_dir": str(output_dir),
    }


def load_trained_autoencoder(config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """Load the latest AE/CAE checkpoint from the configured output directory."""
    output_dir = Path(config.get("paths", {}).get("autoencoder_dir", "data/results/autoencoders"))
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = load_torch(checkpoint_path, map_location="cpu")
    model = build_autoencoder(
        {"autoencoder": checkpoint["autoencoder_config"]},
        image_shape=checkpoint["image_shape"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint
