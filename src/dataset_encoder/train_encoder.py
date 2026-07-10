"""Training helpers for dataset-to-autoencoder-latent encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cond_AE.datasets import train_validation_split
from src.cond_AE.train_ae import load_trained_autoencoder
from src.dataset_encoder.datasets import collate_dataset_latent_pairs, load_dataset_latent_pair_dataset
from src.dataset_encoder.models import build_dataset_encoder
from src.evaluation.plots import plot_reconstruction_grid
from src.utils.io import load_torch, save_json, save_torch
from src.utils.paths import create_output_dirs


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _target_latents(autoencoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return autoencoder.encode(images).detach()


def _run_train_epoch(
    model: nn.Module,
    autoencoder: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in data_loader:
        rows = batch["rows"].to(device)
        mask = batch["mask"].to(device)
        images = batch["image"].to(device)
        target_latents = _target_latents(autoencoder, images)

        optimizer.zero_grad(set_to_none=True)
        predicted_latents = model(rows, mask)
        loss = F.mse_loss(predicted_latents, target_latents)
        loss.backward()
        optimizer.step()

        batch_size = int(rows.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate_dataset_encoder(
    model: nn.Module,
    autoencoder: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "cpu",
    include_decoded_metrics: bool = False,
) -> dict[str, float]:
    """Evaluate latent prediction quality and optional decoded-image quality."""
    model.eval()
    latent_squared_error = 0.0
    latent_absolute_error = 0.0
    latent_elements = 0
    cosine_similarity_sum = 0.0
    total_items = 0

    image_squared_error = 0.0
    image_absolute_error = 0.0
    image_elements = 0
    image_relative_l2_sum = 0.0

    with torch.no_grad():
        for batch in data_loader:
            rows = batch["rows"].to(device)
            mask = batch["mask"].to(device)
            images = batch["image"].to(device)
            target_latents = autoencoder.encode(images)
            predicted_latents = model(rows, mask)
            latent_diff = predicted_latents - target_latents

            latent_squared_error += float(torch.sum(latent_diff.square()).item())
            latent_absolute_error += float(torch.sum(latent_diff.abs()).item())
            latent_elements += int(latent_diff.numel())
            cosine_similarity_sum += float(
                F.cosine_similarity(predicted_latents, target_latents, dim=1).sum().item()
            )
            total_items += int(rows.shape[0])

            if include_decoded_metrics:
                decoded_images = autoencoder.decode(predicted_latents)
                image_diff = decoded_images - images
                image_squared_error += float(torch.sum(image_diff.square()).item())
                image_absolute_error += float(torch.sum(image_diff.abs()).item())
                image_elements += int(image_diff.numel())

                per_image_diff = image_diff.reshape(image_diff.shape[0], -1)
                per_image_target = images.reshape(images.shape[0], -1)
                numerator = torch.linalg.norm(per_image_diff, dim=1)
                denominator = torch.linalg.norm(per_image_target, dim=1).clamp_min(1e-12)
                image_relative_l2_sum += float((numerator / denominator).sum().item())

    metrics = {
        "latent_mse": latent_squared_error / max(latent_elements, 1),
        "latent_mae": latent_absolute_error / max(latent_elements, 1),
        "latent_cosine_similarity": cosine_similarity_sum / max(total_items, 1),
    }
    if include_decoded_metrics:
        metrics.update(
            {
                "decoded_image_mse": image_squared_error / max(image_elements, 1),
                "decoded_image_mae": image_absolute_error / max(image_elements, 1),
                "decoded_image_relative_l2_error": image_relative_l2_sum / max(total_items, 1),
            }
        )
    return metrics


def _collect_decoded_examples(
    model: nn.Module,
    autoencoder: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device,
    n_examples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    originals: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            rows = batch["rows"].to(device)
            mask = batch["mask"].to(device)
            images = batch["image"].to(device)
            predicted_latents = model(rows, mask)
            decoded_images = autoencoder.decode(predicted_latents)
            originals.append(images.detach().cpu())
            reconstructions.append(decoded_images.detach().cpu())
            if sum(tensor.shape[0] for tensor in originals) >= n_examples:
                break

    return torch.cat(originals, dim=0)[:n_examples], torch.cat(reconstructions, dim=0)[:n_examples]


def _latent_dim_from_checkpoint(checkpoint: dict[str, Any]) -> int:
    autoencoder_config = checkpoint.get("autoencoder_config", {})
    if "latent_dim" in autoencoder_config:
        return int(autoencoder_config["latent_dim"])
    metadata = checkpoint.get("metadata", {})
    if "latent_dim" in metadata:
        return int(metadata["latent_dim"])
    raise KeyError("Could not determine latent_dim from the autoencoder checkpoint.")


def load_trained_dataset_encoder(config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained dataset encoder checkpoint."""
    output_dir = Path(config.get("paths", {}).get("dataset_encoder_dir", "data/results/dataset_encoders"))
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = load_torch(checkpoint_path, map_location="cpu")
    model = build_dataset_encoder(
        {"dataset_encoder": checkpoint["dataset_encoder_config"]},
        row_dim=int(checkpoint["row_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def train_dataset_encoder(config: dict[str, Any]) -> dict[str, Any]:
    """Train a dataset encoder to predict frozen autoencoder latents."""
    create_output_dirs(config)
    paths = config.get("paths", {})
    encoder_config = config.get("dataset_encoder", {})
    eval_config = config.get("evaluation", {})
    seed = int(config.get("seed", 42))
    device = config.get("device", "cpu")

    autoencoder, autoencoder_checkpoint = load_trained_autoencoder(config)
    autoencoder = autoencoder.to(device)
    _freeze_module(autoencoder)
    latent_dim = _latent_dim_from_checkpoint(autoencoder_checkpoint)

    dataset = load_dataset_latent_pair_dataset(
        processed_dir=paths.get("processed_dir", "data/processed"),
        weight_image_dir=paths.get("weight_image_dir", "data/weight_images"),
        input_split=str(encoder_config.get("input_split", "all")),
        include_labels=bool(encoder_config.get("include_labels", True)),
    )
    train_dataset, val_dataset = train_validation_split(
        dataset,
        train_split=float(encoder_config.get("train_split", 0.8)),
        seed=seed,
    )

    batch_size = int(encoder_config.get("batch_size", 16))
    num_workers = int(encoder_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_dataset_latent_pairs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dataset_latent_pairs,
    )

    model = build_dataset_encoder(config, row_dim=dataset.row_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(encoder_config.get("lr", 1e-3)),
        weight_decay=float(encoder_config.get("weight_decay", 0.0)),
    )

    output_dir = Path(paths.get("dataset_encoder_dir", "data/results/dataset_encoders"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    n_epochs = int(encoder_config.get("epochs", 50))
    progress = tqdm(range(1, n_epochs + 1), desc="Training dataset encoder")
    for epoch in progress:
        train_loss = _run_train_epoch(model, autoencoder, train_loader, optimizer, device)
        val_metrics = evaluate_dataset_encoder(model, autoencoder, val_loader, device=device)
        val_loss = val_metrics["latent_mse"]

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        progress.set_postfix({"train": f"{train_loss:.4g}", "val": f"{val_loss:.4g}"})

    final_metrics = {
        "train": evaluate_dataset_encoder(model, autoencoder, train_loader, device=device),
        "validation": evaluate_dataset_encoder(
            model,
            autoencoder,
            val_loader,
            device=device,
            include_decoded_metrics=True,
        ),
    }
    metadata = {
        "model_type": encoder_config.get("model_type", "deepsets"),
        "input_split": encoder_config.get("input_split", "all"),
        "include_labels": bool(encoder_config.get("include_labels", True)),
        "input_dim": int(dataset.input_dim),
        "row_dim": int(dataset.row_dim),
        "latent_dim": int(latent_dim),
        "n_total": len(dataset),
        "n_train": len(train_dataset),
        "n_validation": len(val_dataset),
        "n_rows_min": min(dataset.n_rows_per_dataset),
        "n_rows_max": max(dataset.n_rows_per_dataset),
        "batch_size": batch_size,
        "epochs": n_epochs,
        "autoencoder_dir": paths.get("autoencoder_dir", "data/results/autoencoders"),
    }

    checkpoint = {
        "state_dict": model.state_dict(),
        "dataset_encoder_config": encoder_config,
        "autoencoder_config": autoencoder_checkpoint.get("autoencoder_config", {}),
        "row_dim": int(dataset.row_dim),
        "latent_dim": int(latent_dim),
        "metadata": metadata,
        "metrics": final_metrics,
    }
    save_torch(checkpoint, output_dir / "checkpoint.pt")
    save_json(history, output_dir / "training_history.json")
    save_json(final_metrics, output_dir / "metrics.json")
    save_json(metadata, output_dir / "metadata.json")

    figure_paths: dict[str, str] = {}
    if bool(eval_config.get("save_reconstructions", True)) and len(val_dataset) > 0:
        n_examples = int(eval_config.get("n_reconstruction_examples", 8))
        plot_format = eval_config.get("plot_format", "png")
        figure_dir = Path(paths.get("figure_dir", "data/results/figures"))
        figure_dir.mkdir(parents=True, exist_ok=True)
        originals, reconstructions = _collect_decoded_examples(
            model,
            autoencoder,
            val_loader,
            device=device,
            n_examples=n_examples,
        )
        figure_path = figure_dir / f"dataset_encoder_reconstructions.{plot_format}"
        plot_reconstruction_grid(
            originals,
            reconstructions,
            figure_path,
            max_images=n_examples,
            title="Dataset-encoder decoded weight images",
        )
        figure_paths["dataset_encoder_reconstructions"] = str(figure_path)

    return {
        "model": model,
        "history": history,
        "metrics": final_metrics,
        "metadata": metadata,
        "figure_paths": figure_paths,
        "output_dir": str(output_dir),
    }
