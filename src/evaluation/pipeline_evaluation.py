"""End-to-end evaluation helpers for trained autoencoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.cond_AE.datasets import load_weight_image_dataset
from src.cond_AE.train_ae import evaluate_autoencoder, load_trained_autoencoder
from src.evaluation.latent_analysis import save_latent_embeddings
from src.evaluation.plots import plot_reconstruction_error_heatmap, plot_reconstruction_grid
from src.utils.io import save_json


def _collect_examples(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    n_examples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    originals: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            reconstructed = model(images)
            originals.append(images.detach().cpu())
            reconstructions.append(reconstructed.detach().cpu())
            if sum(tensor.shape[0] for tensor in originals) >= n_examples:
                break

    original_tensor = torch.cat(originals, dim=0)[:n_examples]
    reconstruction_tensor = torch.cat(reconstructions, dim=0)[:n_examples]
    return original_tensor, reconstruction_tensor


def evaluate_trained_autoencoder(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the configured trained AE/CAE and save metrics/plots."""
    paths = config.get("paths", {})
    eval_config = config.get("evaluation", {})
    device = config.get("device", "cpu")
    plot_format = eval_config.get("plot_format", "png")
    figure_dir = Path(paths.get("figure_dir", "data/results/figures"))
    metric_dir = Path(paths.get("metric_dir", "data/results/metrics"))
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_trained_autoencoder(config)
    model = model.to(device)
    dataset = load_weight_image_dataset(paths.get("weight_image_dir", "data/weight_images"))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("autoencoder", {}).get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config.get("autoencoder", {}).get("num_workers", 0)),
    )

    metrics = evaluate_autoencoder(model, loader, device=device)
    metrics_path = metric_dir / "reconstruction_metrics.json"
    metrics_payload: dict[str, Any] = {
        "reconstruction": metrics,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
    save_json(metrics_payload, metrics_path)

    figure_paths: dict[str, str] = {}
    if bool(eval_config.get("save_reconstructions", True)):
        n_examples = int(eval_config.get("n_reconstruction_examples", 8))
        originals, reconstructions = _collect_examples(model, loader, device, n_examples=n_examples)

        grid_path = figure_dir / f"reconstructions.{plot_format}"
        heatmap_path = figure_dir / f"reconstruction_error_heatmap.{plot_format}"
        plot_reconstruction_grid(
            originals,
            reconstructions,
            grid_path,
            max_images=n_examples,
        )
        plot_reconstruction_error_heatmap(
            originals[0],
            reconstructions[0],
            heatmap_path,
        )
        figure_paths["reconstructions"] = str(grid_path)
        figure_paths["reconstruction_error_heatmap"] = str(heatmap_path)

    latent_paths: dict[str, str] = {}
    if bool(eval_config.get("save_latent_embeddings", True)):
        latent_paths = save_latent_embeddings(
            model,
            loader,
            metric_dir,
            device=device,
        )
        metrics_payload["latent_embeddings"] = latent_paths
        save_json(metrics_payload, metrics_path)

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "figure_paths": figure_paths,
        "latent_paths": latent_paths,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
