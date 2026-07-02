#!/usr/bin/env python3
"""Evaluate trained AE/CAE reconstructions and latent embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cond_AE.datasets import load_weight_image_dataset
from src.cond_AE.train_ae import evaluate_autoencoder, load_trained_autoencoder
from src.evaluation.latent_analysis import save_latent_embeddings
from src.evaluation.plots import plot_reconstruction_error_heatmap, plot_reconstruction_grid
from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.paths import create_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    create_output_dirs(config)

    paths = config.get("paths", {})
    eval_config = config.get("evaluation", {})
    device = config.get("device", "cpu")
    plot_format = eval_config.get("plot_format", "png")
    figure_dir = Path(paths.get("figure_dir", "data/results/figures"))
    metric_dir = Path(paths.get("metric_dir", "data/results/metrics"))
    metric_dir.mkdir(parents=True, exist_ok=True)

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
    metrics_payload = {
        "reconstruction": metrics,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
    save_json(metrics_payload, metric_dir / "reconstruction_metrics.json")

    if bool(eval_config.get("save_reconstructions", True)):
        n_examples = int(eval_config.get("n_reconstruction_examples", 8))
        originals, reconstructions = _collect_examples(model, loader, device, n_examples=n_examples)
        plot_reconstruction_grid(
            originals,
            reconstructions,
            figure_dir / f"reconstructions.{plot_format}",
            max_images=n_examples,
        )
        plot_reconstruction_error_heatmap(
            originals[0],
            reconstructions[0],
            figure_dir / f"reconstruction_error_heatmap.{plot_format}",
        )

    if bool(eval_config.get("save_latent_embeddings", True)):
        paths_written = save_latent_embeddings(
            model,
            loader,
            metric_dir,
            device=device,
        )
        metrics_payload["latent_embeddings"] = paths_written
        save_json(metrics_payload, metric_dir / "reconstruction_metrics.json")

    print(f"Saved reconstruction metrics to {metric_dir / 'reconstruction_metrics.json'}")


if __name__ == "__main__":
    main()

