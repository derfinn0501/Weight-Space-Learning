#!/usr/bin/env python3
"""Minimal autoencoder training script for generated weight images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cond_AE.ae_models import FullyConnectedAE, SmallConvAE
from src.cond_AE.datasets import load_weight_image_dataset
from src.cond_AE.train_ae import train_autoencoder
from src.utils.config import load_config
from src.utils.io import save_json, save_torch
from src.utils.paths import create_output_dirs
from src.utils.seed import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_random_seed(int(config.get("seed", 42)))
    create_output_dirs(config)

    dataset = load_weight_image_dataset(config["paths"]["weight_image_dir"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["autoencoder"].get("batch_size", 32)),
        shuffle=True,
    )

    sample_image = dataset[0]["image"]
    latent_dim = int(config["autoencoder"].get("latent_dim", 32))
    model_type = config["autoencoder"].get("model_type", "cae")
    if model_type == "ae":
        model = FullyConnectedAE(input_dim=int(sample_image.numel()), latent_dim=latent_dim)
    elif model_type == "cae":
        model = SmallConvAE(latent_dim=latent_dim)
    else:
        raise ValueError("autoencoder.model_type must be 'ae' or 'cae'.")

    model, history = train_autoencoder(
        model,
        loader,
        epochs=int(config["autoencoder"].get("epochs", 50)),
        lr=float(config["autoencoder"].get("lr", 1e-3)),
        device=config.get("device", "cpu"),
    )

    out_dir = Path(config["paths"]["result_dir"]) / "autoencoder"
    save_torch({"state_dict": model.state_dict(), "config": config["autoencoder"]}, out_dir / "checkpoint.pt")
    save_json(history, out_dir / "training_history.json")
    print(f"Saved autoencoder outputs to {out_dir}")


if __name__ == "__main__":
    main()

