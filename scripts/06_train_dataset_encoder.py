#!/usr/bin/env python3
"""Train a dataset encoder into the frozen autoencoder latent space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_encoder.train_encoder import train_dataset_encoder
from src.evaluation.plots import plot_loss_curve
from src.utils.config import load_config
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

    result = train_dataset_encoder(config)
    plot_format = config.get("evaluation", {}).get("plot_format", "png")
    figure_dir = Path(config["paths"].get("figure_dir", "data/results/figures"))
    plot_loss_curve(
        result["history"],
        figure_dir / f"dataset_encoder_loss_curve.{plot_format}",
        title="Dataset encoder training loss",
    )

    print(f"Saved dataset encoder outputs to {result['output_dir']}")


if __name__ == "__main__":
    main()
