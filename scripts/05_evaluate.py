#!/usr/bin/env python3
"""Create a few lightweight diagnostic plots from generated artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.plots import plot_synthetic_dataset, plot_weight_image
from src.utils.config import load_config
from src.utils.io import load_torch
from src.utils.paths import create_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    create_output_dirs(config)

    processed_dir = Path(config["paths"]["processed_dir"])
    weight_image_dir = Path(config["paths"]["weight_image_dir"])
    figure_dir = Path(config["paths"]["result_dir"]) / "figures"

    dataset_paths = sorted((processed_dir / "datasets").glob("dataset_*.pt"))
    if dataset_paths:
        dataset = load_torch(dataset_paths[0], map_location="cpu")
        plot_synthetic_dataset(
            dataset["X_train"],
            dataset["y_train"],
            figure_dir / "first_dataset.png",
            title=dataset["dataset_id"],
        )

    image_paths = sorted((weight_image_dir / "images").glob("dataset_*.pt"))
    if image_paths:
        payload = load_torch(image_paths[0], map_location="cpu")
        plot_weight_image(
            payload["image"],
            figure_dir / "first_weight_image.png",
            title=payload["dataset_id"],
        )

    print(f"Saved available diagnostic plots to {figure_dir}")


if __name__ == "__main__":
    main()

