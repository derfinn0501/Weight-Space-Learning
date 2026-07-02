#!/usr/bin/env python3
"""Convert trained target-network weights into deterministic image tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.image_gen.image_layouts import append_input_data_block, build_weight_image
from src.image_gen.normalization import normalize_image
from src.image_gen.weight_extraction import extract_parameters_from_state_dict
from src.utils.config import load_config
from src.utils.io import load_torch, save_json, save_torch
from src.utils.paths import create_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def _prepare_dirs(weight_image_dir: Path) -> dict[str, Path]:
    directories = {
        "images": weight_image_dir / "images",
        "layouts": weight_image_dir / "layouts",
        "raw_weights": weight_image_dir / "raw_weights",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def generate_weight_images(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate one weight-image tensor for every trained target model."""
    paths_cfg = config.get("paths", {})
    model_zoo_dir = Path(paths_cfg.get("model_zoo_dir", "data/model_zoo"))
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    weight_image_dir = Path(paths_cfg.get("weight_image_dir", "data/weight_images"))
    dirs = _prepare_dirs(weight_image_dir)

    image_cfg = config.get("image", {})
    representation = image_cfg.get("representation", "block_matrix")
    include_bias = bool(image_cfg.get("include_bias", True))
    include_input_data = bool(image_cfg.get("include_input_data", False))
    include_labels = bool(image_cfg.get("include_labels", True))
    normalize = image_cfg.get("normalize", "none")
    block_gap = int(image_cfg.get("block_gap", 1))

    model_paths = sorted((model_zoo_dir / "models").glob("dataset_*.pt"))
    if not model_paths:
        raise FileNotFoundError(
            f"No trained models found in {model_zoo_dir / 'models'}. "
            "Run scripts/02_train_target_networks.py first."
        )

    records: list[dict[str, Any]] = []
    for model_path in model_paths:
        checkpoint = load_torch(model_path, map_location="cpu")
        dataset_id = checkpoint.get("dataset_id", model_path.stem)
        parameters = extract_parameters_from_state_dict(checkpoint, include_bias=include_bias)
        image, layout_metadata = build_weight_image(
            parameters,
            representation=representation,
            block_gap=block_gap,
        )

        if include_input_data:
            dataset = load_torch(processed_dir / "datasets" / f"{dataset_id}.pt", map_location="cpu")
            image, layout_metadata = append_input_data_block(
                image,
                layout_metadata,
                X_train=dataset["X_train"],
                y_train=dataset["y_train"],
                include_labels=include_labels,
                block_gap=block_gap,
            )

        image, normalization_metadata = normalize_image(image, method=normalize)
        layout_metadata["dataset_id"] = dataset_id
        layout_metadata["source_model_path"] = str(model_path)
        layout_metadata["include_bias"] = include_bias
        layout_metadata["include_input_data"] = include_input_data
        layout_metadata["normalization"] = normalization_metadata

        image_path = dirs["images"] / f"{dataset_id}.pt"
        layout_path = dirs["layouts"] / f"{dataset_id}.json"
        raw_weight_path = dirs["raw_weights"] / f"{dataset_id}.pt"

        save_torch(
            {
                "dataset_id": dataset_id,
                "image": image,
                "layout": layout_metadata,
                "normalization": normalization_metadata,
            },
            image_path,
        )
        save_json(layout_metadata, layout_path)
        save_torch(
            {
                "dataset_id": dataset_id,
                "parameters": parameters,
                "include_bias": include_bias,
            },
            raw_weight_path,
        )

        records.append(
            {
                "dataset_id": dataset_id,
                "model_path": str(model_path),
                "image_path": str(image_path),
                "layout_path": str(layout_path),
                "raw_weight_path": str(raw_weight_path),
                "image_height": int(image.shape[0]),
                "image_width": int(image.shape[1]),
                "representation": layout_metadata["representation"],
                "normalization": normalize,
                "include_bias": include_bias,
                "include_input_data": include_input_data,
            }
        )

    save_json(records, weight_image_dir / "metadata.json")
    pd.DataFrame(records).to_csv(weight_image_dir / "metadata.csv", index=False)
    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    create_output_dirs(config)
    records = generate_weight_images(config)
    print(f"Generated {len(records)} weight images in {config['paths']['weight_image_dir']}/images")


if __name__ == "__main__":
    main()

