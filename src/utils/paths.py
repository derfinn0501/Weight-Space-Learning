"""Path creation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_output_dirs(config: dict[str, Any]) -> None:
    """Create the standard output directories used by the pipeline."""
    paths = config.get("paths", {})
    required = [
        paths.get("data_root", "data"),
        paths.get("raw_dir", "data/raw"),
        paths.get("processed_dir", "data/processed"),
        paths.get("model_zoo_dir", "data/model_zoo"),
        paths.get("weight_image_dir", "data/weight_images"),
        paths.get("result_dir", "data/results"),
        paths.get("autoencoder_dir", "data/results/autoencoders"),
        paths.get("figure_dir", "data/results/figures"),
        paths.get("metric_dir", "data/results/metrics"),
    ]

    for path in required:
        Path(path).mkdir(parents=True, exist_ok=True)
