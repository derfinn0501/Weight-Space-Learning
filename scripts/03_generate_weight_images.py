#!/usr/bin/env python3
"""Convert trained target-network weights into deterministic image tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.image_gen.generate_collection import generate_weight_images
from src.utils.config import load_config
from src.utils.paths import create_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    create_output_dirs(config)
    records = generate_weight_images(config)
    print(f"Generated {len(records)} weight images in {config['paths']['weight_image_dir']}/images")


if __name__ == "__main__":
    main()
