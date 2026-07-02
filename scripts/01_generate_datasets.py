#!/usr/bin/env python3
"""Generate a collection of synthetic train/test datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_gen.dataset_collection import generate_dataset_collection
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
    records = generate_dataset_collection(config)
    print(f"Generated {len(records)} datasets in {config['paths']['processed_dir']}/datasets")


if __name__ == "__main__":
    main()

