#!/usr/bin/env python3
"""Train one target network per generated dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.network_learning.train_collection import train_model_collection
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
    records = train_model_collection(config)
    print(f"Trained {len(records)} target networks in {config['paths']['model_zoo_dir']}/models")


if __name__ == "__main__":
    main()

