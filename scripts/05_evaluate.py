#!/usr/bin/env python3
"""Evaluate trained AE/CAE reconstructions and latent embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.pipeline_evaluation import evaluate_trained_autoencoder
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
    result = evaluate_trained_autoencoder(config)
    print(f"Saved reconstruction metrics to {result['metrics_path']}")


if __name__ == "__main__":
    main()
