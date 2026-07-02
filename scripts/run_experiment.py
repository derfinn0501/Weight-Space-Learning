#!/usr/bin/env python3
"""Run the full Weight-Space-Learning pipeline as one tracked experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.experiment_runner import STAGE_ORDER, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=None,
        help="Optional subset of stages to run. Dependencies must already exist for skipped earlier stages.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Run the pipeline without creating an MLflow run, regardless of config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        config_path=args.config,
        stages=args.stages,
        disable_mlflow=args.no_mlflow,
    )
    print(f"Completed stages: {', '.join(results['stages'])}")


if __name__ == "__main__":
    main()
