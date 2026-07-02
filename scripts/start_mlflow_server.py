#!/usr/bin/env python3
"""Start a local MLflow server using paths from the experiment config."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def main() -> None:
    args = parse_args()
    config_path = _resolve_repo_path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    paths = config.get("paths", {})
    backend_store = os.environ.get(
        "MLFLOW_BACKEND_STORE",
        paths.get("mlflow_backend_store", "data/mlflow/mlflow.db"),
    )
    artifact_root = os.environ.get(
        "MLFLOW_ARTIFACT_ROOT",
        paths.get("mlflow_artifact_dir", "data/mlflow/artifacts"),
    )
    host = args.host or os.environ.get("MLFLOW_HOST", "127.0.0.1")
    port = str(args.port or os.environ.get("MLFLOW_PORT", "5000"))

    backend_path = _resolve_repo_path(str(backend_store))
    artifact_path = _resolve_repo_path(str(artifact_root))
    backend_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)

    command = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        port,
        "--backend-store-uri",
        f"sqlite:///{backend_path}",
        "--default-artifact-root",
        str(artifact_path),
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
