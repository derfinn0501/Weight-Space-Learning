"""Train one target network for every generated dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.network_learning.train_single import train_one_model
from src.utils.io import load_torch, save_json, save_torch


def train_model_collection(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load generated datasets, train one model per dataset, and save checkpoints."""
    paths_cfg = config.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    model_zoo_dir = Path(paths_cfg.get("model_zoo_dir", "data/model_zoo"))
    dataset_dir = processed_dir / "datasets"

    model_dir = model_zoo_dir / "models"
    metrics_dir = model_zoo_dir / "metrics"
    metadata_dir = model_zoo_dir / "metadata"
    for directory in (model_dir, metrics_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(dataset_dir.glob("dataset_*.pt"))
    if not dataset_paths:
        raise FileNotFoundError(
            f"No generated datasets found in {dataset_dir}. "
            "Run scripts/01_generate_datasets.py first."
        )

    device = config.get("device", "cpu")
    network_config = dict(config["network"])
    records: list[dict[str, Any]] = []

    for dataset_path in dataset_paths:
        dataset = load_torch(dataset_path, map_location="cpu")
        dataset_id = dataset["dataset_id"]
        model, metrics, history = train_one_model(dataset, network_config, device=device)

        model_path = model_dir / f"{dataset_id}.pt"
        metrics_path = metrics_dir / f"{dataset_id}.json"
        metadata_path = metadata_dir / f"{dataset_id}.json"

        state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        checkpoint = {
            "dataset_id": dataset_id,
            "state_dict": state_dict,
            "network_config": network_config,
            "dataset_path": str(dataset_path),
            "metrics": metrics,
        }
        save_torch(checkpoint, model_path)

        metrics_payload = {
            "dataset_id": dataset_id,
            **metrics,
            "history": history,
        }
        save_json(metrics_payload, metrics_path)

        metadata = {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "network_config": network_config,
        }
        save_json(metadata, metadata_path)
        records.append({**metadata, **metrics})

    save_json(records, model_zoo_dir / "collection_metadata.json")
    pd.DataFrame(records).to_csv(model_zoo_dir / "collection_metadata.csv", index=False)
    return records

