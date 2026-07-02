"""Generation and storage of meta-dataset collections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset_gen.registry import get_dataset_generator
from src.utils.io import save_json, save_torch


def generate_dataset_collection(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate M synthetic datasets and save each one as a .pt file."""
    data_cfg = config["data"]
    paths_cfg = config.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    dataset_dir = processed_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    generator_name = data_cfg.get("generator", "moons")
    generator = get_dataset_generator(generator_name)
    n_datasets = int(data_cfg.get("M", 1))
    random_state_start = int(data_cfg.get("random_state_start", 0))

    generator_kwargs = {
        key: value
        for key, value in data_cfg.items()
        if key not in {"generator", "M", "N_train", "N_test", "random_state_start"}
    }

    records: list[dict[str, Any]] = []
    for index in range(n_datasets):
        dataset_id = f"dataset_{index:05d}"
        random_state = random_state_start + index
        X_train, y_train, X_test, y_test = generator(
            n_train=int(data_cfg["N_train"]),
            n_test=int(data_cfg["N_test"]),
            random_state=random_state,
            **generator_kwargs,
        )

        dataset_path = dataset_dir / f"{dataset_id}.pt"
        metadata = {
            "dataset_id": dataset_id,
            "generator": generator_name,
            "random_state": random_state,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "input_dim": int(X_train.shape[1]),
            "n_classes": int(data_cfg.get("n_classes", int(y_train.max().item()) + 1)),
            "dataset_path": str(dataset_path),
            "generator_parameters": generator_kwargs,
        }

        save_torch(
            {
                "dataset_id": dataset_id,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "metadata": metadata,
            },
            dataset_path,
        )
        records.append(metadata)

    save_json(records, processed_dir / "dataset_metadata.json")
    csv_records = [
        {
            **record,
            "generator_parameters": json.dumps(record["generator_parameters"], sort_keys=True),
        }
        for record in records
    ]
    pd.DataFrame(csv_records).to_csv(processed_dir / "dataset_metadata.csv", index=False)
    return records

