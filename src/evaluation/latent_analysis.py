"""Latent embedding extraction for trained autoencoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.io import save_torch


def _batch_ids(batch: Any) -> list[str]:
    if isinstance(batch, dict) and "dataset_id" in batch:
        dataset_ids = batch["dataset_id"]
        if isinstance(dataset_ids, str):
            return [dataset_ids]
        return [str(dataset_id) for dataset_id in dataset_ids]
    return []


def extract_latent_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Run the autoencoder encoder over all weight images."""
    if not hasattr(model, "encode"):
        raise ValueError("The model must expose an encode(x) method.")

    model.eval()
    embeddings: list[torch.Tensor] = []
    dataset_ids: list[str] = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
            latent = model.encode(images).detach().cpu()
            embeddings.append(latent)
            dataset_ids.extend(_batch_ids(batch))

    return torch.cat(embeddings, dim=0), dataset_ids


def save_latent_embeddings(
    model: nn.Module,
    data_loader: DataLoader,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    basename: str = "latent_embeddings",
) -> dict[str, str]:
    """Extract latent embeddings and save them as .pt and .csv files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings, dataset_ids = extract_latent_embeddings(model, data_loader, device=device)

    pt_path = output_path / f"{basename}.pt"
    csv_path = output_path / f"{basename}.csv"
    save_torch({"embeddings": embeddings, "dataset_ids": dataset_ids}, pt_path)

    rows = []
    for index, vector in enumerate(embeddings):
        row = {
            "dataset_id": dataset_ids[index] if index < len(dataset_ids) else f"image_{index:05d}",
        }
        row.update({f"z_{dim:03d}": float(value) for dim, value in enumerate(vector.tolist())})
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {"pt_path": str(pt_path), "csv_path": str(csv_path)}

