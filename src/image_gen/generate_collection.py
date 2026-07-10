"""Generate deterministic weight-image tensors from trained model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from src.image_gen.image_layouts import append_input_data_block, build_weight_image, w1_h_w2_layout
from src.image_gen.normalization import normalize_image
from src.image_gen.weight_extraction import extract_parameters_from_state_dict
from src.utils.io import load_torch, save_json, save_torch


def _prepare_dirs(weight_image_dir: Path) -> dict[str, Path]:
    directories = {
        "images": weight_image_dir / "images",
        "layouts": weight_image_dir / "layouts",
        "raw_weights": weight_image_dir / "raw_weights",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _hidden_activation(pre_activation: torch.Tensor, activation: str) -> torch.Tensor:
    activation = activation.lower()
    if activation == "relu":
        return torch.relu(pre_activation)
    if activation == "tanh":
        return torch.tanh(pre_activation)
    if activation == "gelu":
        return F.gelu(pre_activation)
    raise ValueError("activation must be one of: relu, tanh, gelu")


def _single_hidden_layer_parameters(checkpoint: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    weight_names = sorted(name for name in state_dict if name.endswith(".weight"))
    if len(weight_names) != 2:
        raise ValueError(
            "representation='w1_h_w2' requires a one-hidden-layer MLP with exactly "
            f"two Linear weight tensors. Found {len(weight_names)} weights: {weight_names}"
        )

    first_base = weight_names[0].rsplit(".", 1)[0]
    second_base = weight_names[1].rsplit(".", 1)[0]
    W1 = state_dict[f"{first_base}.weight"].detach().cpu().float()
    b1 = state_dict.get(f"{first_base}.bias")
    W2 = state_dict[f"{second_base}.weight"].detach().cpu().float()
    b2 = state_dict.get(f"{second_base}.bias")

    b1 = b1.detach().cpu().float() if isinstance(b1, torch.Tensor) else torch.zeros(W1.shape[0])
    b2 = b2.detach().cpu().float() if isinstance(b2, torch.Tensor) else None
    return W1, b1, W2, b2


def _generate_w1_h_w2_images(
    *,
    checkpoint: dict[str, Any],
    dataset: dict[str, Any],
    model_path: Path,
    dirs: dict[str, Path],
    image_cfg: dict[str, Any],
    normalize: str,
) -> list[dict[str, Any]]:
    dataset_id = checkpoint.get("dataset_id", model_path.stem)
    block_gap = int(image_cfg.get("block_gap", 1))
    activation = checkpoint.get("network_config", {}).get("activation", "relu")
    input_split = image_cfg.get("activation_input_split", "train")
    if input_split != "train":
        raise ValueError("Only image.activation_input_split='train' is implemented for w1_h_w2.")

    X = dataset["X_train"]
    y = dataset["y_train"]
    W1, b1, W2, b2 = _single_hidden_layer_parameters(checkpoint)
    raw_weight_path = dirs["raw_weights"] / f"{dataset_id}.pt"
    save_torch(
        {
            "dataset_id": dataset_id,
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "representation": "w1_h_w2",
        },
        raw_weight_path,
    )

    records: list[dict[str, Any]] = []
    for sample_index, (x_i, y_i) in enumerate(zip(X, y, strict=True)):
        pre_activation = W1 @ x_i.detach().cpu().float() + b1
        h_i = _hidden_activation(pre_activation, activation=activation)
        image, layout_metadata = w1_h_w2_layout(W1, h_i, W2, block_gap=block_gap)
        image, normalization_metadata = normalize_image(image, method=normalize)

        image_id = f"{dataset_id}_sample_{sample_index:05d}"
        image_path = dirs["images"] / f"{image_id}.pt"
        layout_path = dirs["layouts"] / f"{image_id}.json"
        logits = W2 @ h_i
        if b2 is not None:
            logits = logits + b2

        layout_metadata["dataset_id"] = dataset_id
        layout_metadata["image_id"] = image_id
        layout_metadata["sample_index"] = sample_index
        layout_metadata["source_model_path"] = str(model_path)
        layout_metadata["activation"] = activation
        layout_metadata["normalization"] = normalization_metadata

        save_torch(
            {
                "dataset_id": dataset_id,
                "image_id": image_id,
                "sample_index": sample_index,
                "image": image,
                "layout": layout_metadata,
                "normalization": normalization_metadata,
                "x": x_i.detach().cpu().float(),
                "y": y_i.detach().cpu().long(),
                "hidden_activation": h_i,
                "logits": logits,
            },
            image_path,
        )
        save_json(layout_metadata, layout_path)
        records.append(
            {
                "dataset_id": dataset_id,
                "image_id": image_id,
                "sample_index": sample_index,
                "model_path": str(model_path),
                "image_path": str(image_path),
                "layout_path": str(layout_path),
                "raw_weight_path": str(raw_weight_path),
                "image_height": int(image.shape[0]),
                "image_width": int(image.shape[1]),
                "representation": layout_metadata["representation"],
                "normalization": normalize,
                "activation_input_split": input_split,
            }
        )

    return records


def generate_weight_images(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate weight-image tensors from every trained target model."""
    paths_cfg = config.get("paths", {})
    model_zoo_dir = Path(paths_cfg.get("model_zoo_dir", "data/model_zoo"))
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    weight_image_dir = Path(paths_cfg.get("weight_image_dir", "data/weight_images"))
    dirs = _prepare_dirs(weight_image_dir)

    image_cfg = config.get("image", {})
    representation = image_cfg.get("representation", "block_matrix")
    include_bias = bool(image_cfg.get("include_bias", True))
    include_input_data = bool(image_cfg.get("include_input_data", False))
    include_labels = bool(image_cfg.get("include_labels", True))
    normalize = image_cfg.get("normalize", "none")
    block_gap = int(image_cfg.get("block_gap", 1))

    model_paths = sorted((model_zoo_dir / "models").glob("dataset_*.pt"))
    if not model_paths:
        raise FileNotFoundError(
            f"No trained models found in {model_zoo_dir / 'models'}. "
            "Run scripts/02_train_target_networks.py first."
        )

    records: list[dict[str, Any]] = []
    for model_path in model_paths:
        checkpoint = load_torch(model_path, map_location="cpu")
        dataset_id = checkpoint.get("dataset_id", model_path.stem)

        if representation == "w1_h_w2":
            dataset = load_torch(processed_dir / "datasets" / f"{dataset_id}.pt", map_location="cpu")
            records.extend(
                _generate_w1_h_w2_images(
                    checkpoint=checkpoint,
                    dataset=dataset,
                    model_path=model_path,
                    dirs=dirs,
                    image_cfg=image_cfg,
                    normalize=normalize,
                )
            )
            continue

        parameters = extract_parameters_from_state_dict(checkpoint, include_bias=include_bias)
        image, layout_metadata = build_weight_image(
            parameters,
            representation=representation,
            block_gap=block_gap,
        )

        if include_input_data:
            dataset = load_torch(processed_dir / "datasets" / f"{dataset_id}.pt", map_location="cpu")
            image, layout_metadata = append_input_data_block(
                image,
                layout_metadata,
                X_train=dataset["X_train"],
                y_train=dataset["y_train"],
                include_labels=include_labels,
                block_gap=block_gap,
            )

        image, normalization_metadata = normalize_image(image, method=normalize)
        layout_metadata["dataset_id"] = dataset_id
        layout_metadata["source_model_path"] = str(model_path)
        layout_metadata["include_bias"] = include_bias
        layout_metadata["include_input_data"] = include_input_data
        layout_metadata["normalization"] = normalization_metadata

        image_path = dirs["images"] / f"{dataset_id}.pt"
        layout_path = dirs["layouts"] / f"{dataset_id}.json"
        raw_weight_path = dirs["raw_weights"] / f"{dataset_id}.pt"

        save_torch(
            {
                "dataset_id": dataset_id,
                "image": image,
                "layout": layout_metadata,
                "normalization": normalization_metadata,
            },
            image_path,
        )
        save_json(layout_metadata, layout_path)
        save_torch(
            {
                "dataset_id": dataset_id,
                "parameters": parameters,
                "include_bias": include_bias,
            },
            raw_weight_path,
        )

        records.append(
            {
                "dataset_id": dataset_id,
                "model_path": str(model_path),
                "image_path": str(image_path),
                "layout_path": str(layout_path),
                "raw_weight_path": str(raw_weight_path),
                "image_height": int(image.shape[0]),
                "image_width": int(image.shape[1]),
                "representation": layout_metadata["representation"],
                "normalization": normalize,
                "include_bias": include_bias,
                "include_input_data": include_input_data,
            }
        )

    save_json(records, weight_image_dir / "metadata.json")
    pd.DataFrame(records).to_csv(weight_image_dir / "metadata.csv", index=False)
    return records
