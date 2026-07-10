"""Deterministic 2D layouts for neural-network weight tensors."""

from __future__ import annotations

import copy
from typing import Any

import torch


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a parameter tensor into a 2D block."""
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def block_matrix_layout(
    parameters: list[dict[str, Any]],
    block_gap: int = 1,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build a structured image by placing each parameter as a padded 2D block.

    Weight matrices keep their natural row/column shape. Bias vectors are column
    blocks. All blocks are concatenated left-to-right and vertically centered in
    a common canvas; metadata records the exact image region for each block.
    """
    if not parameters:
        raise ValueError("Cannot build a block_matrix image from an empty parameter list.")
    if block_gap < 0:
        raise ValueError("block_gap must be non-negative.")

    matrices = [_as_matrix(parameter["tensor"]) for parameter in parameters]
    max_height = max(int(matrix.shape[0]) for matrix in matrices)
    total_width = sum(int(matrix.shape[1]) for matrix in matrices)
    total_width += block_gap * (len(matrices) - 1)

    image = torch.full((max_height, total_width), float(fill_value), dtype=torch.float32)
    blocks: list[dict[str, Any]] = []

    left = 0
    for parameter, matrix in zip(parameters, matrices, strict=True):
        height, width = int(matrix.shape[0]), int(matrix.shape[1])
        top = (max_height - height) // 2
        image[top : top + height, left : left + width] = matrix
        blocks.append(
            {
                "parameter_index": int(parameter["index"]),
                "name": parameter["name"],
                "kind": parameter["kind"],
                "original_shape": list(parameter["shape"]),
                "matrix_shape": [height, width],
                "top": top,
                "left": left,
                "height": height,
                "width": width,
            }
        )
        left += width + block_gap

    metadata = {
        "representation": "block_matrix",
        "image_shape": list(image.shape),
        "block_gap": block_gap,
        "fill_value": fill_value,
        "blocks": blocks,
    }
    return image, metadata


def build_weight_image(
    parameters: list[dict[str, Any]],
    representation: str = "block_matrix",
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a weight image using a named representation."""
    if representation != "block_matrix":
        raise ValueError("Only representation='block_matrix' is implemented.")
    return block_matrix_layout(parameters, **kwargs)


def w1_h_w2_layout(
    W1: torch.Tensor,
    hidden_activation: torch.Tensor,
    W2: torch.Tensor,
    block_gap: int = 1,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build the professor-style image [W1 | h(x) | W2^T] for one hidden-layer MLPs.

    PyTorch stores Linear weights as [out_features, in_features]. For a network
    input -> hidden -> output, W1 has shape [hidden_dim, input_dim] and W2 has
    shape [output_dim, hidden_dim]. W2 is transposed so every block shares the
    hidden-unit row axis.
    """
    if block_gap < 0:
        raise ValueError("block_gap must be non-negative.")

    W1 = W1.detach().cpu().float()
    h = hidden_activation.detach().cpu().float().reshape(-1, 1)
    W2_t = W2.detach().cpu().float().T

    hidden_dim = int(W1.shape[0])
    if h.shape[0] != hidden_dim:
        raise ValueError(f"h has height {h.shape[0]}, expected hidden_dim={hidden_dim}.")
    if W2_t.shape[0] != hidden_dim:
        raise ValueError(f"W2.T has height {W2_t.shape[0]}, expected hidden_dim={hidden_dim}.")

    blocks = [
        ("W1", W1),
        ("h", h),
        ("W2_transposed", W2_t),
    ]
    total_width = sum(int(block.shape[1]) for _, block in blocks) + block_gap * (len(blocks) - 1)
    image = torch.full((hidden_dim, total_width), float(fill_value), dtype=torch.float32)

    metadata_blocks: list[dict[str, Any]] = []
    left = 0
    for name, block in blocks:
        height, width = int(block.shape[0]), int(block.shape[1])
        image[:, left : left + width] = block
        metadata_blocks.append(
            {
                "name": name,
                "top": 0,
                "left": left,
                "height": height,
                "width": width,
                "matrix_shape": [height, width],
            }
        )
        left += width + block_gap

    metadata = {
        "representation": "w1_h_w2",
        "image_shape": list(image.shape),
        "block_gap": block_gap,
        "fill_value": fill_value,
        "row_axis": "hidden_units",
        "blocks": metadata_blocks,
        "notes": "Canvas is [W1 | h(x) | W2^T] for a one-hidden-layer MLP.",
    }
    return image, metadata


def append_input_data_block(
    weight_image: torch.Tensor,
    layout_metadata: dict[str, Any],
    X_train: torch.Tensor,
    y_train: torch.Tensor | None = None,
    include_labels: bool = True,
    block_gap: int = 1,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Concatenate the training inputs as a deterministic block to the right.

    The data block is X_train with an optional final class-label column. It is
    stored in the layout metadata so downstream code can separate it from the
    trained-weight region.
    """
    data_block = X_train.detach().cpu().float()
    label_column_added = False
    if include_labels and y_train is not None:
        label_column = y_train.detach().cpu().float().reshape(-1, 1)
        data_block = torch.cat([data_block, label_column], dim=1)
        label_column_added = True

    weight_height, weight_width = int(weight_image.shape[0]), int(weight_image.shape[1])
    data_height, data_width = int(data_block.shape[0]), int(data_block.shape[1])
    composite_height = max(weight_height, data_height)
    composite_width = weight_width + block_gap + data_width

    composite = torch.full(
        (composite_height, composite_width),
        float(fill_value),
        dtype=torch.float32,
    )
    composite[:weight_height, :weight_width] = weight_image
    input_left = weight_width + block_gap
    composite[:data_height, input_left : input_left + data_width] = data_block

    metadata = {
        "representation": "block_matrix_with_input_data",
        "image_shape": list(composite.shape),
        "block_gap": block_gap,
        "fill_value": fill_value,
        "components": {
            "weight_image": {
                "top": 0,
                "left": 0,
                "height": weight_height,
                "width": weight_width,
            },
            "input_data": {
                "top": 0,
                "left": input_left,
                "height": data_height,
                "width": data_width,
                "columns": "X_train columns followed by y_train label column"
                if label_column_added
                else "X_train columns",
                "label_column_added": label_column_added,
            },
        },
        "weight_layout": copy.deepcopy(layout_metadata),
    }
    return composite, metadata
