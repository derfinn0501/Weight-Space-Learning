from __future__ import annotations

import torch

from src.image_gen.image_layouts import build_weight_image, w1_h_w2_layout
from src.image_gen.weight_extraction import extract_parameters_from_state_dict
from src.network_learning.models import MLP


def test_weight_image_shape_and_metadata() -> None:
    model = MLP(input_dim=2, output_dim=2, hidden_layers=[4], activation="relu")
    parameters = extract_parameters_from_state_dict(model.state_dict(), include_bias=True)
    image, metadata = build_weight_image(parameters, representation="block_matrix")

    assert isinstance(image, torch.Tensor)
    assert image.ndim == 2
    assert metadata["representation"] == "block_matrix"
    assert metadata["image_shape"] == list(image.shape)
    assert len(metadata["blocks"]) == len(parameters)


def test_w1_h_w2_layout_shape_and_metadata() -> None:
    W1 = torch.randn(4, 2)
    h = torch.randn(4)
    W2 = torch.randn(2, 4)
    image, metadata = w1_h_w2_layout(W1, h, W2)

    assert image.shape == (4, 7)
    assert metadata["representation"] == "w1_h_w2"
    assert [block["name"] for block in metadata["blocks"]] == ["W1", "h", "W2_transposed"]

