from __future__ import annotations

import torch

from src.evaluation.metrics import mae, mse, relative_l2_error


def test_reconstruction_metrics_simple_values() -> None:
    prediction = torch.tensor([1.0, 3.0])
    target = torch.tensor([1.0, 1.0])

    assert mse(prediction, target) == 2.0
    assert mae(prediction, target) == 1.0
    assert abs(relative_l2_error(prediction, target) - (2.0 / (2.0**0.5))) < 1e-6

