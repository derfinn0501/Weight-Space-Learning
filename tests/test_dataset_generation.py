from __future__ import annotations

import torch

from src.dataset_gen.generators import make_blobs_dataset, make_moons_dataset


def test_moons_dataset_shapes() -> None:
    X_train, y_train, X_test, y_test = make_moons_dataset(
        n_train=16,
        n_test=20,
        input_dim=2,
        n_classes=2,
        noise=0.1,
        random_state=0,
    )

    assert X_train.shape == (16, 2)
    assert y_train.shape == (16,)
    assert X_test.shape == (20, 2)
    assert y_test.shape == (20,)
    assert X_train.dtype == torch.float32
    assert y_train.dtype == torch.long


def test_blobs_dataset_shapes() -> None:
    X_train, y_train, X_test, y_test = make_blobs_dataset(
        n_train=16,
        n_test=20,
        input_dim=3,
        n_classes=2,
        centers=2,
        cluster_std=1.0,
        random_state=0,
    )

    assert X_train.shape == (16, 3)
    assert y_train.shape == (16,)
    assert X_test.shape == (20, 3)
    assert y_test.shape == (20,)

