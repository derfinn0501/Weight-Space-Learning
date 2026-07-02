"""Synthetic classification dataset generators."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TensorDatasetTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _split_and_convert(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_test: int,
    random_state: int | None,
    standardize: bool,
) -> TensorDatasetTuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=n_train,
        test_size=n_test,
        random_state=random_state,
        stratify=y,
    )

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return (
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long),
    )


def make_moons_dataset(
    n_train: int,
    n_test: int,
    input_dim: int = 2,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: int | None = None,
    standardize: bool = True,
    **_: Any,
) -> TensorDatasetTuple:
    """Generate one train/test split from the two-moons dataset."""
    if n_classes != 2:
        raise ValueError("make_moons_dataset supports n_classes=2.")
    if input_dim < 2:
        raise ValueError("make_moons_dataset requires input_dim >= 2.")

    n_total = n_train + n_test
    X, y = make_moons(n_samples=n_total, noise=noise, random_state=random_state)

    if input_dim > 2:
        rng = np.random.default_rng(random_state)
        extra_dims = rng.normal(size=(n_total, input_dim - 2))
        X = np.concatenate([X, extra_dims], axis=1)

    return _split_and_convert(X, y, n_train, n_test, random_state, standardize)


def make_blobs_dataset(
    n_train: int,
    n_test: int,
    input_dim: int = 2,
    n_classes: int = 2,
    centers: int | list[list[float]] | None = None,
    cluster_std: float | list[float] = 1.0,
    random_state: int | None = None,
    standardize: bool = True,
    **_: Any,
) -> TensorDatasetTuple:
    """Generate one train/test split from Gaussian blobs."""
    if centers is None:
        centers = n_classes

    if isinstance(centers, int) and centers != n_classes:
        raise ValueError("For integer centers, centers must match n_classes.")
    if isinstance(centers, list) and len(centers) != n_classes:
        raise ValueError("For explicit centers, len(centers) must match n_classes.")

    n_total = n_train + n_test
    X, y = make_blobs(
        n_samples=n_total,
        n_features=input_dim,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    return _split_and_convert(X, y, n_train, n_test, random_state, standardize)

