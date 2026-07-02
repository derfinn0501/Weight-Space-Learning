"""Small plotting helpers for generated datasets and weight images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_synthetic_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    output_path: str | Path,
    title: str = "Synthetic dataset",
) -> None:
    """Save a scatter plot for the first two input dimensions."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), s=12, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weight_image(
    image: torch.Tensor,
    output_path: str | Path,
    title: str = "Weight image",
) -> None:
    """Save a plot of one 2D weight image tensor."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    im = ax.imshow(image.detach().cpu(), aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

