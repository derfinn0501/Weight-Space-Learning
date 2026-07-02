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


def _to_2d(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    if image.ndim == 3:
        return image.squeeze(0)
    return image


def plot_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: str | Path,
    max_images: int = 8,
    title: str = "Weight-image reconstructions",
) -> None:
    """Save a grid with original, reconstruction, and absolute error columns."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_examples = min(int(max_images), int(originals.shape[0]))
    fig, axes = plt.subplots(
        n_examples,
        3,
        figsize=(9, max(2.0, 2.0 * n_examples)),
        dpi=150,
        squeeze=False,
    )
    column_titles = ["original", "reconstruction", "absolute error"]
    for col, column_title in enumerate(column_titles):
        axes[0, col].set_title(column_title)

    for row in range(n_examples):
        original = _to_2d(originals[row])
        reconstruction = _to_2d(reconstructions[row])
        error = (reconstruction - original).abs()
        images = [original, reconstruction, error]
        cmaps = ["coolwarm", "coolwarm", "magma"]
        for col, (image, cmap) in enumerate(zip(images, cmaps, strict=True)):
            axes[row, col].imshow(image, aspect="auto", interpolation="nearest", cmap=cmap)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reconstruction_error_heatmap(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    output_path: str | Path,
    title: str = "Reconstruction absolute error",
) -> None:
    """Save a heatmap of absolute reconstruction error for one example."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    error = (_to_2d(reconstruction) - _to_2d(original)).abs()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    im = ax.imshow(error, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curve(
    history: dict[str, list[float]],
    output_path: str | Path,
    title: str = "Autoencoder training loss",
) -> None:
    """Save a train/validation loss curve."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", [])
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="validation")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
