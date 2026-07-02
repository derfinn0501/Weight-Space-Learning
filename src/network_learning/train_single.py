"""Train one target network on one generated dataset."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import classification_accuracy
from src.network_learning.registry import get_model_class


def _evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            batch_size = int(y_batch.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y_batch).sum().item())
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train_one_model(
    dataset: dict[str, Any],
    network_config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, float], dict[str, list[float]]]:
    """Train one configured target model and return the model, metrics, and history."""
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    if int(network_config["input_dim"]) != int(X_train.shape[1]):
        raise ValueError(
            f"network.input_dim={network_config['input_dim']} does not match "
            f"dataset input_dim={X_train.shape[1]}."
        )

    observed_classes = int(max(y_train.max().item(), y_test.max().item()) + 1)
    if int(network_config["output_dim"]) < observed_classes:
        raise ValueError(
            f"network.output_dim={network_config['output_dim']} is smaller than "
            f"the observed number of classes={observed_classes}."
        )

    model_class = get_model_class(network_config.get("model_type", "mlp"))
    model = model_class(
        input_dim=int(network_config["input_dim"]),
        output_dim=int(network_config["output_dim"]),
        hidden_layers=network_config.get("hidden_layers", [32, 32]),
        activation=network_config.get("activation", "relu"),
    ).to(device)

    batch_size = int(network_config.get("batch_size", 64))
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    eval_train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(network_config.get("lr", 1e-3)),
        weight_decay=float(network_config.get("weight_decay", 0.0)),
    )

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(int(network_config.get("epochs", 50))):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = _evaluate(model, eval_train_loader, criterion, device)
        test_loss, test_accuracy = _evaluate(model, test_loader, criterion, device)
        history["epoch"].append(float(epoch + 1))
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

    train_loss, train_accuracy = _evaluate(model, eval_train_loader, criterion, device)
    test_loss, test_accuracy = _evaluate(model, test_loader, criterion, device)
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    return model, metrics, history

