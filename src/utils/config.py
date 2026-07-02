"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and resolve runtime-only values."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config["device"] = resolve_device(config.get("device", "auto"))
    return config


def resolve_device(device: str | torch.device) -> str:
    """Resolve the configured device string to a concrete PyTorch device name."""
    if isinstance(device, torch.device):
        return str(device)

    if str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return str(device)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary and return the updated dictionary."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

