"""Small IO helpers used across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a path and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(payload: dict[str, Any] | list[Any], path: str | Path) -> None:
    """Save a JSON-serializable object with stable indentation."""
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_torch(payload: Any, path: str | Path) -> None:
    """Save a PyTorch object and ensure the parent directory exists."""
    output_path = ensure_parent(path)
    torch.save(payload, output_path)


def load_torch(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch object from disk."""
    return torch.load(Path(path), map_location=map_location)

