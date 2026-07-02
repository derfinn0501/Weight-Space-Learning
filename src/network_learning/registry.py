"""Registry for target-network classes."""

from __future__ import annotations

from src.network_learning.models import MLP


MODEL_REGISTRY = {
    "mlp": MLP,
}


def get_model_class(name: str):
    """Return a target-network class by registry name."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model type '{name}'. Available: {available}") from exc

