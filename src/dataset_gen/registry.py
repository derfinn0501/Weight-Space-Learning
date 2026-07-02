"""Registry for synthetic dataset generators."""

from __future__ import annotations

from src.dataset_gen.generators import make_blobs_dataset, make_moons_dataset


DATASET_REGISTRY = {
    "moons": make_moons_dataset,
    "blobs": make_blobs_dataset,
}


def get_dataset_generator(name: str):
    """Return a dataset generator by registry name."""
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset generator '{name}'. Available: {available}") from exc

