"""Synthetic dataset generation package."""

from src.dataset_gen.dataset_collection import generate_dataset_collection
from src.dataset_gen.generators import make_blobs_dataset, make_moons_dataset

__all__ = [
    "generate_dataset_collection",
    "make_blobs_dataset",
    "make_moons_dataset",
]

