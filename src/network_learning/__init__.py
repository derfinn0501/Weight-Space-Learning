"""Target-network models and training utilities."""

from src.network_learning.models import MLP
from src.network_learning.train_collection import train_model_collection
from src.network_learning.train_single import train_one_model

__all__ = ["MLP", "train_model_collection", "train_one_model"]

