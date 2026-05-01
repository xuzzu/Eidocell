"""Classification processors: interface for classifiers."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class ClassificationProcessor(ABC):
    """Interface for classifiers trained on extracted features."""

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray, **params) -> None:
        """Train the classifier on labeled features."""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a trained model from disk."""
        ...
