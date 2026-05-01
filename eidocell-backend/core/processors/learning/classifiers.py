"""Concrete classifier implementations: KNN and Linear Probe."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from core.processors.inference.classification import ClassificationProcessor


class KNNClassifier(ClassificationProcessor):
    """K-Nearest Neighbors classifier."""

    def __init__(self):
        self.model_: KNeighborsClassifier | None = None

    def train(self, features: np.ndarray, labels: np.ndarray, **params) -> None:
        n_neighbors = params.get("n_neighbors", 5)
        # Clamp n_neighbors to number of samples
        n_neighbors = min(n_neighbors, len(features))
        self.model_ = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model_.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not trained yet")
        return self.model_.predict(features)

    def save(self, path: Path) -> None:
        joblib.dump({"classifier": self.model_}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self.model_ = data["classifier"]


class LinearProbeClassifier(ClassificationProcessor):
    """Logistic Regression linear probe classifier."""

    def __init__(self):
        self.model_: LogisticRegression | None = None

    def train(self, features: np.ndarray, labels: np.ndarray, **params) -> None:
        max_iter = params.get("max_iter", 1000)
        C = params.get("C", 1.0)
        self.model_ = LogisticRegression(
            max_iter=max_iter, C=C, solver="lbfgs"
        )
        self.model_.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not trained yet")
        return self.model_.predict(features)

    def save(self, path: Path) -> None:
        joblib.dump({"classifier": self.model_}, path)

    def load(self, path: Path) -> None:
        data = joblib.load(path)
        self.model_ = data["classifier"]


# ── Registry ────────────────────────────────────────────────────────────

CLASSIFIER_REGISTRY: dict[str, type[ClassificationProcessor]] = {
    "knn": KNNClassifier,
    "linear_probe": LinearProbeClassifier,
}


def get_classifier(method: str) -> ClassificationProcessor:
    cls = CLASSIFIER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown classifier: {method}. Options: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    return cls()
