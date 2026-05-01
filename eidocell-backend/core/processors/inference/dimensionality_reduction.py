"""Dimensionality reduction processors: interface and concrete implementations."""

from abc import ABC, abstractmethod

import numpy as np

from core.processors.errors import UnknownProcessorError


class DimensionalityReductionProcessor(ABC):
    """Interface for dimensionality reduction algorithms."""

    @abstractmethod
    def fit_transform(self, features: np.ndarray, **params) -> np.ndarray:
        """Reduce dimensionality of feature matrix. Returns (n_samples, n_components)."""
        ...


class PCAReduction(DimensionalityReductionProcessor):
    def fit_transform(
        self,
        features: np.ndarray,
        n_components: int | float = 2,
        **_,
    ) -> np.ndarray:
        from sklearn.decomposition import PCA

        # Clamp integer n_components to available features
        if isinstance(n_components, int):
            n_components = min(n_components, features.shape[1], features.shape[0])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(features)


class UMAPReduction(DimensionalityReductionProcessor):
    def fit_transform(
        self,
        features: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 10,
        metric: str = "euclidean",
        **_,
    ) -> np.ndarray:
        import umap.umap_ as umap

        n_neighbors = min(n_neighbors, len(features) - 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
        )
        return reducer.fit_transform(features)


# ── Registry ────────────────────────────────────────────────────────────

DR_REGISTRY: dict[str, DimensionalityReductionProcessor] = {
    "pca": PCAReduction(),
    "umap": UMAPReduction(),
}


def get_processor(method: str) -> DimensionalityReductionProcessor:
    if method not in DR_REGISTRY:
        raise UnknownProcessorError(
            f"Unknown DR method: {method}. Options: {list(DR_REGISTRY.keys())}"
        )
    return DR_REGISTRY[method]
