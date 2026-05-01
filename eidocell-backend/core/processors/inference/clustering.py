"""Clustering processors: interface and concrete implementations."""

from abc import ABC, abstractmethod

import numpy as np


class ClusteringProcessor(ABC):
    """Interface for clustering algorithms."""

    @abstractmethod
    def fit_predict(self, features: np.ndarray, n_clusters: int | None = None, **params) -> np.ndarray:
        """Return cluster labels (int array of shape [n_samples])."""
        ...


class KMeansClustering(ClusteringProcessor):
    def fit_predict(
        self,
        features: np.ndarray,
        n_clusters: int | None = 10,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int = 42,
        **_,
    ) -> np.ndarray:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        return kmeans.fit_predict(features)


class EVoCClustering(ClusteringProcessor):
    """Embedding Vector Oriented Clustering.

    Auto-selects cluster count. Works directly on high-dimensional
    feature vectors without requiring prior dimensionality reduction.
    """

    def fit_predict(
        self,
        features: np.ndarray,
        n_clusters: int | None = None,
        *,
        min_cluster_size: int = 5,
        min_samples: int = 5,
        n_neighbors: int = 15,
        neighbor_scale: float = 1.0,
        noise_level: float = 0.5,
        **_,
    ) -> np.ndarray:
        import evoc

        clusterer = evoc.EVoC(
            approx_n_clusters=n_clusters,
            base_min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_neighbors=n_neighbors,
            neighbor_scale=neighbor_scale,
            noise_level=noise_level,
        )
        labels = clusterer.fit_predict(features)

        # EVoC may produce -1 for noise points — reassign them to nearest cluster
        unique_labels = set(labels)
        if -1 in unique_labels and len(unique_labels) > 1:
            from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

            mask = labels != -1
            if mask.sum() > 0:
                knn = KNeighborsClassifier(n_neighbors=min(5, mask.sum()))
                knn.fit(features[mask], labels[mask])
                labels[~mask] = knn.predict(features[~mask])

        # Renumber labels to be contiguous 0..K-1
        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        return np.array([remap[l] for l in labels], dtype=np.intp)


# ── Registry ────────────────────────────────────────────────────────────

CLUSTERING_REGISTRY: dict[str, ClusteringProcessor] = {
    "kmeans": KMeansClustering(),
    "evoc": EVoCClustering(),
}


def get_processor(method: str = "kmeans") -> ClusteringProcessor:
    if method not in CLUSTERING_REGISTRY:
        raise ValueError(f"Unknown clustering method: {method}. Options: {list(CLUSTERING_REGISTRY.keys())}")
    return CLUSTERING_REGISTRY[method]
