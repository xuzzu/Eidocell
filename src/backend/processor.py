# backend/processor.py
import logging
from typing import Any, Dict, Optional

import numpy as np
import umap.umap_ as umap

# import faiss # Switched to sklearn KMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .onnx_processor import ONNXFeatureExtractor
from .pytorch_processor import PyTorchFeatureExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Processor:
    def __init__(
        self,
        model_name: str,
        available_models: dict,
        execution_provider="CPUExecutionProvider",
    ):
        super().__init__()
        self.model_name = model_name
        self.available_models = available_models

        model_info = self.available_models.get(self.model_name)
        if not model_info:
            logging.error(
                f"Selected model '{self.model_name}' not found. Falling back if possible."
            )
            pyt_models = [
                m for m, i in available_models.items() if i.get("type") == "pytorch"
            ]
            onnx_models = [
                m for m, i in available_models.items() if i.get("type") == "onnx"
            ]

            if pyt_models:
                fallback_name = pyt_models[0]
            elif onnx_models:
                fallback_name = onnx_models[0]
            else:
                fallback_name = None

            if fallback_name:
                self.model_name = fallback_name
                model_info = self.available_models[fallback_name]
                logging.warning(f"Falling back to model: {fallback_name}")
            else:
                logging.critical(
                    "No models available or configured for Processor. Feature extraction will fail."
                )
                raise ValueError("No models available for Processor.")

        self.model_type = model_info.get("type", "pytorch")
        self.model_path = model_info.get("path")
        self.feature_dim = model_info.get("dimension")
        self.architecture = model_info.get("architecture")

        if (
            not self.model_path
            or not isinstance(self.feature_dim, int)
            or self.feature_dim <= 0
        ):
            raise ValueError(
                f"Invalid configuration for model '{self.model_name}'. Missing path or valid dimension."
            )

        self.engine: Optional[Any] = None

        try:
            if self.model_type == "pytorch":
                self.engine = PyTorchFeatureExtractor(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    feature_dim=self.feature_dim,
                    architecture=self.architecture,
                )
                logging.info(
                    f"Processor initialized with PyTorch engine for model: {self.model_name}"
                )
            elif self.model_type == "onnx":
                self.engine = ONNXFeatureExtractor(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    feature_dim=self.feature_dim,
                    architecture=self.architecture,
                    execution_provider=execution_provider,
                )
                logging.info(
                    f"Processor initialized with ONNX engine for model: {self.model_name}"
                )
            else:
                raise ValueError(
                    f"Unsupported model type '{self.model_type}' for model '{self.model_name}'."
                )
        except Exception as e:
            logging.error(
                f"Failed to initialize feature extraction engine for {self.model_name} (type: {self.model_type}): {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Engine initialization failed for {self.model_name}"
            ) from e

    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        if not self.engine:
            logging.error(
                f"Feature extraction engine not initialized for model {self.model_name}. Cannot extract features for {image_path}."
            )
            return (
                np.zeros(self.feature_dim, dtype=np.float32)
                if self.feature_dim
                else None
            )

        try:
            if self.model_type == "pytorch":
                feature_tensor = self.engine.extract_features(image_path)
                feature_np = feature_tensor.numpy()
            elif self.model_type == "onnx":
                feature_np = self.engine.extract_features(image_path)
            else:
                logging.error(
                    f"Unknown model type '{self.model_type}' during feature extraction."
                )
                return np.zeros(self.feature_dim, dtype=np.float32)

            if feature_np.shape[0] != self.feature_dim:
                logging.error(
                    f"Final feature dimension mismatch for {self.model_name}. Expected {self.feature_dim}, got {feature_np.shape[0]}. Returning zeros."
                )
                return np.zeros(self.feature_dim, dtype=np.float32)

            return feature_np.astype(np.float32)

        except Exception as e:
            logging.error(
                f"Error in Processor.extract_features for {image_path} with model {self.model_name}: {e}",
                exc_info=True,
            )
            return np.zeros(self.feature_dim, dtype=np.float32)

    def reduce_dimensions_umap(
        self, features: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[0] == 0:
            logging.warning("UMAP: features array is empty.")
            return np.array([])
        n_components = params.get("n_components", 2)
        if features.shape[0] <= n_components:
            logging.warning(
                f"UMAP: Not enough samples ({features.shape[0]}) for n_components={n_components}. Returning original features."
            )
            return features
        n_neighbors = params.get("n_neighbors", 15)
        if n_neighbors >= features.shape[0]:
            n_neighbors = max(2, features.shape[0] - 1)
            logging.warning(
                f"UMAP: n_neighbors ({params.get('n_neighbors', 15)}) was >= n_samples ({features.shape[0]}). Adjusted to {n_neighbors}."
            )
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=params.get("min_dist", 0.1),
                n_components=n_components,
                metric=params.get("metric", "euclidean"),
                random_state=42,
            )
            return reducer.fit_transform(features)
        except Exception as e:
            logging.error(f"Error during UMAP: {e}", exc_info=True)
            return features

    def reduce_dimensions_pca(
        self, features: np.ndarray, n_components=0.95
    ) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[0] == 0:
            logging.warning("PCA: features array is empty.")
            return np.array([])
        if (
            features.shape[0] == 1
            and isinstance(n_components, float)
            and n_components <= 1.0
        ):
            logging.warning(
                f"PCA: Cannot compute variance-based PCA on a single sample. Returning original features."
            )
            return features
        if (
            features.shape[0] < 2
            and isinstance(n_components, float)
            and n_components <= 1.0
        ):
            logging.warning(
                f"PCA: n_components={n_components} (variance explained) requires >= 2 samples. Found {features.shape[0]}. Returning original features."
            )
            return features
        actual_n_components = n_components
        if isinstance(n_components, int):
            max_possible_components = min(features.shape)
            if n_components > max_possible_components:
                actual_n_components = max_possible_components
                logging.warning(
                    f"PCA: Requested n_components {n_components} > min(n_samples, n_features) ({max_possible_components}). Using {actual_n_components}."
                )
        try:
            reducer = PCA(n_components=actual_n_components, random_state=42)
            return reducer.fit_transform(features)
        except Exception as e:
            logging.error(f"Error during PCA: {e}", exc_info=True)
            return features

    def cluster_images(
        self,
        features: np.ndarray,
        n_clusters: int = 10,
        n_iter: int = 100,  # Sklearn KMeans uses `max_iter`
        n_redo: int = 3,  # Sklearn KMeans uses `n_init`
        find_k_elbow: bool = False,
    ) -> list:
        num_samples = features.shape[0]
        logging.info(
            "Processor: Using scikit-learn KMeans for clustering."
        )  # MODIFICATION

        if num_samples == 0:
            logging.warning("Processor: Cannot cluster. Features array is empty.")
            return []
        if features.ndim == 1:
            features = features.reshape(1, -1)
            num_samples = 1

        original_n_clusters_requested = n_clusters

        if num_samples < n_clusters:
            logging.warning(
                f"Processor: Number of samples ({num_samples}) is less than initially requested n_clusters ({n_clusters}). "
                f"Adjusting n_clusters to {num_samples} for scikit-learn KMeans."
            )
            n_clusters = num_samples

        if n_clusters <= 0:
            if num_samples > 0:
                logging.warning(
                    f"Processor: n_clusters became <= 0 ({n_clusters}) after initial adjustment with {num_samples} samples. Setting n_clusters to 1."
                )
                n_clusters = 1
            else:
                logging.warning(
                    "Processor: No samples and n_clusters is 0. Returning empty labels."
                )
                return []

        # Elbow method (not used based on logs, but keep placeholder)
        if find_k_elbow:
            # Implement or call elbow method logic here if needed.
            # For now, this path is not taken.
            logging.info(
                "Processor: Elbow method for K selection is not currently active."
            )
            pass

        try:
            # Parameters for sklearn.cluster.KMeans:
            # n_clusters: The number of clusters to form as well as the number of centroids to generate.
            # init: Method for initialization, defaults to ‘k-means++’.
            # n_init: Number of times the k-means algorithm will be run with different centroid seeds.
            #         The final results will be the best output of n_init consecutive runs in terms of inertia.
            #         Default is 10. Let's use `n_redo` from UI, or default to 10 if n_redo is small.
            # max_iter: Maximum number of iterations of the k-means algorithm for a single run. Default is 300.
            #           Let's use `n_iter` from UI.
            # random_state: Determines random number generation for centroid initialization.

            # Adjust n_init based on n_redo. Sklearn's default is 10. If n_redo is small, use 10.
            # If n_redo is large, it might be too slow. For now, let's use n_redo directly,
            # but cap it or ensure it's reasonable.
            # A common value for n_init in sklearn is 10. Let's use the `n_redo` from UI,
            # but ensure it's at least 1. `n_redo` from UI is 10 in logs.

            effective_n_init = max(1, n_redo)  # Ensure at least 1 run
            if (
                effective_n_init == 1 and n_clusters > 1
            ):  # k-means++ might be slow for n_init=1, but let's try
                logging.info(
                    f"Processor: scikit-learn KMeans using n_init={effective_n_init} (from n_redo={n_redo})."
                )

            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=effective_n_init,
                max_iter=n_iter,
                random_state=42,
                # algorithm="lloyd", # Default, faster than elkan for many cases
            )
            logging.info(
                f"Processor: Fitting sklearn KMeans with k={n_clusters}, max_iter={n_iter}, n_init={effective_n_init}..."
            )
            kmeans.fit(
                features.astype(np.float64)
            )  # KMeans prefers float64 for stability
            cluster_labels = kmeans.labels_
            logging.info("Processor: sklearn KMeans fitting complete.")
            return cluster_labels.tolist()
        except Exception as e:
            logging.error(f"Error during scikit-learn K-means: {e}", exc_info=True)
            return [0] * num_samples

    def split_cluster(
        self,
        features: np.ndarray,
        n_clusters: int = 2,  # Number of sub-clusters
        n_iter: int = 100,
        n_redo: int = 3,
    ) -> list:
        num_samples = features.shape[0]
        logging.info("Processor (split): Using scikit-learn KMeans for splitting.")

        if num_samples == 0:
            logging.warning(
                "Processor (split): Cannot split cluster. Features array is empty."
            )
            return []
        if features.ndim == 1:
            features = features.reshape(1, -1)
            num_samples = 1

        if num_samples < n_clusters:
            logging.warning(
                f"Processor (split): Samples ({num_samples}) < n_sub_clusters ({n_clusters}). Adjusting n_sub_clusters to {num_samples}."
            )
            n_clusters = num_samples

        if n_clusters <= 0:
            if num_samples > 0:
                logging.warning(
                    f"Processor (split): n_sub_clusters became <= 0. Setting to 1."
                )
                n_clusters = 1
            else:
                return []

        try:
            effective_n_init = max(1, n_redo)
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=effective_n_init,
                max_iter=n_iter,
                random_state=42,
            )
            logging.info(
                f"Processor (split): Fitting sklearn KMeans with k={n_clusters}, max_iter={n_iter}, n_init={effective_n_init}..."
            )
            kmeans.fit(features.astype(np.float64))
            new_labels = kmeans.labels_
            logging.info(
                "Processor (split): sklearn KMeans fitting complete for split."
            )
            return new_labels.tolist()
        except Exception as e:
            logging.error(
                f"Error during scikit-learn K-means for split: {e}", exc_info=True
            )
            return []
