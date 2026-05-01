"""Feature extraction processors: interface and concrete implementations.

Concrete DL-based extractors (ONNX, PyTorch) will be added as needed.
"""

from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractionProcessor(ABC):
    """Interface for feature extraction from images."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a 1-D feature vector from a single image."""
        ...

    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        ...


class MorphologicalFeatureExtraction(FeatureExtractionProcessor):
    """Extract features from pre-computed mask attributes.

    This is a passthrough that converts a mask attribute dict
    into a fixed-order numeric vector.
    """

    # Ordered list of attribute keys used as features
    FEATURE_KEYS = [
        "area", 
        "perimeter", 
        "equivalent_diameter",
        "major_axis_length", 
        "minor_axis_length", 
        "aspect_ratio",
        "eccentricity", 
        "form_factor", 
        "compactness", 
        "solidity",
        "convexity", 
        "mean_intensity", 
        "std_intensity",
        "thickness_mean", 
        "gradient_rms",
    ]

    def extract(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MorphologicalFeatureExtraction operates on pre-computed attributes, "
            "not raw images. Use extract_from_attributes() instead."
        )

    def extract_from_attributes(self, attrs: dict) -> np.ndarray:
        return np.array([attrs.get(k, 0.0) for k in self.FEATURE_KEYS], dtype=np.float32)

    def feature_dim(self) -> int:
        return len(self.FEATURE_KEYS)


# ── Registry ────────────────────────────────────────────────────────────


def _build_registry() -> dict[str, FeatureExtractionProcessor]:
    registry: dict[str, FeatureExtractionProcessor] = {
        "morphological": MorphologicalFeatureExtraction(),
    }
    try:
        from core.processors.inference.deep_feature_extraction import MobileNetV3Extraction
        registry["mobilenetv3"] = MobileNetV3Extraction()
    except ImportError:
        pass  # torch not installed
    return registry


FEATURE_EXTRACTION_REGISTRY: dict[str, FeatureExtractionProcessor] = _build_registry()


def get_processor(method: str) -> FeatureExtractionProcessor:
    if method not in FEATURE_EXTRACTION_REGISTRY:
        raise ValueError(
            f"Unknown feature extraction method: {method}. "
            f"Options: {list(FEATURE_EXTRACTION_REGISTRY.keys())}"
        )
    return FEATURE_EXTRACTION_REGISTRY[method]
