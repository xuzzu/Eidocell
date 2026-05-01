"""Segmentation processors: interface and concrete implementations."""

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np

from core.processors.image_utils import to_gray


class SegmentationProcessor(ABC):
    """Interface for segmentation algorithms."""

    @abstractmethod
    def segment(self, image: np.ndarray, **params) -> np.ndarray:
        """Produce a binary mask (0/255) from an input image."""
        ...

    @staticmethod
    def method_info() -> dict:
        """Return method metadata: id, name, and parameter definitions."""
        raise NotImplementedError


class OtsuIntensitySegmentation(SegmentationProcessor):
    def segment(
        self,
        image: np.ndarray,
        distance_from_center: int = 30,
        min_component_size: int = 15,
        **_,
    ) -> np.ndarray:
        gray = to_gray(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = _filter_by_center_distance(binary, distance_from_center / 100.0)
        binary = _remove_small_components(binary, min_component_size)
        return binary

    @staticmethod
    def method_info() -> dict:
        return {
            "id": "otsu_intensity",
            "name": "Otsu (Intensity)",
            "params": [
                {"name": "distance_from_center", "label": "Distance from center %",
                 "min": 0, "max": 100, "default": 30, "step": 1},
                {"name": "min_component_size", "label": "Min component size",
                 "min": 10, "max": 100, "default": 15, "step": 1},
            ],
        }


class OtsuEdgesSegmentation(SegmentationProcessor):
    def segment(
        self,
        image: np.ndarray,
        distance_from_center: int = 30,
        min_component_size: int = 15,
        close_radius: int = 5,
        **_,
    ) -> np.ndarray:
        gray = to_gray(image)
        p2, p98 = np.percentile(gray, (2, 98))
        stretched = np.clip(
            (gray.astype(float) - p2) / max(p98 - p2, 1) * 255, 0, 255
        ).astype(np.uint8)
        sx = cv2.Scharr(stretched, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(stretched, cv2.CV_64F, 0, 1)
        edges = np.sqrt(sx**2 + sy**2)
        edges = (
            (edges / edges.max() * 255).astype(np.uint8)
            if edges.max() > 0
            else edges.astype(np.uint8)
        )
        _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = _fill_holes(binary)
        binary = _filter_by_center_distance(binary, distance_from_center / 100.0)
        binary = _remove_small_components(binary, min_component_size)
        return binary

    @staticmethod
    def method_info() -> dict:
        return {
            "id": "otsu_edges",
            "name": "Otsu (Edges)",
            "params": [
                {"name": "distance_from_center", "label": "Distance from center %",
                 "min": 0, "max": 100, "default": 30, "step": 1},
                {"name": "min_component_size", "label": "Min component size",
                 "min": 10, "max": 100, "default": 15, "step": 1},
                {"name": "close_radius", "label": "Close radius",
                 "min": 1, "max": 20, "default": 5, "step": 1},
            ],
        }


class AdaptiveSegmentation(SegmentationProcessor):
    def segment(
        self,
        image: np.ndarray,
        block_size: int = 35,
        c_value: int = 2,
        **_,
    ) -> np.ndarray:
        gray = to_gray(image)
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_value,
        )
        return binary

    @staticmethod
    def method_info() -> dict:
        return {
            "id": "adaptive",
            "name": "Adaptive Thresholding",
            "params": [
                {"name": "block_size", "label": "Block size (odd)",
                 "min": 3, "max": 101, "default": 35, "step": 2},
                {"name": "c_value", "label": "C value",
                 "min": -100, "max": 100, "default": 2, "step": 1},
            ],
        }


class WatershedSegmentation(SegmentationProcessor):
    def segment(
        self,
        image: np.ndarray,
        foreground_threshold: int = 70,
        morph_kernel_size: int = 3,
        **_,
    ) -> np.ndarray:
        gray = to_gray(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_size = max(3, morph_kernel_size | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        thresh = dist.max() * (foreground_threshold / 100.0)
        _, sure_fg = cv2.threshold(dist, thresh, 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        img_color = (
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if len(image.shape) < 3
            else image.copy()
        )
        cv2.watershed(img_color, markers)

        result = np.zeros_like(gray)
        result[markers > 1] = 255
        return result

    @staticmethod
    def method_info() -> dict:
        return {
            "id": "watershed",
            "name": "Watershed",
            "params": [
                {"name": "foreground_threshold", "label": "Foreground threshold %",
                 "min": 0, "max": 100, "default": 70, "step": 1},
                {"name": "morph_kernel_size", "label": "Morph kernel size (odd)",
                 "min": 1, "max": 11, "default": 3, "step": 2},
            ],
        }


# ── Registry ────────────────────────────────────────────────────────────

SEGMENTATION_REGISTRY: dict[str, SegmentationProcessor] = {
    "otsu_intensity": OtsuIntensitySegmentation(),
    "otsu_edges": OtsuEdgesSegmentation(),
    "adaptive": AdaptiveSegmentation(),
    "watershed": WatershedSegmentation(),
}


def get_processor(method: str) -> SegmentationProcessor:
    if method not in SEGMENTATION_REGISTRY:
        raise ValueError(f"Unknown segmentation method: {method}. Options: {list(SEGMENTATION_REGISTRY.keys())}")
    return SEGMENTATION_REGISTRY[method]


def list_methods() -> list[dict]:
    return [proc.method_info() for proc in SEGMENTATION_REGISTRY.values()]


# ── Shared helpers ──────────────────────────────────────────────────────



def _filter_by_center_distance(binary: np.ndarray, max_ratio: float) -> np.ndarray:
    h, w = binary.shape
    cx, cy = w / 2, h / 2
    max_dist = math.sqrt(cx**2 + cy**2) * max_ratio
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)
    for i in range(1, n_labels):
        comp_cx, comp_cy = centroids[i]
        dist = math.sqrt((comp_cx - cx) ** 2 + (comp_cy - cy) ** 2)
        if dist <= max_dist:
            result[labels == i] = 255
    return result


def _remove_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255
    return result


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    return binary | cv2.bitwise_not(flood)
