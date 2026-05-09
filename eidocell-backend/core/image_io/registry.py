"""Image format adapter registry.

Today: cv2-based reader handles png/jpg/bmp/tiff. New formats (CZI, LIF,
OME-TIFF, etc.) register their own readers via ``register_reader``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# A reader returns (image_array, channel_meta).
# `channel_meta` is None for legacy single-channel-set images; future multi-
# channel readers populate it with per-channel descriptors.
ReaderFn = Callable[[Path], "tuple[np.ndarray | None, dict | None]"]

_READERS: dict[str, ReaderFn] = {}


def register_reader(extension: str, fn: ReaderFn) -> None:
    """Register a reader for a file extension (case-insensitive, must include leading dot)."""
    if not extension.startswith("."):
        raise ValueError("extension must start with '.'")
    _READERS[extension.lower()] = fn


def supported_extensions() -> set[str]:
    return set(_READERS.keys())


def read(path: str | Path) -> "tuple[np.ndarray | None, dict | None]":
    """Read an image via the registered reader for its suffix."""
    p = Path(path)
    fn = _READERS.get(p.suffix.lower())
    if fn is None:
        return None, None
    return fn(p)


def _cv2_reader(path: Path) -> "tuple[np.ndarray | None, dict | None]":
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    return img, None


for _ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
    register_reader(_ext, _cv2_reader)
