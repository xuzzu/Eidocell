"""Per-sample binary mask IO. One PNG per sample under sessions/{sid}/masks/."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MASKS_SUBDIR = "masks"


def masks_dir(session_folder: str | Path) -> Path:
    return Path(session_folder) / MASKS_SUBDIR


def mask_path(session_folder: str | Path, sample_id: str) -> Path:
    return masks_dir(session_folder) / f"{sample_id}.png"


def write_mask(session_folder: str | Path, sample_id: str, mask: np.ndarray) -> Path:
    """Write a binary or label mask as PNG. Values >0 are written as 255 for binary."""
    out = mask_path(session_folder, sample_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask)
    if arr.dtype == bool:
        arr = (arr.astype(np.uint8) * 255)
    elif arr.dtype != np.uint8:
        if arr.max() <= 1:
            arr = (arr.astype(np.uint8) * 255)
        else:
            arr = arr.astype(np.uint8)
    cv2.imwrite(str(out), arr)
    return out


def read_mask(session_folder: str | Path, sample_id: str) -> np.ndarray | None:
    p = mask_path(session_folder, sample_id)
    if not p.is_file():
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def delete_mask(session_folder: str | Path, sample_id: str) -> bool:
    p = mask_path(session_folder, sample_id)
    if p.is_file():
        p.unlink()
        return True
    return False
