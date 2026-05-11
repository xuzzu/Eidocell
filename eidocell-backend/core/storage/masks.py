"""Per-sample, per-channel binary mask IO. One PNG per (sample, channel) under sessions/{sid}/masks/."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MASKS_SUBDIR = "masks"


def masks_dir(session_folder: str | Path) -> Path:
    return Path(session_folder) / MASKS_SUBDIR


def mask_path(session_folder: str | Path, sample_id: str, channel_index: int = 0) -> Path:
    return masks_dir(session_folder) / f"{sample_id}_ch{int(channel_index)}.png"


def write_mask(
    session_folder: str | Path,
    sample_id: str,
    mask: np.ndarray,
    channel_index: int = 0,
) -> Path:
    """Write a binary or label mask as PNG. Values >0 are written as 255 for binary."""
    out = mask_path(session_folder, sample_id, channel_index)
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


def read_mask(
    session_folder: str | Path,
    sample_id: str,
    channel_index: int = 0,
) -> np.ndarray | None:
    p = mask_path(session_folder, sample_id, channel_index)
    if not p.is_file():
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def delete_mask(
    session_folder: str | Path,
    sample_id: str,
    channel_index: int = 0,
) -> bool:
    p = mask_path(session_folder, sample_id, channel_index)
    if p.is_file():
        p.unlink()
        return True
    return False


def list_mask_channels(session_folder: str | Path, sample_id: str) -> list[int]:
    """Return sorted channel indices for which a mask file exists on disk."""
    d = masks_dir(session_folder)
    if not d.is_dir():
        return []
    prefix = f"{sample_id}_ch"
    out: list[int] = []
    for p in d.iterdir():
        if not p.is_file() or not p.name.startswith(prefix) or not p.name.endswith(".png"):
            continue
        stem = p.name[len(prefix):-len(".png")]
        try:
            out.append(int(stem))
        except ValueError:
            continue
    return sorted(out)


def delete_all_masks_for_sample(session_folder: str | Path, sample_id: str) -> int:
    """Delete every mask file belonging to a sample (across all channels)."""
    d = masks_dir(session_folder)
    if not d.is_dir():
        return 0
    prefix = f"{sample_id}_ch"
    count = 0
    for p in list(d.iterdir()):
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(".png"):
            p.unlink()
            count += 1
    return count
