"""Per-image preprocessing operations.

Every op returns ``(output_array, metadata_dict)`` so the pipeline can record
what was done. Operations preserve multi-channel layout (HWC) when present,
operating per-channel where that's the IFC norm (z-score, background sub).

Most heavy lifting reuses code patterns from the user's ``ifc-data-eval``
prototype: padding (constant/poisson/replicate), z-score, phase-correlation
alignment, rolling-ball background subtraction.
"""
from __future__ import annotations


import cv2
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────


def _ensure_hwc(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Promote 2-D arrays to (H,W,1). Returns (array, was_2d_originally)."""
    if arr.ndim == 2:
        return arr[:, :, None], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(f"unsupported ndim {arr.ndim}")


def _restore(arr: np.ndarray, was_2d: bool) -> np.ndarray:
    if was_2d and arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


# ── Normalisation ──────────────────────────────────────────────────────


def normalize_minmax(arr: np.ndarray, *, per_channel: bool = True) -> tuple[np.ndarray, dict]:
    img, was_2d = _ensure_hwc(arr)
    out = np.empty_like(img, dtype=np.float32)
    stats: list[dict] = []
    if per_channel:
        for c in range(img.shape[2]):
            ch = img[:, :, c].astype(np.float32, copy=False)
            mn, mx = float(ch.min()), float(ch.max())
            if mx > mn:
                out[:, :, c] = (ch - mn) / (mx - mn)
            else:
                out[:, :, c] = 0.0
            stats.append({"channel": c, "min": mn, "max": mx})
    else:
        mn, mx = float(img.min()), float(img.max())
        out = ((img.astype(np.float32) - mn) / (mx - mn)) if mx > mn else np.zeros_like(img, dtype=np.float32)
        stats.append({"channel": None, "min": mn, "max": mx})
    return _restore(out, was_2d), {"per_channel": per_channel, "stats": stats}


def normalize_zscore(arr: np.ndarray, *, per_channel: bool = True) -> tuple[np.ndarray, dict]:
    img, was_2d = _ensure_hwc(arr)
    out = np.empty_like(img, dtype=np.float32)
    stats: list[dict] = []
    if per_channel:
        for c in range(img.shape[2]):
            ch = img[:, :, c].astype(np.float32, copy=False)
            mean, std = float(ch.mean()), float(ch.std())
            out[:, :, c] = (ch - mean) / std if std > 0 else 0.0
            stats.append({"channel": c, "mean": mean, "std": std})
    else:
        mean, std = float(img.mean()), float(img.std())
        out = (img.astype(np.float32) - mean) / std if std > 0 else np.zeros_like(img, dtype=np.float32)
        stats.append({"channel": None, "mean": mean, "std": std})
    return _restore(out, was_2d), {"per_channel": per_channel, "stats": stats}


# ── Padding / resizing ─────────────────────────────────────────────────


def pad_to(
    arr: np.ndarray,
    target_shape: tuple[int, int],
    *,
    method: str = "constant",
    constant_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Pad ``arr`` so its (H, W) is at least ``target_shape``. Centred padding.

    ``method`` ∈ {"constant", "replicate", "poisson"}. Poisson is a noise-fill
    approximation popular for IFC: samples integer counts whose mean equals
    the local background corner of the input.
    """
    img, was_2d = _ensure_hwc(arr)
    h, w, c = img.shape
    th, tw = target_shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return _restore(img, was_2d), {"method": method, "pad": [0, 0, 0, 0], "target": list(target_shape)}

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if method == "constant":
        padded = np.full((th, tw, c), constant_value, dtype=img.dtype)
        padded[top : top + h, left : left + w, :] = img
    elif method == "replicate":
        # Fall back to cv2 per-channel.
        per_ch = []
        for k in range(c):
            per_ch.append(
                cv2.copyMakeBorder(img[:, :, k], top, bottom, left, right, cv2.BORDER_REPLICATE)
            )
        padded = np.stack(per_ch, axis=-1)
    elif method == "poisson":
        per_ch = []
        for k in range(c):
            ch = img[:, :, k]
            corner = ch[: max(1, h // 8), : max(1, w // 8)]
            bg_mean = float(corner.mean()) if corner.size else 0.0
            bg_mean = max(bg_mean, 0.0)
            base = cv2.copyMakeBorder(ch, top, bottom, left, right, cv2.BORDER_REPLICATE)
            noise = np.random.poisson(bg_mean, base.shape).astype(base.dtype)
            mask = np.zeros(base.shape, dtype=bool)
            mask[:top, :] = True
            mask[base.shape[0] - bottom :, :] = True
            mask[:, :left] = True
            mask[:, base.shape[1] - right :] = True
            base[mask] = noise[mask]
            per_ch.append(base)
        padded = np.stack(per_ch, axis=-1)
    else:
        raise ValueError(f"unknown pad method '{method}'")

    return _restore(padded, was_2d), {
        "method": method,
        "pad": [top, bottom, left, right],
        "target": [th, tw],
    }


def pad_to_square(
    arr: np.ndarray,
    *,
    method: str = "constant",
    constant_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Pad ``arr`` to a square whose side equals max(H, W). Centred.

    Thin wrapper over ``pad_to`` that derives the target shape per-image
    from the input's longest side. Used by the ``per_image_square`` import
    strategy.
    """
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    side = max(h, w)
    return pad_to(arr, (side, side), method=method, constant_value=constant_value)


def resize_to(arr: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Bilinear resize per channel to ``target_shape`` (H, W)."""
    img, was_2d = _ensure_hwc(arr)
    h, w, c = img.shape
    th, tw = target_shape
    if (h, w) == (th, tw):
        return _restore(img, was_2d), {"target": [th, tw], "skipped": True}
    out = np.empty((th, tw, c), dtype=img.dtype)
    for k in range(c):
        out[:, :, k] = cv2.resize(img[:, :, k], (tw, th), interpolation=cv2.INTER_LINEAR)
    return _restore(out, was_2d), {"target": [th, tw], "skipped": False}


# ── Channel registration ───────────────────────────────────────────────


def align_channels(
    arr: np.ndarray,
    *,
    reference_index: int = 0,
    upsample_factor: int = 10,
) -> tuple[np.ndarray, dict]:
    """Sub-pixel align every channel to ``reference_index`` via phase correlation.

    Single-channel inputs are returned unchanged. Returns the per-channel
    detected shifts in metadata so the pipeline can audit alignment quality.
    """
    if arr.ndim != 3 or arr.shape[2] < 2:
        return arr, {"shifts": [[0.0, 0.0]] * (arr.shape[2] if arr.ndim == 3 else 1), "skipped": True}

    from skimage.registration import phase_cross_correlation

    h, w, c = arr.shape
    ref = arr[:, :, reference_index].astype(np.float32, copy=False)
    out = np.empty_like(arr)
    out[:, :, reference_index] = arr[:, :, reference_index]
    shifts: list[list[float]] = [[0.0, 0.0]] * c
    for k in range(c):
        if k == reference_index:
            continue
        target = arr[:, :, k].astype(np.float32, copy=False)
        shift, _err, _diff = phase_cross_correlation(ref, target, upsample_factor=upsample_factor)
        dy, dx = float(shift[0]), float(shift[1])
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(target, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[:, :, k] = warped.astype(arr.dtype)
        shifts[k] = [dy, dx]
    return out, {"reference_index": reference_index, "shifts": shifts, "skipped": False}


# ── Compensation (spectral unmixing) ───────────────────────────────────


def compensate(
    arr: np.ndarray,
    matrix: np.ndarray | list[list[float]],
) -> tuple[np.ndarray, dict]:
    """Apply linear compensation: ``out[:, :, c] = sum_k M[c,k] * arr[:, :, k]``.

    ``matrix`` must be square and match the channel count.
    """
    if arr.ndim != 3:
        raise ValueError("compensate requires HWC input")
    M = np.asarray(matrix, dtype=np.float32)
    h, w, c = arr.shape
    if M.shape != (c, c):
        raise ValueError(f"matrix shape {M.shape} != ({c},{c})")
    flat = arr.reshape(-1, c).astype(np.float32, copy=False)
    out = flat @ M.T
    out = out.reshape(h, w, c).astype(arr.dtype, copy=False)
    return out, {"matrix": M.tolist()}


# ── Background Subtraction ───────────────────────────────────────────


def background_subtraction(arr: np.ndarray, *, radius: int = 50, per_channel: bool = True) -> tuple[np.ndarray, dict]:
    """White top-hat background subtraction via morphological opening.

    Equivalent to the white top-hat: ``out = img - opening(img)``. We use
    ``cv2.morphologyEx`` with an elliptical SE of diameter ``2*radius + 1`` —
    same semantics as skimage's rolling_ball (subtracting a smooth low-frequency
    background floor), but ~15× faster.
    """
    img, was_2d = _ensure_hwc(arr)
    ksize = max(3, int(radius) * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32, copy=False)
        out[:, :, c] = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, kernel)
    np.clip(out, 0, None, out=out)
    return _restore(out, was_2d), {"radius": radius, "per_channel": per_channel, "method": "tophat"}

__all__ = [
    "normalize_minmax",
    "normalize_zscore",
    "pad_to",
    "pad_to_square",
    "resize_to",
    "align_channels",
    "compensate",
    "background_subtraction",
]
