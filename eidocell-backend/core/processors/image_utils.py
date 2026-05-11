"""Image utility functions: overlays, thumbnails, collages."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

THUMBNAIL_SIZE = (256, 256)


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
COLLAGE_GRID = 5
COLLAGE_TILE = 64
COLLAGE_SIZE = COLLAGE_GRID * COLLAGE_TILE
MAX_COLLAGE_SAMPLES = COLLAGE_GRID * COLLAGE_GRID


def generate_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.4,
    contour_color: tuple = (0, 255, 0),
    fill_color: tuple = (0, 100, 255),
) -> np.ndarray:
    """Generate an RGB overlay image with mask visualization."""
    if len(image.shape) == 2:
        display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        display = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        display = image.copy()
        if display.shape[2] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    binary = (mask > 0).astype(np.uint8) * 255

    if binary.shape[:2] != display.shape[:2]:
        binary = cv2.resize(
            binary, (display.shape[1], display.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = display.copy()
    overlay[binary > 0] = fill_color
    blended = cv2.addWeighted(display, 1 - opacity, overlay, opacity, 0)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Scale outline thickness by image size so it stays visually consistent
    # across resolutions (1 px per ~256 px of the shorter side).
    h, w = display.shape[:2]
    thickness = max(1, round(min(h, w) / 256))
    cv2.drawContours(blended, contours, -1, contour_color, thickness, lineType=cv2.LINE_AA)

    return blended


def generate_thumbnail(source: Path, dest: Path) -> None:
    """Generate a JPEG thumbnail from a source image."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img.thumbnail(THUMBNAIL_SIZE)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(dest, "JPEG", quality=85)


def array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise any-dtype ndarray to uint8 for visualization.

    Floats and signed ints (e.g. post-zscore) are min-max scaled. uint8
    arrays pass through. Multi-channel (>3) arrays are reduced to channel 0
    — multi-channel UI is out of scope.
    """
    if arr.ndim == 3 and arr.shape[2] not in (1, 3):
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32, copy=False)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        scaled = (a - mn) / (mx - mn) * 255.0
    else:
        scaled = np.zeros_like(a)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def render_array_to_thumbnail(arr: np.ndarray, dest: Path) -> None:
    """Render a numpy ndarray to a thumbnail JPEG at ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    u8 = array_to_uint8(arr)
    img = Image.fromarray(u8)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail(THUMBNAIL_SIZE)
    img.save(dest, "JPEG", quality=85)


def channel_to_uint8(arr: np.ndarray, channel: int) -> np.ndarray:
    """Extract a single channel as a min-max normalised uint8 2-D array."""
    if arr.ndim == 2:
        if channel != 0:
            raise IndexError(f"channel {channel} out of range (single-channel image)")
        ch = arr
    elif arr.ndim == 3:
        n = arr.shape[2]
        if channel < 0 or channel >= n:
            raise IndexError(f"channel {channel} out of range (n_channels={n})")
        ch = arr[..., channel]
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    if ch.dtype == np.uint8:
        return ch
    a = ch.astype(np.float32, copy=False)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        scaled = (a - mn) / (mx - mn) * 255.0
    else:
        scaled = np.zeros_like(a)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def render_channel_to_thumbnail(arr: np.ndarray, channel: int, dest: Path) -> None:
    """Render one channel of a multi-channel array to a thumbnail JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    u8 = channel_to_uint8(arr, channel)
    img = Image.fromarray(u8, mode="L")
    img.thumbnail(THUMBNAIL_SIZE)
    img.save(dest, "JPEG", quality=85)


def render_array_to_full(arr: np.ndarray, dest: Path) -> None:
    """Render a numpy ndarray to a PNG at full resolution at ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    u8 = array_to_uint8(arr)
    img = Image.fromarray(u8)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(dest, "PNG")


def generate_collage(image_paths: list[Path], dest: Path) -> None:
    """Generate a grid collage JPEG from a list of image paths."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    collage = Image.new("RGB", (COLLAGE_SIZE, COLLAGE_SIZE), (40, 40, 40))

    for idx, src_path in enumerate(image_paths[:MAX_COLLAGE_SAMPLES]):
        if not src_path.is_file():
            continue
        try:
            with Image.open(src_path) as img:
                img.thumbnail((COLLAGE_TILE, COLLAGE_TILE))
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                row, col = divmod(idx, COLLAGE_GRID)
                x = col * COLLAGE_TILE + (COLLAGE_TILE - img.width) // 2
                y = row * COLLAGE_TILE + (COLLAGE_TILE - img.height) // 2
                collage.paste(img, (x, y))
        except Exception:
            continue

    collage.save(dest, "JPEG", quality=85)
