"""CIF / RIF container parser.

Both formats are TIFF variants with two custom compressions:

* ``30817`` — packed image pixels (cell brightfield / fluorescence channels).
* ``30818`` — RLE-encoded IDEAS binary masks (same crop window as the
  preceding image IFD).

IDEAS exports interleave these as ``image, mask, image, mask, …`` per object.
``iter_objects`` walks the IFDs in that order: it yields one record per
**image** IFD and attaches the paired mask IFD (when present) as a 2-D array.
Channel splitting (``n_channels > 1``) is applied to both image and mask.

Decompressors are ported verbatim from
``ifc-data-eval/decomp_numba.py`` and ``ifc-data-eval/decomp_30818_numba.py``;
the IFD walker is generalised from ``ifc-data-eval/extract_chunk.py``.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Iterator

import numba
import numpy as np

from core.image_io.errors import ContainerParseError, ImageReadError

logger = logging.getLogger("eidocell.image_io.cif_rif")

_COMPRESSION_30817 = 30817
_COMPRESSION_30818 = 30818
_TAG_WIDTH = 256
_TAG_HEIGHT = 257
_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_STRIP_BYTECOUNTS = 279


# ── Decompressors (ported from ifc-data-eval) ──────────────────────────


@numba.jit(
    numba.int32[:, :](numba.uint8[::1], numba.int64, numba.int64),
    nopython=True,
    locals={"nibble": numba.int32, "value": numba.int32, "shift": numba.int32, "k": numba.int64},
)
def _decompress_30817(raw_chnk, img_width, img_height):
    nbytes = len(raw_chnk)
    last_row = np.zeros(img_width + 1, dtype=np.int32)
    img = np.zeros((img_height, img_width + 1), dtype=np.int32)

    odd = False
    k = 0

    for y in range(img_height):
        for x in range(1, img_width + 1):
            value = 0
            shift = 0
            nibble = 8

            while (nibble & 8) != 0:
                if odd:
                    nibble = int(raw_chnk[k]) >> 4
                    k += 1
                else:
                    if k >= nbytes:
                        raise ValueError("Buffer overrun")
                    nibble = int(raw_chnk[k]) & 15
                odd = not odd

                value += (nibble & 7) << shift
                shift += 3

            if (nibble & 4) != 0:
                value |= -(1 << shift)

            last_row[x] += value
            img[y, x] = img[y, x - 1] + last_row[x]

    return img[:, 1:]


@numba.jit(
    numba.int32[:, :](numba.uint8[::1], numba.int64, numba.int64),
    nopython=True,
)
def _decompress_30818(raw_chnk, img_width, img_height):
    nbytes = len(raw_chnk)
    L = img_width * img_height
    img_flat = np.zeros(L, dtype=np.int32)

    run_length = 0
    k = 0

    while k < nbytes:
        value = raw_chnk[k]
        k += 1
        off = run_length
        run_length = off + int(raw_chnk[k]) + 1
        k += 1

        if run_length > L:
            raise ValueError("Buffer overrun")

        for j in range(off, run_length):
            img_flat[j] = value

    img = img_flat.reshape((img_width, img_height)).T
    return img


# ── IFD walker ─────────────────────────────────────────────────────────


def _walk_image_ifds(fh) -> Iterator[dict]:
    """Yield each image-strip IFD as a dict.

    Each yielded record contains: width, height, compression, strip_offset,
    strip_bytes. Non-image IFDs (any compression other than 30817/30818) are
    skipped silently.
    """
    fh.seek(0)
    header = fh.read(8)
    if len(header) < 8:
        raise ContainerParseError("file too small to be a TIFF")
    endian = "<" if header[:2] == b"II" else ">"
    magic = struct.unpack(endian + "H", header[2:4])[0]
    if magic not in (42, 43):
        raise ContainerParseError(f"not a TIFF (magic={magic})")
    offset = struct.unpack(endian + "I", header[4:8])[0]

    seen_offsets: set[int] = set()
    while offset != 0:
        if offset in seen_offsets:
            raise ContainerParseError(f"IFD cycle detected at {offset}")
        seen_offsets.add(offset)
        fh.seek(offset)
        nt_bytes = fh.read(2)
        if len(nt_bytes) < 2:
            return
        num_tags = struct.unpack(endian + "H", nt_bytes)[0]

        tags: dict[int, int] = {}
        for _ in range(num_tags):
            entry = fh.read(12)
            if len(entry) < 12:
                return
            tag, _typ, _count, value_offset = struct.unpack(endian + "HHII", entry)
            tags[tag] = value_offset

        next_bytes = fh.read(4)
        if len(next_bytes) < 4:
            offset = 0
        else:
            offset = struct.unpack(endian + "I", next_bytes)[0]

        comp = tags.get(_TAG_COMPRESSION)
        if comp not in (_COMPRESSION_30817, _COMPRESSION_30818):
            continue
        width = tags.get(_TAG_WIDTH)
        height = tags.get(_TAG_HEIGHT)
        strip_off = tags.get(_TAG_STRIP_OFFSETS)
        strip_bytes = tags.get(_TAG_STRIP_BYTECOUNTS)
        if None in (width, height, strip_off, strip_bytes):
            continue
        yield {
            "width": int(width),
            "height": int(height),
            "compression": int(comp),
            "strip_offset": int(strip_off),
            "strip_bytes": int(strip_bytes),
        }


# ── Public API ─────────────────────────────────────────────────────────


def count_objects(path: str | Path) -> int:
    """Count image IFDs (compression 30817) only — masks are paired, not counted."""
    with open(path, "rb") as fh:
        return sum(
            1 for ifd in _walk_image_ifds(fh)
            if ifd["compression"] == _COMPRESSION_30817
        )


def _split_horizontal(arr: np.ndarray, n_channels: int) -> list[np.ndarray]:
    if n_channels == 1:
        return [arr]
    w = arr.shape[1]
    if w % n_channels != 0:
        raise ImageReadError(
            f"width {w} not divisible by n_channels {n_channels}"
        )
    tile = w // n_channels
    return [arr[:, c * tile : (c + 1) * tile] for c in range(n_channels)]


def _read_strip(fh, ifd: dict) -> np.ndarray:
    fh.seek(ifd["strip_offset"])
    chunk = fh.read(ifd["strip_bytes"])
    return np.frombuffer(chunk, dtype=np.uint8).copy()


def iter_objects(
    path: str | Path,
    *,
    n_channels: int = 1,
) -> Iterator[dict]:
    """Yield one record per **image** IFD (compression 30817).

    Each record:
        {
          "object_index": int,                 # 0-based image-IFD index
          "width": int, "height": int,
          "compression": int,                  # 30817 (image)
          "image": np.ndarray,                 # 2-D int32 (full strip width)
          "channels": list[np.ndarray],        # n_channels horizontal tiles
          "mask": np.ndarray | None,           # paired IDEAS mask (uint8 0/1)
          "mask_channels": list[np.ndarray],   # n_channels tiles of the mask
        }

    IDEAS interleaves the file as ``image, mask, image, mask, …``. When the
    IFD that follows an image is a 30818 mask of the same width/height, it's
    decompressed and attached. Orphan masks (no preceding image) are skipped
    with a warning. When ``n_channels > 1``, both the image and the mask are
    split into ``n_channels`` equal horizontal tiles.

    Skips an IFD pair with a warning if either side refuses to decompress, so
    a partially-damaged file still yields what it can.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")
    p = Path(path)
    with open(p, "rb") as fh:
        pending: dict | None = None  # buffered image IFD awaiting its mask
        object_idx = 0

        def _emit(image_ifd: dict, mask_ifd: dict | None):
            nonlocal object_idx
            try:
                img = _decompress_30817(
                    _read_strip(fh, image_ifd), image_ifd["width"], image_ifd["height"]
                )
            except Exception as e:
                logger.warning(
                    "object %d: image decompression failed (%s); skipping",
                    object_idx, e,
                )
                object_idx += 1
                return None

            mask: np.ndarray | None = None
            if mask_ifd is not None:
                try:
                    mask_raw = _decompress_30818(
                        _read_strip(fh, mask_ifd),
                        mask_ifd["width"], mask_ifd["height"],
                    )
                    mask = (mask_raw != 0).astype(np.uint8)
                except Exception as e:
                    logger.warning(
                        "object %d: mask decompression failed (%s); dropping mask",
                        object_idx, e,
                    )
                    mask = None

            channels = _split_horizontal(img, n_channels)
            mask_channels = _split_horizontal(mask, n_channels) if mask is not None else []
            rec = {
                "object_index": object_idx,
                "width": image_ifd["width"],
                "height": image_ifd["height"],
                "compression": image_ifd["compression"],
                "image": img,
                "channels": channels,
                "mask": mask,
                "mask_channels": mask_channels,
            }
            object_idx += 1
            return rec

        for ifd in _walk_image_ifds(fh):
            if ifd["compression"] == _COMPRESSION_30817:
                if pending is not None:
                    rec = _emit(pending, None)
                    if rec is not None:
                        yield rec
                pending = ifd
            else:  # 30818 mask
                if pending is None:
                    logger.warning("orphan mask IFD at offset %d; skipping", ifd["strip_offset"])
                    continue
                if ifd["width"] != pending["width"] or ifd["height"] != pending["height"]:
                    logger.warning(
                        "mask shape %dx%d != image %dx%d at object %d; dropping mask",
                        ifd["width"], ifd["height"],
                        pending["width"], pending["height"], object_idx,
                    )
                    rec = _emit(pending, None)
                    if rec is not None:
                        yield rec
                    pending = None
                    continue
                rec = _emit(pending, ifd)
                if rec is not None:
                    yield rec
                pending = None

        if pending is not None:
            rec = _emit(pending, None)
            if rec is not None:
                yield rec


def split_channels(arr: np.ndarray, n_channels: int) -> list[np.ndarray]:
    """Horizontal split helper for callers that already have a decoded array."""
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")
    return _split_horizontal(arr, n_channels)
