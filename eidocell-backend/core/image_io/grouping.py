"""Group flat image filenames into multi-channel samples.

Default convention: trailing ``_ch<N>`` or ``_c<N>`` (case-insensitive) marks a
channel; the stem before the suffix is the group key. Files without a channel
suffix become singleton groups (one channel).

Example:
    cell001_ch1.png, cell001_ch2.png  -> group "cell001" with channels [1, 2]
    blob.tif                           -> group "blob"    with channels [0]
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# Match suffixes like _ch1, _CH01, _c2 just before the extension.
_CHANNEL_SUFFIX_RE = re.compile(r"_(?:ch|c)(\d+)$", re.IGNORECASE)


def parse_filename(filename: str) -> tuple[str, int | None]:
    """Return (group_stem, channel_index) for a filename.

    ``channel_index`` is None when no channel suffix is present. The returned
    stem strips both the extension and the channel suffix.
    """
    stem = Path(filename).stem
    m = _CHANNEL_SUFFIX_RE.search(stem)
    if not m:
        return stem, None
    return stem[: m.start()], int(m.group(1))


def group_paths(
    paths: Iterable[Path],
    *,
    detect_channels: bool = True,
) -> list[dict]:
    """Group input paths by stem when channel-suffix detection is enabled.

    Returns a list of dicts shaped:
        {
          "group_stem": str,
          "paths": [Path, ...],            # sorted by detected channel index
          "channel_indices": [int, ...],   # 0-based; [0] for singletons
        }

    When ``detect_channels`` is False, every path becomes its own group.
    """
    paths = list(paths)
    if not detect_channels:
        return [
            {"group_stem": p.stem, "paths": [p], "channel_indices": [0]}
            for p in paths
        ]

    by_stem: dict[str, list[tuple[int, Path]]] = {}
    no_suffix: list[Path] = []
    for p in paths:
        stem, ch = parse_filename(p.name)
        if ch is None:
            no_suffix.append(p)
        else:
            by_stem.setdefault(stem, []).append((ch, p))

    groups: list[dict] = []
    for stem, items in by_stem.items():
        items.sort(key=lambda t: t[0])
        # Normalise to 0-based contiguous indices for downstream use.
        indices = [i for i, _ in items]
        if min(indices) >= 1:
            zero_based = [i - 1 for i in indices]
        else:
            zero_based = indices
        groups.append({
            "group_stem": stem,
            "paths": [p for _, p in items],
            "channel_indices": zero_based,
        })
    for p in no_suffix:
        groups.append({
            "group_stem": p.stem,
            "paths": [p],
            "channel_indices": [0],
        })
    groups.sort(key=lambda g: g["group_stem"])
    return groups
