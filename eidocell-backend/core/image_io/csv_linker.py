"""Parse a user-provided CSV and link rows to image filenames.

Stdlib-only (no pandas dependency). Only numeric columns are exposed; the
filename column itself is dropped from the attribute payload.
"""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from core.image_io.errors import CsvLinkError

logger = logging.getLogger("eidocell.image_io.csv_linker")


def _to_float(v: str) -> float | None:
    if v is None:
        return None
    s = v.strip()
    if not s or s.lower() in ("nan", "inf", "-inf", "null", "none"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if not math.isfinite(f):
        return None
    return f


def parse_csv(path: str | Path, filename_col: str) -> dict[str, dict]:
    """Return ``{filename_or_stem: {numeric_col: value}}``.

    A column is considered numeric when at least one row parses as a finite
    float. Within a numeric column, individual unparseable cells are dropped.
    Lookup keys are indexed under both the original filename and its stem.
    """
    p = Path(path)
    if not p.is_file():
        raise CsvLinkError(f"CSV not found: {p}")

    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise CsvLinkError("CSV is empty")
        rows = list(reader)

    if filename_col not in header:
        raise CsvLinkError(
            f"CSV is missing filename column '{filename_col}'. "
            f"Available columns: {header}"
        )
    fname_idx = header.index(filename_col)

    # Identify numeric columns: at least one row parses as finite float.
    candidate_idx = [i for i, c in enumerate(header) if i != fname_idx]
    numeric_idx: list[int] = []
    for i in candidate_idx:
        for row in rows:
            if i >= len(row):
                continue
            if _to_float(row[i]) is not None:
                numeric_idx.append(i)
                break
    if not numeric_idx:
        logger.warning("CSV has no numeric columns besides %s", filename_col)

    out: dict[str, dict] = {}
    for row in rows:
        if fname_idx >= len(row):
            continue
        key = row[fname_idx].strip()
        if not key:
            continue
        attrs: dict[str, float] = {}
        for i in numeric_idx:
            if i >= len(row):
                continue
            v = _to_float(row[i])
            if v is None:
                continue
            attrs[header[i]] = v
        out[key] = attrs
        stem = Path(key).stem
        if stem != key:
            out.setdefault(stem, attrs)

    if not out:
        raise CsvLinkError("CSV produced zero rows")
    return out


def lookup(linked: dict[str, dict], filename: str) -> dict:
    """Return numeric attrs for a filename. Tries full name then stem."""
    if filename in linked:
        return linked[filename]
    stem = Path(filename).stem
    return linked.get(stem, {})
