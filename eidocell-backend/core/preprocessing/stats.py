"""Dataset-level passes used by the preprocessing pipeline.

Today: shape stats from the staging table. Future: per-channel intensity
distributions for adaptive compensation, etc.
"""
from __future__ import annotations

from core.storage import import_staging


def compute_size_stats(session_id: str, import_id: str) -> dict:
    """Min / max / mean (H, W) over the staging table; 0 when empty."""
    return import_staging.shape_summary(session_id, import_id)
