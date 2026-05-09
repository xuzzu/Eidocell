"""Shared test helpers for the LanceDB-backed storage layout."""

from __future__ import annotations

import numpy as np

from core.storage import features as lance_features
from core.storage import mask_attrs as lance_mask_attrs


def list_samples(client, session_id: str, *, limit: int = 1000):
    """Return samples ordered by filename (stable across runs)."""
    return client.post(
        f"/sessions/{session_id}/samples/list",
        json={"offset": 0, "limit": limit, "sort_by": "filename", "sort_order": "asc"},
    ).json()["items"]


def seed_features(
    session_id: str,
    sample_ids: list[str],
    vectors: np.ndarray,
    *,
    method: str = "mobilenetv3",
) -> None:
    """Bulk-insert (sample_id, vector) rows into the per-session features table."""
    if vectors.ndim != 2 or vectors.shape[0] != len(sample_ids):
        raise ValueError("vectors must be 2D with one row per sample_id")
    rows = [
        {"sample_id": sid, "vector": vectors[i].astype(np.float32, copy=False)}
        for i, sid in enumerate(sample_ids)
    ]
    lance_features.upsert_features(session_id, method, rows)


def seed_attrs(session_id: str, rows: list[dict]) -> None:
    """Insert mask-attribute rows. Each row needs at least `sample_id`."""
    lance_mask_attrs.upsert_attrs(session_id, rows)
