"""Typed wrappers over the per-session features Lance table."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pyarrow as pa

from core.storage import lance as lance_store

logger = logging.getLogger("eidocell.storage.features")

DEFAULT_CHANNEL_SET = "default"
DEFAULT_VERSION = 1


def _ensure_table(session_id: str, dim: int):
    return lance_store.features_table(session_id, dim=dim, create_if_missing=True)


def upsert_features(
    session_id: str,
    method: str,
    rows: list[dict],
) -> int:
    """Upsert (sample_id, method, channel_set) feature rows.

    Each row must have keys: sample_id, vector (np.ndarray | list[float]).
    Optional: channel_set, version. dim is inferred from the first row's vector.
    Returns the number of rows written.
    """
    if not rows:
        return 0

    first_vec = np.asarray(rows[0]["vector"], dtype=np.float32).ravel()
    dim = int(first_vec.shape[0])

    now = datetime.now(timezone.utc)
    payload = []
    for r in rows:
        vec = np.asarray(r["vector"], dtype=np.float32).ravel()
        if vec.shape[0] != dim:
            raise ValueError(
                f"vector dim mismatch for sample {r['sample_id']}: "
                f"got {vec.shape[0]}, expected {dim}"
            )
        payload.append(
            {
                "sample_id": r["sample_id"],
                "method": method,
                "vector": vec.tolist(),
                "dim": dim,
                "version": int(r.get("version", DEFAULT_VERSION)),
                "channel_set": r.get("channel_set", DEFAULT_CHANNEL_SET),
                "created_at": now,
            }
        )

    table = _ensure_table(session_id, dim)
    arrow_data = pa.Table.from_pylist(payload, schema=lance_store.features_schema(dim))

    def _do_write():
        table.merge_insert(["sample_id", "method", "channel_set"]) \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(arrow_data)

    lance_store.with_retry(_do_write)
    return len(payload)


def has_method(
    session_id: str,
    method: str,
    sample_ids: Iterable[str],
    *,
    channel_set: str = DEFAULT_CHANNEL_SET,
) -> set[str]:
    """Return the subset of sample_ids that already have a feature row for (method, channel_set)."""
    sample_ids = list(sample_ids)
    if not sample_ids:
        return set()
    try:
        table = lance_store.features_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return set()
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    where = (
        f"method = '{method}' AND channel_set = '{channel_set}' "
        f"AND sample_id IN ({ids_literal})"
    )
    arrow = table.search().where(where).select(["sample_id"]).to_arrow()
    if arrow.num_rows == 0:
        return set()
    return set(arrow.column("sample_id").to_pylist())


def load_vectors(
    session_id: str,
    method: str,
    sample_ids: list[str],
    *,
    channel_set: str = DEFAULT_CHANNEL_SET,
) -> tuple[list[str], np.ndarray]:
    """Return (ids_in_order, vectors_2d) for the requested sample_ids that have features.

    Order of the returned list matches the input order, dropping any ids that
    don't have a row. The returned 2D array is float32 with shape (n_found, dim).
    """
    if not sample_ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    try:
        table = lance_store.features_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return [], np.zeros((0, 0), dtype=np.float32)

    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    where = (
        f"method = '{method}' AND channel_set = '{channel_set}' "
        f"AND sample_id IN ({ids_literal})"
    )
    arrow = (
        table.search()
        .where(where)
        .select(["sample_id", "vector"])
        .limit(len(sample_ids))
        .to_arrow()
    )
    if arrow.num_rows == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    found_by_id: dict[str, np.ndarray] = {}
    sids_col = arrow.column("sample_id").to_pylist()
    vecs_col = arrow.column("vector").to_pylist()
    for sid, vec in zip(sids_col, vecs_col):
        found_by_id[sid] = np.asarray(vec, dtype=np.float32)

    ordered_ids: list[str] = []
    ordered_vecs: list[np.ndarray] = []
    for sid in sample_ids:
        if sid in found_by_id:
            ordered_ids.append(sid)
            ordered_vecs.append(found_by_id[sid])
    if not ordered_vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ordered_ids, np.stack(ordered_vecs, axis=0)


def load_all_vectors(
    session_id: str,
    method: str,
    *,
    channel_set: str = DEFAULT_CHANNEL_SET,
) -> tuple[list[str], np.ndarray]:
    """Return (sample_ids, vectors_2d) for every row of (method, channel_set)."""
    try:
        table = lance_store.features_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return [], np.zeros((0, 0), dtype=np.float32)
    where = f"method = '{method}' AND channel_set = '{channel_set}'"
    arrow = table.search().where(where).select(["sample_id", "vector"]).to_arrow()
    if arrow.num_rows == 0:
        return [], np.zeros((0, 0), dtype=np.float32)
    sids = arrow.column("sample_id").to_pylist()
    vecs = arrow.column("vector").to_pylist()
    return sids, np.asarray(vecs, dtype=np.float32)


def cosine_search(
    session_id: str,
    method: str,
    query_vector: np.ndarray,
    *,
    k: int | None = None,
    candidate_sample_ids: Iterable[str] | None = None,
    channel_set: str = DEFAULT_CHANNEL_SET,
) -> list[tuple[str, float]]:
    """Cosine similarity search. Returns [(sample_id, similarity_in_[-1,1])] sorted desc.

    `candidate_sample_ids`, if given, restricts the search to those ids.
    `k` is the result limit; None returns the full candidate set.
    """
    try:
        table = lance_store.features_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return []

    query = np.asarray(query_vector, dtype=np.float32).ravel()
    where_clauses = [f"method = '{method}'", f"channel_set = '{channel_set}'"]
    if candidate_sample_ids is not None:
        cand_list = list(candidate_sample_ids)
        if not cand_list:
            return []
        ids_literal = ",".join(f"'{sid}'" for sid in cand_list)
        where_clauses.append(f"sample_id IN ({ids_literal})")
    where = " AND ".join(where_clauses)

    builder = (
        table.search(query.tolist(), vector_column_name="vector")
        .metric("cosine")
        .where(where, prefilter=True)
        .select(["sample_id"])
    )
    if k is not None:
        builder = builder.limit(int(k))
    arrow = builder.to_arrow()
    if arrow.num_rows == 0:
        return []

    sids = arrow.column("sample_id").to_pylist()
    # Lance returns "_distance" as cosine distance in [0, 2]; similarity = 1 - distance
    distances = arrow.column("_distance").to_pylist()
    return [(sid, 1.0 - float(d)) for sid, d in zip(sids, distances)]


def delete_session(session_id: str) -> None:
    """Convenience wrapper — mostly used by tests / session_service."""
    from core.storage.lance import drop_session_tables
    drop_session_tables(session_id)
