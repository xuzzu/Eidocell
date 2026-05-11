"""Typed wrappers over the per-session mask_attrs Lance table.

This is the columnar replacement for the old ``Mask.attributes`` JSON blob.
All scalar mask attributes computed by ``compute_mask_attributes`` live here as
typed columns, keyed by ``(sample_id, channel_index)``. Schema evolution
(adding new columns) is supported by Lance natively.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pyarrow as pa

from core.storage import lance as lance_store

logger = logging.getLogger("eidocell.storage.mask_attrs")

DEFAULT_CHANNEL_INDEX = 0

# Source of truth for which scalar attributes the Lance table stores. Order is
# the canonical CSV/export ordering. Every attr is stored as float32 — integer-
# valued attrs (component counts, skeleton points) cast losslessly within the
# ranges seen in practice.
ATTRIBUTE_NAMES: list[str] = [
    "area", "perimeter",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "major_axis_length", "minor_axis_length", "orientation_deg", "eccentricity",
    "equivalent_diameter", "aspect_ratio", "elongatedness",
    "form_factor", "compactness",
    "convex_area", "convex_perimeter", "solidity", "convexity", "convex_deficiency",
    "feret_max", "feret_min",
    "thickness_mean", "thickness_std", "thickness_max", "thickness_min",
    "component_count", "holes_count", "euler_number",
    "skeleton_length", "skeleton_branch_points", "skeleton_end_points",
    "centroid_x", "centroid_y",
    "mean_intensity", "std_intensity", "median_intensity",
    "min_intensity", "max_intensity", "intensity_sum",
    "background_mean", "background_std", "modulation", "snr",
    "gradient_rms", "gradient_max",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
]

_KEY_FIELDS = ["sample_id", "channel_index"]
_META_FIELDS = ["created_at"]


def attrs_schema() -> pa.Schema:
    fields = [
        pa.field("sample_id", pa.string()),
        pa.field("channel_index", pa.int32()),
    ]
    for name in ATTRIBUTE_NAMES:
        fields.append(pa.field(name, pa.float32()))
    fields.append(pa.field("created_at", pa.timestamp("us")))
    return pa.schema(fields)


def _ensure_table(session_id: str):
    return lance_store.mask_attrs_table(session_id, create_if_missing=True)


def _coerce_value(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    return None


def upsert_attrs(
    session_id: str,
    rows: list[dict],
) -> int:
    """Upsert (sample_id, channel_index) rows. Missing attrs become null."""
    if not rows:
        return 0

    now = datetime.now(timezone.utc)
    payload = []
    for r in rows:
        sid = r["sample_id"]
        ci = int(r.get("channel_index", DEFAULT_CHANNEL_INDEX))
        record = {"sample_id": sid, "channel_index": ci}
        for name in ATTRIBUTE_NAMES:
            record[name] = _coerce_value(r.get(name))
        record["created_at"] = now
        payload.append(record)

    table = _ensure_table(session_id)
    arrow_data = pa.Table.from_pylist(payload, schema=attrs_schema())

    def _do_write():
        table.merge_insert(_KEY_FIELDS) \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(arrow_data)

    lance_store.with_retry(_do_write)
    return len(payload)


def delete_for_samples(session_id: str, sample_ids: Iterable[str]) -> int:
    sample_ids = list(sample_ids)
    if not sample_ids:
        return 0
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return 0
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    where = f"sample_id IN ({ids_literal})"
    table.delete(where)
    return len(sample_ids)


def delete_for_sample_channel(session_id: str, sample_id: str, channel_index: int) -> int:
    """Delete the row for a specific (sample_id, channel_index) pair."""
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return 0
    where = f"sample_id = '{sample_id}' AND channel_index = {int(channel_index)}"
    table.delete(where)
    return 1


def get_attrs(
    session_id: str,
    sample_id: str,
    *,
    channel_index: int = DEFAULT_CHANNEL_INDEX,
) -> dict | None:
    """Return the attribute dict for one (sample, channel), or None if absent."""
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return None
    where = f"sample_id = '{sample_id}' AND channel_index = {int(channel_index)}"
    arrow = table.search().where(where).limit(1).to_arrow()
    if arrow.num_rows == 0:
        return None
    out: dict = {}
    for name in ATTRIBUTE_NAMES:
        if name not in arrow.schema.names:
            continue
        val = arrow.column(name)[0].as_py()
        if val is None:
            continue
        out[name] = val
    return out


def get_attrs_bulk(
    session_id: str,
    sample_ids: Iterable[str],
    *,
    channel_index: int = DEFAULT_CHANNEL_INDEX,
) -> dict[str, dict]:
    """Fetch attribute dicts for many samples on one channel, keyed by sample_id."""
    sample_ids = list(sample_ids)
    if not sample_ids:
        return {}
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return {}
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    where = f"channel_index = {int(channel_index)} AND sample_id IN ({ids_literal})"
    arrow = table.search().where(where).to_arrow()
    if arrow.num_rows == 0:
        return {}
    out: dict[str, dict] = {}
    sids = arrow.column("sample_id").to_pylist()
    cols = {
        name: arrow.column(name).to_pylist()
        for name in ATTRIBUTE_NAMES
        if name in arrow.schema.names
    }
    for i, sid in enumerate(sids):
        d: dict = {}
        for name, col in cols.items():
            v = col[i]
            if v is None:
                continue
            d[name] = v
        out[sid] = d
    return out


def has_sample_ids(
    session_id: str,
    *,
    channel_index: int = DEFAULT_CHANNEL_INDEX,
) -> set[str]:
    """Return the set of sample_ids that already have an attrs row for the channel."""
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return set()
    where = f"channel_index = {int(channel_index)}"
    arrow = table.search().where(where).select(["sample_id"]).to_arrow()
    if arrow.num_rows == 0:
        return set()
    return set(arrow.column("sample_id").to_pylist())


def channels_with_attrs_bulk(
    session_id: str,
    sample_ids: Iterable[str],
) -> dict[str, list[int]]:
    """Return {sample_id: sorted list of channel_index values with mask attrs}."""
    sample_ids = list(sample_ids)
    if not sample_ids:
        return {}
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return {}
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    where = f"sample_id IN ({ids_literal})"
    arrow = table.search().where(where).select(["sample_id", "channel_index"]).to_arrow()
    if arrow.num_rows == 0:
        return {}
    out: dict[str, set[int]] = {}
    sids = arrow.column("sample_id").to_pylist()
    chans = arrow.column("channel_index").to_pylist()
    for sid, ci in zip(sids, chans):
        out.setdefault(sid, set()).add(int(ci))
    return {sid: sorted(s) for sid, s in out.items()}


def query_predicate(
    session_id: str,
    where: str,
    *,
    sample_ids: Iterable[str] | None = None,
    channel_index: int = DEFAULT_CHANNEL_INDEX,
) -> list[str]:
    """Return sample_ids matching a Lance SQL predicate over attribute columns.

    `where` is a SQL fragment using attribute column names (e.g.
    `"area > 100 AND eccentricity < 0.5"`). The caller is responsible for
    sanitizing column names against the canonical attribute list.
    """
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return []
    clauses = [f"channel_index = {int(channel_index)}"]
    if where:
        clauses.append(f"({where})")
    if sample_ids is not None:
        sids = list(sample_ids)
        if not sids:
            return []
        ids_literal = ",".join(f"'{sid}'" for sid in sids)
        clauses.append(f"sample_id IN ({ids_literal})")
    final_where = " AND ".join(clauses)
    arrow = table.search().where(final_where).select(["sample_id"]).to_arrow()
    if arrow.num_rows == 0:
        return []
    return arrow.column("sample_id").to_pylist()


def fetch_columns(
    session_id: str,
    columns: list[str],
    *,
    sample_ids: Iterable[str] | None = None,
    channel_index: int = DEFAULT_CHANNEL_INDEX,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return aligned (sample_ids, {col: np.ndarray}) for the given columns.

    Skips rows that have null in any of the requested columns. Use this for
    polygon/ellipse gates that need numeric arrays for in-Python checks.
    """
    bad = [c for c in columns if c not in ATTRIBUTE_NAMES]
    if bad:
        raise ValueError(f"unknown attribute(s): {bad}")
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return [], {c: np.zeros(0) for c in columns}

    clauses = [f"channel_index = {int(channel_index)}"]
    null_filter = " AND ".join(f"{c} IS NOT NULL" for c in columns)
    if null_filter:
        clauses.append(null_filter)
    if sample_ids is not None:
        sids = list(sample_ids)
        if not sids:
            return [], {c: np.zeros(0) for c in columns}
        ids_literal = ",".join(f"'{sid}'" for sid in sids)
        clauses.append(f"sample_id IN ({ids_literal})")
    where = " AND ".join(clauses)
    arrow = (
        table.search().where(where).select(["sample_id"] + columns).to_arrow()
    )
    if arrow.num_rows == 0:
        return [], {c: np.zeros(0) for c in columns}
    out_ids = arrow.column("sample_id").to_pylist()
    out_cols = {
        c: np.asarray(arrow.column(c).to_pylist(), dtype=np.float64)
        for c in columns
    }
    return out_ids, out_cols


def list_attribute_names(session_id: str) -> list[str]:
    """Return the attribute columns available in a session's table.

    Returns [] when the table is missing or has no rows (matches the old
    "no masks → no sortable attrs" behaviour). Once any row has been written,
    the canonical schema columns are returned.
    """
    try:
        table = lance_store.mask_attrs_table(session_id, create_if_missing=False)
    except FileNotFoundError:
        return []
    if table.count_rows() == 0:
        return []
    schema = table.schema
    return [name for name in ATTRIBUTE_NAMES if name in schema.names]
