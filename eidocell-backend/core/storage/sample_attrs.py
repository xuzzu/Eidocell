"""Per-session, dynamically-schema'd Lance table for CSV-imported sample attributes.

Distinct from ``mask_attrs`` (which has a fixed schema of morphological
descriptors). Here, each session's table has columns chosen by whatever CSV
the user linked at import. Subsequent imports that introduce new columns use
Lance's ``add_columns`` API to extend the schema in place.

Cells store ``float64`` (matches CSV numeric semantics). ``sample_id`` is the
join key; only one row per sample (no channel_set partition — these are
sample-level not channel-level).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pyarrow as pa

from core.storage import lance as lance_store

logger = logging.getLogger("eidocell.storage.sample_attrs")

_KEY_FIELDS = ["sample_id"]


def _schema(columns: list[str]) -> pa.Schema:
    fields = [pa.field("sample_id", pa.string())]
    for c in columns:
        fields.append(pa.field(c, pa.float64()))
    return pa.schema(fields)


def _table_name(session_id: str) -> str:
    return lance_store.sample_attrs_table_name(session_id)


def _coerce(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    return None


def upsert_attrs(session_id: str, rows: list[dict]) -> int:
    """Upsert per-sample attribute rows.

    Each row: ``{"sample_id": str, **{col: numeric|None}}``. Columns absent
    from the table schema are added on the fly. Missing values become null.
    """
    if not rows:
        return 0

    cols_in_rows: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k == "sample_id":
                continue
            if k not in seen:
                seen.add(k)
                cols_in_rows.append(k)

    db = lance_store.connect()
    name = _table_name(session_id)
    try:
        table = db.open_table(name)
        existing = set(table.schema.names) - {"sample_id"}
    except (FileNotFoundError, ValueError):
        table = db.create_table(name, schema=_schema(cols_in_rows))
        existing = set(cols_in_rows)

    new_cols = [c for c in cols_in_rows if c not in existing]
    if new_cols:
        try:
            table.add_columns({c: "CAST(NULL AS DOUBLE)" for c in new_cols})
        except Exception:
            logger.exception("add_columns failed; reopening table")
            table = db.open_table(name)

    final_cols = [c for c in table.schema.names if c != "sample_id"]
    payload = []
    for r in rows:
        rec = {"sample_id": r["sample_id"]}
        for c in final_cols:
            rec[c] = _coerce(r.get(c))
        payload.append(rec)

    arrow = pa.Table.from_pylist(payload, schema=_schema(final_cols))

    def _do_write():
        table.merge_insert(_KEY_FIELDS) \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(arrow)

    lance_store.with_retry(_do_write)
    return len(payload)


def list_columns(session_id: str) -> list[str]:
    try:
        table = lance_store.open_or_create_table(
            _table_name(session_id), _schema([]), create_if_missing=False
        )
    except FileNotFoundError:
        return []
    return [c for c in table.schema.names if c != "sample_id"]


def get_attrs_bulk(
    session_id: str,
    sample_ids: Iterable[str],
) -> dict[str, dict]:
    sample_ids = list(sample_ids)
    if not sample_ids:
        return {}
    try:
        table = lance_store.open_or_create_table(
            _table_name(session_id), _schema([]), create_if_missing=False
        )
    except FileNotFoundError:
        return {}
    cols = [c for c in table.schema.names if c != "sample_id"]
    if not cols:
        return {}
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    arrow = table.search().where(f"sample_id IN ({ids_literal})").to_arrow()
    if arrow.num_rows == 0:
        return {}
    sids = arrow.column("sample_id").to_pylist()
    out: dict[str, dict] = {}
    col_data = {c: arrow.column(c).to_pylist() for c in cols}
    for i, sid in enumerate(sids):
        d: dict = {}
        for c, vals in col_data.items():
            v = vals[i]
            if v is None:
                continue
            d[c] = v
        out[sid] = d
    return out


def delete_for_samples(session_id: str, sample_ids: Iterable[str]) -> int:
    sample_ids = list(sample_ids)
    if not sample_ids:
        return 0
    try:
        table = lance_store.open_or_create_table(
            _table_name(session_id), _schema([]), create_if_missing=False
        )
    except FileNotFoundError:
        return 0
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    table.delete(f"sample_id IN ({ids_literal})")
    return len(sample_ids)
