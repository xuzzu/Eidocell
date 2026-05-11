"""Per-import staging Lance table.

Holds raw image bytes + parsed CSV attrs after the I/O phase. Read by the
preprocessing phase to produce the final ``images_{sid}`` rows + Sample SQL
records. Dropped on import success (or sweep on session delete via
``lance.drop_session_tables``).

Schema is fixed; each row represents one *grouped* candidate (one Sample
post-import), with its raw multi-channel array stored flat.
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

import numpy as np
import pyarrow as pa

from core.storage import lance as lance_store

logger = logging.getLogger("eidocell.storage.import_staging")


def schema() -> pa.Schema:
    return pa.schema([
        pa.field("tmp_id", pa.string()),
        pa.field("group_stem", pa.string()),
        pa.field("source_paths", pa.list_(pa.string())),
        pa.field("channel_meta", pa.string()),  # JSON: {"order": [...], "names": [...]}
        pa.field("height", pa.int32()),
        pa.field("width", pa.int32()),
        pa.field("n_channels", pa.int32()),
        pa.field("dtype", pa.string()),
        pa.field("raw_data", pa.binary()),
        pa.field("parsed_attrs", pa.string()),  # JSON: {col: number} (may be "")
        # Optional pre-existing binary mask (e.g. IDEAS CIF 30818 IFDs).
        # Stored as raw uint8 0/1 bytes of shape (mask_h, mask_w); empty when absent.
        pa.field("mask_h", pa.int32()),
        pa.field("mask_w", pa.int32()),
        pa.field("mask_data", pa.binary()),
    ])


def open_table(session_id: str, import_id: str, *, create_if_missing: bool = False):
    return lance_store.open_or_create_table(
        lance_store.import_staging_table_name(session_id, import_id),
        schema(),
        create_if_missing=create_if_missing,
    )


def upsert_rows(session_id: str, import_id: str, rows: list[dict]) -> int:
    """Append staged candidates. Each row dict shape:

        {
          "tmp_id": str,
          "group_stem": str,
          "source_paths": list[str],
          "channel_meta": dict,
          "array": np.ndarray (HWC or HW),
          "parsed_attrs": dict,
        }
    """
    if not rows:
        return 0
    table = lance_store.open_or_create_table(
        lance_store.import_staging_table_name(session_id, import_id),
        schema(),
        create_if_missing=True,
    )
    payload = []
    for r in rows:
        arr = np.ascontiguousarray(r["array"])
        if arr.ndim == 2:
            h, w = arr.shape
            c = 1
        elif arr.ndim == 3:
            h, w, c = arr.shape
        else:
            raise ValueError(f"unsupported ndim {arr.ndim} for tmp_id={r['tmp_id']}")
        mask = r.get("mask")
        if mask is not None:
            mask_arr = np.ascontiguousarray(mask).astype(np.uint8, copy=False)
            mh, mw = mask_arr.shape
            mask_bytes = mask_arr.tobytes()
        else:
            mh, mw, mask_bytes = 0, 0, b""
        payload.append({
            "tmp_id": r["tmp_id"],
            "group_stem": r.get("group_stem", ""),
            "source_paths": list(r.get("source_paths", [])),
            "channel_meta": json.dumps(r.get("channel_meta") or {}),
            "height": int(h),
            "width": int(w),
            "n_channels": int(c),
            "dtype": str(arr.dtype),
            "raw_data": arr.tobytes(),
            "parsed_attrs": json.dumps(r.get("parsed_attrs") or {}),
            "mask_h": int(mh),
            "mask_w": int(mw),
            "mask_data": mask_bytes,
        })
    arrow = pa.Table.from_pylist(payload, schema=schema())

    def _do_write():
        table.add(arrow)

    lance_store.with_retry(_do_write)
    return len(payload)


def iter_rows(session_id: str, import_id: str, *, batch_size: int = 64) -> Iterator[dict]:
    """Stream staged rows as decoded ndarrays + parsed metadata.

    Yields ``{tmp_id, group_stem, source_paths, channel_meta, array, parsed_attrs}``.
    """
    try:
        table = open_table(session_id, import_id, create_if_missing=False)
    except FileNotFoundError:
        return
    arrow = table.search().to_arrow()
    rows = arrow.num_rows
    for start in range(0, rows, batch_size):
        slice_ = arrow.slice(start, batch_size)
        for i in range(slice_.num_rows):
            h = slice_.column("height")[i].as_py()
            w = slice_.column("width")[i].as_py()
            c = slice_.column("n_channels")[i].as_py()
            dtype = slice_.column("dtype")[i].as_py()
            data = slice_.column("raw_data")[i].as_py()
            arr = np.frombuffer(data, dtype=dtype).copy()
            arr = arr.reshape((h, w)) if c == 1 else arr.reshape((h, w, c))
            channel_meta_s = slice_.column("channel_meta")[i].as_py() or "{}"
            parsed_s = slice_.column("parsed_attrs")[i].as_py() or "{}"
            mask: np.ndarray | None = None
            mh = slice_.column("mask_h")[i].as_py()
            mw = slice_.column("mask_w")[i].as_py()
            if mh and mw:
                mdata = slice_.column("mask_data")[i].as_py()
                mask = np.frombuffer(mdata, dtype=np.uint8).copy().reshape((mh, mw))
            yield {
                "tmp_id": slice_.column("tmp_id")[i].as_py(),
                "group_stem": slice_.column("group_stem")[i].as_py(),
                "source_paths": list(slice_.column("source_paths")[i].as_py() or []),
                "channel_meta": json.loads(channel_meta_s),
                "array": arr,
                "parsed_attrs": json.loads(parsed_s),
                "mask": mask,
            }


def count(session_id: str, import_id: str) -> int:
    try:
        table = open_table(session_id, import_id, create_if_missing=False)
    except FileNotFoundError:
        return 0
    return table.count_rows()


def drop(session_id: str, import_id: str) -> None:
    lance_store.drop_table(lance_store.import_staging_table_name(session_id, import_id))


def shape_summary(session_id: str, import_id: str) -> dict:
    """Return aggregate shape stats over staged rows for preprocessing planning."""
    try:
        table = open_table(session_id, import_id, create_if_missing=False)
    except FileNotFoundError:
        return {"count": 0}
    arrow = table.search().select(["height", "width", "n_channels"]).to_arrow()
    if arrow.num_rows == 0:
        return {"count": 0}
    heights = np.asarray(arrow.column("height").to_pylist(), dtype=np.int32)
    widths = np.asarray(arrow.column("width").to_pylist(), dtype=np.int32)
    chans = np.asarray(arrow.column("n_channels").to_pylist(), dtype=np.int32)
    return {
        "count": int(arrow.num_rows),
        "min_h": int(heights.min()), "max_h": int(heights.max()), "mean_h": int(round(heights.mean())),
        "min_w": int(widths.min()),  "max_w": int(widths.max()),  "mean_w": int(round(widths.mean())),
        "n_channels": int(chans.max()),
    }
