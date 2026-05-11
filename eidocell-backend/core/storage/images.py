"""Per-session image-array storage in LanceDB.

Each session gets one ``images_{sid}`` table holding flattened image bytes
plus shape + dtype, so any number of channels and any pre/post-preprocessing
shape can be stored without per-file disk artifacts. Reading reconstructs the
ndarray via ``np.frombuffer(data, dtype).reshape((h, w, c))`` with the channel
axis dropped if ``n_channels == 1`` and the source ndarray was 2-D.

Variable-length ``list<uint8>`` is intentional — we store both raw (variable
shape) and preprocessed (uniform shape) arrays in the same table, keyed by
``(sample_id, channel_set)``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pyarrow as pa

from core.storage import lance as lance_store

logger = logging.getLogger("eidocell.storage.images")

DEFAULT_CHANNEL_SET = "default"

_KEY_FIELDS = ["sample_id", "channel_set"]


def schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("channel_set", pa.string()),
            pa.field("height", pa.int32()),
            pa.field("width", pa.int32()),
            pa.field("n_channels", pa.int32()),
            pa.field("dtype", pa.string()),
            pa.field("data", pa.list_(pa.uint8())),
            pa.field("source", pa.string()),  # "raw" | "preprocessed"
            pa.field("created_at", pa.timestamp("us")),
        ]
    )


def _ensure_table(session_id: str):
    return lance_store.open_or_create_table(
        lance_store.images_table_name(session_id),
        schema(),
        create_if_missing=True,
    )


def _open(session_id: str):
    return lance_store.open_or_create_table(
        lance_store.images_table_name(session_id),
        schema(),
        create_if_missing=False,
    )


def _shape_for(arr: np.ndarray) -> tuple[int, int, int]:
    if arr.ndim == 2:
        h, w = arr.shape
        return h, w, 1
    if arr.ndim == 3:
        h, w, c = arr.shape
        return h, w, c
    raise ValueError(f"unsupported image ndim {arr.ndim}")


def _row_from_array(
    sample_id: str,
    arr: np.ndarray,
    *,
    channel_set: str,
    source: str,
    now: datetime,
) -> dict:
    arr = np.ascontiguousarray(arr)
    h, w, c = _shape_for(arr)
    return {
        "sample_id": sample_id,
        "channel_set": channel_set,
        "height": h,
        "width": w,
        "n_channels": c,
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
        "source": source,
        "created_at": now,
    }


def upsert_arrays(
    session_id: str,
    rows: list[dict],
) -> int:
    """Upsert ``[{sample_id, array, channel_set?, source?}]`` rows.

    ``array`` is a numpy ndarray (2-D grayscale or 3-D HWC). Existing rows for
    the same (sample_id, channel_set) key are replaced.
    """
    if not rows:
        return 0
    now = datetime.now(timezone.utc)
    payload = []
    for r in rows:
        arr = r["array"]
        if not isinstance(arr, np.ndarray):
            raise TypeError("rows[*]['array'] must be a numpy.ndarray")
        payload.append(_row_from_array(
            sample_id=r["sample_id"],
            arr=arr,
            channel_set=r.get("channel_set", DEFAULT_CHANNEL_SET),
            source=r.get("source", "preprocessed"),
            now=now,
        ))
    table = _ensure_table(session_id)
    arrow = pa.Table.from_pylist(payload, schema=schema())

    def _do_write():
        table.merge_insert(_KEY_FIELDS) \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(arrow)

    lance_store.with_retry(_do_write)
    return len(payload)


def read_array(
    session_id: str,
    sample_id: str,
    *,
    channel_set: str = DEFAULT_CHANNEL_SET,
) -> np.ndarray | None:
    """Read one sample's image as a numpy ndarray, or None if absent."""
    try:
        table = _open(session_id)
    except FileNotFoundError:
        return None
    where = f"sample_id = '{sample_id}' AND channel_set = '{channel_set}'"
    arrow = table.search().where(where).limit(1).to_arrow()
    if arrow.num_rows == 0:
        return None
    h = arrow.column("height")[0].as_py()
    w = arrow.column("width")[0].as_py()
    c = arrow.column("n_channels")[0].as_py()
    dtype = arrow.column("dtype")[0].as_py()
    data = bytes(arrow.column("data")[0].as_py())
    arr = np.frombuffer(data, dtype=dtype).copy()
    if c == 1:
        return arr.reshape((h, w))
    return arr.reshape((h, w, c))


def has_sample_ids(
    session_id: str,
    *,
    channel_set: str = DEFAULT_CHANNEL_SET,
) -> set[str]:
    try:
        table = _open(session_id)
    except FileNotFoundError:
        return set()
    where = f"channel_set = '{channel_set}'"
    arrow = table.search().where(where).select(["sample_id"]).to_arrow()
    if arrow.num_rows == 0:
        return set()
    return set(arrow.column("sample_id").to_pylist())


def delete_for_samples(session_id: str, sample_ids: Iterable[str]) -> int:
    sample_ids = list(sample_ids)
    if not sample_ids:
        return 0
    try:
        table = _open(session_id)
    except FileNotFoundError:
        return 0
    ids_literal = ",".join(f"'{sid}'" for sid in sample_ids)
    table.delete(f"sample_id IN ({ids_literal})")
    return len(sample_ids)
