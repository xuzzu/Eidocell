"""LanceDB connection + per-session table accessors.

Single shared LanceDB instance under ~/.eidocell/lance/. Tables are namespaced
per session via name suffix (features_{sid}, mask_attrs_{sid}). Session
deletion drops the matching tables; everything else lives in raw files.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import lancedb
import pyarrow as pa

from core.config import LANCE_DIR

logger = logging.getLogger("eidocell.storage.lance")


FEATURES_PREFIX = "features_"
MASK_ATTRS_PREFIX = "mask_attrs_"
IMAGES_PREFIX = "images_"
SAMPLE_ATTRS_PREFIX = "sample_attrs_"
IMPORT_STAGING_PREFIX = "import_staging_"


@lru_cache(maxsize=1)
def connect() -> "lancedb.DBConnection":
    """Return the cached connection to the shared LanceDB instance."""
    LANCE_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(LANCE_DIR))


def features_table_name(session_id: str) -> str:
    return f"{FEATURES_PREFIX}{session_id}"


def mask_attrs_table_name(session_id: str) -> str:
    return f"{MASK_ATTRS_PREFIX}{session_id}"


def images_table_name(session_id: str) -> str:
    return f"{IMAGES_PREFIX}{session_id}"


def sample_attrs_table_name(session_id: str) -> str:
    return f"{SAMPLE_ATTRS_PREFIX}{session_id}"


def import_staging_table_name(session_id: str, import_id: str) -> str:
    return f"{IMPORT_STAGING_PREFIX}{session_id}_{import_id}"


def features_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("method", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("dim", pa.int32()),
            pa.field("version", pa.int32()),
            pa.field("channel_set", pa.string()),
            pa.field("created_at", pa.timestamp("us")),
        ]
    )


def features_table(
    session_id: str,
    *,
    dim: int | None = None,
    create_if_missing: bool = False,
):
    """Return the features table for a session.

    `dim` is required when creating; ignored when opening. Raises if the table
    does not exist and `create_if_missing` is False.
    """
    db = connect()
    name = features_table_name(session_id)
    try:
        return db.open_table(name)
    except (FileNotFoundError, ValueError):
        pass
    if not create_if_missing:
        raise FileNotFoundError(f"Lance table '{name}' does not exist")
    if dim is None or dim <= 0:
        raise ValueError("dim must be a positive int when creating features table")
    try:
        return db.create_table(name, schema=features_schema(dim))
    except (ValueError, OSError):
        # Lost a race with another writer — re-open.
        return db.open_table(name)


def open_or_create_table(
    name: str,
    schema: pa.Schema,
    *,
    create_if_missing: bool = True,
):
    """Generic opener used by per-session image / sample_attrs / staging tables."""
    db = connect()
    try:
        return db.open_table(name)
    except (FileNotFoundError, ValueError):
        pass
    if not create_if_missing:
        raise FileNotFoundError(f"Lance table '{name}' does not exist")
    try:
        return db.create_table(name, schema=schema)
    except (ValueError, OSError):
        return db.open_table(name)


def drop_table(name: str) -> None:
    db = connect()
    try:
        db.drop_table(name)
    except (FileNotFoundError, ValueError):
        return
    except Exception:
        logger.exception("failed to drop lance table %s", name)


def mask_attrs_table(
    session_id: str,
    *,
    create_if_missing: bool = True,
):
    """Return the mask_attrs table for a session.

    Schema is fixed by `attrs_schema()` in mask_attrs module — passed in here
    by callers to avoid a circular import.
    """
    from core.storage.mask_attrs import attrs_schema

    db = connect()
    name = mask_attrs_table_name(session_id)
    try:
        return db.open_table(name)
    except (FileNotFoundError, ValueError):
        pass
    if not create_if_missing:
        raise FileNotFoundError(f"Lance table '{name}' does not exist")
    try:
        return db.create_table(name, schema=attrs_schema())
    except (ValueError, OSError):
        return db.open_table(name)


def drop_session_tables(session_id: str) -> None:
    """Best-effort drop of every Lance table belonging to a session.

    Includes any leftover import_staging_{sid}_* tables (from cancelled or
    crashed imports).
    """
    db = connect()
    fixed = (
        features_table_name(session_id),
        mask_attrs_table_name(session_id),
        images_table_name(session_id),
        sample_attrs_table_name(session_id),
    )
    for name in fixed:
        try:
            db.drop_table(name)
        except (FileNotFoundError, ValueError):
            continue
        except Exception:
            logger.exception("failed to drop lance table %s", name)
    # Sweep any staging tables for this session.
    staging_prefix = f"{IMPORT_STAGING_PREFIX}{session_id}_"
    try:
        names = list(db.table_names())
    except Exception:
        names = []
    for name in names:
        if name.startswith(staging_prefix):
            try:
                db.drop_table(name)
            except Exception:
                logger.exception("failed to drop staging table %s", name)


def with_retry(fn, *, attempts: int = 3, backoff: float = 0.05):
    """Retry a write op a few times to absorb Lance's optimistic-concurrency conflicts."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # lance raises a generic CommitConflict in some versions
            last_exc = e
            time.sleep(backoff * (i + 1))
    assert last_exc is not None
    raise last_exc
