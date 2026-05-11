from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from core.notifications import notification_manager
from core.processors.image_utils import (
    channel_to_uint8,
    generate_thumbnail,
    render_array_to_full,
    render_array_to_thumbnail,
    render_channel_to_thumbnail,
)
from core.storage import images as lance_images
from core.storage import mask_attrs as lance_mask_attrs
from models.models import Mask, Sample, SampleClass
from schemas.workspace.gallery import (
    BulkClassAssignment,
    ClassCreate,
    ClassUpdate,
    FilterCondition,
    SampleListParams,
)


# ── Samples ──────────────────────────────────────────────────────────────


def list_samples(db: DbSession, session_id: str, params: SampleListParams) -> tuple[list[dict], int]:
    """List samples with filters/sort. attr:* sorts are resolved via Lance."""
    base_query = (
        db.query(Sample, SampleClass)
        .outerjoin(SampleClass, Sample.class_id == SampleClass.id)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
    )
    for f in params.filters:
        base_query = _apply_filter(base_query, f)

    if params.sort_by.startswith("attr:"):
        attr_name, attr_channel = _parse_attr_sort(params.sort_by)
        if attr_name not in lance_mask_attrs.ATTRIBUTE_NAMES:
            raise HTTPException(status_code=400, detail=f"Unknown attribute: {attr_name}")

        # Pull unsorted candidate ids/classes from SQL, then order by Lance.
        candidate_rows = base_query.all()
        sample_by_id = {s.id: (s, cls) for s, cls in candidate_rows}
        if not sample_by_id:
            return [], 0

        ordered_ids = _attr_sorted_sample_ids(
            session_id,
            attr_name,
            list(sample_by_id.keys()),
            descending=(params.sort_order == "desc"),
            channel_index=attr_channel,
        )
        # Append samples missing the attribute at the end (preserves total count)
        missing = [sid for sid in sample_by_id if sid not in set(ordered_ids)]
        ordered_ids.extend(missing)

        total = len(ordered_ids)
        page_ids = ordered_ids[params.offset : params.offset + params.limit]
        rows = [sample_by_id[sid] for sid in page_ids if sid in sample_by_id]
    else:
        sort_col = _get_sort_column(params.sort_by)
        if params.sort_order == "desc":
            sort_col = sort_col.desc()
        ordered_query = base_query.order_by(sort_col)
        total = ordered_query.count()
        rows = ordered_query.offset(params.offset).limit(params.limit).all()

    sample_ids_in_page = [sample.id for sample, _ in rows]
    mask_channels_by_sid: dict[str, list[int]] = {}
    if sample_ids_in_page:
        for m_sid, ci in (
            db.query(Mask.sample_id, Mask.channel_index)
            .filter(Mask.sample_id.in_(sample_ids_in_page))
            .all()
        ):
            mask_channels_by_sid.setdefault(m_sid, []).append(int(ci))
    for sid in mask_channels_by_sid:
        mask_channels_by_sid[sid].sort()

    items = []
    for sample, sample_class in rows:
        chans = mask_channels_by_sid.get(sample.id, [])
        items.append({
            "id": sample.id,
            "filename": sample.filename,
            "path": sample.path,
            "is_active": sample.is_active,
            "class_id": sample.class_id,
            "class_name": sample_class.name if sample_class else None,
            "class_color": sample_class.color if sample_class else None,
            "has_mask": bool(chans),
            "mask_channels": chans,
            "n_channels": _n_channels_from_meta(sample.channels),
        })

    return items, total


def _n_channels_from_meta(channels_meta: dict | None) -> int:
    if not channels_meta:
        return 1
    shape = channels_meta.get("shape")
    if shape and len(shape) >= 3:
        try:
            return int(shape[2]) or 1
        except (TypeError, ValueError):
            pass
    order = channels_meta.get("order") or []
    return max(1, len(order))


def _parse_attr_sort(sort_by: str) -> tuple[str, int]:
    """Parse ``attr:<name>`` (back-compat, ch 0) or ``attr:<name>:ch:<idx>``.

    Returns ``(attr_name, channel_index)``.
    """
    parts = sort_by.split(":")
    if len(parts) == 2 and parts[0] == "attr":
        return parts[1], 0
    if len(parts) == 4 and parts[0] == "attr" and parts[2] == "ch":
        try:
            ch = int(parts[3])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid channel in sort_by: {sort_by}")
        return parts[1], ch
    raise HTTPException(status_code=400, detail=f"Invalid sort_by grammar: {sort_by}")


def _attr_sorted_sample_ids(
    session_id: str,
    attr: str,
    candidate_ids: list[str],
    *,
    descending: bool,
    channel_index: int = 0,
) -> list[str]:
    """Return candidate_ids ordered by Lance attribute value. Missing values dropped."""
    if not candidate_ids:
        return []
    sids, cols = lance_mask_attrs.fetch_columns(
        session_id, [attr], sample_ids=candidate_ids, channel_index=channel_index,
    )
    if not sids:
        return []
    arr = cols[attr]
    order = arr.argsort(kind="stable")
    if descending:
        order = order[::-1]
    return [sids[i] for i in order]


def get_sample(db: DbSession, sample_id: str) -> dict:
    row = (
        db.query(Sample, SampleClass)
        .outerjoin(SampleClass, Sample.class_id == SampleClass.id)
        .filter(Sample.id == sample_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Sample not found")
    sample, sample_class = row
    mask_chans = sorted(
        int(ci) for (ci,) in db.query(Mask.channel_index).filter(Mask.sample_id == sample.id).all()
    )
    return {
        "id": sample.id,
        "filename": sample.filename,
        "path": sample.path,
        "is_active": sample.is_active,
        "class_id": sample.class_id,
        "class_name": sample_class.name if sample_class else None,
        "class_color": sample_class.color if sample_class else None,
        "has_mask": bool(mask_chans),
        "mask_channels": mask_chans,
        "n_channels": _n_channels_from_meta(sample.channels),
    }


def get_image_path(db: DbSession, sample_id: str, *, channel: int | None = None) -> Path:
    """Return a disk path serving the full-resolution image.

    Reads the array from the per-session ``images_{sid}`` Lance table and
    caches a rendered PNG under ``previews/full/{sample_id}.png`` (or
    ``previews/full/{sample_id}_ch{idx}.png`` when ``channel`` is set).
    Falls back to the legacy ``Sample.path`` if no Lance row exists.
    """
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    arr = lance_images.read_array(sample.session_id, sample.id)
    if arr is None:
        if channel is not None and channel != 0:
            raise HTTPException(status_code=404, detail="Per-channel image not available for legacy sample")
        path = Path(sample.path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return path

    from models.models import Session
    session = db.query(Session).filter(Session.id == sample.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    full_dir = Path(session.session_folder) / "previews" / "full"
    if channel is None:
        cache = full_dir / f"{sample.id}.png"
        if not cache.exists():
            render_array_to_full(arr, cache)
        return cache

    if channel < 0:
        raise HTTPException(status_code=400, detail="channel must be >= 0")
    if arr.ndim == 3 and channel >= arr.shape[2]:
        raise HTTPException(status_code=400, detail=f"channel {channel} out of range (n_channels={arr.shape[2]})")
    cache = full_dir / f"{sample.id}_ch{channel}.png"
    if not cache.exists():
        from PIL import Image
        cache.parent.mkdir(parents=True, exist_ok=True)
        u8 = channel_to_uint8(arr, channel)
        Image.fromarray(u8, mode="L").save(cache, "PNG")
    return cache


def get_thumbnail_path(db: DbSession, sample_id: str, session_folder: str) -> Path:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    thumb_dir = Path(session_folder) / "previews" / "thumbnails"
    thumb_path = thumb_dir / f"{sample.id}.jpg"
    if thumb_path.exists():
        return thumb_path

    arr = lance_images.read_array(sample.session_id, sample.id)
    if arr is not None:
        render_array_to_thumbnail(arr, thumb_path)
        return thumb_path

    # Legacy fallback for sessions imported before the Lance images store.
    source = Path(sample.path)
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Source image not found on disk")
    generate_thumbnail(source, thumb_path)
    return thumb_path


def get_channel_thumbnail_path(
    db: DbSession, sample_id: str, session_folder: str, channel: int
) -> Path:
    """Return a cached JPEG of a single channel of a sample's image array."""
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    if channel < 0:
        raise HTTPException(status_code=400, detail="channel must be >= 0")

    thumb_dir = Path(session_folder) / "previews" / "thumbnails"
    thumb_path = thumb_dir / f"{sample.id}_ch{channel}.jpg"
    if thumb_path.exists():
        return thumb_path

    arr = lance_images.read_array(sample.session_id, sample.id)
    if arr is None:
        raise HTTPException(status_code=404, detail="Image array not found")
    try:
        render_channel_to_thumbnail(arr, channel, thumb_path)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return thumb_path


# ── Classes ──────────────────────────────────────────────────────────────


def list_classes(db: DbSession, session_id: str) -> list[dict]:
    classes = db.query(SampleClass).filter(SampleClass.session_id == session_id).all()
    results = []
    for c in classes:
        count = db.query(func.count(Sample.id)).filter(
            Sample.class_id == c.id, Sample.is_active == True
        ).scalar()
        results.append({
            "id": c.id,
            "name": c.name,
            "color": c.color,
            "sample_count": count,
        })
    return results


def create_class(db: DbSession, session_id: str, data: ClassCreate) -> dict:
    existing = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == session_id, SampleClass.name == data.name)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"Class '{data.name}' already exists")

    cls = SampleClass(session_id=session_id, name=data.name, color=data.color)
    db.add(cls)
    db.commit()
    db.refresh(cls)
    return {"id": cls.id, "name": cls.name, "color": cls.color, "sample_count": 0}


def update_class(db: DbSession, class_id: str, data: ClassUpdate) -> dict:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    if data.name is not None:
        dup = (
            db.query(SampleClass)
            .filter(
                SampleClass.session_id == cls.session_id,
                SampleClass.name == data.name,
                SampleClass.id != class_id,
            )
            .first()
        )
        if dup:
            raise HTTPException(status_code=409, detail=f"Class '{data.name}' already exists")
        cls.name = data.name
    if data.color is not None:
        cls.color = data.color

    db.commit()
    db.refresh(cls)
    count = db.query(func.count(Sample.id)).filter(
        Sample.class_id == cls.id, Sample.is_active == True
    ).scalar()
    return {"id": cls.id, "name": cls.name, "color": cls.color, "sample_count": count}


def delete_class(db: DbSession, class_id: str) -> None:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    if cls.name == "Uncategorized":
        raise HTTPException(status_code=400, detail="Cannot delete the Uncategorized class")

    uncategorized = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == cls.session_id, SampleClass.name == "Uncategorized")
        .first()
    )
    db.query(Sample).filter(Sample.class_id == class_id).update(
        {"class_id": uncategorized.id if uncategorized else None}
    )
    db.delete(cls)
    db.commit()


# ── Bulk assignment ──────────────────────────────────────────────────────


def assign_samples_to_class(db: DbSession, data: BulkClassAssignment) -> int:
    cls = None
    if data.class_id is not None:
        cls = db.query(SampleClass).filter(SampleClass.id == data.class_id).first()
        if not cls:
            raise HTTPException(status_code=404, detail="Class not found")

    updated = (
        db.query(Sample)
        .filter(Sample.id.in_(data.sample_ids))
        .update({"class_id": data.class_id}, synchronize_session="fetch")
    )
    db.commit()

    if updated > 0:
        if cls is not None:
            notification_manager.broadcast(
                title="Samples assigned",
                message=f"Assigned {updated} sample{'s' if updated != 1 else ''} to {cls.name}.",
                level="success",
                data={"class_id": cls.id, "class_name": cls.name, "count": updated},
            )
        else:
            notification_manager.broadcast(
                title="Samples unassigned",
                message=f"Removed class from {updated} sample{'s' if updated != 1 else ''}.",
                level="info",
                data={"count": updated},
            )

    return updated


# ── Sortable attributes ─────────────────────────────────────────────────


def list_sortable_attributes(db: DbSession, session_id: str) -> list[dict]:
    """Return sortable attributes per channel.

    Each entry: ``{name, channel_index, channel_name, value}`` where ``value`` is the
    sort_by token (e.g. ``attr:area:ch:0``). One entry per (attr, channel) with a mask.
    """
    from models.models import Mask, Session as SessionModel
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        return []
    names = sorted(lance_mask_attrs.list_attribute_names(session_id))
    if not names:
        return []
    channel_indices_with_mask: set[int] = {
        int(ci) for (ci,) in db.query(Mask.channel_index).distinct().all()
        if ci is not None
    }
    # Restrict to masks that actually belong to this session via samples.
    from models.models import Sample
    session_sample_ids = {sid for (sid,) in db.query(Sample.id).filter(Sample.session_id == session_id).all()}
    channel_indices_with_mask = {
        int(ci) for (sid, ci) in db.query(Mask.sample_id, Mask.channel_index).all()
        if sid in session_sample_ids and ci is not None
    }
    if not channel_indices_with_mask:
        channel_indices_with_mask = {0}

    channel_names = session.channel_names or []
    out: list[dict] = []
    for ch in sorted(channel_indices_with_mask):
        ch_name = channel_names[ch] if ch < len(channel_names) else f"Channel {ch + 1}"
        for attr in names:
            out.append({
                "name": attr,
                "channel_index": ch,
                "channel_name": ch_name,
                "value": f"attr:{attr}:ch:{ch}",
            })
    return out


# ── Filtering helpers ────────────────────────────────────────────────────


def _get_sort_column(sort_by: str):
    mapping = {
        "filename": Sample.filename,
        "class_name": SampleClass.name,
    }
    col = mapping.get(sort_by)
    if col is None:
        raise HTTPException(status_code=400, detail=f"Unknown sort field: {sort_by}")
    return col


def _apply_filter(query, f: FilterCondition):
    if f.field == "filename":
        return _apply_string_filter(query, Sample.filename, f)
    elif f.field == "class_name":
        return _apply_string_filter(query, SampleClass.name, f)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown filter field: {f.field}")


def _apply_string_filter(query, column, f: FilterCondition):
    val = str(f.value)
    if f.operator == "==":
        return query.filter(column == val)
    elif f.operator == "!=":
        return query.filter(column != val)
    elif f.operator == "contains":
        return query.filter(column.ilike(f"%{val}%"))
    elif f.operator == "not_contains":
        return query.filter(~column.ilike(f"%{val}%"))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported operator '{f.operator}' for string field")
