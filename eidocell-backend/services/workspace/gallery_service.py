from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from core.processors.image_utils import generate_thumbnail
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
        attr_name = params.sort_by.split(":", 1)[1]
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
    mask_sample_ids = (
        {m.sample_id for m in db.query(Mask.sample_id).filter(Mask.sample_id.in_(sample_ids_in_page)).all()}
        if sample_ids_in_page else set()
    )

    items = []
    for sample, sample_class in rows:
        items.append({
            "id": sample.id,
            "filename": sample.filename,
            "path": sample.path,
            "is_active": sample.is_active,
            "class_id": sample.class_id,
            "class_name": sample_class.name if sample_class else None,
            "class_color": sample_class.color if sample_class else None,
            "has_mask": sample.id in mask_sample_ids,
        })

    return items, total


def _attr_sorted_sample_ids(
    session_id: str, attr: str, candidate_ids: list[str], *, descending: bool
) -> list[str]:
    """Return candidate_ids ordered by Lance attribute value. Missing values dropped."""
    if not candidate_ids:
        return []
    sids, cols = lance_mask_attrs.fetch_columns(
        session_id, [attr], sample_ids=candidate_ids
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
    has_mask = db.query(Mask.id).filter(Mask.sample_id == sample.id).first() is not None
    return {
        "id": sample.id,
        "filename": sample.filename,
        "path": sample.path,
        "is_active": sample.is_active,
        "class_id": sample.class_id,
        "class_name": sample_class.name if sample_class else None,
        "class_color": sample_class.color if sample_class else None,
        "has_mask": has_mask,
    }


def get_image_path(db: DbSession, sample_id: str) -> Path:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    path = Path(sample.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return path


def get_thumbnail_path(db: DbSession, sample_id: str, session_folder: str) -> Path:
    sample = db.query(Sample).filter(Sample.id == sample_id).first()
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    thumb_dir = Path(session_folder) / "previews" / "thumbnails"
    thumb_path = thumb_dir / f"{sample.id}.jpg"

    if not thumb_path.exists():
        source = Path(sample.path)
        if not source.is_file():
            raise HTTPException(status_code=404, detail="Source image not found on disk")
        generate_thumbnail(source, thumb_path)

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
    return updated


# ── Sortable attributes ─────────────────────────────────────────────────


def list_sortable_attributes(db: DbSession, session_id: str) -> list[str]:
    """Return attribute names available for sorting (driven by Lance schema)."""
    return sorted(lance_mask_attrs.list_attribute_names(session_id))


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
