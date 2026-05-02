from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from models.models import Sample, SampleClass, Mask
from core.processors.image_utils import generate_thumbnail
from schemas.workspace.gallery import (
    SampleListParams,
    FilterCondition,
    ClassCreate,
    ClassUpdate,
    BulkClassAssignment,
)


# ── Samples ──────────────────────────────────────────────────────────────


def list_samples(db: DbSession, session_id: str, params: SampleListParams) -> tuple[list[dict], int]:
    query = (
        db.query(Sample, SampleClass)
        .outerjoin(SampleClass, Sample.class_id == SampleClass.id)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
    )

    # Apply filters
    for f in params.filters:
        query = _apply_filter(query, f)

    total = query.count()

    # Sorting
    if params.sort_by.startswith("attr:"):
        attr_name = params.sort_by.split(":", 1)[1]
        sort_expr = func.json_extract(Mask.attributes, f'$.{attr_name}')
        query = query.outerjoin(Mask, Mask.sample_id == Sample.id)
        if params.sort_order == "desc":
            sort_expr = sort_expr.desc()
        query = query.order_by(sort_expr)
    else:
        sort_col = _get_sort_column(params.sort_by)
        if params.sort_order == "desc":
            sort_col = sort_col.desc()
        query = query.order_by(sort_col)

    # Pagination
    rows = query.offset(params.offset).limit(params.limit).all()

    sample_ids_in_page = [sample.id for sample, _ in rows]
    mask_sample_ids = {
        m.sample_id for m in db.query(Mask.sample_id).filter(Mask.sample_id.in_(sample_ids_in_page)).all()
    } if sample_ids_in_page else set()

    items = []
    for sample, sample_class in rows:
        has_mask = sample.id in mask_sample_ids
        items.append({
            "id": sample.id,
            "filename": sample.filename,
            "path": sample.path,
            "storage_index": sample.storage_index,
            "is_active": sample.is_active,
            "class_id": sample.class_id,
            "class_name": sample_class.name if sample_class else None,
            "class_color": sample_class.color if sample_class else None,
            "has_mask": has_mask,
        })

    return items, total


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
        "storage_index": sample.storage_index,
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

    thumb_dir = Path(session_folder) / "thumbnails"
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
        # Check uniqueness within session
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

    # Move samples back to Uncategorized
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
    """Return a list of attribute names available for sorting (from mask attributes)."""
    mask = (
        db.query(Mask)
        .join(Sample, Mask.sample_id == Sample.id)
        .filter(Sample.session_id == session_id, Mask.attributes.isnot(None))
        .first()
    )
    if not mask or not mask.attributes:
        return []
    return sorted(mask.attributes.keys())


# ── Filtering helpers ────────────────────────────────────────────────────


def _get_sort_column(sort_by: str):
    mapping = {
        "filename": Sample.filename,
        "storage_index": Sample.storage_index,
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
