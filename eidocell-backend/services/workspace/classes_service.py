import statistics
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from models.models import Mask, Sample, SampleClass, Session
from core.processors.image_utils import generate_collage, MAX_COLLAGE_SAMPLES

# Key attributes to aggregate for class statistics
_STAT_ATTRIBUTES = [
    "area", "perimeter", "equivalent_diameter", "aspect_ratio",
    "solidity", "form_factor", "mean_intensity", "std_intensity",
    "thickness_mean", "snr",
]


def get_class_summary(db: DbSession, class_id: str) -> dict:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    count = db.query(func.count(Sample.id)).filter(
        Sample.class_id == class_id, Sample.is_active == True
    ).scalar()

    return {
        "id": cls.id,
        "name": cls.name,
        "color": cls.color,
        "sample_count": count,
    }


def list_class_samples(
    db: DbSession, class_id: str, offset: int = 0, limit: int = 100
) -> tuple[list[dict], int]:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    base = db.query(Sample).filter(
        Sample.class_id == class_id, Sample.is_active == True
    )
    total = base.count()
    samples = base.order_by(Sample.filename).offset(offset).limit(limit).all()

    items = []
    for s in samples:
        items.append({
            "id": s.id,
            "filename": s.filename,
            "path": s.path,
            "storage_index": s.storage_index,
            "is_active": s.is_active,
            "class_id": s.class_id,
            "class_name": cls.name,
            "class_color": cls.color,
            "has_mask": s.mask is not None,
        })
    return items, total


def get_class_statistics(db: DbSession, class_id: str) -> dict:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    rows = (
        db.query(Mask.attributes)
        .join(Sample, Sample.id == Mask.sample_id)
        .filter(Sample.class_id == class_id, Sample.is_active == True, Mask.attributes.isnot(None))
        .all()
    )

    sample_count = db.query(func.count(Sample.id)).filter(
        Sample.class_id == class_id, Sample.is_active == True
    ).scalar()

    attr_stats = []
    for attr_name in _STAT_ATTRIBUTES:
        values = []
        for (attrs,) in rows:
            if attrs and attr_name in attrs:
                v = attrs[attr_name]
                if v is not None and isinstance(v, (int, float)):
                    values.append(float(v))

        if values:
            attr_stats.append({
                "name": attr_name,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            })
        else:
            attr_stats.append({"name": attr_name})

    return {
        "id": cls.id,
        "name": cls.name,
        "color": cls.color,
        "sample_count": sample_count,
        "attributes": attr_stats,
    }


def get_or_generate_collage(db: DbSession, class_id: str) -> Path:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    session = db.query(Session).filter(Session.id == cls.session_id).first()
    collage_dir = Path(session.session_folder) / "collages"
    collage_dir.mkdir(exist_ok=True)
    collage_path = collage_dir / f"{class_id}.jpg"

    # Always regenerate — class contents may have changed
    samples = (
        db.query(Sample)
        .filter(Sample.class_id == class_id, Sample.is_active == True)
        .order_by(Sample.storage_index)
        .limit(MAX_COLLAGE_SAMPLES)
        .all()
    )
    image_paths = [Path(s.path) for s in samples]
    generate_collage(image_paths, collage_path)
    return collage_path
