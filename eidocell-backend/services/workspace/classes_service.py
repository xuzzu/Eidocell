import statistics
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from core.processors.image_utils import generate_collage, MAX_COLLAGE_SAMPLES
from core.storage import mask_attrs as lance_mask_attrs
from models.models import Sample, SampleClass, Session

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

    sample_ids = [
        sid for (sid,) in db.query(Sample.id)
        .filter(Sample.class_id == class_id, Sample.is_active == True)
        .all()
    ]
    sample_count = len(sample_ids)

    attrs_by_sid = lance_mask_attrs.get_attrs_bulk(cls.session_id, sample_ids)

    attr_stats = []
    for attr_name in _STAT_ATTRIBUTES:
        values: list[float] = []
        for attrs in attrs_by_sid.values():
            v = attrs.get(attr_name)
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


def get_session_attribute_distributions(
    db: DbSession, session_id: str, bins: int = 30
) -> dict:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    sample_ids = [
        sid for (sid,) in db.query(Sample.id)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .all()
    ]
    attrs_by_sid = lance_mask_attrs.get_attrs_bulk(session_id, sample_ids)

    distributions = []
    for attr_name in _STAT_ATTRIBUTES:
        values: list[float] = []
        for attrs in attrs_by_sid.values():
            v = attrs.get(attr_name)
            if v is not None and isinstance(v, (int, float)):
                values.append(float(v))

        if not values:
            distributions.append({
                "name": attr_name,
                "bin_edges": [],
                "bin_counts": [],
            })
            continue

        arr = np.asarray(values, dtype=float)
        counts, edges = np.histogram(arr, bins=bins)
        distributions.append({
            "name": attr_name,
            "bin_edges": edges.tolist(),
            "bin_counts": counts.astype(int).tolist(),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        })

    return {
        "sample_count": len(attrs_by_sid),
        "attributes": distributions,
    }


def get_or_generate_collage(db: DbSession, class_id: str) -> Path:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    session = db.query(Session).filter(Session.id == cls.session_id).first()
    collage_dir = Path(session.session_folder) / "previews" / "collages"
    collage_dir.mkdir(parents=True, exist_ok=True)
    collage_path = collage_dir / f"{class_id}.jpg"

    samples = (
        db.query(Sample)
        .filter(Sample.class_id == class_id, Sample.is_active == True)
        .order_by(Sample.filename)
        .limit(MAX_COLLAGE_SAMPLES)
        .all()
    )
    image_paths = [Path(s.path) for s in samples]
    generate_collage(image_paths, collage_path)
    return collage_path
