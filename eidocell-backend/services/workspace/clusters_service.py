import statistics
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy import case, func
from sqlalchemy.orm import Session as DbSession

from core.notifications import notification_manager
from core.processors.image_utils import generate_collage, MAX_COLLAGE_SAMPLES
from core.processors.inference.clustering import get_processor as get_clustering_processor
from core.storage import features as lance_features
from core.storage import mask_attrs as lance_mask_attrs
from core.utils import get_active_samples, random_color
from models.models import Cluster, Mask, Sample, SampleClass, Session, sample_clusters
from services._pipeline_utils import compute_quality
from services.workspace.classes_service import _STAT_ATTRIBUTES


def _run_clustering_on_features(features: np.ndarray, n_clusters: int, **kwargs) -> np.ndarray:
    """Run clustering via the processor interface."""
    processor = get_clustering_processor("kmeans")
    return processor.fit_predict(features, n_clusters, **kwargs)


def _load_features_for_samples(
    session_id: str, method: str, samples: list[Sample]
) -> tuple[list[Sample], np.ndarray]:
    """Pull feature vectors for samples that have them; returns aligned (samples, matrix)."""
    sample_ids = [s.id for s in samples]
    found_ids, matrix = lance_features.load_vectors(session_id, method, sample_ids)
    by_id = {s.id: s for s in samples}
    aligned = [by_id[sid] for sid in found_ids if sid in by_id]
    return aligned, matrix


def _quality_for_subset(
    session_id: str, method: str, samples: list[Sample]
) -> float | None:
    """Mean distance to centroid for one cluster's worth of samples.

    Returns None if features aren't available or the subset is too small.
    """
    if len(samples) < 2:
        return None
    _, matrix = _load_features_for_samples(session_id, method, samples)
    if matrix.shape[0] < 2:
        return None
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    return compute_quality(matrix, labels, 0)


# ── Run clustering ──────────────────────────────────────────────────────


def _is_labeled(sample: Sample, uncat_id: str | None) -> bool:
    if sample.class_id is None:
        return False
    return sample.class_id != uncat_id


def run_clustering(
    db: DbSession, session_id: str, n_clusters: int, feature_method: str = "mobilenetv3"
) -> dict:
    samples = get_active_samples(db, session_id)
    if len(samples) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {n_clusters} active samples, got {len(samples)}",
        )

    aligned, features = _load_features_for_samples(session_id, feature_method, samples)
    if features.shape[0] < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {n_clusters} samples with extracted features for "
                f"method '{feature_method}', got {features.shape[0]}"
            ),
        )

    labels = _run_clustering_on_features(features, n_clusters)

    _clear_clusters(db, session_id)
    uncat_id = _uncategorized_class_id(db, session_id)

    clusters_out = []
    for k in range(n_clusters):
        quality = compute_quality(features, labels, k)
        cluster = Cluster(
            session_id=session_id,
            color=random_color(),
            quality_score=quality,
            feature_method=feature_method,
        )
        db.add(cluster)
        db.flush()

        cluster_samples = [aligned[i] for i, lbl in enumerate(labels) if lbl == k]
        for s in cluster_samples:
            s.clusters.append(cluster)

        clusters_out.append({
            "id": cluster.id,
            "color": cluster.color,
            "sample_count": len(cluster_samples),
            "labeled_count": sum(1 for s in cluster_samples if _is_labeled(s, uncat_id)),
            "quality_score": quality,
            "feature_method": feature_method,
        })

    db.commit()
    total = sum(c["sample_count"] for c in clusters_out)
    return {"clusters": clusters_out, "total_samples_clustered": total}


# ── CRUD ────────────────────────────────────────────────────────────────


def _uncategorized_class_id(db: DbSession, session_id: str) -> str | None:
    cls = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == session_id, SampleClass.name == "Uncategorized")
        .first()
    )
    return cls.id if cls else None


def list_clusters(db: DbSession, session_id: str) -> list[dict]:
    """Return cluster summaries with sample_count and labeled_count in one query."""
    uncat_id = _uncategorized_class_id(db, session_id)
    labeled_expr = case(
        (
            (Sample.class_id.is_not(None)) & (Sample.class_id != uncat_id),
            1,
        ),
        else_=0,
    ) if uncat_id else case((Sample.class_id.is_not(None), 1), else_=0)

    rows = (
        db.query(
            Cluster.id,
            Cluster.color,
            Cluster.quality_score,
            Cluster.feature_method,
            func.count(sample_clusters.c.sample_id).label("sample_count"),
            func.coalesce(func.sum(labeled_expr), 0).label("labeled_count"),
        )
        .outerjoin(sample_clusters, sample_clusters.c.cluster_id == Cluster.id)
        .outerjoin(Sample, Sample.id == sample_clusters.c.sample_id)
        .filter(Cluster.session_id == session_id)
        .group_by(Cluster.id, Cluster.color, Cluster.quality_score, Cluster.feature_method)
        .all()
    )
    return [
        {
            "id": r.id,
            "color": r.color,
            "sample_count": int(r.sample_count or 0),
            "labeled_count": int(r.labeled_count or 0),
            "quality_score": r.quality_score,
            "feature_method": r.feature_method,
        }
        for r in rows
    ]


def get_cluster_samples(
    db: DbSession, cluster_id: str, offset: int = 0, limit: int = 100
) -> tuple[list[dict], int]:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    base = (
        db.query(Sample)
        .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
        .filter(sample_clusters.c.cluster_id == cluster_id, Sample.is_active == True)
    )
    total = base.count()
    samples = base.order_by(Sample.filename).offset(offset).limit(limit).all()

    sample_ids = [s.id for s in samples]
    has_mask_sids: set[str] = set()
    if sample_ids:
        has_mask_sids = {
            sid for (sid,) in db.query(Mask.sample_id)
            .filter(Mask.sample_id.in_(sample_ids))
            .distinct()
        }

    items = [
        {
            "id": s.id,
            "filename": s.filename,
            "path": s.path,
            "is_active": s.is_active,
            "class_id": s.class_id,
            "class_name": s.sample_class.name if s.sample_class else None,
            "class_color": s.sample_class.color if s.sample_class else None,
            "has_mask": s.id in has_mask_sids,
        }
        for s in samples
    ]
    return items, total


def delete_cluster(db: DbSession, cluster_id: str) -> None:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")
    db.delete(cluster)
    db.commit()


def clear_clusters(db: DbSession, session_id: str) -> int:
    return _clear_clusters(db, session_id)


def _clear_clusters(db: DbSession, session_id: str) -> int:
    clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
    count = len(clusters)
    for c in clusters:
        db.delete(c)
    db.commit()
    return count


# ── Split & Merge ───────────────────────────────────────────────────────


def split_cluster(
    db: DbSession, cluster_id: str, n_sub_clusters: int
) -> dict:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    parent_feature_method = cluster.feature_method or "mobilenetv3"
    deleted_id = cluster.id
    session_id = cluster.session_id

    samples = (
        db.query(Sample)
        .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
        .filter(sample_clusters.c.cluster_id == cluster_id, Sample.is_active == True)
        .order_by(Sample.filename)
        .all()
    )
    if len(samples) < n_sub_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Cluster has {len(samples)} samples, need at least {n_sub_clusters}",
        )

    aligned, features = _load_features_for_samples(session_id, parent_feature_method, samples)
    if features.shape[0] < n_sub_clusters:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cluster has {features.shape[0]} samples with features for "
                f"method '{parent_feature_method}', need at least {n_sub_clusters}"
            ),
        )
    labels = _run_clustering_on_features(features, n_sub_clusters, max_iter=100, n_init=5)

    uncat_id = _uncategorized_class_id(db, session_id)

    new_clusters = []
    for k in range(n_sub_clusters):
        quality = compute_quality(features, labels, k)
        new_cluster = Cluster(
            session_id=session_id,
            color=random_color(),
            quality_score=quality,
            feature_method=parent_feature_method,
        )
        db.add(new_cluster)
        db.flush()

        sub_samples = [aligned[i] for i, lbl in enumerate(labels) if lbl == k]
        for s in sub_samples:
            s.clusters.append(new_cluster)

        new_clusters.append({
            "id": new_cluster.id,
            "color": new_cluster.color,
            "sample_count": len(sub_samples),
            "labeled_count": sum(1 for s in sub_samples if _is_labeled(s, uncat_id)),
            "quality_score": quality,
            "feature_method": parent_feature_method,
        })

    db.delete(cluster)
    db.commit()
    return {"new_clusters": new_clusters, "deleted_cluster_id": deleted_id}


def merge_clusters(db: DbSession, cluster_ids: list[str]) -> dict:
    clusters = db.query(Cluster).filter(Cluster.id.in_(cluster_ids)).all()
    if len(clusters) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid clusters to merge")

    session_ids = {c.session_id for c in clusters}
    if len(session_ids) > 1:
        raise HTTPException(status_code=400, detail="All clusters must belong to the same session")

    session_id = session_ids.pop()
    deleted_ids = [c.id for c in clusters]

    methods = {c.feature_method for c in clusters}
    merged_method = methods.pop() if len(methods) == 1 else None

    all_samples: set[Sample] = set()
    for c in clusters:
        samples = (
            db.query(Sample)
            .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id == c.id)
            .all()
        )
        all_samples.update(samples)

    sample_list = list(all_samples)
    quality = (
        _quality_for_subset(session_id, merged_method, sample_list)
        if merged_method is not None
        else None
    )

    merged = Cluster(
        session_id=session_id,
        color=random_color(),
        quality_score=quality,
        feature_method=merged_method,
    )
    db.add(merged)
    db.flush()

    for s in sample_list:
        s.clusters.append(merged)

    for c in clusters:
        db.delete(c)

    db.commit()
    uncat_id = _uncategorized_class_id(db, session_id)
    return {
        "new_cluster": {
            "id": merged.id,
            "color": merged.color,
            "sample_count": len(sample_list),
            "labeled_count": sum(1 for s in sample_list if _is_labeled(s, uncat_id)),
            "quality_score": quality,
            "feature_method": merged_method,
        },
        "deleted_cluster_ids": deleted_ids,
    }


# ── Assign to class ────────────────────────────────────────────────────


def assign_clusters_to_class(
    db: DbSession, cluster_ids: list[str], class_id: str
) -> int:
    cls = db.query(SampleClass).filter(SampleClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    sample_ids = [
        sid for (sid,) in (
            db.query(sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id.in_(cluster_ids))
            .distinct()
        )
    ]
    if sample_ids:
        updated = (
            db.query(Sample)
            .filter(Sample.id.in_(sample_ids), Sample.is_active == True)
            .update({"class_id": class_id}, synchronize_session=False)
        )
    else:
        updated = 0

    db.commit()

    if updated > 0:
        n_clusters = len(cluster_ids)
        notification_manager.broadcast(
            title="Samples assigned",
            message=(
                f"Assigned {updated} sample{'s' if updated != 1 else ''} from "
                f"{n_clusters} cluster{'s' if n_clusters != 1 else ''} to {cls.name}."
            ),
            level="success",
            data={
                "class_id": cls.id,
                "class_name": cls.name,
                "count": updated,
                "cluster_count": n_clusters,
            },
        )

    return updated


# ── Statistics ──────────────────────────────────────────────────────────


def get_cluster_statistics(db: DbSession, cluster_id: str) -> dict:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    sample_ids = [
        sid for (sid,) in (
            db.query(Sample.id)
            .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id == cluster_id, Sample.is_active == True)
            .all()
        )
    ]
    sample_count = len(sample_ids)

    attrs_by_sid = lance_mask_attrs.get_attrs_bulk(cluster.session_id, sample_ids)

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
        "id": cluster.id,
        "color": cluster.color,
        "sample_count": sample_count,
        "attributes": attr_stats,
    }


# ── Preview collage ─────────────────────────────────────────────────────


def get_or_generate_collage(db: DbSession, cluster_id: str) -> Path:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    session = db.query(Session).filter(Session.id == cluster.session_id).first()
    collage_dir = Path(session.session_folder) / "previews" / "collages"
    collage_dir.mkdir(parents=True, exist_ok=True)
    collage_path = collage_dir / f"cluster_{cluster_id}.jpg"

    samples = (
        db.query(Sample)
        .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
        .filter(sample_clusters.c.cluster_id == cluster_id, Sample.is_active == True)
        .order_by(Sample.filename)
        .limit(MAX_COLLAGE_SAMPLES)
        .all()
    )
    # Prefer the pre-generated channel-0 thumbnail (default brightfield channel).
    thumb_dir = Path(session.session_folder) / "previews" / "thumbnails"
    image_paths: list[Path] = []
    for s in samples:
        ch0 = thumb_dir / f"{s.id}_ch0.jpg"
        combined = thumb_dir / f"{s.id}.jpg"
        if ch0.is_file():
            image_paths.append(ch0)
        elif combined.is_file():
            image_paths.append(combined)
        elif s.path:
            image_paths.append(Path(s.path))
    generate_collage(image_paths, collage_path)
    return collage_path
