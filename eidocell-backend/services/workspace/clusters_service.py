from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy import case, func
from sqlalchemy.orm import Session as DbSession

from core.processors.inference.clustering import get_processor as get_clustering_processor
from core.processors.image_utils import generate_collage, MAX_COLLAGE_SAMPLES
from core.utils import random_color, load_session_features, get_active_samples
from models.models import Session, Sample, SampleClass, Cluster, sample_clusters
from services._pipeline_utils import compute_quality


def _run_clustering_on_features(features: np.ndarray, n_clusters: int, **kwargs) -> np.ndarray:
    """Run clustering via the processor interface."""
    processor = get_clustering_processor("kmeans")
    return processor.fit_predict(features, n_clusters, **kwargs)


def _quality_for_subset(
    session_folder: str, storage_indices: list[int]
) -> float | None:
    """Mean distance to centroid for one cluster's worth of samples.

    Loads features lazily; returns None if features aren't available or the
    subset is too small. Used by merge to compute the new cluster's quality.
    """
    if len(storage_indices) < 2:
        return None
    try:
        features = load_session_features(session_folder, storage_indices)
    except HTTPException:
        return None
    if features.size == 0:
        return None
    labels = np.zeros(len(features), dtype=np.int64)
    return compute_quality(features, labels, 0)


# ── Run clustering ──────────────────────────────────────────────────────


def _is_labeled(sample: Sample, uncat_id: str | None) -> bool:
    """A sample counts as 'labeled' if it has a non-Uncategorized class."""
    if sample.class_id is None:
        return False
    return sample.class_id != uncat_id


def run_clustering(db: DbSession, session_id: str, n_clusters: int) -> dict:
    samples = get_active_samples(db, session_id)
    if len(samples) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {n_clusters} active samples, got {len(samples)}",
        )

    session = db.query(Session).filter(Session.id == session_id).first()
    indices = [s.storage_index for s in samples]
    features = load_session_features(session.session_folder, indices)

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
        )
        db.add(cluster)
        db.flush()

        cluster_samples = [samples[i] for i, lbl in enumerate(labels) if lbl == k]
        for s in cluster_samples:
            s.clusters.append(cluster)

        clusters_out.append({
            "id": cluster.id,
            "color": cluster.color,
            "sample_count": len(cluster_samples),
            "labeled_count": sum(1 for s in cluster_samples if _is_labeled(s, uncat_id)),
            "quality_score": quality,
            "feature_method": None,
        })

    db.commit()
    total = sum(c["sample_count"] for c in clusters_out)
    return {"clusters": clusters_out, "total_samples_clustered": total}


# ── CRUD ────────────────────────────────────────────────────────────────


def _uncategorized_class_id(db: DbSession, session_id: str) -> str | None:
    """Each session has a default 'Uncategorized' class assigned on sample import.

    Returns its id, or None if the session is missing it (legacy data).
    """
    cls = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == session_id, SampleClass.name == "Uncategorized")
        .first()
    )
    return cls.id if cls else None


def list_clusters(db: DbSession, session_id: str) -> list[dict]:
    """Return cluster summaries with sample_count and labeled_count in one query."""
    uncat_id = _uncategorized_class_id(db, session_id)
    # A sample is "labeled" if it has a class assignment that is not the
    # auto-created Uncategorized class.
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

    items = []
    for s in samples:
        items.append({
            "id": s.id,
            "filename": s.filename,
            "path": s.path,
            "storage_index": s.storage_index,
            "is_active": s.is_active,
            "class_id": s.class_id,
            "class_name": s.sample_class.name if s.sample_class else None,
            "class_color": s.sample_class.color if s.sample_class else None,
            "has_mask": s.mask is not None,
        })
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

    parent_feature_method = cluster.feature_method
    deleted_id = cluster.id
    session = db.query(Session).filter(Session.id == cluster.session_id).first()

    samples = (
        db.query(Sample)
        .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
        .filter(sample_clusters.c.cluster_id == cluster_id, Sample.is_active == True)
        .order_by(Sample.storage_index)
        .all()
    )
    if len(samples) < n_sub_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Cluster has {len(samples)} samples, need at least {n_sub_clusters}",
        )

    indices = [s.storage_index for s in samples]
    features = load_session_features(session.session_folder, indices)
    labels = _run_clustering_on_features(features, n_sub_clusters, max_iter=100, n_init=5)

    uncat_id = _uncategorized_class_id(db, cluster.session_id)

    new_clusters = []
    for k in range(n_sub_clusters):
        quality = compute_quality(features, labels, k)
        new_cluster = Cluster(
            session_id=cluster.session_id,
            color=random_color(),
            quality_score=quality,
            feature_method=parent_feature_method,
        )
        db.add(new_cluster)
        db.flush()

        sub_samples = [samples[i] for i, lbl in enumerate(labels) if lbl == k]
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

    # If all source clusters share a feature_method, the merged cluster's
    # quality is comparable; otherwise mark mixed (None).
    methods = {c.feature_method for c in clusters}
    merged_method = methods.pop() if len(methods) == 1 else None

    session = db.query(Session).filter(Session.id == session_id).first()

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
        _quality_for_subset(
            session.session_folder, [s.storage_index for s in sample_list]
        )
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

    updated = 0
    for cid in cluster_ids:
        samples = (
            db.query(Sample)
            .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id == cid, Sample.is_active == True)
            .all()
        )
        for s in samples:
            s.class_id = class_id
            updated += 1

    db.commit()
    return updated


# ── Preview collage ─────────────────────────────────────────────────────


def get_or_generate_collage(db: DbSession, cluster_id: str) -> Path:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    session = db.query(Session).filter(Session.id == cluster.session_id).first()
    collage_dir = Path(session.session_folder) / "collages"
    collage_dir.mkdir(exist_ok=True)
    collage_path = collage_dir / f"cluster_{cluster_id}.jpg"

    samples = (
        db.query(Sample)
        .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
        .filter(sample_clusters.c.cluster_id == cluster_id, Sample.is_active == True)
        .order_by(Sample.storage_index)
        .limit(MAX_COLLAGE_SAMPLES)
        .all()
    )
    image_paths = [Path(s.path) for s in samples]
    generate_collage(image_paths, collage_path)
    return collage_path
