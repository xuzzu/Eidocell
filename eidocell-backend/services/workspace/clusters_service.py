from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from core.processors.inference.clustering import get_processor as get_clustering_processor
from core.processors.image_utils import generate_collage, MAX_COLLAGE_SAMPLES
from core.utils import random_color, load_session_features, get_active_samples
from models.models import Session, Sample, SampleClass, Cluster, sample_clusters


def _run_clustering_on_features(features: np.ndarray, n_clusters: int, **kwargs) -> np.ndarray:
    """Run clustering via the processor interface."""
    processor = get_clustering_processor("kmeans")
    return processor.fit_predict(features, n_clusters, **kwargs)


# ── Run clustering ──────────────────────────────────────────────────────


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

    clusters_out = []
    for k in range(n_clusters):
        cluster = Cluster(session_id=session_id, color=random_color())
        db.add(cluster)
        db.flush()

        cluster_samples = [samples[i] for i, lbl in enumerate(labels) if lbl == k]
        for s in cluster_samples:
            s.clusters.append(cluster)

        clusters_out.append({
            "id": cluster.id,
            "color": cluster.color,
            "sample_count": len(cluster_samples),
        })

    db.commit()
    total = sum(c["sample_count"] for c in clusters_out)
    return {"clusters": clusters_out, "total_samples_clustered": total}


# ── CRUD ────────────────────────────────────────────────────────────────


def list_clusters(db: DbSession, session_id: str) -> list[dict]:
    clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
    results = []
    for c in clusters:
        count = (
            db.query(func.count(sample_clusters.c.sample_id))
            .filter(sample_clusters.c.cluster_id == c.id)
            .scalar()
        )
        results.append({"id": c.id, "color": c.color, "sample_count": count})
    return results


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
) -> list[dict]:
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

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

    new_clusters = []
    for k in range(n_sub_clusters):
        new_cluster = Cluster(session_id=cluster.session_id, color=random_color())
        db.add(new_cluster)
        db.flush()

        sub_samples = [samples[i] for i, lbl in enumerate(labels) if lbl == k]
        for s in sub_samples:
            s.clusters.append(new_cluster)

        new_clusters.append({
            "id": new_cluster.id,
            "color": new_cluster.color,
            "sample_count": len(sub_samples),
        })

    db.delete(cluster)
    db.commit()
    return new_clusters


def merge_clusters(db: DbSession, cluster_ids: list[str]) -> dict:
    clusters = db.query(Cluster).filter(Cluster.id.in_(cluster_ids)).all()
    if len(clusters) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid clusters to merge")

    session_ids = {c.session_id for c in clusters}
    if len(session_ids) > 1:
        raise HTTPException(status_code=400, detail="All clusters must belong to the same session")

    session_id = session_ids.pop()

    all_samples = set()
    for c in clusters:
        samples = (
            db.query(Sample)
            .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id == c.id)
            .all()
        )
        all_samples.update(samples)

    merged = Cluster(session_id=session_id, color=random_color())
    db.add(merged)
    db.flush()

    for s in all_samples:
        s.clusters.append(merged)

    for c in clusters:
        db.delete(c)

    db.commit()
    return {
        "id": merged.id,
        "color": merged.color,
        "sample_count": len(all_samples),
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
