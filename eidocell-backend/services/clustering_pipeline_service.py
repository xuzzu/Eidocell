"""Clustering pipeline service: orchestrates feature extraction → dim reduction → clustering.

Runs as a background task with progress reporting.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DbSession, sessionmaker

from core.processors.inference.clustering import get_processor as get_clustering_processor
from core.processors.inference.dimensionality_reduction import get_processor as get_dr_processor
from core.processors.inference.feature_extraction import (
    get_processor as get_fe_processor,
    MorphologicalFeatureExtraction,
)
from core.task_manager import task_manager
from core.utils import random_color, npy_save, npy_load, get_active_samples
from models.models import Session, Sample, Mask, Cluster, sample_clusters

logger = logging.getLogger("eidocell.pipeline")


def run_clustering_pipeline(
    db: DbSession,
    session_id: str,
    *,
    feature_method: str = "mobilenetv3",
    dim_reduction_method: str | None = "pca",
    dim_reduction_params: dict | None = None,
    clustering_method: str = "kmeans",
    n_clusters: int | None = 10,
    clustering_params: dict | None = None,
) -> str:
    """Start the full clustering pipeline as a background task. Returns task_id."""
    from fastapi import HTTPException

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    samples = get_active_samples(db, session_id)
    if not samples:
        raise HTTPException(status_code=400, detail="No active samples")
    if n_clusters is not None and len(samples) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {n_clusters} active samples, got {len(samples)}",
        )

    # Validate processors exist
    try:
        get_fe_processor(feature_method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if dim_reduction_method:
        try:
            get_dr_processor(dim_reduction_method)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        get_clustering_processor(clustering_method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Snapshot data for background thread
    sample_data = [
        {"id": s.id, "path": s.path, "storage_index": s.storage_index}
        for s in samples
    ]
    session_folder = session.session_folder

    # Pre-load mask attributes for morphological
    masks = {}
    fe_processor = get_fe_processor(feature_method)
    if isinstance(fe_processor, MorphologicalFeatureExtraction):
        sample_ids = [s["id"] for s in sample_data]
        masks = {
            m.sample_id: m.attributes
            for m in db.query(Mask).filter(Mask.sample_id.in_(sample_ids)).all()
            if m.attributes
        }

    # Get DB URL for background thread
    db_url = str(db.get_bind().url)

    task_id = task_manager.submit(
        name="Clustering pipeline",
        func=_background_pipeline,
        session_id=session_id,
        sample_data=sample_data,
        session_folder=session_folder,
        masks=masks,
        db_url=db_url,
        feature_method=feature_method,
        dim_reduction_method=dim_reduction_method,
        dim_reduction_params=dim_reduction_params or {},
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        clustering_params=clustering_params or {},
    )
    return task_id


def _background_pipeline(
    *,
    session_id: str,
    sample_data: list[dict],
    session_folder: str,
    masks: dict,
    db_url: str,
    feature_method: str,
    dim_reduction_method: str | None,
    dim_reduction_params: dict,
    clustering_method: str,
    n_clusters: int | None,
    clustering_params: dict,
    on_progress,
    is_cancelled=None,
):
    """Run the full pipeline in a background thread."""
    total_steps = len(sample_data) + 3  # feature extraction per-sample + dr + clustering + embed
    current = 0

    # ── Step 1: Feature extraction ──────────────────────────────────────
    on_progress(0, total_steps, "Starting feature extraction...")

    fe_processor = get_fe_processor(feature_method)
    feature_dim = fe_processor.feature_dim()
    max_index = max(s["storage_index"] for s in sample_data) + 1

    features_dir = Path(session_folder) / "features"
    features_dir.mkdir(exist_ok=True)
    features_path = features_dir / "session_features.npy"

    # Check if features already exist with correct dimension
    need_extraction = True
    if features_path.exists():
        existing = npy_load(features_path)
        if existing.shape[0] >= max_index and existing.shape[1] == feature_dim:
            # Check that the samples we need have non-zero features
            indices = [s["storage_index"] for s in sample_data]
            sample_features = existing[indices]
            if np.any(np.abs(sample_features).sum(axis=1) > 0):
                need_extraction = False
                all_features = existing
                on_progress(len(sample_data), total_steps, "Features already extracted, reusing...")

    if need_extraction:
        all_features = np.zeros((max_index, feature_dim), dtype=np.float32)

        for i, sd in enumerate(sample_data):
            if is_cancelled and is_cancelled():
                pass # on_progress will raise TaskCancelledException shortly

            if isinstance(fe_processor, MorphologicalFeatureExtraction):
                attrs = masks.get(sd["id"])
                if attrs:
                    all_features[sd["storage_index"]] = fe_processor.extract_from_attributes(attrs)
            else:
                image = cv2.imread(sd["path"], cv2.IMREAD_UNCHANGED)
                if image is not None:
                    try:
                        all_features[sd["storage_index"]] = fe_processor.extract(image)
                    except Exception:
                        pass

            current = i + 1
            on_progress(current, total_steps, f"Extracting features... {current}/{len(sample_data)}")

        npy_save(features_path, all_features)
    else:
        current = len(sample_data)

    # Get features for active samples only
    indices = [s["storage_index"] for s in sample_data]
    active_features = all_features[indices]  # (n_samples, feature_dim)

    # ── Step 2: Dimensionality reduction for clustering ─────────────────
    current += 1
    on_progress(current, total_steps, "Reducing dimensionality...")
    if is_cancelled and is_cancelled(): pass


    if dim_reduction_method:
        dr_processor = get_dr_processor(dim_reduction_method)
        # Clean zero-variance features
        stds = np.std(active_features, axis=0)
        valid_cols = stds > 0
        features_clean = active_features[:, valid_cols] if np.any(valid_cols) else active_features
        reduced_features = dr_processor.fit_transform(features_clean, **dim_reduction_params)
    else:
        reduced_features = active_features

    # ── Step 3: Clustering ──────────────────────────────────────────────
    current += 1
    on_progress(current, total_steps, "Clustering...")
    if is_cancelled and is_cancelled(): pass


    cl_processor = get_clustering_processor(clustering_method)
    labels = cl_processor.fit_predict(reduced_features, n_clusters, **clustering_params)
    actual_n_clusters = int(labels.max()) + 1

    # ── Step 4: Generate 2D embeddings for visualization ────────────────
    current += 1
    on_progress(current, total_steps, "Generating 2D projection...")
    if is_cancelled and is_cancelled(): pass


    from sklearn.decomposition import PCA as PCA2D
    if reduced_features.shape[1] > 2:
        n_comp = min(2, reduced_features.shape[0], reduced_features.shape[1])
        pca_2d = PCA2D(n_components=n_comp)
        embeddings_2d = pca_2d.fit_transform(reduced_features)
    elif reduced_features.shape[1] == 2:
        embeddings_2d = reduced_features
    else:
        # 1D features, pad with zeros
        embeddings_2d = np.column_stack([reduced_features, np.zeros(len(reduced_features))])

    # ── Step 5: Save clusters to DB ─────────────────────────────────────
    on_progress(current, total_steps, "Saving clusters...")

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Clear existing clusters
        existing_clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
        for c in existing_clusters:
            db.delete(c)
        db.commit()

        # Create new clusters
        cluster_colors = {}
        clusters_out = []
        for k in range(actual_n_clusters):
            color = random_color()
            cluster = Cluster(session_id=session_id, color=color)
            db.add(cluster)
            db.flush()

            cluster_sample_ids = [
                sample_data[i]["id"] for i, lbl in enumerate(labels) if lbl == k
            ]
            # Add samples to cluster via the junction table
            for sid in cluster_sample_ids:
                sample = db.query(Sample).filter(Sample.id == sid).first()
                if sample:
                    sample.clusters.append(cluster)

            cluster_colors[k] = {"id": cluster.id, "color": color}
            clusters_out.append({
                "id": cluster.id,
                "color": color,
                "sample_count": len(cluster_sample_ids),
            })

        db.commit()

        # Build embeddings response
        embeddings = []
        for i, sd in enumerate(sample_data):
            label = int(labels[i])
            ci = cluster_colors.get(label, {})
            embeddings.append({
                "sample_id": sd["id"],
                "x": float(embeddings_2d[i, 0]),
                "y": float(embeddings_2d[i, 1]),
                "cluster_id": ci.get("id"),
                "cluster_color": ci.get("color"),
            })

        total_clustered = sum(c["sample_count"] for c in clusters_out)

        return {
            "clusters": clusters_out,
            "total_samples_clustered": total_clustered,
            "embeddings": embeddings,
        }
    finally:
        db.close()
        engine.dispose()
