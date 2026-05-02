"""Clustering pipeline: feature extraction → dim reduction → clustering → save.

Thin orchestrator over the shared pipeline helpers. Runs as a background task
with progress reporting.
"""

from services._pipeline_utils import project_to_2d
from services._pipeline_utils import features_cached_for
import logging
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.processors.errors import UnknownProcessorError
from core.processors.inference.clustering import get_processor as get_clustering_processor
from core.processors.inference.dimensionality_reduction import (
    get_processor as get_dr_processor,
)
from core.processors.inference.feature_extraction import (
    get_processor as get_fe_processor,
)
from core.task_manager import task_manager
from core.utils import npy_load, random_color
from models.models import Cluster, Sample
from services._pipeline_utils import (
    clean_zero_variance,
    extract_features_into,
    preload_morphological_masks,
    snapshot_samples,
    task_context,
    thread_db_session,
    validate_session_and_active_samples,
)

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
    """Submit the full clustering pipeline as a background task. Returns task_id."""
    session, samples = validate_session_and_active_samples(db, session_id)
    if n_clusters is not None and len(samples) < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {n_clusters} active samples, got {len(samples)}",
        )

    try:
        fe_processor = get_fe_processor(feature_method)
        if dim_reduction_method:
            get_dr_processor(dim_reduction_method)
        get_clustering_processor(clustering_method)
    except UnknownProcessorError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sample_data = snapshot_samples(samples)
    masks = preload_morphological_masks(db, sample_data, fe_processor)

    return task_manager.submit(
        name="Clustering pipeline",
        func=_background_pipeline,
        session_id=session_id,
        sample_data=sample_data,
        session_folder=session.session_folder,
        masks=masks,
        db_url=str(db.get_bind().url),
        feature_method=feature_method,
        dim_reduction_method=dim_reduction_method,
        dim_reduction_params=dim_reduction_params or {},
        clustering_method=clustering_method,
        n_clusters=n_clusters,
        clustering_params=clustering_params or {},
    )


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
    fe_processor = get_fe_processor(feature_method)
    feature_dim = fe_processor.feature_dim()
    features_path = Path(session_folder) / "features" / "session_features.npy"

    # 1: Feature extraction (cache if they already exist)
    if not features_cached_for(features_path, sample_data, feature_dim):
        extract_features_into(
            sample_data=sample_data,
            processor=fe_processor,
            masks=masks,
            features_path=features_path,
            feature_dim=feature_dim,
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )

    all_features = npy_load(features_path)
    indices = [s["storage_index"] for s in sample_data]
    active_features = all_features[indices]

    # 2-4: DR → cluster → 2D projection
    with task_context(on_progress, is_cancelled, total=len(sample_data) + 3) as ctx:
        ctx.current = len(sample_data)

        ctx.stage("Reducing dimensionality...", advance=1)
        if dim_reduction_method:
            dr_processor = get_dr_processor(dim_reduction_method)
            features_for_clustering = dr_processor.fit_transform(
                clean_zero_variance(active_features), **dim_reduction_params
            )
        else:
            features_for_clustering = active_features

        ctx.stage(f"Clustering ({clustering_method})...", advance=1)
        cl_processor = get_clustering_processor(clustering_method)
        labels = cl_processor.fit_predict(
            features_for_clustering, n_clusters, **clustering_params
        )
        actual_n_clusters = int(labels.max()) + 1

        # Visualization
        ctx.stage("Generating 2D projection...", advance=1)
        embeddings_2d = project_to_2d(features_for_clustering)

    # 5: Save clusters
    on_progress(len(sample_data) + 3, len(sample_data) + 3, "Saving clusters...")
    return _save_clusters(
        db_url=db_url,
        session_id=session_id,
        sample_data=sample_data,
        labels=labels,
        embeddings_2d=embeddings_2d,
        n_clusters=actual_n_clusters,
    )


def _save_clusters(
    *,
    db_url: str,
    session_id: str,
    sample_data: list[dict],
    labels: np.ndarray,
    embeddings_2d: np.ndarray,
    n_clusters: int,
) -> dict:
    with thread_db_session(db_url) as db:
        # Clear existing clusters
        for c in db.query(Cluster).filter(Cluster.session_id == session_id).all():
            db.delete(c)
        db.commit()

        clusters_out = []
        cluster_meta: dict[int, dict] = {}
        for k in range(n_clusters):
            cluster = Cluster(session_id=session_id, color=random_color())
            db.add(cluster)
            db.flush()

            sample_ids_for_cluster = [
                sample_data[i]["id"] for i, lbl in enumerate(labels) if lbl == k
            ]
            for sid in sample_ids_for_cluster:
                sample = db.query(Sample).filter(Sample.id == sid).first()
                if sample:
                    sample.clusters.append(cluster)

            cluster_meta[k] = {"id": cluster.id, "color": cluster.color}
            clusters_out.append({
                "id": cluster.id,
                "color": cluster.color,
                "sample_count": len(sample_ids_for_cluster),
            })

        db.commit()

        embeddings = []
        for i, sd in enumerate(sample_data):
            meta = cluster_meta.get(int(labels[i]), {})
            embeddings.append({
                "sample_id": sd["id"],
                "x": float(embeddings_2d[i, 0]),
                "y": float(embeddings_2d[i, 1]),
                "cluster_id": meta.get("id"),
                "cluster_color": meta.get("color"),
            })

        return {
            "clusters": clusters_out,
            "total_samples_clustered": sum(c["sample_count"] for c in clusters_out),
            "embeddings": embeddings,
        }
