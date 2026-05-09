from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session as DbSession

from db.session import get_db
from schemas.workspace.clusters import (
    RunClusteringRequest,
    RunClusteringPipelineRequest,
    SplitClusterRequest,
    MergeClustersRequest,
    AssignClustersToClassRequest,
    ClusterOut,
    ClusteringResult,
    SplitClusterResult,
    MergeClustersResult,
)
from schemas.workspace.gallery import SamplePage
from services.workspace import clusters_service
from services import clustering_pipeline_service

router = APIRouter(prefix="/sessions/{session_id}/clusters", tags=["clusters"])


@router.post("/run-pipeline")
def run_clustering_pipeline(
    session_id: str,
    data: RunClusteringPipelineRequest,
    db: DbSession = Depends(get_db),
):
    """Run the full clustering pipeline (extract → reduce → cluster) as a background task."""
    task_id = clustering_pipeline_service.run_clustering_pipeline(
        db,
        session_id,
        feature_method=data.feature_method,
        dim_reduction_method=data.dim_reduction_method,
        dim_reduction_params=data.dim_reduction_params,
        clustering_method=data.clustering_method,
        n_clusters=data.n_clusters,
        clustering_params=data.clustering_params,
        scope=data.scope,
    )
    return {"task_id": task_id}


@router.post("/run", response_model=ClusteringResult)
def run_clustering(
    session_id: str,
    data: RunClusteringRequest,
    db: DbSession = Depends(get_db),
):
    """Run K-Means clustering on stored features."""
    return clusters_service.run_clustering(
        db, session_id, data.n_clusters, feature_method=data.feature_method
    )


@router.get("/", response_model=list[ClusterOut])
def list_clusters(session_id: str, db: DbSession = Depends(get_db)):
    return clusters_service.list_clusters(db, session_id)


@router.delete("/", status_code=204)
def clear_clusters(session_id: str, db: DbSession = Depends(get_db)):
    """Delete all clusters in this session."""
    clusters_service.clear_clusters(db, session_id)


@router.get("/{cluster_id}/samples", response_model=SamplePage)
def get_cluster_samples(
    session_id: str,
    cluster_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: DbSession = Depends(get_db),
):
    items, total = clusters_service.get_cluster_samples(db, cluster_id, offset, limit)
    return SamplePage(items=items, total=total, offset=offset, limit=limit)


@router.get("/{cluster_id}/preview")
def get_cluster_preview(
    session_id: str, cluster_id: str, db: DbSession = Depends(get_db)
):
    path = clusters_service.get_or_generate_collage(db, cluster_id)
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/{cluster_id}", status_code=204)
def delete_cluster(
    session_id: str, cluster_id: str, db: DbSession = Depends(get_db)
):
    clusters_service.delete_cluster(db, cluster_id)


@router.post("/{cluster_id}/split", response_model=SplitClusterResult)
def split_cluster(
    session_id: str,
    cluster_id: str,
    data: SplitClusterRequest,
    db: DbSession = Depends(get_db),
):
    return clusters_service.split_cluster(db, cluster_id, data.n_sub_clusters)


@router.post("/merge", response_model=MergeClustersResult)
def merge_clusters(
    session_id: str,
    data: MergeClustersRequest,
    db: DbSession = Depends(get_db),
):
    return clusters_service.merge_clusters(db, data.cluster_ids)


@router.post("/assign-class")
def assign_clusters_to_class(
    session_id: str,
    data: AssignClustersToClassRequest,
    db: DbSession = Depends(get_db),
):
    updated = clusters_service.assign_clusters_to_class(db, data.cluster_ids, data.class_id)
    return {"updated": updated}
