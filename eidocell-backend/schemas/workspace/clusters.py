from pydantic import BaseModel, Field


class RunClusteringRequest(BaseModel):
    n_clusters: int = Field(10, ge=2, le=100)


class RunClusteringPipelineRequest(BaseModel):
    """Full pipeline: feature extraction → dim reduction → clustering."""
    # Feature extraction
    feature_method: str = "mobilenetv3"

    # Dimensionality reduction (None = skip)
    dim_reduction_method: str | None = "pca"
    dim_reduction_params: dict = {}

    # Clustering
    clustering_method: str = "kmeans"
    n_clusters: int | None = Field(10, ge=2, le=100)
    clustering_params: dict = {}


class SplitClusterRequest(BaseModel):
    n_sub_clusters: int = Field(2, ge=2, le=20)


class MergeClustersRequest(BaseModel):
    cluster_ids: list[str] = Field(..., min_length=2)


class AssignClustersToClassRequest(BaseModel):
    cluster_ids: list[str]
    class_id: str


class ClusterOut(BaseModel):
    id: str
    color: str
    sample_count: int

    model_config = {"from_attributes": True}


class EmbeddingPoint(BaseModel):
    sample_id: str
    x: float
    y: float
    cluster_id: str | None = None
    cluster_color: str | None = None


class ClusteringResult(BaseModel):
    clusters: list[ClusterOut]
    total_samples_clustered: int


class ClusteringPipelineResult(BaseModel):
    clusters: list[ClusterOut]
    total_samples_clustered: int
    embeddings: list[EmbeddingPoint]
