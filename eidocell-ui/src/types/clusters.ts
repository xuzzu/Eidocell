export interface ClusterOut {
  id: string
  color: string
  sample_count: number
}

export interface ClusteringResult {
  clusters: ClusterOut[]
  total_samples_clustered: number
}

export interface RunClusteringRequest {
  n_clusters: number
}

export interface RunClusteringPipelineRequest {
  feature_method: string
  dim_reduction_method: string | null
  dim_reduction_params: Record<string, unknown>
  clustering_method: string
  n_clusters: number | null
  clustering_params: Record<string, unknown>
}

export interface ClusterEmbeddingPoint {
  sample_id: string
  x: number
  y: number
  cluster_id: string | null
  cluster_color: string | null
}

export interface ClusteringPipelineResult {
  clusters: ClusterOut[]
  total_samples_clustered: number
  embeddings: ClusterEmbeddingPoint[]
}

export interface SplitClusterRequest {
  n_sub_clusters: number
}

export interface MergeClustersRequest {
  cluster_ids: string[]
}

export interface AssignClustersToClassRequest {
  cluster_ids: string[]
  class_id: string
}

