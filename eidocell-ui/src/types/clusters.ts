export interface ClusterOut {
  id: string
  color: string
  sample_count: number
  labeled_count: number
  quality_score: number | null
  feature_method: string | null
}

export interface ClusteringResult {
  clusters: ClusterOut[]
  total_samples_clustered: number
}

export interface RunClusteringRequest {
  n_clusters: number
}

export type ClusteringScopeMode = 'all' | 'unlabeled' | 'class' | 'samples'

export interface ClusteringScope {
  mode: ClusteringScopeMode
  class_id?: string | null
  sample_ids?: string[] | null
}

export interface RunClusteringPipelineRequest {
  feature_method: string
  dim_reduction_method: string | null
  dim_reduction_params: Record<string, unknown>
  clustering_method: string
  n_clusters: number | null
  clustering_params: Record<string, unknown>
  scope?: ClusteringScope
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

export interface SplitClusterResult {
  new_clusters: ClusterOut[]
  deleted_cluster_id: string
}

export interface MergeClustersRequest {
  cluster_ids: string[]
}

export interface MergeClustersResult {
  new_cluster: ClusterOut
  deleted_cluster_ids: string[]
}

export interface AssignClustersToClassRequest {
  cluster_ids: string[]
  class_id: string
}

export type ClusterSortMode =
  | 'manual'
  | 'count_desc'
  | 'count_asc'
  | 'labeled_desc'
  | 'quality_desc'

export interface ClusterAttributeStatistics {
  name: string
  mean: number | null
  std: number | null
  min: number | null
  max: number | null
  median: number | null
}

export interface ClusterStatistics {
  id: string
  color: string
  sample_count: number
  attributes: ClusterAttributeStatistics[]
}
