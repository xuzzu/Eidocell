export interface FeatureExtractionMethod {
  id: string
  name: string
  feature_dim: number
}

export interface RunFeatureExtractionRequest {
  method?: string
}

export interface FeatureExtractionResult {
  processed: number
  skipped: number
  total: number
  feature_dim: number
}

export interface RunDimReductionRequest {
  method?: string
  n_components?: number
  params?: Record<string, unknown>
}

export interface EmbeddingPoint {
  sample_id: string
  filename: string
  x: number
  y: number
  z?: number
  class_id: string | null
  class_name: string | null
  class_color: string | null
}

export interface DimReductionResult {
  method: string
  n_components: number
  n_samples: number
  embeddings: EmbeddingPoint[]
}
