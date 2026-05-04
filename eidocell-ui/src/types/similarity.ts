import type { SampleOut } from './gallery'

export type SimilarityFilter = 'all' | 'unlabeled'

export interface SimilaritySearchRequest {
  reference_sample_ids: string[]
  filter_mode?: SimilarityFilter
  feature_method?: string
  min_similarity_pct?: number
  top_k?: number | null
}

export interface SimilarityHit {
  sample: SampleOut
  similarity_pct: number
  bucket: number
}

export interface SimilaritySearchResponse {
  reference_sample_ids: string[]
  total_candidates: number
  returned: number
  hits: SimilarityHit[]
}
