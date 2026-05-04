import { apiPost } from './client'
import type { SimilaritySearchRequest, SimilaritySearchResponse } from '@/types'

export const searchSimilar = (sid: string, data: SimilaritySearchRequest) =>
  apiPost<SimilaritySearchResponse>(`/sessions/${sid}/similarity/search`, data)
