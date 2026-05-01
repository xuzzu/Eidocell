import { apiGet, apiPost } from './client'
import type {
  FeatureExtractionMethod, RunFeatureExtractionRequest,
  RunDimReductionRequest,
} from '@/types'

export const listFeatureMethods = (sid: string) =>
  apiGet<FeatureExtractionMethod[]>(`/sessions/${sid}/features/methods`)

export const runFeatureExtraction = (sid: string, data: RunFeatureExtractionRequest) =>
  apiPost<{ task_id: string }>(`/sessions/${sid}/features/extract`, data)

export const listDimReductionMethods = (sid: string) =>
  apiGet<{ id: string; name: string }[]>(`/sessions/${sid}/features/dim-reduction/methods`)

export const runDimReduction = (sid: string, data: RunDimReductionRequest) =>
  apiPost<{ task_id: string }>(`/sessions/${sid}/features/dim-reduction/run`, data)
