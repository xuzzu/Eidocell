import { apiGet, apiPost, apiDelete } from './client'
import type {
  TrainModelRequest, TrainModelResult,
  RunInferenceRequest, InferenceResult,
  ApplyModelRequest, TrainedModelOut,
} from '@/types'

export const trainModel = (sid: string, data: TrainModelRequest) =>
  apiPost<TrainModelResult>(`/sessions/${sid}/learning/train`, data)

export const runInference = (sid: string, data: RunInferenceRequest) =>
  apiPost<InferenceResult>(`/sessions/${sid}/learning/infer`, data)

export const listSessionModels = (sid: string) =>
  apiGet<TrainedModelOut[]>(`/sessions/${sid}/learning/models`)

export const listAllModels = () =>
  apiGet<TrainedModelOut[]>('/models/')

export const getModel = (modelId: string) =>
  apiGet<TrainedModelOut>(`/models/${modelId}`)

export const deleteModel = (modelId: string) =>
  apiDelete(`/models/${modelId}`)

export const applyModel = (modelId: string, data: ApplyModelRequest) =>
  apiPost<InferenceResult>(`/models/${modelId}/apply`, data)
