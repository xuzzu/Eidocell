import { apiGet, apiPost, imageUrl } from './client'
import type { SegmentationMethod, RunSegmentationRequest, SegmentationResult } from '@/types'

export const listSegmentationMethods = (sid: string) =>
  apiGet<SegmentationMethod[]>(`/sessions/${sid}/segmentation/methods`)

export const runSegmentation = (sid: string, data: RunSegmentationRequest) =>
  apiPost<SegmentationResult>(`/sessions/${sid}/segmentation/run`, data)

export const runSegmentationAsync = (sid: string, data: RunSegmentationRequest) =>
  apiPost<{ task_id: string }>(`/sessions/${sid}/segmentation/run-async`, data)

export const getMaskAttributes = (sid: string, sampleId: string) =>
  apiGet<Record<string, number>>(`/sessions/${sid}/samples/${sampleId}/mask/attributes`)

export const runSegmentationPreview = (sid: string, data: { method: string; params: Record<string, number>; sample_ids: string[] }) =>
  apiPost<SegmentationResult>(`/sessions/${sid}/segmentation/preview`, data)

export const maskOverlayUrl = (sid: string, sampleId: string) =>
  imageUrl(`/sessions/${sid}/samples/${sampleId}/mask/overlay`)
