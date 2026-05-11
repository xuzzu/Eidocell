import { apiGet, apiPost, imageUrl } from './client'
import type { SegmentationMethod, RunSegmentationRequest, SegmentationResult } from '@/types'

export const listSegmentationMethods = (sid: string) =>
  apiGet<SegmentationMethod[]>(`/sessions/${sid}/segmentation/methods`)

export const runSegmentation = (sid: string, data: RunSegmentationRequest) =>
  apiPost<SegmentationResult>(`/sessions/${sid}/segmentation/run`, data)

export const runSegmentationAsync = (sid: string, data: RunSegmentationRequest) =>
  apiPost<{ task_id: string }>(`/sessions/${sid}/segmentation/run-async`, data)

export const getMaskAttributes = (sid: string, sampleId: string, channel = 0) =>
  apiGet<Record<string, number>>(
    `/sessions/${sid}/samples/${sampleId}/mask/attributes?channel=${channel}`,
  )

export const maskOverlayUrl = (
  sid: string,
  sampleId: string,
  version?: number,
  channel = 0,
) => {
  const params = new URLSearchParams()
  params.set('channel', String(channel))
  if (version) params.set('v', String(version))
  return imageUrl(`/sessions/${sid}/samples/${sampleId}/mask/overlay?${params}`)
}
