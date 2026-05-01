export interface SegmentationMethodParam {
  name: string
  label: string
  min: number
  max: number
  default: number
  step: number
}

export interface SegmentationMethod {
  id: string
  name: string
  params: SegmentationMethodParam[]
}

export interface RunSegmentationRequest {
  method: string
  params?: Record<string, number>
}

export interface SegmentationResult {
  processed: number
  failed: number
  total: number
}
