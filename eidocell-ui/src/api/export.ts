import { apiPost } from './client'
import type { ExportRequest, ExportResult } from '@/types'

export const exportSession = (sid: string, data: ExportRequest) =>
  apiPost<ExportResult>(`/sessions/${sid}/export/`, data)
