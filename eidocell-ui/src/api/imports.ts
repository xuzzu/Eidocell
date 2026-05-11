import { apiGet, apiPost, apiDelete } from './client'
import type { ImportCreate, ImportOut, ImportSubmitOut } from '@/types'

export const submitImport = (sessionId: string, data: ImportCreate) =>
  apiPost<ImportSubmitOut>(`/sessions/${sessionId}/imports/`, data)

export const listImports = (sessionId: string) =>
  apiGet<ImportOut[]>(`/sessions/${sessionId}/imports/`)

export const getImport = (sessionId: string, importId: string) =>
  apiGet<ImportOut>(`/sessions/${sessionId}/imports/${importId}`)

export const cancelImport = (sessionId: string, importId: string) =>
  apiDelete<ImportOut>(`/sessions/${sessionId}/imports/${importId}`)

export const previewPreprocessing = (sessionId: string, payload: any) =>
  apiPost<{ image_data: string }>(`/sessions/${sessionId}/imports/preview`, payload)
