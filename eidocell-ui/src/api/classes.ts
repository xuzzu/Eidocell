import { apiGet, imageUrl } from './client'
import type { ClassSummary, ClassStatistics, SamplePage, SessionDistributions } from '@/types'

export const getClassSummary = (sid: string, classId: string) =>
  apiGet<ClassSummary>(`/sessions/${sid}/classes/${classId}/summary`)

export const getClassStatistics = (sid: string, classId: string) =>
  apiGet<ClassStatistics>(`/sessions/${sid}/classes/${classId}/statistics`)

export const getClassSamples = (sid: string, classId: string, offset = 0, limit = 100) =>
  apiGet<SamplePage>(`/sessions/${sid}/classes/${classId}/samples?offset=${offset}&limit=${limit}`)

export const getSessionDistributions = (sid: string) =>
  apiGet<SessionDistributions>(`/sessions/${sid}/classes/distributions`)

export const classPreviewUrl = (sid: string, classId: string) =>
  imageUrl(`/sessions/${sid}/classes/${classId}/preview`)
