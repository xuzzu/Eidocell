import { apiGet, apiPost, apiPatch, apiDelete, imageUrl } from './client'
import type {
  SampleOut, SamplePage, SampleListParams,
  ClassOut, ClassCreate, ClassUpdate, BulkClassAssignment,
} from '@/types'

export const listSamples = (sid: string, params: SampleListParams) =>
  apiPost<SamplePage>(`/sessions/${sid}/samples/list`, params)

export const getSample = (sid: string, sampleId: string) =>
  apiGet<SampleOut>(`/sessions/${sid}/samples/${sampleId}`)

export const sampleImageUrl = (sid: string, sampleId: string) =>
  imageUrl(`/sessions/${sid}/samples/${sampleId}/image`)

export const sampleThumbnailUrl = (sid: string, sampleId: string) =>
  imageUrl(`/sessions/${sid}/samples/${sampleId}/thumbnail`)

export const listClasses = (sid: string) =>
  apiGet<ClassOut[]>(`/sessions/${sid}/classes`)

export const createClass = (sid: string, data: ClassCreate) =>
  apiPost<ClassOut>(`/sessions/${sid}/classes`, data)

export const updateClass = (sid: string, classId: string, data: ClassUpdate) =>
  apiPatch<ClassOut>(`/sessions/${sid}/classes/${classId}`, data)

export const deleteClass = (sid: string, classId: string) =>
  apiDelete(`/sessions/${sid}/classes/${classId}`)

export const assignSamplesToClass = (sid: string, data: BulkClassAssignment) =>
  apiPost<{ updated: number }>(`/sessions/${sid}/samples/assign-class`, data)

export const listSortableAttributes = (sid: string) =>
  apiGet<string[]>(`/sessions/${sid}/sortable-attributes`)
