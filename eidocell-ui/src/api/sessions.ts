import { apiGet, apiPost, apiPatch, apiDelete } from './client'
import type { SessionCreate, SessionUpdate, SessionListItem, SessionOut } from '@/types'

export const listSessions = () =>
  apiGet<SessionListItem[]>('/sessions/')

export const createSession = (data: SessionCreate) =>
  apiPost<SessionOut>('/sessions/', data)

export const getSession = (id: string) =>
  apiGet<SessionOut>(`/sessions/${id}`)

export const updateSession = (id: string, data: SessionUpdate) =>
  apiPatch<SessionOut>(`/sessions/${id}`, data)

export const deleteSession = (id: string) =>
  apiDelete(`/sessions/${id}`)
