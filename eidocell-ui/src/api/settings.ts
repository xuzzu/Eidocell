import { apiGet, apiPatch, apiPost } from './client'
import type { AppSettings, AppSettingsUpdate } from '@/types'

export const getSettings = () =>
  apiGet<AppSettings>('/settings/')

export const updateSettings = (data: AppSettingsUpdate) =>
  apiPatch<AppSettings>('/settings/', data)

export const resetSettings = () =>
  apiPost<AppSettings>('/settings/reset')
