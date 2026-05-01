import { apiGet, apiPost } from './client'
import type { TaskInfo } from '@/types'

export const getTask = (taskId: string) =>
  apiGet<TaskInfo>(`/tasks/${taskId}`)

export const listTasks = () =>
  apiGet<TaskInfo[]>('/tasks/')

export const cleanupTasks = () =>
  apiPost<{ removed: number }>('/tasks/cleanup')
