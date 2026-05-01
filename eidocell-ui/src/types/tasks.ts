export interface TaskInfo {
  id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  total: number
  percentage: number
  message: string | null
  result: unknown
  error: string | null
  created_at: string
  completed_at: string | null
}
