export interface SessionCreate {
  name: string
}

export interface SessionUpdate {
  name?: string
  scale_factor?: number
  scale_units?: string
  channel_names?: string[] | null
}

export interface SessionListItem {
  id: string
  name: string
  images_directory: string | null
  created_at: string
  last_opened_at: string
  sample_count: number
}

export interface SessionOut extends SessionListItem {
  session_folder: string
  scale_factor: number
  scale_units: string
  channel_count: number
  channel_names: string[] | null
}

export type PreviewPhase = 'ready' | 'importing' | 'pregenerating' | 'failed'

export interface PreviewStatus {
  ready: boolean
  progress: number
  message: string
  phase: PreviewPhase
}
