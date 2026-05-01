export interface SessionCreate {
  name: string
  images_directory: string
}

export interface SessionUpdate {
  name?: string
  scale_factor?: number
  scale_units?: string
}

export interface SessionListItem {
  id: string
  name: string
  images_directory: string
  created_at: string
  last_opened_at: string
  sample_count: number
}

export interface SessionOut extends SessionListItem {
  session_folder: string
  scale_factor: number
  scale_units: string
}
