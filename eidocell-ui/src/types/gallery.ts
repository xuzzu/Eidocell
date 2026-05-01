export interface SampleOut {
  id: string
  filename: string
  path: string
  storage_index: number
  is_active: boolean
  class_id: string | null
  class_name: string | null
  class_color: string | null
  has_mask: boolean
}

export interface SamplePage {
  items: SampleOut[]
  total: number
  offset: number
  limit: number
}

export interface FilterCondition {
  field: string
  operator: string
  value: string | number
  value2?: string | number | null
}

export interface SampleListParams {
  offset?: number
  limit?: number
  sort_by?: string
  sort_order?: string
  filters?: FilterCondition[]
}

export interface ClassCreate {
  name: string
  color?: string
}

export interface ClassUpdate {
  name?: string
  color?: string
}

export interface ClassOut {
  id: string
  name: string
  color: string
  sample_count: number
}

export interface BulkClassAssignment {
  sample_ids: string[]
  class_id: string | null
}
