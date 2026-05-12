export interface SampleOut {
  id: string
  filename: string
  path: string
  is_active: boolean
  class_id: string | null
  class_name: string | null
  class_color: string | null
  has_mask: boolean
  mask_channels: number[]
  n_channels: number
}

export interface SamplePage {
  items: SampleOut[]
  total: number
  offset: number
  limit: number
  session_has_any_mask: boolean
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

export interface SortableAttribute {
  name: string
  channel_index: number
  channel_name: string
  /** sort_by token, e.g. "attr:area:ch:0" */
  value: string
}
