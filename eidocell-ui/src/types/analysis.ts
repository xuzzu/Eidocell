export type ChartType = 'histogram' | 'scatter' | 'density' | 'contour'
export type GateType = 'interval' | 'rectangular' | 'polygon' | 'ellipse' | 'quadrant'

export interface PlotCreate {
  name?: string
  chart_type: ChartType
  parameters: Record<string, unknown>
}

export interface PlotOut {
  id: string
  name: string
  chart_type: ChartType
  parameters: Record<string, unknown>
  created_at: string
  gate_count: number
}

export interface PlotDataPoint {
  sample_id: string
  values: Record<string, number>
  class_name: string | null
  class_color: string | null
  cluster_ids: string[]
}

export interface PlotData {
  plot_id: string
  chart_type: ChartType
  parameters: Record<string, unknown>
  data: PlotDataPoint[]
  total?: number
}

export interface GateCreate {
  name?: string
  gate_type: GateType
  definition: Record<string, unknown>
  color?: string
  parameters: string[]
  is_active?: boolean
  parent_gate_id?: string | null
}

export interface GateUpdate {
  name?: string
  color?: string
  definition?: Record<string, unknown>
  is_active?: boolean
}

export interface GateOut {
  id: string
  plot_id: string
  name: string
  gate_type: GateType
  definition: Record<string, unknown>
  color: string
  parameters: string[]
  is_active: boolean
  sample_count: number
  percentage: number
  parent_gate_id?: string | null
}

export interface PlotLayout {
  i: string    // plot.id
  x: number
  y: number
  w: number    // grid units (1-12)
  h: number    // grid units
}
