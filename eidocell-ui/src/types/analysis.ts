export type ChartType = 'histogram' | 'scatter' | 'density' | 'contour'
export type HierarchicalGateType = 'interval' | 'rectangular' | 'polygon' | 'ellipse' | 'quadrant'
export type GateType = HierarchicalGateType | 'boolean' | 'root'
export type BooleanOperator = 'AND' | 'OR'

export interface PlotCreate {
  name?: string
  chart_type: ChartType
  parameters: Record<string, unknown>
  parent_gate_id?: string | null
}

export interface PlotOut {
  id: string
  name: string
  chart_type: ChartType
  parameters: Record<string, unknown>
  parent_gate_id?: string | null
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
  gate_type: HierarchicalGateType
  definition: Record<string, unknown>
  color?: string
  parameters: string[]
  parent_gate_id?: string | null
}

export interface BooleanGateCreate {
  name: string
  operator: BooleanOperator
  source_gate_ids: [string, string]
  color?: string
}

export interface GateUpdate {
  name?: string
  color?: string
  definition?: Record<string, unknown>
  parent_gate_id?: string | null
}

export interface GateOut {
  id: string
  plot_id: string | null
  name: string
  gate_type: GateType
  definition: Record<string, unknown>
  color: string
  parameters: string[]
  sample_count: number
  percentage: number
  parent_gate_id?: string | null
  operator?: BooleanOperator | null
  source_gate_ids?: string[] | null
}

export interface PopulationTreeNode extends GateOut {
  children: PopulationTreeNode[]
}

export interface PopulationTreeResponse {
  root: PopulationTreeNode
  booleans: PopulationTreeNode[]
}

export interface SelectedPopulation {
  selected_gate_id: string | null
}

export interface PlotLayout {
  i: string    // plot.id
  x: number
  y: number
  w: number    // grid units (1-12)
  h: number    // grid units
}
