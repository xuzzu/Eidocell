export interface ClassSummary {
  id: string
  name: string
  color: string
  sample_count: number
}

export interface AttributeStatistics {
  name: string
  mean: number | null
  std: number | null
  min: number | null
  max: number | null
  median: number | null
}

export interface ClassStatistics {
  id: string
  name: string
  color: string
  sample_count: number
  attributes: AttributeStatistics[]
}
