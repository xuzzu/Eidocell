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

export interface AttributeDistribution {
  name: string
  bin_edges: number[]
  bin_counts: number[]
  min?: number | null
  max?: number | null
  mean?: number | null
  std?: number | null
}

export interface SessionDistributions {
  sample_count: number
  attributes: AttributeDistribution[]
}
