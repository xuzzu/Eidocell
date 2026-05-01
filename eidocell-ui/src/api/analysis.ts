import { apiGet, apiPost, apiPatch, apiDelete, BASE_URL } from './client'
import type { PlotCreate, PlotOut, PlotData, GateCreate, GateUpdate, GateOut } from '@/types'

export const listParameters = (sid: string) =>
  apiGet<string[]>(`/sessions/${sid}/analysis/parameters`)

export const createPlot = (sid: string, data: PlotCreate) =>
  apiPost<PlotOut>(`/sessions/${sid}/analysis/plots`, data)

export const listPlots = (sid: string) =>
  apiGet<PlotOut[]>(`/sessions/${sid}/analysis/plots`)

export const getPlot = (sid: string, plotId: string) =>
  apiGet<PlotOut>(`/sessions/${sid}/analysis/plots/${plotId}`)

export const getPlotData = (sid: string, plotId: string) =>
  apiGet<PlotData>(`/sessions/${sid}/analysis/plots/${plotId}/data`)

export const deletePlot = (sid: string, plotId: string) =>
  apiDelete(`/sessions/${sid}/analysis/plots/${plotId}`)

export const createGate = (sid: string, plotId: string, data: GateCreate) =>
  apiPost<GateOut>(`/sessions/${sid}/analysis/plots/${plotId}/gates`, data)

export const listGates = (sid: string) =>
  apiGet<GateOut[]>(`/sessions/${sid}/analysis/gates`)

export const listPlotGates = (sid: string, plotId: string) =>
  apiGet<GateOut[]>(`/sessions/${sid}/analysis/plots/${plotId}/gates`)

export const updateGate = (sid: string, gateId: string, data: GateUpdate) =>
  apiPatch<GateOut>(`/sessions/${sid}/analysis/gates/${gateId}`, data)

export const deleteGate = (sid: string, gateId: string) =>
  apiDelete(`/sessions/${sid}/analysis/gates/${gateId}`)

export const getGatePopulation = (sid: string, gateId: string) =>
  apiGet<string[]>(`/sessions/${sid}/analysis/gates/${gateId}/population`)

export const getActiveSamples = (sid: string) =>
  apiGet<string[]>(`/sessions/${sid}/analysis/active-samples`)

export const resetGates = (sid: string) =>
  apiPost<{ reactivated: number }>(`/sessions/${sid}/analysis/reset-gates`)

export const getPopulationTree = (sid: string) =>
  apiGet<GateTreeNode[]>(`/sessions/${sid}/analysis/population-tree`)

export interface GateTreeNode extends GateOut {
  children: GateTreeNode[]
}


// ── Batch plot data ────────────────────────────────────────────────────

export const batchPlotData = (sid: string, plotIds: string[], maxPoints = 0) =>
  apiPost<{ plots: Record<string, PlotData> }>(
    `/sessions/${sid}/analysis/batch-plot-data`,
    { plot_ids: plotIds, max_points: maxPoints },
  )

// ── Binary data for high-performance rendering ─────────────────────────

export interface BinaryPlotData {
  meta: {
    plot_id: string
    chart_type: string
    axes: string[]
    total: number
    returned: number
    sample_ids: string[]
    colors: string[]
  }
  xData: Float32Array
  yData: Float32Array | null
}

export async function getPlotDataBinary(
  sid: string,
  plotId: string,
  maxPoints = 50000,
): Promise<BinaryPlotData> {
  const resp = await fetch(
    `${BASE_URL}/sessions/${sid}/analysis/plots/${plotId}/data/binary?max_points=${maxPoints}`,
  )
  if (!resp.ok) throw new Error(`Binary data fetch failed: ${resp.status}`)

  const buffer = await resp.arrayBuffer()
  const view = new DataView(buffer)

  // Read header length (uint32 LE)
  const headerLen = view.getUint32(0, true)

  // Read JSON header
  const headerBytes = new Uint8Array(buffer, 4, headerLen)
  const headerStr = new TextDecoder().decode(headerBytes)
  const meta = JSON.parse(headerStr)

  const dataOffset = 4 + headerLen
  const returned = meta.returned as number
  const numAxes = meta.axes.length

  // Read x data
  const xData = new Float32Array(buffer, dataOffset, returned)

  // Read y data (if 2D)
  let yData: Float32Array | null = null
  if (numAxes > 1) {
    yData = new Float32Array(buffer, dataOffset + returned * 4, returned)
  }

  return { meta, xData, yData }
}
