import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useSessionStore } from './session'
import * as analysisApi from '@/api/analysis'
import type { PlotOut, PlotData, GateOut, PlotCreate, GateCreate, GateUpdate, PlotLayout } from '@/types'

export const useAnalysisStore = defineStore('analysis', () => {
  const sessionStore = useSessionStore()

  const parameters = ref<string[]>([])
  const plots = ref<PlotOut[]>([])
  const selectedPlotId = ref<string | null>(null)
  const gates = ref<GateOut[]>([])
  const activeSamples = ref<string[]>([])
  const loading = ref(false)

  // Hierarchical gating: selected parent gate for sub-gating
  const parentGateId = ref<string | null>(null)
  // Cache of sample IDs inside the parent gate population
  const parentPopulationIds = ref<Set<string> | null>(null)

  // Workspace layout
  const plotLayouts = ref<PlotLayout[]>([])

  // Per-plot data/gates cache
  const plotDataCache = ref<Map<string, PlotData>>(new Map())
  const plotGatesCache = ref<Map<string, GateOut[]>>(new Map())

  // Computed
  const hasActiveGates = computed(() => gates.value.some(g => g.is_active))

  const parentGate = computed(() => {
    if (!parentGateId.value) return null
    for (const gateList of plotGatesCache.value.values()) {
      const found = gateList.find(g => g.id === parentGateId.value)
      if (found) return found
    }
    return gates.value.find(g => g.id === parentGateId.value) ?? null
  })

  function $reset() {
    parameters.value = []
    plots.value = []
    selectedPlotId.value = null
    parentGateId.value = null
    parentPopulationIds.value = null
    gates.value = []
    activeSamples.value = []
    loading.value = false
    plotLayouts.value = []
    plotDataCache.value.clear()
    plotGatesCache.value.clear()
  }

  async function fetchParameters() {
    if (!sessionStore.currentSessionId) return
    parameters.value = await analysisApi.listParameters(sessionStore.currentSessionId)
  }

  async function fetchPlots() {
    if (!sessionStore.currentSessionId) return
    plots.value = await analysisApi.listPlots(sessionStore.currentSessionId)
  }

  async function createPlot(data: PlotCreate) {
    if (!sessionStore.currentSessionId) return
    const plot = await analysisApi.createPlot(sessionStore.currentSessionId, data)
    await fetchPlots()
    // Auto-add to workspace
    addPlotToWorkspace(plot.id)
    await selectPlot(plot.id)
    return plot
  }

  function addPlotToWorkspace(plotId: string) {
    if (plotLayouts.value.some(l => l.i === plotId)) return
    // Find next available Y position
    const maxY = plotLayouts.value.length > 0
      ? Math.max(...plotLayouts.value.map(l => l.y + l.h))
      : 0
    plotLayouts.value.push({ i: plotId, x: 0, y: maxY, w: 6, h: 5 })
    // Fetch data for this plot
    fetchPlotData(plotId)
  }

  function removePlotFromWorkspace(plotId: string) {
    plotLayouts.value = plotLayouts.value.filter(l => l.i !== plotId)
    plotDataCache.value.delete(plotId)
    plotGatesCache.value.delete(plotId)
  }

  async function fetchPlotData(plotId: string) {
    if (!sessionStore.currentSessionId) return
    const data = await analysisApi.getPlotData(sessionStore.currentSessionId, plotId)
    plotDataCache.value.set(plotId, data)
    const plotGates = await analysisApi.listPlotGates(sessionStore.currentSessionId, plotId)
    plotGatesCache.value.set(plotId, plotGates)
  }

  async function selectPlot(plotId: string) {
    if (!sessionStore.currentSessionId) return
    selectedPlotId.value = plotId
    loading.value = true
    try {
      // Fetch data for this specific plot (update cache)
      await fetchPlotData(plotId)
      // Also set the "current" gates for the gate panel
      gates.value = plotGatesCache.value.get(plotId) ?? []
    } finally {
      loading.value = false
    }
  }

  async function deletePlot(plotId: string) {
    if (!sessionStore.currentSessionId) return
    await analysisApi.deletePlot(sessionStore.currentSessionId, plotId)
    removePlotFromWorkspace(plotId)
    if (selectedPlotId.value === plotId) {
      selectedPlotId.value = null
      gates.value = []
    }
    await fetchPlots()
  }

  async function selectParentGate(gateId: string | null) {
    parentGateId.value = gateId
    if (!gateId || !sessionStore.currentSessionId) {
      parentPopulationIds.value = null
      return
    }
    // Fetch the population of the parent gate
    const ids = await analysisApi.getGatePopulation(sessionStore.currentSessionId, gateId)
    parentPopulationIds.value = new Set(ids)
  }

  function clearParentGate() {
    parentGateId.value = null
    parentPopulationIds.value = null
  }

  async function createGate(plotId: string, data: GateCreate) {
    if (!sessionStore.currentSessionId) return
    // If sub-gating within a parent, attach the parent_gate_id
    if (parentGateId.value) {
      data = { ...data, parent_gate_id: parentGateId.value }
    }
    await analysisApi.createGate(sessionStore.currentSessionId, plotId, data)
    // Refresh gates for this plot
    const plotGates = await analysisApi.listPlotGates(sessionStore.currentSessionId, plotId)
    plotGatesCache.value.set(plotId, plotGates)
    if (selectedPlotId.value === plotId) {
      gates.value = plotGates
    }
    activeSamples.value = await analysisApi.getActiveSamples(sessionStore.currentSessionId)
  }

  async function updateGate(gateId: string, data: GateUpdate) {
    if (!sessionStore.currentSessionId) return
    const updated = await analysisApi.updateGate(sessionStore.currentSessionId, gateId, data)
    // Refresh gates for the plot this gate belongs to
    const plotId = updated.plot_id
    const plotGates = await analysisApi.listPlotGates(sessionStore.currentSessionId, plotId)
    plotGatesCache.value.set(plotId, plotGates)
    if (selectedPlotId.value === plotId) {
      gates.value = plotGates
    }
    activeSamples.value = await analysisApi.getActiveSamples(sessionStore.currentSessionId)
  }

  async function deleteGate(gateId: string) {
    if (!sessionStore.currentSessionId) return
    // Find which plot this gate belongs to before deletion
    const gate = gates.value.find(g => g.id === gateId)
    const plotId = gate?.plot_id
    await analysisApi.deleteGate(sessionStore.currentSessionId, gateId)
    if (plotId) {
      const plotGates = await analysisApi.listPlotGates(sessionStore.currentSessionId, plotId)
      plotGatesCache.value.set(plotId, plotGates)
      if (selectedPlotId.value === plotId) {
        gates.value = plotGates
      }
    }
    activeSamples.value = await analysisApi.getActiveSamples(sessionStore.currentSessionId)
  }

  async function resetAllGates() {
    if (!sessionStore.currentSessionId) return
    await analysisApi.resetGates(sessionStore.currentSessionId)
    // Refresh all gates
    for (const layout of plotLayouts.value) {
      const plotGates = await analysisApi.listPlotGates(sessionStore.currentSessionId, layout.i)
      plotGatesCache.value.set(layout.i, plotGates)
    }
    if (selectedPlotId.value) {
      gates.value = plotGatesCache.value.get(selectedPlotId.value) ?? []
    }
    activeSamples.value = await analysisApi.getActiveSamples(sessionStore.currentSessionId)
  }

  async function fetchActiveSamples() {
    if (!sessionStore.currentSessionId) return
    activeSamples.value = await analysisApi.getActiveSamples(sessionStore.currentSessionId)
  }

  // Refresh all visible plots' data
  async function refreshWorkspace() {
    for (const layout of plotLayouts.value) {
      await fetchPlotData(layout.i)
    }
  }

  return {
    parameters, plots, selectedPlotId, gates, activeSamples, loading,
    plotLayouts, plotDataCache, plotGatesCache, hasActiveGates,
    parentGateId, parentPopulationIds, parentGate,
    $reset,
    fetchParameters, fetchPlots, createPlot, selectPlot, deletePlot,
    addPlotToWorkspace, removePlotFromWorkspace, fetchPlotData,
    createGate, updateGate, deleteGate, resetAllGates,
    selectParentGate, clearParentGate,
    fetchActiveSamples, refreshWorkspace,
  }
})
