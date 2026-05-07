import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useSessionStore } from './session'
import { useGalleryStore } from './gallery'
import * as analysisApi from '@/api/analysis'
import type {
  PlotOut, PlotData, GateOut,
  PlotCreate, GateCreate, GateUpdate, BooleanGateCreate,
  PlotLayout, PopulationTreeResponse,
} from '@/types'

export const useAnalysisStore = defineStore('analysis', () => {
  const sessionStore = useSessionStore()
  const galleryStore = useGalleryStore()

  const parameters = ref<string[]>([])
  const plots = ref<PlotOut[]>([])
  const selectedPlotId = ref<string | null>(null)
  // Session-wide gate list. Populated by fetchAllGates and refreshed after every mutation.
  const allGates = ref<GateOut[]>([])
  const populationTree = ref<PopulationTreeResponse | null>(null)
  const selectedGateId = ref<string | null>(null)
  const activeSamples = ref<string[]>([])
  const loading = ref(false)

  const plotLayouts = ref<PlotLayout[]>([])
  const plotDataCache = ref<Map<string, PlotData>>(new Map())

  const gatesByPlotId = computed(() => {
    const map = new Map<string, GateOut[]>()
    for (const g of allGates.value) {
      if (!g.plot_id) continue
      const list = map.get(g.plot_id) ?? []
      list.push(g)
      map.set(g.plot_id, list)
    }
    return map
  })

  const hierarchicalGates = computed(() =>
    allGates.value.filter(g => g.gate_type !== 'boolean'),
  )
  const booleanGates = computed(() =>
    allGates.value.filter(g => g.gate_type === 'boolean'),
  )

  function $reset() {
    parameters.value = []
    plots.value = []
    selectedPlotId.value = null
    allGates.value = []
    populationTree.value = null
    selectedGateId.value = null
    activeSamples.value = []
    loading.value = false
    plotLayouts.value = []
    plotDataCache.value.clear()
  }

  async function fetchParameters() {
    if (!sessionStore.currentSessionId) return
    parameters.value = await analysisApi.listParameters(sessionStore.currentSessionId)
  }

  async function fetchPlots() {
    if (!sessionStore.currentSessionId) return
    plots.value = await analysisApi.listPlots(sessionStore.currentSessionId)
  }

  async function fetchAllGates() {
    if (!sessionStore.currentSessionId) return
    allGates.value = await analysisApi.listGates(sessionStore.currentSessionId)
  }

  async function fetchPopulationTree() {
    if (!sessionStore.currentSessionId) return
    populationTree.value = await analysisApi.getPopulationTree(sessionStore.currentSessionId)
  }

  async function fetchSelectedPopulation() {
    if (!sessionStore.currentSessionId) return
    const resp = await analysisApi.getSelectedPopulation(sessionStore.currentSessionId)
    selectedGateId.value = resp.selected_gate_id
  }

  async function fetchActiveSamples() {
    if (!sessionStore.currentSessionId) return
    activeSamples.value = await analysisApi.getActiveSamples(sessionStore.currentSessionId)
  }

  async function refreshGatingState() {
    await Promise.all([
      fetchAllGates(),
      fetchPopulationTree(),
      fetchSelectedPopulation(),
      fetchActiveSamples(),
    ])
  }

  async function createPlot(data: PlotCreate) {
    if (!sessionStore.currentSessionId) return
    const plot = await analysisApi.createPlot(sessionStore.currentSessionId, data)
    await fetchPlots()
    addPlotToWorkspace(plot.id)
    selectedPlotId.value = plot.id
    return plot
  }

  function addPlotToWorkspace(plotId: string) {
    if (plotLayouts.value.some(l => l.i === plotId)) return
    const maxY = plotLayouts.value.length > 0
      ? Math.max(...plotLayouts.value.map(l => l.y + l.h))
      : 0
    plotLayouts.value.push({ i: plotId, x: 0, y: maxY, w: 6, h: 5 })
    fetchPlotData(plotId)
  }

  function removePlotFromWorkspace(plotId: string) {
    plotLayouts.value = plotLayouts.value.filter(l => l.i !== plotId)
    plotDataCache.value.delete(plotId)
  }

  async function fetchPlotData(plotId: string) {
    if (!sessionStore.currentSessionId) return
    const data = await analysisApi.getPlotData(sessionStore.currentSessionId, plotId)
    plotDataCache.value.set(plotId, data)
  }

  async function selectPlot(plotId: string) {
    selectedPlotId.value = plotId
    if (!plotDataCache.value.has(plotId)) {
      await fetchPlotData(plotId)
    }
  }

  async function deletePlot(plotId: string) {
    if (!sessionStore.currentSessionId) return
    await analysisApi.deletePlot(sessionStore.currentSessionId, plotId)
    removePlotFromWorkspace(plotId)
    if (selectedPlotId.value === plotId) selectedPlotId.value = null
    await fetchPlots()
    await refreshGatingState()
    galleryStore.fetchSamples()
  }

  async function createGate(plotId: string, data: GateCreate) {
    if (!sessionStore.currentSessionId) return
    await analysisApi.createGate(sessionStore.currentSessionId, plotId, data)
    await refreshGatingState()
    await refreshWorkspace()
  }

  async function createBooleanGate(data: BooleanGateCreate) {
    if (!sessionStore.currentSessionId) return
    await analysisApi.createBooleanGate(sessionStore.currentSessionId, data)
    await refreshGatingState()
  }

  async function updateGate(gateId: string, data: GateUpdate) {
    if (!sessionStore.currentSessionId) return
    await analysisApi.updateGate(sessionStore.currentSessionId, gateId, data)
    await refreshGatingState()
    // Plots inheriting from any gate need to recompute their point set when
    // a gate's definition or parent changes.
    await refreshWorkspace()
    galleryStore.fetchSamples()
  }

  async function deleteGate(gateId: string) {
    if (!sessionStore.currentSessionId) return
    await analysisApi.deleteGate(sessionStore.currentSessionId, gateId)
    await refreshGatingState()
    await refreshWorkspace()
    galleryStore.fetchSamples()
  }

  async function selectPopulation(gateId: string | null) {
    if (!sessionStore.currentSessionId) return
    const resp = await analysisApi.selectPopulation(sessionStore.currentSessionId, gateId)
    selectedGateId.value = resp.selected_gate_id
    await Promise.all([
      fetchActiveSamples(),
      fetchPopulationTree(),
    ])
    galleryStore.fetchSamples()
  }

  async function resetAllGates() {
    await selectPopulation(null)
  }

  async function refreshWorkspace() {
    for (const layout of plotLayouts.value) {
      await fetchPlotData(layout.i)
    }
  }

  return {
    parameters, plots, selectedPlotId,
    allGates, populationTree, selectedGateId,
    activeSamples, loading,
    plotLayouts, plotDataCache,
    gatesByPlotId, hierarchicalGates, booleanGates,
    $reset,
    fetchParameters, fetchPlots, fetchAllGates, fetchPopulationTree,
    fetchSelectedPopulation, fetchActiveSamples, refreshGatingState,
    createPlot, selectPlot, deletePlot,
    addPlotToWorkspace, removePlotFromWorkspace, fetchPlotData,
    createGate, createBooleanGate, updateGate, deleteGate,
    selectPopulation, resetAllGates,
    refreshWorkspace,
  }
})
