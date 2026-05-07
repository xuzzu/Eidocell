<script setup lang="ts">
import { ref, computed, onMounted, toRef } from 'vue'
import { Square, Pentagon, GripHorizontal, Circle, Crosshair, ArrowLeftRight, X, Filter } from 'lucide-vue-next'
import { usePlotRenderer, type GateDrawEvent, type GateEditEvent } from '@/composables/usePlotRenderer'
import { nextGateColor } from '@/utils/gateColors'
import { useAnalysisStore } from '@/stores/analysis'
import type { PlotOut, PlotData, GateOut, ChartType, GateCreate } from '@/types'

const analysis = useAnalysisStore()

const props = defineProps<{
  plot: PlotOut
  plotData: PlotData | null
  gates: GateOut[]
  isActive: boolean
  highlightedSampleIds?: Set<string> | null
}>()

const emit = defineEmits<{
  select: []
  gateCreated: [gate: GateCreate]
  gateUpdated: [gateId: string, definition: Record<string, unknown>]
  gateDeleted: [gateId: string]
  remove: []
}>()

const plotContainer = ref<HTMLElement | null>(null)

const renderer = usePlotRenderer({
  container: plotContainer,
  chartType: toRef(() => props.plot.chart_type as ChartType),
  plotData: toRef(() => props.plotData),
  gates: toRef(() => props.gates),
  highlightedSampleIds: toRef(() => props.highlightedSampleIds ?? null),
})

// Pre-assign next gate color so drawing preview matches final gate
let pendingColor = nextGateColor()
renderer.drawPreviewColor.value = pendingColor

renderer.setOnGateDrawn((e: GateDrawEvent) => {
  const gate: GateCreate = {
    gate_type: e.gateType as GateCreate['gate_type'],
    definition: e.definition,
    parameters: e.parameters,
    color: pendingColor,
  }
  emit('gateCreated', gate)
  pendingColor = nextGateColor()
  renderer.drawPreviewColor.value = pendingColor
})

renderer.setOnGateEdited((e: GateEditEvent) => {
  emit('gateUpdated', e.gateId, e.definition)
})

onMounted(async () => {
  if (plotContainer.value) {
    await renderer.init()
  }
})

// Gate tools based on chart type
interface GateTool {
  type: string
  icon: any
  label: string
}

const scatterTools: GateTool[] = [
  { type: 'rectangular', icon: Square, label: 'Rectangle' },
  { type: 'polygon', icon: Pentagon, label: 'Polygon' },
  { type: 'ellipse', icon: Circle, label: 'Ellipse' },
  { type: 'quadrant', icon: Crosshair, label: 'Quadrant' },
]

const histogramTools: GateTool[] = [
  { type: 'interval', icon: ArrowLeftRight, label: 'Interval' },
]

const gateTools = computed(() => {
  const ct = props.plot.chart_type
  if (ct === 'histogram') return histogramTools
  return scatterTools
})

const pointCount = computed(() => props.plotData?.data.length ?? 0)

const parentPopulationName = computed(() => {
  const pid = props.plot.parent_gate_id
  if (!pid) return null
  return analysis.allGates.find(g => g.id === pid)?.name ?? null
})
</script>

<template>
  <div
    class="flex flex-col h-full bg-base-100 border rounded-[2px] overflow-hidden transition-colors"
    :class="isActive ? 'border-neutral' : 'border-base-300'"
    @click.stop="emit('select')"
  >
    <!-- Toolbar / drag handle -->
    <div class="widget-drag-handle h-8 flex items-center gap-1 px-2 border-b border-base-300 shrink-0 cursor-grab select-none bg-base-100">
      <GripHorizontal class="w-3 h-3 text-neutral/25 shrink-0" />
      <span class="text-[9px] font-mono font-bold text-neutral/50 truncate ml-1">{{ plot.name }}</span>
      <span
        v-if="parentPopulationName"
        class="ml-1 inline-flex items-center gap-1 px-1 py-0.5 rounded-[2px] text-[8px] font-mono font-bold tracking-wider uppercase bg-info/10 text-info shrink-0"
        :title="`Restricted to population: ${parentPopulationName}`"
      >
        <Filter class="w-2.5 h-2.5 stroke-[2px]" />
        {{ parentPopulationName }}
      </span>
      <span class="text-[8px] font-mono text-neutral/30 ml-1 shrink-0">{{ pointCount.toLocaleString() }} pts</span>

      <div class="ml-auto flex items-center gap-0.5">
        <!-- Gate tools -->
        <button
          v-for="tool in gateTools"
          :key="tool.type"
          class="h-6 w-6 flex items-center justify-center rounded-[2px] transition-colors"
          :class="renderer.activeTool.value === tool.type
            ? 'bg-neutral text-neutral-content'
            : 'text-neutral/40 hover:bg-neutral/10 hover:text-neutral'"
          :title="tool.label"
          @click.stop="renderer.activeTool.value = renderer.activeTool.value === tool.type ? null : tool.type"
        >
          <component :is="tool.icon" class="w-3 h-3 stroke-[2px]" />
        </button>

        <!-- Remove plot from workspace -->
        <button
          class="h-6 w-6 flex items-center justify-center rounded-[2px] text-neutral/25 hover:bg-error/10 hover:text-error transition-colors ml-1"
          title="Remove from workspace"
          @click.stop="emit('remove')"
        >
          <X class="w-3 h-3 stroke-[2px]" />
        </button>
      </div>
    </div>

    <!-- Rendering container -->
    <div ref="plotContainer" class="flex-1 relative min-h-0 bg-base-100"></div>
  </div>
</template>
