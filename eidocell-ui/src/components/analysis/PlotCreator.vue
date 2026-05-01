<script setup lang="ts">
import { ref, computed } from 'vue'
import { Plus } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import type { ChartType } from '@/types'

const analysis = useAnalysisStore()

const chartTypes: { value: ChartType; label: string }[] = [
  { value: 'histogram', label: 'Histogram' },
  { value: 'scatter', label: 'Scatter / Dot Plot' },
  { value: 'density', label: 'Density Heatmap' },
  { value: 'contour', label: 'Contour' },
]

const chartType = ref<ChartType>('histogram')
const xVariable = ref('')
const yVariable = ref('')
const numBins = ref(30)

const needsY = computed(() => chartType.value !== 'histogram')

const canCreate = computed(() => {
  if (!xVariable.value) return false
  if (needsY.value && !yVariable.value) return false
  return true
})

async function create() {
  const params: Record<string, unknown> = { x_variable: xVariable.value }
  if (chartType.value === 'histogram') {
    params.num_bins = numBins.value
  } else {
    params.y_variable = yVariable.value
  }
  await analysis.createPlot({ chart_type: chartType.value, parameters: params })
}
</script>

<template>
  <div class="flex flex-col gap-4">
    <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">New Plot</span>

    <!-- Chart type -->
    <div class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Type</span>
      </label>
      <select
        v-model="chartType"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option v-for="ct in chartTypes" :key="ct.value" :value="ct.value">{{ ct.label }}</option>
      </select>
    </div>

    <!-- X axis -->
    <div class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">X Axis</span>
      </label>
      <select
        v-model="xVariable"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option value="">Select parameter...</option>
        <option v-for="p in analysis.parameters" :key="p" :value="p">{{ p }}</option>
      </select>
    </div>

    <!-- Y axis (scatter/density/contour) -->
    <div v-if="needsY" class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Y Axis</span>
      </label>
      <select
        v-model="yVariable"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option value="">Select parameter...</option>
        <option v-for="p in analysis.parameters" :key="p" :value="p">{{ p }}</option>
      </select>
    </div>

    <!-- Bins (histogram only) -->
    <div v-if="chartType === 'histogram'" class="form-control">
      <div class="flex items-center justify-between pb-2">
        <label class="label p-0">
          <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Bins</span>
        </label>
        <span class="text-[10px] font-mono text-neutral font-bold">{{ numBins }}</span>
      </div>
      <input
        v-model.number="numBins"
        type="range"
        min="5"
        max="100"
        class="range range-xs range-neutral flex-1 transition-none"
      />
      <div class="flex justify-between text-[9px] font-mono text-neutral/40 px-0.5 mt-1">
        <span>5</span>
        <span>100</span>
      </div>
    </div>

    <!-- Create button -->
    <button
      class="h-10 w-full rounded-[2px] flex items-center justify-center gap-2 text-[11px] font-bold tracking-widest uppercase transition-opacity"
      :class="canCreate ? 'bg-neutral text-neutral-content hover:opacity-80' : 'bg-base-200 text-neutral/40'"
      :disabled="!canCreate"
      @click="create"
    >
      <Plus class="w-4 h-4 stroke-[2px]" />
      Create Plot
    </button>
  </div>
</template>
