<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { X } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import type { PlotOut, ChartType } from '@/types'

const props = defineProps<{
  plot: PlotOut
  x: number
  y: number
}>()

const emit = defineEmits<{
  close: []
}>()

const analysis = useAnalysisStore()

type ScaleKind = 'linear' | 'log'

function paramStr(key: string): string {
  const v = props.plot.parameters[key]
  return typeof v === 'string' ? v : ''
}

function paramScale(key: string): ScaleKind {
  const v = props.plot.parameters[key]
  return v === 'log' ? 'log' : 'linear'
}

const xVariable = ref(paramStr('x_variable'))
const yVariable = ref(paramStr('y_variable'))
const xScaleType = ref<ScaleKind>(paramScale('x_scale_type'))
const yScaleType = ref<ScaleKind>(paramScale('y_scale_type'))
const numBins = ref<number>((props.plot.parameters.num_bins as number) ?? 30)

const chartType = computed<ChartType>(() => props.plot.chart_type as ChartType)
const needsY = computed(() => chartType.value !== 'histogram')
const allowsLog = computed(() => chartType.value === 'histogram' || chartType.value === 'scatter')

const saving = ref(false)
const root = ref<HTMLElement>()

const style = computed(() => {
  const w = 280
  const h = 360
  const x = Math.min(props.x, window.innerWidth - w - 8)
  const y = Math.min(props.y, window.innerHeight - h - 8)
  return { left: `${x}px`, top: `${y}px`, width: `${w}px` }
})

const dirty = computed(() => {
  if (xVariable.value !== paramStr('x_variable')) return true
  if (needsY.value && yVariable.value !== paramStr('y_variable')) return true
  if (allowsLog.value && xScaleType.value !== paramScale('x_scale_type')) return true
  if (allowsLog.value && needsY.value && yScaleType.value !== paramScale('y_scale_type')) return true
  if (chartType.value === 'histogram'
      && numBins.value !== ((props.plot.parameters.num_bins as number) ?? 30)) return true
  return false
})

const canApply = computed(() => {
  if (!xVariable.value) return false
  if (needsY.value && !yVariable.value) return false
  return dirty.value
})

async function apply() {
  if (!canApply.value) return
  saving.value = true
  try {
    const params: Record<string, unknown> = {
      ...props.plot.parameters,
      x_variable: xVariable.value,
    }
    if (needsY.value) {
      params.y_variable = yVariable.value
    } else {
      delete params.y_variable
    }
    if (chartType.value === 'histogram') {
      params.num_bins = numBins.value
    }
    if (allowsLog.value) {
      params.x_scale_type = xScaleType.value
      if (needsY.value) {
        params.y_scale_type = yScaleType.value
      } else {
        delete params.y_scale_type
      }
    } else {
      delete params.x_scale_type
      delete params.y_scale_type
    }
    await analysis.updatePlot(props.plot.id, { parameters: params })
    emit('close')
  } finally {
    saving.value = false
  }
}

function onClickOutside(e: MouseEvent) {
  if (root.value && !root.value.contains(e.target as Node)) {
    emit('close')
  }
}

onMounted(() => {
  setTimeout(() => document.addEventListener('mousedown', onClickOutside), 0)
})
onUnmounted(() => document.removeEventListener('mousedown', onClickOutside))

// Re-read defaults if the plot prop swaps to a different instance
watch(() => props.plot.id, () => {
  xVariable.value = paramStr('x_variable')
  yVariable.value = paramStr('y_variable')
  xScaleType.value = paramScale('x_scale_type')
  yScaleType.value = paramScale('y_scale_type')
  numBins.value = (props.plot.parameters.num_bins as number) ?? 30
})
</script>

<template>
  <div
    ref="root"
    class="fixed z-[600] bg-base-100 border border-base-300 shadow-xl rounded-[2px] p-3"
    :style="style"
    @mousedown.stop
  >
    <div class="flex items-center justify-between mb-3">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Plot Settings</span>
      <button
        class="w-5 h-5 flex items-center justify-center rounded-[2px] text-neutral/30 hover:bg-neutral/10 hover:text-neutral transition-colors"
        @click="emit('close')"
      >
        <X class="w-3 h-3 stroke-[2px]" />
      </button>
    </div>

    <div class="flex flex-col gap-2.5">
      <!-- X axis param -->
      <div>
        <label class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 block mb-1">X Axis</label>
        <select
          v-model="xVariable"
          class="select select-bordered select-xs rounded-[2px] w-full font-mono text-[10px] focus:outline-neutral"
        >
          <option v-for="p in analysis.parameters" :key="p" :value="p">{{ p }}</option>
        </select>
      </div>

      <!-- X scale -->
      <div v-if="allowsLog">
        <label class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 block mb-1">X Scale</label>
        <div class="flex gap-1">
          <button
            class="flex-1 h-7 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
            :class="xScaleType === 'linear' ? 'bg-neutral text-neutral-content' : 'bg-base-200 hover:bg-neutral/10 text-neutral/70'"
            @click="xScaleType = 'linear'"
          >Linear</button>
          <button
            class="flex-1 h-7 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
            :class="xScaleType === 'log' ? 'bg-neutral text-neutral-content' : 'bg-base-200 hover:bg-neutral/10 text-neutral/70'"
            @click="xScaleType = 'log'"
          >Log</button>
        </div>
      </div>

      <!-- Y axis param -->
      <div v-if="needsY">
        <label class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 block mb-1">Y Axis</label>
        <select
          v-model="yVariable"
          class="select select-bordered select-xs rounded-[2px] w-full font-mono text-[10px] focus:outline-neutral"
        >
          <option v-for="p in analysis.parameters" :key="p" :value="p">{{ p }}</option>
        </select>
      </div>

      <!-- Y scale -->
      <div v-if="needsY && allowsLog">
        <label class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 block mb-1">Y Scale</label>
        <div class="flex gap-1">
          <button
            class="flex-1 h-7 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
            :class="yScaleType === 'linear' ? 'bg-neutral text-neutral-content' : 'bg-base-200 hover:bg-neutral/10 text-neutral/70'"
            @click="yScaleType = 'linear'"
          >Linear</button>
          <button
            class="flex-1 h-7 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
            :class="yScaleType === 'log' ? 'bg-neutral text-neutral-content' : 'bg-base-200 hover:bg-neutral/10 text-neutral/70'"
            @click="yScaleType = 'log'"
          >Log</button>
        </div>
      </div>

      <!-- Histogram bins -->
      <div v-if="chartType === 'histogram'">
        <div class="flex items-center justify-between mb-1">
          <label class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Bins</label>
          <span class="text-[10px] font-mono font-bold">{{ numBins }}</span>
        </div>
        <input
          v-model.number="numBins"
          type="range"
          min="5"
          max="100"
          class="range range-xs range-neutral"
        />
      </div>

      <!-- Apply button -->
      <button
        class="h-8 mt-1 rounded-[2px] flex items-center justify-center gap-2 text-[10px] font-bold tracking-widest uppercase transition-opacity"
        :class="canApply
          ? 'bg-neutral text-neutral-content hover:opacity-80'
          : 'bg-base-200 text-neutral/40 cursor-not-allowed'"
        :disabled="!canApply || saving"
        @click="apply"
      >
        <span v-if="saving" class="loading loading-spinner loading-xs"></span>
        <span v-else>Apply</span>
      </button>
    </div>
  </div>
</template>
