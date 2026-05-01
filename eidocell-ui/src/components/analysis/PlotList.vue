<script setup lang="ts">
import { Trash2, Plus } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'

const analysis = useAnalysisStore()

function isInWorkspace(plotId: string) {
  return analysis.plotLayouts.some(l => l.i === plotId)
}

function addToWorkspace(plotId: string) {
  if (!isInWorkspace(plotId)) {
    analysis.addPlotToWorkspace(plotId)
  }
}

function removePlot(plotId: string) {
  analysis.deletePlot(plotId)
}
</script>

<template>
  <div v-if="analysis.plots.length > 0" class="flex flex-col gap-3">
    <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Plots</span>

    <div class="flex flex-col gap-1">
      <div
        v-for="plot in analysis.plots"
        :key="plot.id"
        class="flex items-center gap-2 px-2 py-1.5 rounded-[2px] group transition-colors cursor-pointer"
        :class="isInWorkspace(plot.id)
          ? 'bg-neutral/5 text-neutral'
          : 'text-neutral/60 hover:bg-neutral/5 hover:text-neutral'"
        @click="addToWorkspace(plot.id)"
      >
        <div class="flex-1 min-w-0">
          <p class="text-[10px] font-mono font-bold truncate">{{ plot.name }}</p>
          <p class="text-[9px] font-mono text-neutral/35">{{ plot.chart_type }} · {{ plot.gate_count }} gates</p>
        </div>

        <button
          v-if="!isInWorkspace(plot.id)"
          class="h-5 w-5 flex items-center justify-center rounded-[2px] text-neutral/30 hover:bg-neutral/10 hover:text-neutral opacity-0 group-hover:opacity-100 transition-all shrink-0"
          title="Add to workspace"
          @click.stop="addToWorkspace(plot.id)"
        >
          <Plus class="w-3 h-3 stroke-[2px]" />
        </button>

        <button
          class="h-5 w-5 flex items-center justify-center rounded-[2px] text-neutral/30 hover:bg-error/10 hover:text-error opacity-0 group-hover:opacity-100 transition-all shrink-0"
          title="Delete plot"
          @click.stop="removePlot(plot.id)"
        >
          <Trash2 class="w-3 h-3 stroke-[2px]" />
        </button>
      </div>
    </div>
  </div>
</template>
