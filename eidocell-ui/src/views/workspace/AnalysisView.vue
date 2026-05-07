<script setup lang="ts">
import { onMounted } from 'vue'
import { BarChart3 } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import PlotCreator from '@/components/analysis/PlotCreator.vue'
import PlotList from '@/components/analysis/PlotList.vue'
import AnalysisWorkspace from '@/components/analysis/AnalysisWorkspace.vue'
import GateImageStrip from '@/components/analysis/GateImageStrip.vue'
import PopulationTree from '@/components/analysis/PopulationTree.vue'
import BooleanGateBuilder from '@/components/analysis/BooleanGateBuilder.vue'
import ActiveSamplesBadge from '@/components/analysis/ActiveSamplesBadge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import type { GateCreate } from '@/types'

const analysis = useAnalysisStore()

onMounted(() => {
  analysis.fetchParameters()
  analysis.fetchPlots()
  analysis.refreshGatingState()
})

async function onGateCreated(plotId: string, gate: GateCreate) {
  await analysis.createGate(plotId, gate)
}

async function onGateUpdated(gateId: string, definition: Record<string, unknown>) {
  await analysis.updateGate(gateId, { definition })
}

async function onGateDeleted(gateId: string) {
  await analysis.deleteGate(gateId)
}
</script>

<template>
  <div class="flex h-full relative overflow-hidden bg-base-200">
    <!-- Left panel: controls -->
    <div class="w-80 min-w-[320px] bg-base-100 border-r border-base-300 p-6 flex flex-col gap-6 overflow-y-auto z-10 shadow-sm shrink-0">
      <h2 class="text-[14px] font-bold tracking-widest uppercase">Analysis</h2>

      <PlotCreator />

      <div class="border-t border-base-300 pt-4">
        <PlotList />
      </div>

      <div class="border-t border-base-300 pt-4">
        <PopulationTree />
      </div>

      <div class="border-t border-base-300 pt-4">
        <BooleanGateBuilder />
      </div>

      <div class="border-t border-base-300 pt-4">
        <GateImageStrip />
      </div>

      <div class="border-t border-base-300 pt-4">
        <ActiveSamplesBadge />
      </div>
    </div>

    <!-- Right panel: workspace -->
    <div class="flex-1 p-4 overflow-y-auto">
      <EmptyState
        v-if="analysis.parameters.length === 0"
        :icon="BarChart3"
        title="No data available"
        description="RUN SEGMENTATION FIRST TO COMPUTE MASK ATTRIBUTES FOR ANALYSIS"
      />

      <div v-else-if="analysis.plotLayouts.length === 0" class="flex items-center justify-center h-full">
        <div class="text-center">
          <BarChart3 class="w-8 h-8 text-neutral/20 mx-auto mb-3" />
          <p class="text-[11px] font-mono font-bold text-neutral/30 tracking-wider uppercase">
            Create a plot to get started
          </p>
        </div>
      </div>

      <!-- Resizable plot workspace -->
      <AnalysisWorkspace
        v-else
        @gate-created="onGateCreated"
        @gate-updated="onGateUpdated"
        @gate-deleted="onGateDeleted"
      />
    </div>
  </div>
</template>
