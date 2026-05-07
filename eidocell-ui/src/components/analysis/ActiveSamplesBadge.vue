<script setup lang="ts">
import { computed } from 'vue'
import { RotateCcw } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import { useGalleryStore } from '@/stores/gallery'

const analysis = useAnalysisStore()
const gallery = useGalleryStore()

const totalSamples = computed(() => gallery.total ?? 0)
const activeCount = computed(() => analysis.activeSamples.length)
const hasSelection = computed(() => analysis.selectedGateId !== null)
const percentage = computed(() =>
  totalSamples.value > 0 ? (activeCount.value / totalSamples.value) * 100 : 100
)
const selectedGate = computed(() =>
  analysis.allGates.find(g => g.id === analysis.selectedGateId) ?? null,
)
</script>

<template>
  <div v-if="hasSelection" class="p-3 bg-base-200 rounded-[2px]">
    <div class="flex items-center justify-between gap-2">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70 truncate">
        {{ selectedGate?.name ?? 'Active population' }}
      </span>
      <span class="text-[11px] font-mono font-bold shrink-0">
        {{ activeCount.toLocaleString() }} / {{ totalSamples.toLocaleString() }}
      </span>
    </div>
    <div class="h-1 bg-base-300 rounded-full mt-2 overflow-hidden">
      <div
        class="h-full bg-neutral rounded-full transition-all duration-300"
        :style="{ width: percentage + '%' }"
      ></div>
    </div>
    <button
      class="mt-2 text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:text-neutral transition-colors flex items-center gap-1.5"
      @click="analysis.resetAllGates()"
    >
      <RotateCcw class="w-3 h-3 stroke-[2px]" />
      Show All Events
    </button>
  </div>
</template>
