<script setup lang="ts">
import { computed } from 'vue'
import { RotateCcw } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import { useGalleryStore } from '@/stores/gallery'

const analysis = useAnalysisStore()
const gallery = useGalleryStore()

const totalSamples = computed(() => gallery.total ?? 0)
const activeCount = computed(() => analysis.activeSamples.length)
const hasActiveGates = computed(() => analysis.gates.some(g => g.is_active))
const percentage = computed(() =>
  totalSamples.value > 0 ? (activeCount.value / totalSamples.value) * 100 : 100
)
</script>

<template>
  <div v-if="hasActiveGates" class="p-3 bg-base-200 rounded-[2px]">
    <div class="flex items-center justify-between">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Active Samples</span>
      <span class="text-[11px] font-mono font-bold">
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
      Reset All Gates
    </button>
  </div>
</template>
