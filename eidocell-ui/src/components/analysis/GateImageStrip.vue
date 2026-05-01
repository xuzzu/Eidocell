<script setup lang="ts">
import { ref, watch } from 'vue'
import { useSessionStore } from '@/stores/session'
import { useAnalysisStore } from '@/stores/analysis'
import * as analysisApi from '@/api/analysis'
import { sampleThumbnailUrl } from '@/api/gallery'
import { Images } from 'lucide-vue-next'

const sessionStore = useSessionStore()
const analysis = useAnalysisStore()

const sampleIds = ref<string[]>([])
const loading = ref(false)

const MAX_THUMBNAILS = 50

// Fetch population sample IDs when selected gate changes
watch(
  () => analysis.selectedPlotId && analysis.gates,
  async () => {
    const sid = sessionStore.currentSessionId
    if (!sid || analysis.gates.length === 0) {
      sampleIds.value = []
      return
    }
    // Use the first active gate on the selected plot, or the first gate
    const activeGate = analysis.gates.find(g => g.is_active) ?? analysis.gates[0]
    if (!activeGate) { sampleIds.value = []; return }

    loading.value = true
    try {
      const ids = await analysisApi.getGatePopulation(sid, activeGate.id)
      sampleIds.value = ids.slice(0, MAX_THUMBNAILS)
    } finally {
      loading.value = false
    }
  },
  { deep: true, immediate: true },
)

function thumbnailSrc(sampleId: string): string {
  const sid = sessionStore.currentSessionId
  if (!sid) return ''
  return sampleThumbnailUrl(sid, sampleId)
}
</script>

<template>
  <div v-if="sampleIds.length > 0" class="flex flex-col gap-2">
    <div class="flex items-center gap-1.5">
      <Images class="w-3 h-3 text-neutral/40 stroke-[2px]" />
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">
        Gate Samples
      </span>
      <span class="text-[9px] font-mono text-neutral/40">{{ sampleIds.length }}{{ sampleIds.length >= MAX_THUMBNAILS ? '+' : '' }}</span>
    </div>

    <div class="flex gap-1 overflow-x-auto py-1 scrollbar-thin">
      <div
        v-for="id in sampleIds"
        :key="id"
        class="w-10 h-10 shrink-0 rounded-[2px] overflow-hidden border border-base-300 hover:border-neutral transition-colors"
      >
        <img
          :src="thumbnailSrc(id)"
          :alt="id"
          class="w-full h-full object-cover"
          loading="lazy"
        />
      </div>
    </div>
  </div>
</template>
