<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { Filter } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'

const analysis = useAnalysisStore()

const hasSelection = computed(() => analysis.selectedGateId !== null)
const activeCount = computed(() => analysis.activeSamples.length)
const selectedGate = computed(() =>
  analysis.allGates.find(g => g.id === analysis.selectedGateId) ?? null,
)

onMounted(() => {
  analysis.fetchActiveSamples()
})
</script>

<template>
  <div
    v-if="hasSelection"
    class="px-4 py-2 bg-warning/10 border-b border-warning/20 flex items-center gap-2"
  >
    <Filter class="w-3 h-3 text-warning shrink-0 stroke-[2px]" />
    <span class="text-[10px] font-mono font-bold text-warning tracking-wider uppercase truncate">
      Population: {{ selectedGate?.name ?? 'Custom' }} — {{ activeCount.toLocaleString() }} samples
    </span>
  </div>
</template>
