<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { Filter } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'

const analysis = useAnalysisStore()

const hasActiveGates = computed(() => analysis.gates.some(g => g.is_active))
const activeCount = computed(() => analysis.activeSamples.length)

onMounted(() => {
  // Ensure we have fresh active samples data
  analysis.fetchActiveSamples()
})
</script>

<template>
  <div
    v-if="hasActiveGates"
    class="px-4 py-2 bg-warning/10 border-b border-warning/20 flex items-center gap-2"
  >
    <Filter class="w-3 h-3 text-warning shrink-0 stroke-[2px]" />
    <span class="text-[10px] font-mono font-bold text-warning tracking-wider uppercase">
      Gating active — showing {{ activeCount.toLocaleString() }} samples
    </span>
  </div>
</template>
