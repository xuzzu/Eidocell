<script setup lang="ts">
import { Trash2, Eye, EyeOff, RotateCcw } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'

const analysis = useAnalysisStore()

async function toggleGate(gateId: string, currentActive: boolean) {
  await analysis.updateGate(gateId, { is_active: !currentActive })
}
</script>

<template>
  <div class="flex flex-col gap-3">
    <div class="flex items-center justify-between">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Gates</span>
      <button
        v-if="analysis.gates.length > 0"
        class="h-6 px-2 rounded-[2px] flex items-center gap-1.5 text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:bg-neutral/10 hover:text-neutral transition-colors"
        @click="analysis.resetAllGates()"
      >
        <RotateCcw class="w-3 h-3 stroke-[2px]" />
        Reset
      </button>
    </div>

    <div v-if="analysis.gates.length === 0" class="text-[10px] font-mono text-neutral/30 tracking-wider">
      No gates on this plot
    </div>

    <div v-for="gate in analysis.gates" :key="gate.id"
      class="flex items-center gap-2 p-2 rounded-[2px] bg-base-200 group"
    >
      <span class="w-2.5 h-2.5 rounded-[2px] shrink-0" :style="{ backgroundColor: gate.color }"></span>
      <div class="flex-1 min-w-0">
        <p class="text-[10px] font-mono font-bold truncate">{{ gate.name }}</p>
        <p class="text-[9px] font-mono text-neutral/40">
          {{ gate.sample_count.toLocaleString() }} samples · {{ gate.percentage.toFixed(1) }}%
        </p>
      </div>
      <button
        class="h-6 w-6 flex items-center justify-center rounded-[2px] transition-colors shrink-0"
        :class="gate.is_active ? 'text-neutral' : 'text-neutral/30'"
        @click="toggleGate(gate.id, gate.is_active)"
      >
        <component :is="gate.is_active ? Eye : EyeOff" class="w-3 h-3 stroke-[2px]" />
      </button>
      <button
        class="h-6 w-6 flex items-center justify-center rounded-[2px] text-neutral/25 hover:bg-error/10 hover:text-error transition-colors opacity-0 group-hover:opacity-100 shrink-0"
        @click="analysis.deleteGate(gate.id)"
      >
        <Trash2 class="w-3 h-3 stroke-[2px]" />
      </button>
    </div>
  </div>
</template>
