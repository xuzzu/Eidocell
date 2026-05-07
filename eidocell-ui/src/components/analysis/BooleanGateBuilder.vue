<script setup lang="ts">
import { ref, computed } from 'vue'
import { GitMerge } from 'lucide-vue-next'
import { useAnalysisStore } from '@/stores/analysis'
import { nextGateColor } from '@/utils/gateColors'
import type { BooleanOperator } from '@/types'

const analysis = useAnalysisStore()

const name = ref('')
const operator = ref<BooleanOperator>('AND')
const gate1 = ref('')
const gate2 = ref('')
const submitting = ref(false)
const error = ref<string | null>(null)

const sourceCandidates = computed(() => analysis.allGates)
const candidatesForGate2 = computed(() =>
  sourceCandidates.value.filter(g => g.id !== gate1.value),
)

const canSubmit = computed(() =>
  !!name.value.trim() && !!gate1.value && !!gate2.value && gate1.value !== gate2.value && !submitting.value,
)

async function submit() {
  if (!canSubmit.value) return
  submitting.value = true
  error.value = null
  try {
    await analysis.createBooleanGate({
      name: name.value.trim(),
      operator: operator.value,
      source_gate_ids: [gate1.value, gate2.value],
      color: nextGateColor(),
    })
    name.value = ''
    gate1.value = ''
    gate2.value = ''
  } catch (e: any) {
    error.value = e?.message ?? 'Failed to create boolean gate'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="flex flex-col gap-2">
    <div class="flex items-center gap-1.5">
      <GitMerge class="w-3 h-3 stroke-[2px] text-purple-500" />
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Boolean population</span>
    </div>

    <div v-if="sourceCandidates.length < 2" class="text-[10px] font-mono text-neutral/30 leading-snug">
      Need at least two gates to combine. Draw gates on plots first.
    </div>

    <div v-else class="flex flex-col gap-2">
      <label class="flex flex-col gap-1">
        <span class="text-[8px] font-mono font-bold tracking-widest uppercase text-neutral/40">Name</span>
        <input
          v-model="name"
          type="text"
          placeholder="e.g. CD4+ AND CD8+"
          class="h-7 px-2 text-[11px] font-mono rounded-[2px] bg-base-200 border border-base-300 focus:outline-none focus:border-neutral/40"
        />
      </label>

      <label class="flex flex-col gap-1">
        <span class="text-[8px] font-mono font-bold tracking-widest uppercase text-neutral/40">Population 1</span>
        <select
          v-model="gate1"
          class="h-7 px-2 text-[11px] font-mono rounded-[2px] bg-base-200 border border-base-300 focus:outline-none focus:border-neutral/40"
        >
          <option value="" disabled>Select…</option>
          <option v-for="g in sourceCandidates" :key="g.id" :value="g.id">{{ g.name }}</option>
        </select>
      </label>

      <div class="flex flex-col gap-1">
        <span class="text-[8px] font-mono font-bold tracking-widest uppercase text-neutral/40">Operator</span>
        <div class="flex gap-1">
          <button
            class="flex-1 h-7 px-2 text-[10px] font-mono font-bold tracking-wider uppercase rounded-[2px] border transition-colors"
            :class="operator === 'AND'
              ? 'bg-purple-100 border-purple-400 text-purple-700'
              : 'bg-base-200 border-base-300 text-neutral/50 hover:bg-base-300'"
            @click="operator = 'AND'"
          >AND ∩</button>
          <button
            class="flex-1 h-7 px-2 text-[10px] font-mono font-bold tracking-wider uppercase rounded-[2px] border transition-colors"
            :class="operator === 'OR'
              ? 'bg-purple-100 border-purple-400 text-purple-700'
              : 'bg-base-200 border-base-300 text-neutral/50 hover:bg-base-300'"
            @click="operator = 'OR'"
          >OR ∪</button>
        </div>
      </div>

      <label class="flex flex-col gap-1">
        <span class="text-[8px] font-mono font-bold tracking-widest uppercase text-neutral/40">Population 2</span>
        <select
          v-model="gate2"
          class="h-7 px-2 text-[11px] font-mono rounded-[2px] bg-base-200 border border-base-300 focus:outline-none focus:border-neutral/40"
        >
          <option value="" disabled>Select…</option>
          <option v-for="g in candidatesForGate2" :key="g.id" :value="g.id">{{ g.name }}</option>
        </select>
      </label>

      <div v-if="error" class="text-[10px] font-mono text-error">{{ error }}</div>

      <button
        class="h-8 px-2 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
        :class="canSubmit
          ? 'bg-purple-600 text-white hover:bg-purple-700'
          : 'bg-base-200 text-neutral/30 cursor-not-allowed'"
        :disabled="!canSubmit"
        @click="submit"
      >Create population</button>
    </div>
  </div>
</template>
