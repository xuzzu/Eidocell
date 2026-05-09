<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { Eye, Scissors, Merge, Plus, BarChart3 } from 'lucide-vue-next'
import type { ClassOut } from '@/types'

const props = defineProps<{
  x: number
  y: number
  selectedIds: string[]
  classes: ClassOut[]
}>()

const emit = defineEmits<{
  view: []
  showStats: []
  split: [n: number]
  merge: []
  assignClass: [classId: string]
  createClassAndAssign: []
  close: []
}>()

const isSingle = computed(() => props.selectedIds.length === 1)
const isMulti = computed(() => props.selectedIds.length >= 2)

const showSplitPrompt = ref(false)
const splitCount = ref(2)

// Clamp position to viewport
const style = computed(() => {
  const menuW = 200
  const menuH = 300
  const x = Math.min(props.x, window.innerWidth - menuW - 8)
  const y = Math.min(props.y, window.innerHeight - menuH - 8)
  return { left: `${x}px`, top: `${y}px` }
})

function onClickOutside() {
  emit('close')
}

onMounted(() => {
  // Use setTimeout so the current mousedown that opened the menu doesn't immediately close it
  setTimeout(() => document.addEventListener('mousedown', onClickOutside), 0)
})
onUnmounted(() => document.removeEventListener('mousedown', onClickOutside))
</script>

<template>
  <div
    class="fixed z-[500] bg-base-100 border border-base-300 shadow-xl rounded-[2px] py-1 min-w-[180px]"
    :style="style"
    @mousedown.stop
  >
    <!-- Single-select options -->
    <template v-if="isSingle">
      <button
        class="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-neutral hover:text-neutral-content transition-colors text-left"
        @click="emit('view'); emit('close')"
      >
        <Eye class="w-3.5 h-3.5 shrink-0 stroke-[2px]" /> View Samples
      </button>
      <button
        class="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-neutral hover:text-neutral-content transition-colors text-left"
        @click="emit('showStats'); emit('close')"
      >
        <BarChart3 class="w-3.5 h-3.5 shrink-0 stroke-[2px]" /> Show Statistics
      </button>
      <div v-if="!showSplitPrompt">
        <button
          class="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-neutral hover:text-neutral-content transition-colors text-left"
          @click.stop="showSplitPrompt = true"
        >
          <Scissors class="w-3.5 h-3.5 shrink-0 stroke-[2px]" /> Split...
        </button>
      </div>
      <div v-else class="px-3 py-2 flex flex-col gap-2">
        <div class="flex items-center justify-between">
          <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Split into</span>
          <span class="text-xs font-mono font-bold">{{ splitCount }}</span>
        </div>
        <input
          v-model.number="splitCount"
          type="range"
          min="2"
          max="20"
          class="range range-xs range-neutral transition-none"
        />
        <button
          class="h-8 w-full rounded-[2px] bg-neutral text-neutral-content text-[10px] font-bold tracking-widest uppercase hover:opacity-80 transition-opacity"
          @click="emit('split', splitCount); emit('close')"
        >
          CONFIRM SPLIT
        </button>
      </div>
    </template>

    <!-- Multi-select options -->
    <template v-if="isMulti">
      <button
        class="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-neutral hover:text-neutral-content transition-colors text-left"
        @click="emit('merge'); emit('close')"
      >
        <Merge class="w-3.5 h-3.5 shrink-0 stroke-[2px]" /> Merge
      </button>
    </template>

    <!-- Assign to class -->
    <div class="border-t border-base-300 my-1"></div>
    <div class="px-3 py-1.5">
      <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/40">Assign to Class</span>
    </div>
    <button
      class="w-full flex items-center gap-2.5 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-neutral hover:text-neutral-content transition-colors text-left"
      @click="emit('createClassAndAssign'); emit('close')"
    >
      <Plus class="w-3.5 h-3.5 shrink-0 stroke-[2px]" />
      New Class...
    </button>
    <div v-if="classes.length > 0" class="max-h-40 overflow-y-auto">
      <button
        v-for="cls in classes"
        :key="cls.id"
        class="w-full flex items-center gap-2.5 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-neutral hover:text-neutral-content transition-colors text-left"
        @click="emit('assignClass', cls.id); emit('close')"
      >
        <span class="w-2.5 h-2.5 rounded-[2px] shrink-0" :style="{ backgroundColor: cls.color }"></span>
        {{ cls.name }}
      </button>
    </div>
  </div>
</template>
