<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import type { ClassOut } from '@/types'

const props = defineProps<{
  x: number
  y: number
  selectedIds: string[]
  classes: ClassOut[]
}>()

const emit = defineEmits<{
  assignClass: [classId: string]
  close: []
}>()

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
    <!-- Assign to class -->
    <template v-if="classes.length > 0">
      <div class="px-3 py-1.5 border-b border-base-300 mb-1">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/40">
          Assign {{ selectedIds.length }} {{ selectedIds.length === 1 ? 'Sample' : 'Samples' }}
        </span>
      </div>
      <div class="max-h-40 overflow-y-auto">
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
    </template>

    <template v-if="classes.length === 0">
      <div class="px-3 py-2 text-[10px] font-mono text-neutral/30 tracking-wider">No classes defined</div>
    </template>
  </div>
</template>
