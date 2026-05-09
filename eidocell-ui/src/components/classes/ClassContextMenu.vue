<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { Eye, BarChart3, Trash2 } from 'lucide-vue-next'

const props = defineProps<{
  x: number
  y: number
  isUncategorized?: boolean
}>()

const emit = defineEmits<{
  view: []
  showStats: []
  delete: []
  close: []
}>()

const style = computed(() => {
  const menuW = 200
  const menuH = 200
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

    <template v-if="!isUncategorized">
      <div class="border-t border-base-300 my-1"></div>
      <button
        class="w-full flex items-center gap-3 px-3 py-2 text-[11px] font-mono font-bold tracking-widest uppercase hover:bg-error hover:text-error-content transition-colors text-left text-error/80"
        @click="emit('delete'); emit('close')"
      >
        <Trash2 class="w-3.5 h-3.5 shrink-0 stroke-[2px]" /> Delete
      </button>
    </template>
  </div>
</template>
