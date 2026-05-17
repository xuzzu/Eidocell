<script setup lang="ts">
import { ArrowUpRight } from 'lucide-vue-next'

defineProps<{
  visible: boolean
  tabName: string
  cursorOutside: boolean
}>()
</script>

<template>
  <Transition name="fade">
    <div
      v-if="visible"
      class="pointer-events-none absolute inset-0 z-30 flex items-center justify-center"
    >
      <div
        class="absolute inset-2 rounded-[2px] border-2 border-dashed transition-colors duration-150"
        :class="cursorOutside
          ? 'border-accent bg-accent/10'
          : 'border-neutral/60 bg-neutral/5'"
      ></div>
      <div
        class="relative flex items-center gap-3 rounded-[2px] border border-neutral bg-base-100/95 px-5 py-3 shadow-2xl backdrop-blur-sm"
      >
        <ArrowUpRight class="w-4 h-4 text-neutral" />
        <span class="text-[11px] font-bold uppercase tracking-widest text-neutral">
          {{ cursorOutside ? `Release to detach ${tabName}` : `Drag outside to detach ${tabName}` }}
        </span>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
</style>
