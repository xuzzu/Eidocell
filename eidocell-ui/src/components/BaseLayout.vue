<script setup lang="ts">
import Sidebar from './Sidebar.vue'
import { useTasksStore } from '@/stores/tasks'

const tasks = useTasksStore()
</script>

<template>
  <div class="flex h-screen overflow-hidden bg-base-200 text-neutral font-sans selection:bg-neutral/20 selection:text-neutral">
    <Sidebar />
    <main class="flex-1 flex flex-col relative overflow-hidden bg-base-200">
      <div class="flex-1 overflow-auto relative">
        <slot />
      </div>

      <!-- Global task indicator -->
      <Transition name="fade">
        <div
          v-if="tasks.hasActiveTasks"
          class="absolute bottom-4 right-4 bg-base-100 border border-base-300 shadow-xl rounded-[2px] p-3 min-w-[200px]"
        >
          <div v-for="task in tasks.activeTaskList" :key="task.id" class="flex items-center gap-3 text-sm">
            <span class="loading loading-spinner loading-xs text-secondary"></span>
            <span class="truncate text-xs font-mono font-medium">{{ task.name }}</span>
            <span class="ml-auto text-xs font-mono text-neutral/50">{{ task.percentage.toFixed(0) }}%</span>
          </div>
        </div>
      </Transition>
    </main>
  </div>
</template>

<style scoped>
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s ease-out;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
</style>
