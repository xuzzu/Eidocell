<script setup lang="ts">
import { toRef } from 'vue'
import { useTaskPoller } from '@/composables/useTaskPoller'
import type { TaskInfo } from '@/types'

const props = defineProps<{
  taskId: string | null
}>()

const emit = defineEmits<{
  complete: [task: TaskInfo]
}>()

const { task, isRunning, isComplete, isFailed, progress } = useTaskPoller(
  toRef(props, 'taskId'),
  (t) => emit('complete', t),
)
</script>

<template>
  <div v-if="taskId" class="w-full">
    <div class="flex items-center gap-2 text-sm">
      <span v-if="isRunning" class="loading loading-spinner loading-xs"></span>
      <span v-if="isComplete" class="text-success">Complete</span>
      <span v-if="isFailed" class="text-error">Failed: {{ task?.error }}</span>
      <span v-if="isRunning">{{ task?.message ?? 'Processing...' }}</span>
      <span v-if="isRunning" class="ml-auto text-xs text-base-content/50">
        {{ task?.progress ?? 0 }}/{{ task?.total ?? '?' }}
      </span>
    </div>
    <progress
      class="progress w-full"
      :class="{ 'progress-primary': isRunning, 'progress-success': isComplete, 'progress-error': isFailed }"
      :value="progress"
      max="100"
    ></progress>
  </div>
</template>
