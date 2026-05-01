import { ref, computed, watch, onUnmounted, type Ref } from 'vue'
import { getTask } from '@/api/tasks'
import type { TaskInfo } from '@/types'

export function useTaskPoller(taskId: Ref<string | null>, onComplete?: (task: TaskInfo) => void) {
  const task = ref<TaskInfo | null>(null)
  const isRunning = computed(() => task.value?.status === 'pending' || task.value?.status === 'running')
  const isComplete = computed(() => task.value?.status === 'completed')
  const isFailed = computed(() => task.value?.status === 'failed')
  const progress = computed(() => task.value?.percentage ?? 0)

  let timer: ReturnType<typeof setInterval> | null = null

  function stopPolling() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  async function poll() {
    if (!taskId.value) return
    try {
      task.value = await getTask(taskId.value)
      if (task.value.status === 'completed' || task.value.status === 'failed') {
        stopPolling()
        if (task.value.status === 'completed' && onComplete) {
          onComplete(task.value)
        }
      }
    } catch {
      stopPolling()
    }
  }

  watch(taskId, (id) => {
    stopPolling()
    task.value = null
    if (id) {
      poll()
      timer = setInterval(poll, 1000)
    }
  }, { immediate: true })

  onUnmounted(stopPolling)

  return { task, isRunning, isComplete, isFailed, progress }
}
