import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getTask } from '@/api/tasks'
import type { TaskInfo } from '@/types'

export const useTasksStore = defineStore('tasks', () => {
  const activeTasks = ref<Map<string, TaskInfo>>(new Map())

  const hasActiveTasks = computed(() => {
    for (const t of activeTasks.value.values()) {
      if (t.status === 'pending' || t.status === 'running') return true
    }
    return false
  })

  const activeTaskList = computed(() => Array.from(activeTasks.value.values()))

  function trackTask(taskId: string, name: string) {
    activeTasks.value.set(taskId, {
      id: taskId, name, status: 'pending',
      progress: 0, total: 0, percentage: 0,
      message: null, result: null, error: null,
      created_at: new Date().toISOString(), completed_at: null,
    })
    pollTask(taskId)
  }

  async function pollTask(taskId: string) {
    const poll = async () => {
      try {
        const task = await getTask(taskId)
        activeTasks.value.set(taskId, task)
        if (task.status === 'completed' || task.status === 'failed') return
        setTimeout(poll, 1000)
      } catch {
        activeTasks.value.delete(taskId)
      }
    }
    poll()
  }

  function clearCompleted() {
    for (const [id, task] of activeTasks.value) {
      if (task.status === 'completed' || task.status === 'failed') {
        activeTasks.value.delete(id)
      }
    }
  }

  return { activeTasks, hasActiveTasks, activeTaskList, trackTask, clearCompleted }
})
