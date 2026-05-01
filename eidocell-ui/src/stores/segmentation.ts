import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useSessionStore } from './session'
import * as segApi from '@/api/segmentation'
import type { SegmentationMethod } from '@/types'

export const useSegmentationStore = defineStore('segmentation', () => {
  const sessionStore = useSessionStore()

  const methods = ref<SegmentationMethod[]>([])
  const selectedMethod = ref('otsu_intensity')
  const params = ref<Record<string, number>>({})
  const taskId = ref<string | null>(null)
  const loading = ref(false)

  function $reset() {
    methods.value = []
    selectedMethod.value = 'otsu_intensity'
    params.value = {}
    taskId.value = null
    loading.value = false
  }

  async function fetchMethods() {
    if (!sessionStore.currentSessionId) return
    methods.value = await segApi.listSegmentationMethods(sessionStore.currentSessionId)
    // Set default params for selected method
    const method = methods.value.find(m => m.id === selectedMethod.value)
    if (method) {
      const defaults: Record<string, number> = {}
      for (const p of method.params) {
        defaults[p.name] = p.default
      }
      params.value = defaults
    }
  }

  function selectMethod(methodId: string) {
    selectedMethod.value = methodId
    const method = methods.value.find(m => m.id === methodId)
    if (method) {
      const defaults: Record<string, number> = {}
      for (const p of method.params) {
        defaults[p.name] = p.default
      }
      params.value = defaults
    }
  }

  async function runSegmentation() {
    if (!sessionStore.currentSessionId) return
    loading.value = true
    try {
      const result = await segApi.runSegmentationAsync(sessionStore.currentSessionId, {
        method: selectedMethod.value,
        params: params.value,
      })
      taskId.value = result.task_id
    } catch {
      loading.value = false
    }
  }

  function onTaskComplete() {
    loading.value = false
    taskId.value = null
  }

  return {
    methods, selectedMethod, params, taskId, loading,
    $reset,
    fetchMethods, selectMethod, runSegmentation, onTaskComplete,
  }
})
