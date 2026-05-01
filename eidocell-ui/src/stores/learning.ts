import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useSessionStore } from './session'
import * as learningApi from '@/api/learning'
import type { TrainedModelOut, TrainModelResult, InferenceResult } from '@/types'

export const useLearningStore = defineStore('learning', () => {
  const sessionStore = useSessionStore()

  const models = ref<TrainedModelOut[]>([])
  const training = ref(false)
  const inferring = ref(false)
  const lastTrainResult = ref<TrainModelResult | null>(null)
  const lastInferenceResult = ref<InferenceResult | null>(null)
  const error = ref('')

  function $reset() {
    models.value = []
    training.value = false
    inferring.value = false
    lastTrainResult.value = null
    lastInferenceResult.value = null
    error.value = ''
  }

  async function fetchModels() {
    if (!sessionStore.currentSessionId) return
    models.value = await learningApi.listSessionModels(sessionStore.currentSessionId)
  }

  async function train(name: string, classifierType: string, params: Record<string, unknown>) {
    if (!sessionStore.currentSessionId) return
    training.value = true
    error.value = ''
    lastTrainResult.value = null
    try {
      lastTrainResult.value = await learningApi.trainModel(sessionStore.currentSessionId, {
        name,
        classifier_type: classifierType,
        params,
      })
      await fetchModels()
    } catch (e: any) {
      error.value = e.message || 'Training failed'
    } finally {
      training.value = false
    }
  }

  async function infer(modelId: string) {
    if (!sessionStore.currentSessionId) return
    inferring.value = true
    error.value = ''
    lastInferenceResult.value = null
    try {
      lastInferenceResult.value = await learningApi.runInference(sessionStore.currentSessionId, {
        model_id: modelId,
      })
    } catch (e: any) {
      error.value = e.message || 'Classification failed'
    } finally {
      inferring.value = false
    }
  }

  async function removeModel(modelId: string) {
    error.value = ''
    try {
      await learningApi.deleteModel(modelId)
      await fetchModels()
    } catch (e: any) {
      error.value = e.message || 'Failed to delete model'
    }
  }

  return {
    models, training, inferring,
    lastTrainResult, lastInferenceResult, error,
    $reset, fetchModels, train, infer, removeModel,
  }
})
