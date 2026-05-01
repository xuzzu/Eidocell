import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useSessionStore } from './session'
import * as classesApi from '@/api/classes'
import * as galleryApi from '@/api/gallery'
import type { ClassOut, SamplePage, ClassCreate } from '@/types'

export const useClassesStore = defineStore('classes', () => {
  const sessionStore = useSessionStore()

  const classes = ref<ClassOut[]>([])
  const selectedClassId = ref<string | null>(null)
  const classSamples = ref<SamplePage | null>(null)
  const loading = ref(false)

  function $reset() {
    classes.value = []
    selectedClassId.value = null
    classSamples.value = null
    loading.value = false
  }

  async function fetchClasses() {
    if (!sessionStore.currentSessionId) return
    classes.value = await galleryApi.listClasses(sessionStore.currentSessionId)
  }

  async function createClass(data: ClassCreate) {
    if (!sessionStore.currentSessionId) return
    await galleryApi.createClass(sessionStore.currentSessionId, data)
    await fetchClasses()
  }

  async function deleteClass(classId: string) {
    if (!sessionStore.currentSessionId) return
    await galleryApi.deleteClass(sessionStore.currentSessionId, classId)
    if (selectedClassId.value === classId) {
      selectedClassId.value = null
      classSamples.value = null
    }
    await fetchClasses()
  }

  async function selectClass(classId: string) {
    selectedClassId.value = classId
    await fetchClassSamples(classId)
  }

  async function fetchClassSamples(classId: string, offset = 0, limit = 100) {
    if (!sessionStore.currentSessionId) return
    classSamples.value = await classesApi.getClassSamples(
      sessionStore.currentSessionId, classId, offset, limit,
    )
  }

  async function getClassSummary(classId: string) {
    if (!sessionStore.currentSessionId) return null
    return classesApi.getClassSummary(sessionStore.currentSessionId, classId)
  }

  return {
    classes, selectedClassId, classSamples, loading,
    $reset,
    fetchClasses, createClass, deleteClass, selectClass, fetchClassSamples, getClassSummary,
  }
})
