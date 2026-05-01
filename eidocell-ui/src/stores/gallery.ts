import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { useSessionStore } from './session'
import * as galleryApi from '@/api/gallery'
import type { SampleOut, FilterCondition, ClassOut, ClassCreate } from '@/types'

export const useGalleryStore = defineStore('gallery', () => {
  const sessionStore = useSessionStore()

  const samples = ref<SampleOut[]>([])
  const total = ref(0)
  const offset = ref(0)
  const limit = ref(100)
  const sortBy = ref('filename')
  const sortOrder = ref('asc')
  const filters = ref<FilterCondition[]>([])
  const selectedIds = ref<Set<string>>(new Set())
  const detailSample = ref<SampleOut | null>(null)
  const classes = ref<ClassOut[]>([])
  const loading = ref(false)
  const sortableAttributes = ref<string[]>([])

  const sid = computed(() => sessionStore.currentSessionId!)

  function $reset() {
    samples.value = []
    total.value = 0
    offset.value = 0
    sortBy.value = 'filename'
    sortOrder.value = 'asc'
    filters.value = []
    selectedIds.value.clear()
    detailSample.value = null
    classes.value = []
    loading.value = false
    sortableAttributes.value = []
  }

  async function fetchSamples() {
    if (!sessionStore.currentSessionId) return
    loading.value = true
    offset.value = 0
    try {
      const page = await galleryApi.listSamples(sid.value, {
        offset: 0,
        limit: limit.value,
        sort_by: sortBy.value,
        sort_order: sortOrder.value,
        filters: filters.value,
      })
      samples.value = page.items
      total.value = page.total
    } finally {
      loading.value = false
    }
  }

  async function loadNextPage() {
    if (!sessionStore.currentSessionId) return
    const newOffset = samples.value.length
    if (newOffset >= total.value) return
    const page = await galleryApi.listSamples(sid.value, {
      offset: newOffset,
      limit: limit.value,
      sort_by: sortBy.value,
      sort_order: sortOrder.value,
      filters: filters.value,
    })
    samples.value = [...samples.value, ...page.items]
    total.value = page.total
  }

  async function fetchClasses() {
    if (!sessionStore.currentSessionId) return
    classes.value = await galleryApi.listClasses(sid.value)
  }

  async function fetchSortableAttributes() {
    if (!sessionStore.currentSessionId) return
    try {
      sortableAttributes.value = await galleryApi.listSortableAttributes(sid.value)
    } catch {
      sortableAttributes.value = []
    }
  }

  async function createClass(data: ClassCreate) {
    const cls = await galleryApi.createClass(sid.value, data)
    await fetchClasses()
    return cls
  }

  async function deleteClass(classId: string) {
    await galleryApi.deleteClass(sid.value, classId)
    await fetchClasses()
    await fetchSamples()
  }

  async function assignSelectedToClass(classId: string | null) {
    if (selectedIds.value.size === 0) return
    await galleryApi.assignSamplesToClass(sid.value, {
      sample_ids: Array.from(selectedIds.value),
      class_id: classId,
    })
    selectedIds.value.clear()
    await fetchSamples()
    await fetchClasses()
  }

  function toggleSelection(id: string) {
    if (selectedIds.value.has(id)) {
      selectedIds.value.delete(id)
    } else {
      selectedIds.value.add(id)
    }
  }

  function setSelection(ids: string[]) {
    selectedIds.value = new Set(ids)
  }

  function clearSelection() {
    selectedIds.value.clear()
  }

  function setSort(field: string, order: string) {
    sortBy.value = field
    sortOrder.value = order
    offset.value = 0
    fetchSamples()
  }

  function setFilters(newFilters: FilterCondition[]) {
    filters.value = newFilters
    offset.value = 0
    fetchSamples()
  }

  return {
    samples, total, offset, limit, sortBy, sortOrder, filters,
    selectedIds, detailSample, classes, loading,
    sortableAttributes,
    $reset,
    fetchSamples, loadNextPage, fetchClasses, fetchSortableAttributes,
    createClass, deleteClass,
    assignSelectedToClass, toggleSelection, setSelection, clearSelection,
    setSort, setFilters,
  }
})
