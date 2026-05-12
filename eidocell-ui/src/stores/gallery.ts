import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { useSessionStore } from './session'
import * as galleryApi from '@/api/gallery'
import type { SampleOut, FilterCondition, ClassOut, ClassCreate, SortableAttribute } from '@/types'

export type ChannelDisplayMode = 'single' | 'multi'

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
  const detailChannel = ref(0)
  const classes = ref<ClassOut[]>([])
  const loading = ref(false)
  const sortableAttributes = ref<SortableAttribute[]>([])

  // Channel display state — picked up by SampleCardGrid + SampleDetailPanel.
  // Defaults are resolved on session swap.
  const channelDisplayMode = ref<ChannelDisplayMode>('multi')
  const selectedChannel = ref(0)
  const selectedChannels = ref<number[]>([])
  // Bumped after segmentation completes — used as a query-string version on
  // mask-overlay URLs to defeat the browser's in-memory image cache when the
  // file at a stable URL has been overwritten.
  const maskVersion = ref(0)
  // True when the session has at least one extracted mask. Drives the
  // FilterBar MASK VIEW toggle's disabled state. Refreshed on every
  // gallery fetch (page-listing also returns this aggregate).
  const sessionHasAnyMask = ref(false)

  // Persists across view switches so the toggle isn't reset when GalleryView
  // remounts after the user navigates away and back.
  const inspectMode = ref(true)

  // Set by GalleryView; called by children to open the SimilaritySearchDialog
  // without prop-drilling its ref through SampleGrid.
  const openSimilarityDialog = ref<((ids: string[]) => void) | null>(null)

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
    detailChannel.value = 0
    classes.value = []
    loading.value = false
    sortableAttributes.value = []
    maskVersion.value = 0
    sessionHasAnyMask.value = false
  }

  function openDetail(sample: SampleOut, channel = 0) {
    detailSample.value = sample
    detailChannel.value = Math.max(0, Math.min(channel, (sample.n_channels ?? 1) - 1))
  }

  // ── Channel display ────────────────────────────────────────────────────

  function _channelStorageKey(sid: string) {
    return `eidocell_channel_display_${sid}`
  }

  function _loadChannelDisplay(sid: string, channelCount: number) {
    const fallback = () => {
      if (channelCount > 1) {
        channelDisplayMode.value = 'multi'
        selectedChannels.value = Array.from({ length: channelCount }, (_, i) => i)
        selectedChannel.value = 0
      } else {
        channelDisplayMode.value = 'single'
        selectedChannel.value = 0
        selectedChannels.value = [0]
      }
    }
    try {
      const raw = localStorage.getItem(_channelStorageKey(sid))
      if (!raw) return fallback()
      const parsed = JSON.parse(raw) as {
        mode?: ChannelDisplayMode
        selectedChannel?: number
        selectedChannels?: number[]
      }
      const mode = parsed.mode === 'single' || parsed.mode === 'multi' ? parsed.mode : null
      const sc = typeof parsed.selectedChannel === 'number' ? parsed.selectedChannel : null
      const scs = Array.isArray(parsed.selectedChannels) ? parsed.selectedChannels.filter(n => Number.isInteger(n)) : null
      if (mode === null) return fallback()
      channelDisplayMode.value = (mode === 'multi' && channelCount === 1) ? 'single' : mode
      selectedChannel.value = (sc !== null && sc >= 0 && sc < channelCount) ? sc : 0
      const filtered = (scs ?? []).filter(c => c >= 0 && c < channelCount)
      selectedChannels.value = filtered.length > 0
        ? filtered
        : Array.from({ length: channelCount }, (_, i) => i)
    } catch {
      fallback()
    }
  }

  function _persistChannelDisplay() {
    const sid = sessionStore.currentSessionId
    if (!sid) return
    try {
      localStorage.setItem(_channelStorageKey(sid), JSON.stringify({
        mode: channelDisplayMode.value,
        selectedChannel: selectedChannel.value,
        selectedChannels: selectedChannels.value,
      }))
    } catch { /* quota / disabled */ }
  }

  function setChannelDisplayMode(mode: ChannelDisplayMode) {
    channelDisplayMode.value = mode
    _persistChannelDisplay()
  }

  function setSelectedChannel(idx: number) {
    selectedChannel.value = idx
    _persistChannelDisplay()
  }

  function setSelectedChannels(channels: number[]) {
    const sorted = [...new Set(channels)].sort((a, b) => a - b)
    selectedChannels.value = sorted
    _persistChannelDisplay()
  }

  const sessionChannelCount = computed(() => sessionStore.currentSession?.channel_count ?? 1)
  const sessionChannelNames = computed<string[]>(() => {
    const names = sessionStore.currentSession?.channel_names
    const count = sessionChannelCount.value
    if (Array.isArray(names) && names.length === count) return names
    return Array.from({ length: count }, (_, i) => `Channel ${i + 1}`)
  })

  watch(
    () => [sessionStore.currentSessionId, sessionStore.currentSession?.channel_count] as const,
    ([sid, count]) => {
      if (!sid) return
      _loadChannelDisplay(sid, Math.max(1, Number(count ?? 1)))
    },
    { immediate: true },
  )

  function bumpMaskVersion() {
    maskVersion.value++
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
      sessionHasAnyMask.value = page.session_has_any_mask
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
    sessionHasAnyMask.value = page.session_has_any_mask
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

  const sortableAttributesByChannel = computed<Record<number, SortableAttribute[]>>(() => {
    const grouped: Record<number, SortableAttribute[]> = {}
    for (const a of sortableAttributes.value) {
      ;(grouped[a.channel_index] ??= []).push(a)
    }
    return grouped
  })

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
    selectedIds, detailSample, detailChannel, classes, loading,
    sortableAttributes, sortableAttributesByChannel,
    maskVersion, sessionHasAnyMask, inspectMode, openSimilarityDialog,
    channelDisplayMode, selectedChannel, selectedChannels,
    sessionChannelCount, sessionChannelNames,
    $reset, openDetail,
    fetchSamples, loadNextPage, fetchClasses, fetchSortableAttributes,
    createClass, deleteClass,
    assignSelectedToClass, toggleSelection, setSelection, clearSelection,
    setSort, setFilters, bumpMaskVersion,
    setChannelDisplayMode, setSelectedChannel, setSelectedChannels,
  }
})
