<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useGalleryStore } from '@/stores/gallery'
import { useSessionStore } from '@/stores/session'
import { getPreviewStatus } from '@/api/sessions'
import type { PreviewStatus } from '@/types'
import GatingActiveBanner from '@/components/common/GatingActiveBanner.vue'
import FilterBar from '@/components/gallery/FilterBar.vue'
import ClassAssignBar from '@/components/gallery/ClassAssignBar.vue'
import SampleGrid from '@/components/gallery/SampleGrid.vue'
import SampleDetailPanel from '@/components/gallery/SampleDetailPanel.vue'
import SimilaritySearchDialog from '@/components/gallery/SimilaritySearchDialog.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

const gallery = useGalleryStore()
const sessionStore = useSessionStore()

const zoomLevel = ref(3)
const maskViewEnabled = ref(false)
const similarityDialog = ref<InstanceType<typeof SimilaritySearchDialog> | null>(null)
const dragTrigger = ref<HTMLDivElement | null>(null)

const previewReady = ref(false)
const previewStatus = ref<PreviewStatus | null>(null)
let previewPollHandle: number | null = null

async function checkPreviewReady() {
  const sid = sessionStore.currentSessionId
  if (!sid) return
  try {
    const status = await getPreviewStatus(sid)
    previewStatus.value = status
    if (status.ready) {
      previewReady.value = true
      stopPreviewPoll()
      gallery.fetchSamples()
      gallery.fetchClasses()
      gallery.fetchSortableAttributes()
    } else if (!previewPollHandle) {
      previewPollHandle = window.setInterval(checkPreviewReady, 500) as unknown as number
    }
  } catch {
    // transient
  }
}

function stopPreviewPoll() {
  if (previewPollHandle !== null) {
    clearInterval(previewPollHandle)
    previewPollHandle = null
  }
}

onMounted(() => {
  gallery.openSimilarityDialog = (ids: string[]) => {
    similarityDialog.value?.open({ referenceSampleIds: ids })
  }
  checkPreviewReady()
})

onUnmounted(() => {
  gallery.openSimilarityDialog = null
  stopPreviewPoll()
})
</script>

<template>
  <div class="flex flex-col h-full relative overflow-hidden">
    <!-- Preview pregen gate. Blocks the gallery until per-channel thumbnails
         are on disk, otherwise the UI hammers thumbnail endpoints. -->
    <div
      v-if="!previewReady"
      class="absolute inset-0 z-[700] bg-base-100/95 flex items-center justify-center"
    >
      <div class="w-full max-w-md bg-base-100 border border-base-300 rounded-[2px] p-6 space-y-3 shadow-xl">
        <div class="flex items-center gap-3">
          <LoadingSpinner size="loading-sm" />
          <h2 class="text-[12px] font-bold tracking-widest uppercase">Pre-generating previews</h2>
        </div>
        <progress
          class="progress progress-neutral w-full rounded-[2px]"
          :value="previewStatus?.progress ?? 0" max="100"
        />
        <p class="text-[10px] font-mono text-neutral/60 truncate">{{ previewStatus?.message ?? 'Starting...' }}</p>
      </div>
    </div>
    <GatingActiveBanner />
    <div class="flex flex-1 min-h-0 overflow-hidden">
    <!-- Outer column: two separate white panes stacked with a gap.
         Used as the drag-select trigger zone so the rectangle can also be
         started from the inter-pane gap; cards are still inside Pane 2's
         scroll container. -->
    <div ref="dragTrigger" class="flex-1 flex flex-col min-w-0 gap-6 overflow-hidden">

      <!-- Pane 1: Filter Controls (excluded from drag-select) -->
      <div data-drag-exclude class="flex-none bg-base-100 border border-base-300">
        <FilterBar
          :zoom-level="zoomLevel"
          :mask-view="maskViewEnabled"
          :mask-available="gallery.sessionHasAnyMask"
          :inspect-mode="gallery.inspectMode"
          @update:zoom-level="zoomLevel = $event"
          @update:mask-view="maskViewEnabled = $event"
          @update:inspect-mode="gallery.inspectMode = $event"
        />
        <ClassAssignBar />
      </div>

      <!-- Pane 2: Sample Grid -->
      <div class="flex-1 bg-base-100 border border-base-300 rounded-[2px] overflow-hidden flex flex-col">
        <div v-if="gallery.loading && gallery.samples.length === 0" class="flex-1 flex items-center justify-center">
          <LoadingSpinner size="loading-lg" />
        </div>

        <SampleGrid
          v-else
          :zoom-level="zoomLevel"
          :mask-view="maskViewEnabled"
          :inspect-mode="gallery.inspectMode"
          :drag-trigger="dragTrigger"
          class="flex-1 overflow-hidden"
        />
      </div>

    </div>

    <!-- Sliding Sidebar Overlay -->
    <SampleDetailPanel />
    </div>
    <SimilaritySearchDialog ref="similarityDialog" />
  </div>
</template>
