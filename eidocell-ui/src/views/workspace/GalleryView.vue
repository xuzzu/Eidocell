<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { useGalleryStore } from '@/stores/gallery'
import GatingActiveBanner from '@/components/common/GatingActiveBanner.vue'
import FilterBar from '@/components/gallery/FilterBar.vue'
import ClassAssignBar from '@/components/gallery/ClassAssignBar.vue'
import SampleGrid from '@/components/gallery/SampleGrid.vue'
import SampleDetailPanel from '@/components/gallery/SampleDetailPanel.vue'
import SimilaritySearchDialog from '@/components/gallery/SimilaritySearchDialog.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

const gallery = useGalleryStore()

const zoomLevel = ref(3)
const maskViewEnabled = ref(false)
const similarityDialog = ref<InstanceType<typeof SimilaritySearchDialog> | null>(null)
const dragTrigger = ref<HTMLDivElement | null>(null)

onMounted(() => {
  gallery.fetchSamples()
  gallery.fetchClasses()
  gallery.fetchSortableAttributes()
  gallery.openSimilarityDialog = (ids: string[]) => {
    similarityDialog.value?.open({ referenceSampleIds: ids })
  }
})

onUnmounted(() => {
  gallery.openSimilarityDialog = null
})
</script>

<template>
  <div class="flex flex-col h-full relative overflow-hidden">
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
