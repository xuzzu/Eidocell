<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useGalleryStore } from '@/stores/gallery'
import GatingActiveBanner from '@/components/common/GatingActiveBanner.vue'
import FilterBar from '@/components/gallery/FilterBar.vue'
import ClassAssignBar from '@/components/gallery/ClassAssignBar.vue'
import SampleGrid from '@/components/gallery/SampleGrid.vue'
import SampleDetailPanel from '@/components/gallery/SampleDetailPanel.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

const gallery = useGalleryStore()

const zoomLevel = ref(3)
const maskViewEnabled = ref(false)
const inspectMode = ref(true)

onMounted(() => {
  gallery.fetchSamples()
  gallery.fetchClasses()
  gallery.fetchSortableAttributes()
})
</script>

<template>
  <div class="flex flex-col h-full relative overflow-hidden">
    <GatingActiveBanner />
    <div class="flex flex-1 min-h-0 overflow-hidden">
    <!-- Outer column: two separate white panes stacked with a gap -->
    <div class="flex-1 flex flex-col min-w-0 gap-4 overflow-hidden">

      <!-- Pane 1: Filter Controls -->
      <div class="flex-none bg-base-100 border border-base-300">
        <FilterBar
          :zoom-level="zoomLevel"
          :mask-view="maskViewEnabled"
          :inspect-mode="inspectMode"
          @update:zoom-level="zoomLevel = $event"
          @update:mask-view="maskViewEnabled = $event"
          @update:inspect-mode="inspectMode = $event"
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
          :inspect-mode="inspectMode"
          class="flex-1 overflow-hidden"
        />
      </div>

    </div>

    <!-- Sliding Sidebar Overlay -->
    <SampleDetailPanel />
    </div>
  </div>
</template>
