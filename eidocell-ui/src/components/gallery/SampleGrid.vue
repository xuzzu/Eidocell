<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, toRef, watch } from 'vue'
import { useGalleryStore } from '@/stores/gallery'
import type { SampleOut } from '@/types'
import { useDragSelect } from '@/composables/useDragSelect'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import GalleryContextMenu from '@/components/gallery/GalleryContextMenu.vue'
import SampleCardGrid from '@/components/gallery/SampleCardGrid.vue'

const gallery = useGalleryStore()

const props = defineProps<{
  zoomLevel: number
  maskView: boolean
  inspectMode: boolean
  dragTrigger?: HTMLElement | null
}>()

// Infinite scroll
const scrollContainer = ref<HTMLDivElement | null>(null)
const loadingMore = ref(false)

const hasMore = computed(() => gallery.samples.length < gallery.total)

async function loadMore() {
  if (loadingMore.value || !hasMore.value || gallery.loading) return
  loadingMore.value = true
  try {
    await gallery.loadNextPage()
  } finally {
    loadingMore.value = false
  }
}

function onScroll() {
  const el = scrollContainer.value
  if (!el) return
  if (el.scrollHeight - el.scrollTop - el.clientHeight < 200) {
    loadMore()
  }
}

function onCardInspect(sample: SampleOut, channelIndex?: number) {
  gallery.openDetail(sample, channelIndex ?? 0)
}

// Drag selection. Listen on the wider `dragTrigger` element (e.g. the gallery
// pane wrapper supplied by GalleryView) so mousedowns in the surrounding
// margins also start a rectangle. The rectangle itself is clamped to
// `scrollContainer` bounds inside the composable.
const dragTriggerRef = computed(() => props.dragTrigger ?? null)
const { selectionRect } = useDragSelect(
  scrollContainer,
  '[data-sample-id]',
  'data-sample-id',
  (ids) => gallery.setSelection(ids),
  toRef(gallery, 'selectedIds'),
  dragTriggerRef,
)

// Context menu
const ctxMenu = ref({ visible: false, x: 0, y: 0 })

function onContextmenu(e: MouseEvent, sampleId: string) {
  if (!gallery.selectedIds.has(sampleId)) {
    gallery.setSelection([sampleId])
  }
  ctxMenu.value = { visible: true, x: e.clientX, y: e.clientY }
}

onMounted(() => {
  scrollContainer.value?.addEventListener('scroll', onScroll, { passive: true })
})

onUnmounted(() => {
  scrollContainer.value?.removeEventListener('scroll', onScroll)
})

watch(scrollContainer, (el, oldEl) => {
  oldEl?.removeEventListener('scroll', onScroll)
  el?.addEventListener('scroll', onScroll, { passive: true })
})
</script>

<template>
  <div class="flex flex-col h-full relative select-none">
    <!-- Scrollable grid -->
    <div ref="scrollContainer" class="flex-1 overflow-y-auto p-2 relative">
      <div
        v-if="selectionRect"
        class="absolute border border-primary/40 bg-primary/10 pointer-events-none z-20"
        :style="{
          left: selectionRect.x + 'px',
          top: selectionRect.y + 'px',
          width: selectionRect.width + 'px',
          height: selectionRect.height + 'px',
        }"
      />

      <SampleCardGrid
        :samples="gallery.samples"
        :selected-ids="gallery.selectedIds"
        :zoom-level="zoomLevel"
        :mask-view="maskView"
        :mask-version="gallery.maskVersion"
        :inspect-mode="inspectMode"
        @update:selected-ids="(ids) => gallery.setSelection(Array.from(ids))"
        @inspect="onCardInspect"
        @contextmenu="onContextmenu"
      />

      <div v-if="loadingMore" class="flex justify-center py-4">
        <LoadingSpinner size="loading-sm" />
      </div>

      <div v-if="gallery.samples.length === 0 && !gallery.loading" class="flex justify-center py-12 text-base-content/40">
        No samples found
      </div>

      <div v-if="!hasMore && gallery.samples.length > 0" class="text-center text-xs text-base-content/40 py-3">
        All {{ gallery.total }} samples loaded
      </div>
    </div>

    <!-- Context menu -->
    <GalleryContextMenu
      v-if="ctxMenu.visible"
      :x="ctxMenu.x"
      :y="ctxMenu.y"
      :selected-ids="Array.from(gallery.selectedIds)"
      :classes="gallery.classes"
      @assign-class="(id) => gallery.assignSelectedToClass(id)"
      @find-similar="gallery.openSimilarityDialog?.(Array.from(gallery.selectedIds))"
      @close="ctxMenu.visible = false"
    />
  </div>
</template>
