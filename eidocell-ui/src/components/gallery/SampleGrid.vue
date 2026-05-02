<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useSessionStore } from '@/stores/session'
import { useGalleryStore } from '@/stores/gallery'
import { sampleThumbnailUrl } from '@/api/gallery'
import { maskOverlayUrl } from '@/api/segmentation'
import type { SampleOut } from '@/types'
import { useDragSelect } from '@/composables/useDragSelect'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import GalleryContextMenu from '@/components/gallery/GalleryContextMenu.vue'

const sessionStore = useSessionStore()

const gallery = useGalleryStore()

const sid = computed(() => sessionStore.currentSessionId!)

const props = defineProps<{
  zoomLevel: number
  maskView: boolean
  inspectMode: boolean
}>()

const colsClass = computed(() => {
  const map: Record<number, string> = {
    1: 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4',
    2: 'grid-cols-3 md:grid-cols-5 lg:grid-cols-6',
    3: 'grid-cols-4 md:grid-cols-6 lg:grid-cols-8 xl:grid-cols-10',
    4: 'grid-cols-6 md:grid-cols-8 lg:grid-cols-12 xl:grid-cols-12',
    5: 'grid-cols-8 md:grid-cols-12 lg:grid-cols-12 xl:grid-cols-12',
  }
  return map[props.zoomLevel] ?? map[3]
})

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

function isSelected(id: string) {
  return gallery.selectedIds.has(id)
}

function getImageSrc(sample: SampleOut) {
  if (props.maskView && sample.has_mask) {
    return maskOverlayUrl(sid.value, sample.id, gallery.maskVersion)
  }
  return sampleThumbnailUrl(sid.value, sample.id)
}

function onClick(sample: SampleOut) {
  if (props.inspectMode) {
    gallery.detailSample = sample
  } else {
    // In selection mode, single click should replace selection
    // wait, for cluster it is selectSingle. We'll do setSelection here
    gallery.setSelection([sample.id])
  }
}

// Drag selection
const { selectionRect } = useDragSelect(
  scrollContainer,
  '[data-sample-id]',
  'data-sample-id',
  (ids) => gallery.setSelection(ids),
  gallery.selectedIds as any,
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

      <div class="grid gap-2" :class="colsClass">
        <div
          v-for="sample in gallery.samples"
          :key="sample.id"
          :data-sample-id="sample.id"
          class="cursor-pointer rounded-[2px] overflow-hidden transition-all flex flex-col bg-base-100"
          :class="isSelected(sample.id) ? 'ring-2 ring-primary ring-offset-1 ring-offset-base-200' : 'hover:ring-1 hover:ring-neutral/30'"
          @click.exact="onClick(sample)"
          @click.ctrl="gallery.toggleSelection(sample.id)"
          @click.meta="gallery.toggleSelection(sample.id)"
          @contextmenu.prevent="onContextmenu($event, sample.id)"
        >
          <!-- Filename label at top with class color background -->
          <div
            class="px-1.5 py-0.5 truncate text-[9px] text-white font-mono font-medium leading-tight shrink-0"
            :style="{ backgroundColor: sample.class_color || '#9ca3af' }"
          >
            {{ sample.filename }}
          </div>
          <div class="w-full aspect-square relative bg-base-200 flex items-center justify-center">
            <img
              :src="getImageSrc(sample)"
              :alt="sample.filename"
              class="absolute inset-0 w-full h-full object-contain"
              loading="lazy"
            />
          </div>
        </div>
      </div>

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
      @close="ctxMenu.visible = false"
    />
  </div>
</template>
