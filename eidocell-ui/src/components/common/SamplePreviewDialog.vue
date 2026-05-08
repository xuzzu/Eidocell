<script setup lang="ts">
import { ref, computed, nextTick, onMounted } from 'vue'
import { X } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useGalleryStore } from '@/stores/gallery'
import { assignSamplesToClass } from '@/api/gallery'
import type { SampleOut } from '@/types'
import SampleCardGrid from '@/components/gallery/SampleCardGrid.vue'
import GalleryContextMenu from '@/components/gallery/GalleryContextMenu.vue'
import SimilaritySearchDialog from '@/components/gallery/SimilaritySearchDialog.vue'

const sessionStore = useSessionStore()
const gallery = useGalleryStore()
const sid = computed(() => sessionStore.currentSessionId!)

const dialogRef = ref<HTMLDialogElement>()
const scrollRef = ref<HTMLElement>()
const title = ref('')
const color = ref<string | null>(null)
const samples = ref<SampleOut[]>([])
const total = ref(0)
const loadingMore = ref(false)

const selectedIds = ref<Set<string>>(new Set())
const ctxMenu = ref({ visible: false, x: 0, y: 0 })
const similarityDialog = ref<InstanceType<typeof SimilaritySearchDialog> | null>(null)

// The caller provides a fetch function for loading more pages
let fetchMore: ((offset: number, limit: number) => Promise<{ items: SampleOut[]; total: number }>) | null = null

const PAGE_SIZE = 100

function open(opts: {
  title: string
  color?: string
  samples: SampleOut[]
  total: number
  fetchMore?: (offset: number, limit: number) => Promise<{ items: SampleOut[]; total: number }>
}) {
  title.value = opts.title
  color.value = opts.color ?? null
  samples.value = opts.samples
  total.value = opts.total
  fetchMore = opts.fetchMore ?? null
  selectedIds.value.clear()
  ctxMenu.value.visible = false
  dialogRef.value?.showModal()
  nextTick(() => scrollRef.value?.scrollTo(0, 0))
}

function close() {
  dialogRef.value?.close()
}

const hasMore = computed(() => samples.value.length < total.value)

async function onScroll(e: Event) {
  if (!hasMore.value || loadingMore.value || !fetchMore) return
  const el = e.target as HTMLElement
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 200) {
    loadingMore.value = true
    try {
      const result = await fetchMore(samples.value.length, PAGE_SIZE)
      samples.value.push(...result.items)
      total.value = result.total
    } finally {
      loadingMore.value = false
    }
  }
}

function onContextmenu(e: MouseEvent, sampleId: string) {
  if (!selectedIds.value.has(sampleId)) {
    selectedIds.value = new Set([sampleId])
  }
  ctxMenu.value = { visible: true, x: e.clientX, y: e.clientY }
}

async function assignSelectionToClass(classId: string) {
  if (selectedIds.value.size === 0) return
  const ids = Array.from(selectedIds.value)
  await assignSamplesToClass(sid.value, { sample_ids: ids, class_id: classId })
  
  // Update class_color of assigned samples in the current view immediately
  const targetClass = gallery.classes.find(c => c.id === classId)
  if (targetClass) {
    for (const sample of samples.value) {
      if (selectedIds.value.has(sample.id)) {
        sample.class_color = targetClass.color
      }
    }
  }
  await gallery.fetchClasses()
}

function findSimilarFromSelection() {
  if (selectedIds.value.size === 0) return
  const ids = Array.from(selectedIds.value)
  similarityDialog.value?.open({ referenceSampleIds: ids })
}

onMounted(() => {
  gallery.fetchClasses()
})

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box max-w-5xl max-h-[85vh] flex flex-col p-6">
      <div class="flex items-center gap-3 mb-4 shrink-0">
        <span
          v-if="color"
          class="w-3 h-3 rounded-[2px] shrink-0"
          :style="{ backgroundColor: color }"
        ></span>
        <h3 class="font-bold text-lg flex-1">{{ title }}</h3>
        
        <span
          v-if="selectedIds.size > 0"
          class="mr-4 text-[10px] font-mono font-bold tracking-widest uppercase text-primary"
        >
          {{ selectedIds.size }} selected
        </span>

        <span class="text-[10px] font-mono text-base-content/50 uppercase tracking-widest">{{ total }} samples</span>
        <button class="btn btn-ghost btn-sm btn-square" @click="close">
          <X class="w-4 h-4" />
        </button>
      </div>
      <div ref="scrollRef" class="flex-1 overflow-y-auto pr-1 -mr-1" @scroll="onScroll">
        <SampleCardGrid
          :samples="samples"
          v-model:selected-ids="selectedIds"
          :zoom-level="3"
          @contextmenu="onContextmenu"
        />
        
        <div v-if="loadingMore" class="flex justify-center py-4">
          <span class="loading loading-spinner loading-sm text-neutral/40"></span>
        </div>
        <div v-if="samples.length === 0" class="text-center py-8 text-base-content/40 text-sm">
          No samples
        </div>
      </div>
    </div>
    
    <GalleryContextMenu
      v-if="ctxMenu.visible"
      :x="ctxMenu.x"
      :y="ctxMenu.y"
      :selected-ids="Array.from(selectedIds)"
      :classes="gallery.classes"
      @assign-class="assignSelectionToClass"
      @find-similar="findSimilarFromSelection"
      @close="ctxMenu.visible = false"
    />

    <form method="dialog" class="modal-backdrop">
      <button @click="close">close</button>
    </form>
  </dialog>

  <SimilaritySearchDialog ref="similarityDialog" />
</template>
