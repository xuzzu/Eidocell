<script setup lang="ts">
import { computed, ref } from 'vue'
import { X } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useGalleryStore } from '@/stores/gallery'
import { useSettingsStore } from '@/stores/settings'
import { searchSimilar } from '@/api/similarity'
import { assignSamplesToClass } from '@/api/gallery'
import { ApiError } from '@/api/client'
import type {
  SimilarityFilter,
  SimilarityHit,
  SimilaritySearchResponse,
} from '@/types'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import SampleCardGrid from '@/components/gallery/SampleCardGrid.vue'
import GalleryContextMenu from '@/components/gallery/GalleryContextMenu.vue'

const sessionStore = useSessionStore()
const gallery = useGalleryStore()
const settings = useSettingsStore()
const sid = computed(() => sessionStore.currentSessionId!)

const dialogRef = ref<HTMLDialogElement>()
const referenceSampleIds = ref<string[]>([])
const filter = ref<SimilarityFilter>('all')
const loading = ref(false)
const error = ref<string | null>(null)
const result = ref<SimilaritySearchResponse | null>(null)
const selectedIds = ref<Set<string>>(new Set())
const ctxMenu = ref({ visible: false, x: 0, y: 0 })

const TOP_K = 2000
const BUCKET_EDGES = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]

const pctBySampleId = computed<Map<string, number>>(() => {
  const m = new Map<string, number>()
  if (result.value) {
    for (const h of result.value.hits) m.set(h.sample.id, h.similarity_pct)
  }
  return m
})

const buckets = computed<{ floor: number; label: string; hits: SimilarityHit[] }[]>(() => {
  if (!result.value) return []
  const byFloor = new Map<number, SimilarityHit[]>()
  for (const h of result.value.hits) {
    const arr = byFloor.get(h.bucket) ?? []
    arr.push(h)
    byFloor.set(h.bucket, arr)
  }
  const out: { floor: number; label: string; hits: SimilarityHit[] }[] = []
  for (const floor of BUCKET_EDGES) {
    const hits = byFloor.get(floor)
    if (!hits || hits.length === 0) continue
    const upper = floor === 90 ? 100 : floor + 10
    out.push({ floor, label: `${floor}–${upper}%`, hits })
  }
  return out
})

const truncated = computed(() =>
  result.value ? result.value.returned < result.value.total_candidates : false,
)

async function runSearch() {
  if (referenceSampleIds.value.length === 0) return
  loading.value = true
  error.value = null
  try {
    result.value = await searchSimilar(sid.value, {
      reference_sample_ids: referenceSampleIds.value,
      filter_mode: filter.value,
      feature_method: settings.settings?.default_feature_method,
      top_k: TOP_K,
    })
    // Drop selections that fell out of the new result set
    const validIds = new Set(result.value.hits.map((h) => h.sample.id))
    const next = new Set<string>()
    for (const id of selectedIds.value) if (validIds.has(id)) next.add(id)
    selectedIds.value = next
  } catch (e) {
    result.value = null
    if (e instanceof ApiError) {
      error.value = e.message
    } else {
      error.value = 'Failed to compute similarity'
    }
  } finally {
    loading.value = false
  }
}

function open(opts: { referenceSampleIds: string[] }) {
  referenceSampleIds.value = [...opts.referenceSampleIds]
  filter.value = 'all'
  result.value = null
  error.value = null
  selectedIds.value = new Set()
  ctxMenu.value.visible = false
  dialogRef.value?.showModal()
  runSearch()
}

function close() {
  dialogRef.value?.close()
}

function setFilter(f: SimilarityFilter) {
  if (filter.value === f) return
  filter.value = f
  runSearch()
}

function onContextmenu(e: MouseEvent, sampleId: string) {
  if (!selectedIds.value.has(sampleId)) {
    selectedIds.value = new Set([sampleId])
  }
  ctxMenu.value = { visible: true, x: e.clientX, y: e.clientY }
}

function findSimilarFromSelection() {
  if (selectedIds.value.size === 0) return
  const ids = Array.from(selectedIds.value)
  open({ referenceSampleIds: ids })
}

async function assignSelectionToClass(classId: string) {
  if (selectedIds.value.size === 0) return
  const ids = Array.from(selectedIds.value)
  await assignSamplesToClass(sid.value, { sample_ids: ids, class_id: classId })
  // Refresh class counts in the gallery store; refetch our result so labels update
  await gallery.fetchClasses()
  await runSearch()
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box max-w-5xl max-h-[85vh] flex flex-col">
      <!-- Header -->
      <div class="flex items-center gap-3 mb-3 shrink-0">
        <h3 class="font-bold text-lg flex-1">
          Find similar to {{ referenceSampleIds.length }}
          {{ referenceSampleIds.length === 1 ? 'image' : 'images' }}
        </h3>
        <span v-if="result" class="text-xs font-mono text-base-content/50">
          {{ result.returned }} / {{ result.total_candidates }}
        </span>
        <button class="btn btn-ghost btn-sm btn-square" @click="close">
          <X class="w-4 h-4" />
        </button>
      </div>

      <!-- Filter toggle + selection indicator -->
      <div class="flex items-center gap-2 mb-3 shrink-0">
        <span class="text-[10px] font-bold tracking-widest uppercase text-base-content/40">
          Filter
        </span>
        <div class="join">
          <button
            class="join-item btn btn-xs rounded-[2px] text-[10px] font-bold tracking-widest uppercase"
            :class="filter === 'all' ? 'btn-neutral' : 'btn-ghost border border-base-300'"
            @click="setFilter('all')"
          >
            All
          </button>
          <button
            class="join-item btn btn-xs rounded-[2px] text-[10px] font-bold tracking-widest uppercase"
            :class="filter === 'unlabeled' ? 'btn-neutral' : 'btn-ghost border border-base-300'"
            @click="setFilter('unlabeled')"
          >
            Unlabeled only
          </button>
        </div>
        <span
          v-if="selectedIds.size > 0"
          class="ml-3 text-[10px] font-mono font-bold tracking-widest uppercase text-primary"
        >
          {{ selectedIds.size }} selected
        </span>
        <span v-if="truncated" class="ml-auto text-[10px] font-mono text-base-content/40">
          Showing top {{ TOP_K }}
        </span>
      </div>

      <!-- Body -->
      <div class="flex-1 overflow-y-auto pr-1 -mr-1">
        <div v-if="loading" class="flex justify-center py-12">
          <LoadingSpinner size="loading-md" />
        </div>

        <div
          v-else-if="error"
          class="flex flex-col items-center justify-center py-12 gap-2 text-center"
        >
          <p class="text-sm text-error font-mono">{{ error }}</p>
          <p
            v-if="error.toLowerCase().includes('features')"
            class="text-xs text-base-content/50"
          >
            Run clustering once on this session to extract features, then try again.
          </p>
        </div>

        <div
          v-else-if="result && buckets.length === 0"
          class="text-center py-12 text-sm text-base-content/40"
        >
          No similar images found.
        </div>

        <div v-else class="space-y-5">
          <section v-for="b in buckets" :key="b.floor">
            <div class="flex items-center gap-2 mb-2 sticky top-0 bg-base-100 py-1 z-10">
              <span class="text-[10px] font-bold tracking-widest uppercase text-base-content/60">
                {{ b.label }}
              </span>
              <span class="text-[10px] font-mono text-base-content/40">
                · {{ b.hits.length }}
              </span>
              <div class="flex-1 h-px bg-base-300 ml-2"></div>
            </div>
            <SampleCardGrid
              :samples="b.hits.map((h) => h.sample)"
              v-model:selected-ids="selectedIds"
              :zoom-level="3"
              @contextmenu="onContextmenu"
            >
              <template #badge="{ sample }">
                <span
                  class="absolute bottom-1 right-1 text-[9px] font-mono font-bold tracking-tight px-1 rounded-[2px] bg-neutral text-neutral-content"
                >
                  {{ (pctBySampleId.get(sample.id) ?? 0).toFixed(1) }}%
                </span>
              </template>
            </SampleCardGrid>
          </section>
        </div>
      </div>
    </div>

    <!-- Context menu inside the dialog -->
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
</template>
