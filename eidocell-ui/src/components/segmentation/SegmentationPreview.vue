<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RefreshCw, Shuffle, Layers, Image as ImageIcon } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useSegmentationStore } from '@/stores/segmentation'
import { useGalleryStore } from '@/stores/gallery'
import { sampleThumbnailUrl } from '@/api/gallery'
import { useSegmentationPreviewWs } from '@/composables/useSegmentationPreviewWs'
import type { SampleOut } from '@/types'

const sessionStore = useSessionStore()
const segStore = useSegmentationStore()
const gallery = useGalleryStore()

const sid = computed(() => sessionStore.currentSessionId)

const previewCounts = [4, 8, 16, 24, 32] as const
type PreviewCount = (typeof previewCounts)[number]
const nPreview = ref<PreviewCount>(8)
type ViewMode = 'overlay' | 'split'
const viewMode = ref<ViewMode>('overlay')
const previewSamples = ref<SampleOut[]>([])

const ws = useSegmentationPreviewWs(() => sid.value)

const gridCols = computed(() => {
  const n = nPreview.value
  if (n <= 4) return 'grid-cols-2 lg:grid-cols-4'
  if (n <= 8) return 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4'
  if (n <= 16) return 'grid-cols-3 md:grid-cols-4 lg:grid-cols-6'
  return 'grid-cols-4 md:grid-cols-6 lg:grid-cols-8'
})

function pickRandomSamples(n: number): SampleOut[] {
  const all = gallery.samples
  if (all.length === 0) return []
  const k = Math.min(n, all.length)
  const indices = all.map((_, i) => i)
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(Math.random() * (indices.length - i))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }
  return indices.slice(0, k).map((i) => all[i])
}

function resamplePreviews() {
  previewSamples.value = pickRandomSamples(nPreview.value)
  ws.clear()
  triggerPreview()
}

function rerollOne(sampleId: string) {
  const all = gallery.samples
  if (all.length === 0) return
  const usedIds = new Set(previewSamples.value.map((s) => s.id))
  const candidates = all.filter((s) => !usedIds.has(s.id))
  if (candidates.length === 0) return
  const replacement = candidates[Math.floor(Math.random() * candidates.length)]
  previewSamples.value = previewSamples.value.map((s) =>
    s.id === sampleId ? replacement : s,
  )
  triggerPreview()
}

let debounceTimer: ReturnType<typeof setTimeout> | null = null

function triggerPreview() {
  if (!sid.value || previewSamples.value.length === 0) return
  if (debounceTimer) clearTimeout(debounceTimer)
  debounceTimer = setTimeout(() => {
    ws.request({
      method: segStore.selectedMethod,
      params: { ...segStore.params },
      sampleIds: previewSamples.value.map((s) => s.id),
      channelIndex: segStore.channelIndex,
    }).catch(() => {
      // surfaced via ws.error
    })
  }, 250)
}

onMounted(async () => {
  if (gallery.samples.length === 0) {
    await gallery.fetchSamples()
  }
  resamplePreviews()
})

watch(
  () => ({ ...segStore.params, method: segStore.selectedMethod, channel: segStore.channelIndex }),
  () => triggerPreview(),
  { deep: true },
)

watch(nPreview, () => resamplePreviews())

function tileFor(sampleId: string) {
  return ws.tiles.value.get(sampleId)
}

function thumbUrl(sampleId: string) {
  return sampleThumbnailUrl(sid.value!, sampleId)
}

const ATTR_DISPLAY: { key: string; label: string; precision: number }[] = [
  { key: 'area', label: 'A', precision: 0 },
  { key: 'form_factor', label: 'C', precision: 2 },
  { key: 'solidity', label: 'S', precision: 2 },
]

function formatAttr(val: number | undefined, precision: number): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '–'
  if (precision === 0) return Math.round(val).toString()
  return val.toFixed(precision)
}
</script>

<template>
  <div class="flex flex-col gap-5">
    <!-- Toolbar -->
    <div class="flex items-center gap-4 flex-wrap">
      <div class="flex items-center gap-2">
        <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Tiles</span>
        <div class="flex items-center gap-1">
          <button
            v-for="count in previewCounts"
            :key="count"
            class="h-7 min-w-[2rem] px-2 rounded-[2px] text-[10px] font-mono font-bold tracking-wider transition-colors"
            :class="nPreview === count
              ? 'bg-neutral text-neutral-content'
              : 'text-neutral/50 hover:bg-neutral/10 hover:text-neutral'"
            @click="nPreview = count"
          >
            {{ count }}
          </button>
        </div>
      </div>

      <div class="flex items-center gap-2">
        <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">View</span>
        <div class="flex items-center gap-1">
          <button
            class="h-7 px-2 rounded-[2px] flex items-center gap-1.5 text-[10px] font-bold tracking-widest uppercase transition-colors"
            :class="viewMode === 'overlay'
              ? 'bg-neutral text-neutral-content'
              : 'text-neutral/50 hover:bg-neutral/10 hover:text-neutral'"
            @click="viewMode = 'overlay'"
          >
            <Layers class="w-3 h-3 stroke-[2px]" />
            Overlay
          </button>
          <button
            class="h-7 px-2 rounded-[2px] flex items-center gap-1.5 text-[10px] font-bold tracking-widest uppercase transition-colors"
            :class="viewMode === 'split'
              ? 'bg-neutral text-neutral-content'
              : 'text-neutral/50 hover:bg-neutral/10 hover:text-neutral'"
            @click="viewMode = 'split'"
          >
            <ImageIcon class="w-3 h-3 stroke-[2px]" />
            Split
          </button>
        </div>
      </div>

      <div class="ml-auto flex items-center gap-3">
        <div
          v-if="ws.inFlight.value"
          class="flex items-center gap-2 text-[10px] font-mono text-neutral/50 tracking-wider"
        >
          <span class="loading loading-spinner loading-xs"></span>
          Streaming…
        </div>
        <button
          class="h-8 px-3 rounded-[2px] flex items-center gap-2 text-[10px] font-bold tracking-widest uppercase transition-colors text-neutral/70 hover:bg-neutral/10 hover:text-neutral"
          @click="resamplePreviews"
        >
          <RefreshCw class="w-3.5 h-3.5 stroke-[2px]" />
          Resample
        </button>
      </div>
    </div>

    <div
      v-if="ws.error.value"
      class="px-3 py-2 bg-error/10 border border-error/30 rounded-[2px] text-[10px] font-mono text-error tracking-wider"
    >
      {{ ws.error.value }}
    </div>

    <!-- Grid -->
    <div v-if="previewSamples.length > 0" class="grid gap-2.5" :class="gridCols">
      <div
        v-for="sample in previewSamples"
        :key="sample.id"
        class="group flex flex-col gap-1 relative"
      >
        <!-- Image area -->
        <div class="relative bg-base-200 border border-base-300 rounded-[2px] overflow-hidden">
          <!-- Split view: original | overlay -->
          <template v-if="viewMode === 'split'">
            <div class="grid grid-cols-2 aspect-square">
              <img
                :src="thumbUrl(sample.id)"
                class="w-full h-full object-contain bg-white border-r border-base-300"
              />
              <img
                v-if="tileFor(sample.id)?.url"
                :src="tileFor(sample.id)!.url!"
                class="w-full h-full object-contain bg-white"
              />
              <div
                v-else
                class="w-full h-full bg-white flex items-center justify-center"
              >
                <span class="loading loading-spinner loading-xs text-neutral/40"></span>
              </div>
            </div>
          </template>

          <!-- Overlay view -->
          <template v-else>
            <div class="aspect-square relative">
              <img
                v-if="tileFor(sample.id)?.url"
                :src="tileFor(sample.id)!.url!"
                class="w-full h-full object-contain bg-white"
              />
              <img
                v-else
                :src="thumbUrl(sample.id)"
                class="w-full h-full object-contain bg-white opacity-60"
              />
              <div
                v-if="!tileFor(sample.id)?.url"
                class="absolute inset-0 flex items-center justify-center"
              >
                <span class="loading loading-spinner loading-xs text-neutral/50"></span>
              </div>
            </div>
          </template>

          <!-- Per-tile re-roll button -->
          <button
            class="absolute top-1 right-1 h-6 w-6 rounded-[2px] bg-neutral/80 text-neutral-content opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center hover:bg-neutral"
            title="Replace with different sample"
            @click="rerollOne(sample.id)"
          >
            <Shuffle class="w-3 h-3 stroke-[2px]" />
          </button>

          <!-- Failure badge -->
          <div
            v-if="tileFor(sample.id) && !tileFor(sample.id)!.ok"
            class="absolute bottom-1 left-1 px-1.5 py-0.5 rounded-[2px] bg-error/80 text-error-content text-[8px] font-bold tracking-widest uppercase"
          >
            Failed
          </div>
        </div>

        <!-- Filename -->
        <p class="text-[9px] font-mono text-neutral/40 truncate px-0.5">
          {{ sample.filename }}
        </p>

        <!-- Attribute footer -->
        <div
          class="flex items-center gap-2 px-0.5 text-[9px] font-mono tracking-wider"
        >
          <template v-for="attr in ATTR_DISPLAY" :key="attr.key">
            <span class="flex items-baseline gap-0.5">
              <span class="text-neutral/40">{{ attr.label }}</span>
              <span class="text-neutral/70 font-bold">
                {{ formatAttr(tileFor(sample.id)?.attrs?.[attr.key], attr.precision) }}
              </span>
            </span>
          </template>
        </div>
      </div>
    </div>

    <div
      v-else
      class="text-[11px] font-mono text-neutral/30 py-12 text-center tracking-wider uppercase"
    >
      No samples available for preview
    </div>
  </div>
</template>
