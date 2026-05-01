<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount, nextTick, computed } from 'vue'
import { RefreshCw } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useSegmentationStore } from '@/stores/segmentation'
import { useGalleryStore } from '@/stores/gallery'
import { maskOverlayUrl, runSegmentationPreview } from '@/api/segmentation'
import { sampleThumbnailUrl } from '@/api/gallery'

const sessionStore = useSessionStore()
const segStore = useSegmentationStore()
const gallery = useGalleryStore()

const sid = computed(() => sessionStore.currentSessionId!)

const previewCounts = [1, 2, 4, 8, 16] as const
const nPreview = ref(4)
const previewSamples = ref<typeof gallery.samples>([])
const previewLoading = ref(false)
const maskVersion = ref(0)
const previewReady = ref(false) // true once current samples have been segmented

const gridCols = computed(() => {
  const n = nPreview.value
  if (n <= 1) return 'grid-cols-1 max-w-xs'
  if (n <= 2) return 'grid-cols-2 max-w-lg'
  if (n <= 4) return 'grid-cols-2 max-w-lg'
  if (n <= 8) return 'grid-cols-4'
  return 'grid-cols-4'
})

onMounted(async () => {
  if (gallery.samples.length === 0) {
    await gallery.fetchSamples()
  }
  resampleAndRun()
})

function resamplePreviews() {
  const all = gallery.samples
  if (all.length === 0) {
    previewSamples.value = []
    return
  }
  const n = Math.min(nPreview.value, all.length)
  const indices = all.map((_, i) => i)
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (indices.length - i))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }
  // Store objects directly — immune to gallery.samples changing after fetchSamples
  previewSamples.value = indices.slice(0, n).map(i => all[i])
}

function resampleAndRun() {
  resamplePreviews()
  previewReady.value = false // show original images until segmentation finishes
  nextTick(() => runPreview())
}

let debounceTimer: ReturnType<typeof setTimeout> | null = null

async function runPreview() {
  if (!sid.value || previewSamples.value.length === 0) return
  // Cancel any pending debounced call
  if (debounceTimer) { clearTimeout(debounceTimer); debounceTimer = null }
  previewLoading.value = true
  try {
    await runSegmentationPreview(sid.value, {
      method: segStore.selectedMethod,
      params: { ...segStore.params },
      sample_ids: previewSamples.value.map(s => s.id),
    })
    maskVersion.value++
    previewReady.value = true
    await gallery.fetchSamples()
  } catch {
    // silently fail for preview
  } finally {
    previewLoading.value = false
  }
}

function debouncedPreview() {
  if (debounceTimer) clearTimeout(debounceTimer)
  debounceTimer = setTimeout(runPreview, 300)
}

onBeforeUnmount(() => {
  if (debounceTimer) {
    clearTimeout(debounceTimer)
    debounceTimer = null
  }
})

// Watch params + method changes to trigger debounced preview
watch(
  () => ({ ...segStore.params, method: segStore.selectedMethod }),
  () => {
    if (previewSamples.value.length > 0) {
      debouncedPreview()
    }
  },
  { deep: true },
)

// Watch nPreview to resample + run when count changes
watch(nPreview, () => {
  resampleAndRun()
})

function imageUrl(sampleId: string) {
  if (previewReady.value) {
    return maskOverlayUrl(sid.value, sampleId) + `?v=${maskVersion.value}`
  }
  return sampleThumbnailUrl(sid.value, sampleId)
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <!-- Preview controls -->
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2">
        <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Previews</span>
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
      <button
        class="h-8 px-3 rounded-[2px] flex items-center gap-2 text-[10px] font-bold tracking-widest uppercase transition-colors ml-auto"
        :class="previewLoading ? 'text-neutral/30' : 'text-neutral/70 hover:bg-neutral/10 hover:text-neutral'"
        :disabled="previewLoading"
        @click="resampleAndRun"
      >
        <RefreshCw class="w-3.5 h-3.5 stroke-[2px]" :class="previewLoading ? 'animate-spin' : ''" />
        Resample
      </button>
    </div>

    <!-- Loading indicator -->
    <div v-if="previewLoading" class="flex items-center gap-2 text-[10px] font-mono text-neutral/40 tracking-wider">
      <span class="loading loading-spinner loading-xs"></span>
      Segmenting previews...
    </div>

    <!-- Preview grid -->
    <div v-if="previewSamples.length > 0" class="grid gap-3" :class="gridCols">
      <div
        v-for="sample in previewSamples"
        :key="sample.id"
        class="flex flex-col gap-1"
      >
        <div class="bg-base-200 border border-base-300 rounded-[2px] overflow-hidden">
          <img
            :src="imageUrl(sample.id)"
            class="w-full aspect-square object-contain bg-white"
          />
        </div>
        <p class="text-[9px] font-mono text-neutral/40 truncate px-0.5">{{ sample.filename }}</p>
      </div>
    </div>

    <div v-else class="text-[11px] font-mono text-neutral/30 py-12 text-center tracking-wider uppercase">
      No samples available for preview
    </div>
  </div>
</template>
