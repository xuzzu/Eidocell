<script setup lang="ts">
import { ref, watch, computed, onMounted, onUnmounted } from 'vue'
import { X, Eye, EyeOff } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useGalleryStore } from '@/stores/gallery'
import { sampleImageUrl } from '@/api/gallery'
import { maskOverlayUrl, getMaskAttributes } from '@/api/segmentation'

const sessionStore = useSessionStore()
const gallery = useGalleryStore()

const sid = computed(() => sessionStore.currentSessionId!)
const sample = computed(() => gallery.detailSample)
const showOverlay = ref(false)
const attributes = ref<Record<string, number> | null>(null)
const panelRef = ref<HTMLElement>()

const isMulti = computed(() => (sample.value?.n_channels ?? 1) > 1)
const channelHasMask = computed(() =>
  !!sample.value?.mask_channels?.includes(gallery.detailChannel),
)

async function loadAttrs() {
  attributes.value = null
  if (!sample.value || !channelHasMask.value) return
  try {
    attributes.value = await getMaskAttributes(sid.value, sample.value.id, gallery.detailChannel)
  } catch { /* no mask */ }
}

watch(sample, () => {
  showOverlay.value = false
  loadAttrs()
})

watch(() => gallery.detailChannel, () => {
  showOverlay.value = false
  loadAttrs()
})

function close() {
  gallery.detailSample = null
}

function onClickOutside(e: MouseEvent) {
  if (sample.value && panelRef.value && !panelRef.value.contains(e.target as Node)) {
    close()
  }
}

onMounted(() => document.addEventListener('mousedown', onClickOutside))
onUnmounted(() => document.removeEventListener('mousedown', onClickOutside))

const imageSource = computed(() => {
  if (!sample.value) return ''
  if (showOverlay.value && channelHasMask.value) {
    return maskOverlayUrl(sid.value, sample.value.id, gallery.maskVersion, gallery.detailChannel)
  }
  if (isMulti.value) {
    return sampleImageUrl(sid.value, sample.value.id, gallery.detailChannel)
  }
  return sampleImageUrl(sid.value, sample.value.id)
})

const attrEntries = computed(() => {
  if (!attributes.value) return []
  return Object.entries(attributes.value)
    .filter(([_, v]) => typeof v === 'number')
    .map(([k, v]) => [k, typeof v === 'number' ? v.toFixed(2) : v])
})
</script>

<template>
  <Transition name="slide">
    <div v-if="sample" ref="panelRef" class="fixed top-0 right-0 bottom-0 z-[100] w-96 shadow-2xl border-l-[2px] border-neutral bg-base-100 flex flex-col overflow-hidden">
      <!-- Header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-base-300 bg-neutral text-neutral-content shrink-0">
        <h3 class="font-mono text-sm font-bold truncate tracking-widest">{{ sample.filename }}</h3>
        <button class="flex items-center justify-center w-8 h-8 rounded-[2px] hover:bg-neutral-content/20 transition-colors" @click="close">
          <X class="w-5 h-5" />
        </button>
      </div>

      <div class="px-6 py-6 flex-1 overflow-y-auto space-y-8 text-neutral">
        <!-- Channel picker (multi-channel sessions only) -->
        <div v-if="isMulti" class="space-y-2">
          <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/60">Channel</span>
          <div class="flex flex-wrap gap-1">
            <button
              v-for="(name, idx) in gallery.sessionChannelNames.slice(0, sample.n_channels)"
              :key="idx"
              class="h-7 px-2 text-[10px] font-mono rounded-[2px] transition-colors flex items-center gap-1.5"
              :class="gallery.detailChannel === idx
                ? 'bg-neutral text-neutral-content'
                : 'bg-base-200 hover:bg-neutral/10 text-neutral/80'"
              @click="gallery.detailChannel = idx"
            >
              <span>{{ name }}</span>
              <span v-if="sample.mask_channels?.includes(idx)" class="w-1.5 h-1.5 rounded-full bg-success" title="Has mask" />
            </button>
          </div>
        </div>

        <!-- Image display -->
        <div class="relative bg-base-200 p-2 rounded-[2px] border border-base-300">
          <img :src="imageSource" class="w-full aspect-square object-contain bg-white" />
          <button
            v-if="channelHasMask"
            class="h-8 px-3 flex items-center gap-2 absolute top-4 right-4 bg-neutral text-neutral-content rounded-[2px] text-[10px] font-bold tracking-widest uppercase transition-opacity hover:opacity-80 shadow-md"
            @click="showOverlay = !showOverlay"
          >
            <component :is="showOverlay ? EyeOff : Eye" class="w-3.5 h-3.5" />
            {{ showOverlay ? 'Hide' : 'Show' }} Mask
          </button>
        </div>

        <!-- Metadata -->
        <div class="space-y-3 font-mono text-xs">
          <div class="flex items-center justify-between pb-2 border-b border-base-300/50">
            <span class="text-neutral/50 tracking-wider">CLASS</span>
            <span class="flex items-center gap-2 font-bold">
              <span v-if="sample.class_color" class="w-3 h-3 rounded-[2px] border border-neutral/20" :style="{ backgroundColor: sample.class_color }"></span>
              {{ sample.class_name ?? 'UNCATEGORIZED' }}
            </span>
          </div>
          <div class="flex items-center justify-between pb-2 border-b border-base-300/50">
            <span class="text-neutral/50 tracking-wider">MASK AVAILABLE</span>
            <span class="font-bold">{{ sample.has_mask ? 'YES' : 'NO' }}</span>
          </div>
        </div>

        <!-- Mask Attributes -->
        <div v-if="attrEntries.length > 0" class="pt-2">
          <h4 class="font-bold text-[11px] text-neutral uppercase tracking-widest mb-4">Mask Attributes</h4>
          <div class="border border-base-300 rounded-[2px] overflow-hidden">
            <table class="table table-xs w-full">
              <tbody class="font-mono text-[11px]">
                <tr v-for="[key, val] in attrEntries" :key="key" class="border-b border-base-300 last:border-none">
                  <td class="text-neutral/60 py-2 pl-3 tracking-wider">{{ key }}</td>
                  <td class="text-right font-bold py-2 pr-3">{{ val }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.slide-enter-active, .slide-leave-active {
  transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.slide-enter-from, .slide-leave-to {
  transform: translateX(100%);
}
</style>

