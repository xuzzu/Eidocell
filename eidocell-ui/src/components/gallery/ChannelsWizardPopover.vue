<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { X } from 'lucide-vue-next'
import { useGalleryStore, type ChannelDisplayMode } from '@/stores/gallery'

const props = defineProps<{
  x: number
  y: number
}>()

const emit = defineEmits<{ close: [] }>()

const gallery = useGalleryStore()
const root = ref<HTMLElement>()

const channelCount = computed(() => gallery.sessionChannelCount)
const channelNames = computed(() => gallery.sessionChannelNames)
const multiAllowed = computed(() => channelCount.value > 1)

const style = computed(() => {
  const w = 240
  const headerEstimate = 84
  const rowH = 26
  const h = Math.min(window.innerHeight - 16, headerEstimate + rowH * channelCount.value + 32)
  const x = Math.max(8, Math.min(props.x, window.innerWidth - w - 8))
  const y = Math.max(8, Math.min(props.y, window.innerHeight - h - 8))
  return { left: `${x}px`, top: `${y}px`, width: `${w}px` }
})

function setMode(mode: ChannelDisplayMode) {
  if (mode === 'multi' && !multiAllowed.value) return
  gallery.setChannelDisplayMode(mode)
  if (mode === 'multi' && gallery.selectedChannels.length === 0) {
    gallery.setSelectedChannels(Array.from({ length: channelCount.value }, (_, i) => i))
  }
}

function pickSingle(idx: number) {
  gallery.setSelectedChannel(idx)
}

function toggleChannel(idx: number) {
  const current = new Set(gallery.selectedChannels)
  if (current.has(idx)) {
    if (current.size <= 1) return  // never leave empty
    current.delete(idx)
  } else {
    current.add(idx)
  }
  gallery.setSelectedChannels(Array.from(current))
}

function isMultiSelected(idx: number) {
  return gallery.selectedChannels.includes(idx)
}

function onClickOutside(e: MouseEvent) {
  if (root.value && !root.value.contains(e.target as Node)) {
    emit('close')
  }
}

onMounted(() => {
  setTimeout(() => document.addEventListener('mousedown', onClickOutside), 0)
})
onUnmounted(() => document.removeEventListener('mousedown', onClickOutside))
</script>

<template>
  <div
    ref="root"
    class="fixed z-[600] bg-base-100 border border-base-300 shadow-xl rounded-[2px] p-3"
    :style="style"
    @mousedown.stop
  >
    <div class="flex items-center justify-between mb-3">
      <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">Channels</span>
      <button
        class="w-5 h-5 flex items-center justify-center rounded-[2px] text-neutral/30 hover:bg-neutral/10 hover:text-neutral transition-colors"
        @click="emit('close')"
      ><X class="w-3 h-3 stroke-[2px]" /></button>
    </div>

    <!-- Mode toggle -->
    <div class="flex gap-1 mb-3">
      <button
        class="flex-1 h-7 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
        :class="gallery.channelDisplayMode === 'single'
          ? 'bg-neutral text-neutral-content'
          : 'bg-base-200 hover:bg-neutral/10 text-neutral/70'"
        @click="setMode('single')"
      >Single</button>
      <button
        :disabled="!multiAllowed"
        class="flex-1 h-7 text-[10px] font-bold tracking-widest uppercase rounded-[2px] transition-colors"
        :class="!multiAllowed
          ? 'bg-base-200 text-neutral/30 cursor-not-allowed'
          : gallery.channelDisplayMode === 'multi'
            ? 'bg-neutral text-neutral-content'
            : 'bg-base-200 hover:bg-neutral/10 text-neutral/70'"
        @click="setMode('multi')"
        :title="multiAllowed ? 'Show selected channels side-by-side' : 'Disabled for single-channel data'"
      >Multi</button>
    </div>

    <!-- Single mode: radio list -->
    <div v-if="gallery.channelDisplayMode === 'single'" class="space-y-1">
      <button
        v-for="(name, idx) in channelNames"
        :key="idx"
        class="w-full h-7 px-2 flex items-center gap-2 text-[11px] font-mono rounded-[2px] transition-colors"
        :class="gallery.selectedChannel === idx
          ? 'bg-neutral text-neutral-content'
          : 'bg-base-200 hover:bg-neutral/10 text-neutral'"
        @click="pickSingle(idx)"
      >
        <span class="w-3 h-3 rounded-full border border-current flex items-center justify-center">
          <span v-if="gallery.selectedChannel === idx" class="w-1.5 h-1.5 rounded-full bg-current" />
        </span>
        <span class="truncate text-left flex-1">{{ name }}</span>
        <span class="text-[9px] opacity-60">ch{{ idx }}</span>
      </button>
    </div>

    <!-- Multi mode: checklist -->
    <div v-else class="space-y-1">
      <button
        v-for="(name, idx) in channelNames"
        :key="idx"
        class="w-full h-7 px-2 flex items-center gap-2 text-[11px] font-mono rounded-[2px] transition-colors"
        :class="isMultiSelected(idx)
          ? 'bg-neutral text-neutral-content'
          : 'bg-base-200 hover:bg-neutral/10 text-neutral'"
        @click="toggleChannel(idx)"
      >
        <span
          class="w-3 h-3 border border-current flex items-center justify-center rounded-[1px]"
        >
          <span v-if="isMultiSelected(idx)" class="text-[8px] font-bold leading-none">×</span>
        </span>
        <span class="truncate text-left flex-1">{{ name }}</span>
        <span class="text-[9px] opacity-60">ch{{ idx }}</span>
      </button>
      <p class="text-[9px] font-mono text-neutral/50 pt-1">At least one channel must remain selected.</p>
    </div>
  </div>
</template>
