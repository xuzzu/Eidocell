<script setup lang="ts">
import { computed, ref } from 'vue'
import { useSessionStore } from '@/stores/session'
import { useGalleryStore } from '@/stores/gallery'
import { sampleChannelThumbnailUrl, sampleThumbnailUrl } from '@/api/gallery'
import { maskOverlayUrl } from '@/api/segmentation'
import type { SampleOut } from '@/types'

const props = withDefaults(defineProps<{
  samples: SampleOut[]
  selectedIds: Set<string>
  zoomLevel?: number
  maskView?: boolean
  maskVersion?: number
  inspectMode?: boolean
}>(), {
  zoomLevel: 3,
  maskView: false,
  maskVersion: 0,
  inspectMode: false,
})

const emit = defineEmits<{
  'update:selectedIds': [ids: Set<string>]
  inspect: [sample: SampleOut, channelIndex?: number]
  contextmenu: [event: MouseEvent, sampleId: string]
}>()

const sessionStore = useSessionStore()
const gallery = useGalleryStore()
const sid = computed(() => sessionStore.currentSessionId!)

// Anchor for shift-click range selection: the last id the user clicked
// without shift. Local to the grid so each grid (e.g. each bucket in the
// similarity dialog) tracks its own range.
const anchorId = ref<string | null>(null)

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

// In multi-channel display mode each sample claims a full row so its channels
// are distributed evenly across the entire gallery width (rather than packed
// into a single grid cell).
const useFullWidthRows = computed(() =>
  gallery.channelDisplayMode === 'multi' && gallery.sessionChannelCount > 1,
)

function isSelected(id: string) {
  return props.selectedIds.has(id)
}

function activeChannel(sample: SampleOut): number {
  const n = sample.n_channels ?? 1
  const c = gallery.selectedChannel
  return Math.max(0, Math.min(c, n - 1))
}

function maskChannel(sample: SampleOut): number {
  if (!sample.mask_channels || sample.mask_channels.length === 0) return 0
  const ac = activeChannel(sample)
  if (sample.mask_channels.includes(ac)) return ac
  return sample.mask_channels[0]
}

function singleChannelSrc(sample: SampleOut) {
  if (props.maskView && sample.has_mask) {
    return maskOverlayUrl(sid.value, sample.id, props.maskVersion, maskChannel(sample))
  }
  const n = sample.n_channels ?? 1
  if (n <= 1) return sampleThumbnailUrl(sid.value, sample.id)
  return sampleChannelThumbnailUrl(sid.value, sample.id, activeChannel(sample))
}

function channelStrip(sample: SampleOut): number[] {
  const n = sample.n_channels ?? 1
  const wanted = gallery.selectedChannels.length > 0
    ? gallery.selectedChannels
    : Array.from({ length: n }, (_, i) => i)
  return wanted.filter(c => c >= 0 && c < n)
}

function showChannelStrip(sample: SampleOut): boolean {
  if (props.maskView && sample.has_mask) return false
  if (gallery.channelDisplayMode !== 'multi') return false
  return (sample.n_channels ?? 1) > 1 && channelStrip(sample).length > 0
}

function channelSrc(sample: SampleOut, channel: number) {
  if (props.maskView && sample.mask_channels?.includes(channel)) {
    return maskOverlayUrl(sid.value, sample.id, props.maskVersion, channel)
  }
  return sampleChannelThumbnailUrl(sid.value, sample.id, channel)
}

function onClick(sample: SampleOut, channelIndex?: number) {
  if (props.inspectMode) {
    emit('inspect', sample, channelIndex)
    return
  }
  anchorId.value = sample.id
  emit('update:selectedIds', new Set([sample.id]))
}

function toggle(id: string) {
  const next = new Set(props.selectedIds)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  anchorId.value = id
  emit('update:selectedIds', next)
}

function rangeSelect(targetId: string) {
  if (props.inspectMode) {
    const sample = props.samples.find((s) => s.id === targetId)
    if (sample) emit('inspect', sample)
    return
  }
  const ids = props.samples.map((s) => s.id)
  const targetIdx = ids.indexOf(targetId)
  if (targetIdx === -1) return

  const anchorIdx = anchorId.value ? ids.indexOf(anchorId.value) : -1
  if (anchorIdx === -1) {
    anchorId.value = targetId
    emit('update:selectedIds', new Set([targetId]))
    return
  }

  const lo = Math.min(anchorIdx, targetIdx)
  const hi = Math.max(anchorIdx, targetIdx)
  const next = new Set<string>()
  for (let i = lo; i <= hi; i++) next.add(ids[i])
  emit('update:selectedIds', next)
}
</script>

<template>
  <div class="grid gap-2" :class="useFullWidthRows ? 'grid-cols-1' : colsClass">
    <div
      v-for="sample in samples"
      :key="sample.id"
      :data-sample-id="sample.id"
      class="cursor-pointer rounded-[2px] overflow-hidden transition-all flex flex-col bg-base-100"
      :class="isSelected(sample.id) ? 'ring-2 ring-primary ring-offset-1 ring-offset-base-200' : 'hover:ring-1 hover:ring-neutral/30'"
      @click.exact="onClick(sample)"
      @click.shift.exact="rangeSelect(sample.id)"
      @click.ctrl="toggle(sample.id)"
      @click.meta="toggle(sample.id)"
      @contextmenu.prevent="emit('contextmenu', $event, sample.id)"
    >
      <div
        class="px-1.5 py-0.5 truncate text-[9px] text-white font-mono font-medium leading-tight shrink-0"
        :style="{ backgroundColor: sample.class_color || '#9ca3af' }"
      >
        {{ sample.filename }}
      </div>
      <div
        v-if="showChannelStrip(sample)"
        class="w-full relative bg-base-200 grid gap-px"
        :style="{ gridTemplateColumns: `repeat(${channelStrip(sample).length}, minmax(0, 1fr))` }"
        @click.exact.stop
        @click.shift.exact.stop
      >
        <div
          v-for="c in channelStrip(sample)"
          :key="c"
          class="relative aspect-square bg-base-300 cursor-pointer"
          @click.exact.stop="onClick(sample, c)"
          @click.shift.exact.stop="rangeSelect(sample.id)"
          @click.ctrl.stop="toggle(sample.id)"
          @click.meta.stop="toggle(sample.id)"
        >
          <img
            :src="channelSrc(sample, c)"
            :alt="`${sample.filename} ${gallery.sessionChannelNames[c] ?? 'ch' + (c + 1)}`"
            class="absolute inset-0 w-full h-full object-contain"
            loading="lazy"
          />
          <div class="absolute bottom-0 right-0 px-1 text-[8px] font-mono text-white/80 bg-black/40 leading-none py-[1px]">
            {{ gallery.sessionChannelNames[c] ?? c + 1 }}
          </div>
        </div>
        <slot name="badge" :sample="sample" />
      </div>
      <div v-else class="w-full aspect-square relative bg-base-200 flex items-center justify-center">
        <img
          :src="singleChannelSrc(sample)"
          :alt="sample.filename"
          class="absolute inset-0 w-full h-full object-contain"
          loading="lazy"
        />
        <slot name="badge" :sample="sample" />
      </div>
    </div>
  </div>
</template>
