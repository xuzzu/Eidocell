<script setup lang="ts">
import { ArrowUp, ArrowDown, Search, SearchCheck, EyeOff, Eye } from 'lucide-vue-next'
import { useGalleryStore } from '@/stores/gallery'

const gallery = useGalleryStore()

const props = defineProps<{
  zoomLevel: number
  maskView: boolean
  inspectMode: boolean
}>()

const emit = defineEmits<{
  'update:zoomLevel': [value: number]
  'update:maskView': [value: boolean]
  'update:inspectMode': [value: boolean]
}>()

function setSortOrder(order: 'asc' | 'desc') {
  gallery.setSort(gallery.sortBy, order)
}

function onSortChange(event: Event) {
  gallery.setSort((event.target as HTMLSelectElement).value, gallery.sortOrder)
}
</script>

<template>
  <div class="flex items-center gap-4 px-6 py-3 border-b border-base-300 bg-base-100">
    <!-- Image Scale -->
    <span class="text-xs font-bold uppercase tracking-widest text-neutral shrink-0">Scale</span>
    <input
      type="range"
      min="1"
      max="5"
      step="1"
      :value="props.zoomLevel"
      class="range range-xs range-neutral w-50 shrink-0 transition-none"
      @input="emit('update:zoomLevel', 6 - Number(($event.target as HTMLInputElement).value))"
    />

    <div class="h-6 w-px bg-base-300 mx-2"></div>

    <!-- Sort direction arrows -->
    <div class="flex items-center gap-1 shrink-0">
      <button
        class="h-8 w-8 flex items-center justify-center rounded-[2px] bg-neutral text-neutral-content transition-opacity duration-200"
        :class="gallery.sortOrder === 'asc' ? 'opacity-100' : 'opacity-40 hover:opacity-100'"
        @click="setSortOrder('asc')"
      >
        <ArrowUp class="w-[18px] h-[18px]" />
      </button>
      <button
        class="h-8 w-8 flex items-center justify-center rounded-[2px] bg-neutral text-neutral-content transition-opacity duration-200"
        :class="gallery.sortOrder === 'desc' ? 'opacity-100' : 'opacity-40 hover:opacity-100'"
        @click="setSortOrder('desc')"
      >
        <ArrowDown class="w-[18px] h-[18px]" />
      </button>
    </div>

    <!-- Sort-by dropdown -->
    <select
      class="select select-bordered select-sm rounded-[2px] outline-none focus:ring-1 focus:ring-neutral text-xs font-mono w-40 shrink-0"
      :value="gallery.sortBy"
      @change="onSortChange"
    >
      <option value="filename">Filename</option>
      <optgroup v-if="gallery.sortableAttributes.length > 0" label="Attributes">
        <option
          v-for="attr in gallery.sortableAttributes"
          :key="attr"
          :value="'attr:' + attr"
        >
          {{ attr }}
        </option>
      </optgroup>
    </select>

    <div class="flex-1" />

    <!-- INSPECT toggle -->
    <button
      class="h-8 px-4 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity duration-200"
      :class="props.inspectMode ? 'opacity-100' : 'opacity-50 hover:opacity-80'"
      @click="emit('update:inspectMode', !props.inspectMode)"
    >
      <component :is="props.inspectMode ? SearchCheck : Search" class="w-4 h-4" />
      INSPECT
    </button>

    <!-- MASK VIEW toggle -->
    <button
      class="h-8 px-4 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity duration-200"
      :class="props.maskView ? 'opacity-100' : 'opacity-50 hover:opacity-80'"
      @click="emit('update:maskView', !props.maskView)"
    >
      <component :is="props.maskView ? Eye : EyeOff" class="w-4 h-4" />
      MASK VIEW
    </button>
  </div>
</template>
