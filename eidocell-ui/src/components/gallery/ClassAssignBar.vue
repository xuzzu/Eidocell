<script setup lang="ts">
import { ref } from 'vue'
import { Sparkles, Tag, X } from 'lucide-vue-next'
import { useGalleryStore } from '@/stores/gallery'

const gallery = useGalleryStore()
const selectedClassId = ref<string | null>(null)

async function assignToClass() {
  await gallery.assignSelectedToClass(selectedClassId.value)
  selectedClassId.value = null
}

function findSimilar() {
  gallery.openSimilarityDialog?.(Array.from(gallery.selectedIds))
}
</script>

<template>
  <div
    v-if="gallery.selectedIds.size > 0"
    class="flex items-center gap-3 px-4 py-2 bg-neutral text-neutral-content border-b border-neutral"
  >
    <Tag class="w-[18px] h-[18px]" />
    <span class="text-xs font-mono tracking-widest uppercase">{{ gallery.selectedIds.size }} selected</span>

    <select v-model="selectedClassId" class="select select-sm rounded-[2px] ml-4 text-xs font-mono text-neutral bg-base-100 outline-none focus:ring-1 focus:ring-neutral">
      <option :value="null">ASSIGN TO CLASS...</option>
      <option v-for="cls in gallery.classes" :key="cls.id" :value="cls.id">
        {{ cls.name }}
      </option>
    </select>

    <button
      class="h-8 px-4 rounded-[2px] text-[11px] font-bold tracking-widest uppercase transition-colors duration-200"
      :class="selectedClassId ? 'bg-secondary text-white hover:bg-secondary/90' : 'bg-base-100/20 text-neutral-content/50 cursor-not-allowed'"
      :disabled="!selectedClassId"
      @click="assignToClass"
    >
      Assign
    </button>

    <button
      class="flex items-center gap-1.5 h-8 px-4 rounded-[2px] text-[11px] font-bold tracking-widest uppercase transition-colors bg-base-100/10 hover:bg-base-100/20"
      @click="findSimilar"
    >
      <Sparkles class="w-3.5 h-3.5" />
      Find Similar
    </button>

    <button class="flex items-center gap-1.5 h-8 px-3 ml-auto rounded-[2px] text-[10px] font-bold tracking-widest uppercase transition-colors hover:bg-base-100/20" @click="gallery.clearSelection()">
      <X class="w-3.5 h-3.5" />
      Clear
    </button>
  </div>
</template>
