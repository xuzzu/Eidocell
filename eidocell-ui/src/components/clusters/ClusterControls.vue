<script setup lang="ts">
import { ref, computed } from 'vue'
import { Scissors, Merge, Tag, X } from 'lucide-vue-next'
import { useClustersStore } from '@/stores/clusters'
import { useGalleryStore } from '@/stores/gallery'

const clusters = useClustersStore()
const gallery = useGalleryStore()

const selectedArray = computed(() => Array.from(clusters.selectedClusterIds))
const splitCount = ref(2)
const assignClassId = ref<string | null>(null)

async function onSplit() {
  if (selectedArray.value.length === 1) {
    await clusters.splitCluster(selectedArray.value[0], splitCount.value)
  }
}

async function onMerge() {
  if (selectedArray.value.length >= 2) {
    await clusters.mergeClusters(selectedArray.value)
  }
}

async function onAssign() {
  if (selectedArray.value.length > 0 && assignClassId.value) {
    await clusters.assignToClass(selectedArray.value, assignClassId.value)
    assignClassId.value = null
    gallery.fetchClasses()
  }
}

function onClear() {
  clusters.selectedClusterIds.clear()
}
</script>

<template>
  <div
    v-if="clusters.selectedClusterIds.size > 0"
    class="flex items-center gap-3 px-4 py-2 bg-neutral text-neutral-content rounded-[2px] flex-wrap"
  >
    <span class="text-xs font-mono tracking-widest uppercase shrink-0">
      {{ clusters.selectedClusterIds.size }} SELECTED
    </span>

    <!-- Split (single) -->
    <div v-if="selectedArray.length === 1" class="flex items-center gap-2 ml-4">
      <input
        v-model.number="splitCount"
        type="number"
        min="2"
        max="20"
        class="w-14 h-7 px-2 rounded-[2px] text-xs font-mono text-neutral bg-base-100 outline-none text-center"
      />
      <button
        class="h-8 px-3 rounded-[2px] text-[10px] font-bold tracking-widest uppercase bg-base-100/20 hover:bg-base-100/30 transition-colors flex items-center gap-2"
        @click="onSplit"
      >
        <Scissors class="w-3 h-3 stroke-[2px]" /> SPLIT
      </button>
    </div>

    <!-- Merge (2+) -->
    <button
      v-if="selectedArray.length >= 2"
      class="h-8 px-3 rounded-[2px] text-[10px] font-bold tracking-widest uppercase bg-base-100/20 hover:bg-base-100/30 transition-colors flex items-center gap-2 ml-4"
      @click="onMerge"
    >
      <Merge class="w-3 h-3 stroke-[2px]" /> MERGE
    </button>

    <!-- Assign to class -->
    <div class="flex items-center gap-2 ml-4">
      <select
        v-model="assignClassId"
        class="select select-sm rounded-[2px] text-xs font-mono text-neutral bg-base-100 outline-none"
      >
        <option :value="null">ASSIGN TO CLASS...</option>
        <option v-for="cls in gallery.classes" :key="cls.id" :value="cls.id">{{ cls.name }}</option>
      </select>
      <button
        class="h-8 px-3 rounded-[2px] text-[10px] font-bold tracking-widest uppercase flex items-center gap-2 transition-colors"
        :class="assignClassId ? 'bg-base-100/20 hover:bg-base-100/30' : 'bg-base-100/10 text-neutral-content/40 cursor-not-allowed'"
        :disabled="!assignClassId"
        @click="onAssign"
      >
        <Tag class="w-3 h-3 stroke-[2px]" /> ASSIGN
      </button>
    </div>

    <!-- Clear -->
    <button
      class="ml-auto h-8 px-3 rounded-[2px] text-[10px] font-bold tracking-widest uppercase hover:bg-base-100/20 transition-colors flex items-center gap-1.5"
      @click="onClear"
    >
      <X class="w-3.5 h-3.5" /> CLEAR
    </button>
  </div>
</template>
