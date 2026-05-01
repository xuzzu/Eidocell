<script setup lang="ts">
import { computed, ref } from 'vue'
import { Users, Eye, Scissors } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useClustersStore } from '@/stores/clusters'
import { clusterPreviewUrl } from '@/api/clusters'
import type { ClusterOut } from '@/types'

const props = defineProps<{
  cluster: ClusterOut
  isSingleSelected: boolean
  isDragTarget: boolean
  isDragging: boolean
}>()

const emit = defineEmits<{
  preview: [id: string]
  split: [id: string, n: number]
  contextmenu: [e: MouseEvent, id: string]
  dragstart: [id: string]
  dragend: []
  dragenter: [id: string]
  dragleave: [id: string]
  drop: [targetId: string]
}>()

const sessionStore = useSessionStore()
const clustersStore = useClustersStore()

const previewSrc = computed(() =>
  clusterPreviewUrl(sessionStore.currentSessionId!, props.cluster.id)
)

const isSelected = computed(() => clustersStore.selectedClusterIds.has(props.cluster.id))

const showSplitPopup = ref(false)
const splitCount = ref(2)

function onSplitConfirm() {
  emit('split', props.cluster.id, splitCount.value)
  showSplitPopup.value = false
}
</script>

<template>
  <div
    :data-cluster-id="cluster.id"
    draggable="true"
    class="bg-base-100 shadow-sm border rounded-[2px] transition-all cursor-pointer group flex flex-col relative"
    :class="[
      isDragging ? 'opacity-40' : '',
      isDragTarget ? 'ring-2 ring-success border-success scale-[1.02]' : (isSelected ? 'border-primary ring-1 ring-primary' : 'border-base-300 hover:border-neutral'),
    ]"
    @click.exact="clustersStore.selectSingle(cluster.id)"
    @click.ctrl="clustersStore.toggleSelection(cluster.id)"
    @click.meta="clustersStore.toggleSelection(cluster.id)"
    @dblclick.stop="emit('preview', cluster.id)"
    @contextmenu.prevent="emit('contextmenu', $event, cluster.id)"
    @dragstart.stop="emit('dragstart', cluster.id)"
    @dragend.stop="emit('dragend')"
    @dragenter.prevent="emit('dragenter', cluster.id)"
    @dragleave="emit('dragleave', cluster.id)"
    @dragover.prevent
    @drop.prevent.stop="emit('drop', cluster.id)"
  >
    <!-- Drag-target overlay -->
    <div
      v-if="isDragTarget"
      class="absolute inset-0 z-10 flex items-center justify-center pointer-events-none rounded-[2px] bg-success/10"
    >
      <span class="text-[9px] font-bold tracking-widest uppercase text-success bg-base-100/90 px-2 py-1 rounded-[2px]">
        DROP TO MERGE
      </span>
    </div>

    <figure class="p-2 pb-0 relative">
      <img :src="previewSrc" class="rounded-[2px] w-full aspect-square object-cover border border-base-200" />
    </figure>

    <div class="px-3 py-3 mt-auto relative">
      <!-- Split popup (above footer) -->
      <div
        v-if="showSplitPopup"
        class="absolute bottom-full left-0 right-0 mb-1 z-30 bg-base-100 border border-neutral rounded-[2px] shadow-lg p-3 flex flex-col gap-3"
        @click.stop
      >
        <div class="flex items-center justify-between">
          <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Split into</span>
          <span class="text-[10px] font-mono font-bold">{{ splitCount }}</span>
        </div>
        <input
          v-model.number="splitCount"
          type="range"
          min="2"
          max="20"
          class="range range-xs range-neutral transition-none"
        />
        <button
          class="h-8 w-full rounded-[2px] bg-neutral text-neutral-content text-[10px] font-bold tracking-widest uppercase hover:opacity-80 transition-opacity"
          @click.stop="onSplitConfirm"
        >
          CONFIRM SPLIT
        </button>
      </div>

      <div class="flex items-center gap-2">
        <span class="w-3 h-3 rounded-[2px]" :style="{ backgroundColor: cluster.color }"></span>
        <span class="flex items-center gap-1.5 text-[10px] font-mono font-bold tracking-widest text-neutral/70">
          <Users class="w-3.5 h-3.5 stroke-[2px]" />
          {{ cluster.sample_count }}
        </span>

        <!-- Scissors (split) — only when single-selected -->
        <button
          v-if="isSingleSelected"
          class="w-7 h-7 flex items-center justify-center rounded-[2px] ml-auto transition-colors"
          :class="showSplitPopup ? 'bg-neutral text-neutral-content' : 'text-neutral/40 hover:bg-neutral/10 hover:text-neutral'"
          @click.stop="showSplitPopup = !showSplitPopup"
        >
          <Scissors class="w-3.5 h-3.5 stroke-[2px]" />
        </button>

        <!-- Eye (view samples) -->
        <button
          class="w-7 h-7 flex items-center justify-center rounded-[2px] transition-colors text-neutral/40 hover:bg-neutral/10 hover:text-neutral"
          :class="isSingleSelected ? '' : 'ml-auto'"
          @click.stop="emit('preview', cluster.id)"
        >
          <Eye class="w-3.5 h-3.5 stroke-[2px]" />
        </button>
      </div>
    </div>
  </div>
</template>
