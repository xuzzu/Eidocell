<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Play, RotateCcw } from 'lucide-vue-next'
import { useSegmentationStore } from '@/stores/segmentation'
import { useGalleryStore } from '@/stores/gallery'
import MethodSelector from '@/components/segmentation/MethodSelector.vue'
import SegmentationPreview from '@/components/segmentation/SegmentationPreview.vue'
import TaskProgressBar from '@/components/common/TaskProgressBar.vue'

const segStore = useSegmentationStore()
const gallery = useGalleryStore()
const resultMessage = ref('')

onMounted(() => {
  segStore.fetchMethods()
})

async function run() {
  resultMessage.value = ''
  await segStore.runSegmentation()
}

function onComplete() {
  segStore.onTaskComplete()
  resultMessage.value = 'Segmentation complete!'
  gallery.fetchSamples()
}
</script>

<template>
  <div class="flex h-full relative overflow-hidden bg-base-200">
    <!-- Left panel: controls -->
    <div class="w-80 min-w-[320px] bg-base-100 border-r border-base-300 p-6 flex flex-col gap-6 overflow-y-auto z-10 shadow-sm shrink-0">
      <h2 class="text-[14px] font-bold tracking-widest uppercase">Segmentation</h2>

      <MethodSelector />

      <div class="space-y-3 pt-2">
        <button
          class="h-10 w-full rounded-[2px] flex items-center justify-center gap-2 text-[11px] font-bold tracking-widest uppercase transition-opacity"
          :class="segStore.loading ? 'bg-base-200 text-neutral/40' : 'bg-neutral text-neutral-content hover:opacity-80'"
          :disabled="segStore.loading"
          @click="run"
        >
          <span v-if="segStore.loading" class="loading loading-spinner loading-sm"></span>
          <Play v-else class="w-4 h-4 stroke-[2px]" />
          Run Segmentation
        </button>

        <button
          class="h-8 w-full rounded-[2px] flex items-center justify-center gap-2 text-[10px] font-bold tracking-widest uppercase transition-colors"
          :class="segStore.loading ? 'text-neutral/30' : 'text-neutral/70 hover:bg-neutral/10 hover:text-neutral'"
          :disabled="segStore.loading"
          @click="segStore.fetchMethods()"
        >
          <RotateCcw class="w-3 h-3 stroke-[2px]" />
          Reset Defaults
        </button>
      </div>

      <TaskProgressBar :task-id="segStore.taskId" @complete="onComplete" />

      <div
        v-if="resultMessage"
        class="p-3 bg-success/10 border border-success/30 rounded-[2px] text-[11px] font-mono font-bold text-success tracking-wider"
      >
        {{ resultMessage }}
      </div>
    </div>

    <!-- Right panel: preview -->
    <div class="flex-1 p-6 overflow-y-auto">
      <SegmentationPreview />
    </div>
  </div>
</template>
