<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Play, RotateCcw, Lock } from 'lucide-vue-next'
import { useSegmentationStore } from '@/stores/segmentation'
import { useGalleryStore } from '@/stores/gallery'
import MethodSelector from '@/components/segmentation/MethodSelector.vue'
import SegmentationPreview from '@/components/segmentation/SegmentationPreview.vue'
import TaskProgressBar from '@/components/common/TaskProgressBar.vue'

type Tab = 'general' | 'advanced'

const segStore = useSegmentationStore()
const gallery = useGalleryStore()
const resultMessage = ref('')
const activeTab = ref<Tab>('general')

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
  // Disk overlays at /mask/overlay are now stale in the browser cache —
  // bump the version so gallery URLs change and force fresh loads.
  gallery.bumpMaskVersion()
  gallery.fetchSamples()
}
</script>

<template>
  <div class="flex flex-col h-full bg-base-200">
    <!-- Tab header -->
    <div class="flex items-center gap-1 px-6 pt-4 border-b border-base-300 bg-base-100 shrink-0">
      <button
        class="h-9 px-4 text-[10px] font-bold tracking-widest uppercase border-b-2 transition-colors"
        :class="activeTab === 'general'
          ? 'border-neutral text-neutral'
          : 'border-transparent text-neutral/40 hover:text-neutral/70'"
        @click="activeTab = 'general'"
      >
        General
      </button>
      <button
        class="h-9 px-4 text-[10px] font-bold tracking-widest uppercase border-b-2 transition-colors flex items-center gap-1.5 cursor-not-allowed"
        :class="'border-transparent text-neutral/25'"
        disabled
        title="Coming soon"
      >
        <Lock class="w-3 h-3 stroke-[2px]" />
        Advanced
      </button>
    </div>

    <!-- General tab -->
    <div v-if="activeTab === 'general'" class="flex flex-1 relative overflow-hidden">
      <!-- Left panel: controls -->
      <div class="w-80 min-w-[320px] bg-base-100 border-r border-base-300 p-6 flex flex-col gap-6 overflow-y-auto z-10 shadow-sm shrink-0">
        <h2 class="text-[14px] font-bold tracking-widest uppercase">Segmentation</h2>

        <div v-if="gallery.sessionChannelCount > 1">
          <label class="text-[10px] font-bold tracking-widest uppercase text-neutral/70 block mb-2">Channel</label>
          <select
            class="select select-bordered select-sm rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
            :value="segStore.channelIndex"
            @change="segStore.setChannelIndex(Number(($event.target as HTMLSelectElement).value))"
          >
            <option
              v-for="(name, idx) in gallery.sessionChannelNames"
              :key="idx"
              :value="idx"
            >{{ name }} (ch{{ idx }})</option>
          </select>
          <p class="text-[10px] font-mono text-neutral/50 mt-1">Mask + attributes will be computed for this channel.</p>
        </div>

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

    <!-- Advanced tab placeholder -->
    <div
      v-else
      class="flex-1 flex flex-col items-center justify-center text-center gap-3 px-6"
    >
      <Lock class="w-8 h-8 text-neutral/20 stroke-[1.5px]" />
      <h3 class="text-[12px] font-bold tracking-widest uppercase text-neutral/60">
        Advanced Segmentation
      </h3>
      <p class="text-[11px] font-mono text-neutral/40 max-w-md tracking-wider">
        Multi-channel pipelines, learned models, and interactive seed-based refinement.
        Coming soon.
      </p>
    </div>
  </div>
</template>
