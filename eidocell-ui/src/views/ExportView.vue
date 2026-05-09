<script setup lang="ts">
import { ref, computed } from 'vue'
import { Download, FolderOpen, CheckCircle, AlertTriangle } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { exportSession } from '@/api/export'
import type { ExportResult } from '@/types'

const sessionStore = useSessionStore()
const outputDirectory = ref('')
const includeClasses = ref(true)
const includeClusters = ref(true)
const includeMasks = ref(true)
const includeBinaryMasks = ref(false)
const includeCsv = ref(true)
const exporting = ref(false)
const result = ref<ExportResult | null>(null)
const error = ref('')

const hasSession = computed(() => !!sessionStore.currentSessionId)

async function pickDirectory() {
  try {
    const dir = await window.ipcRenderer.invoke('select-directory')
    if (dir) outputDirectory.value = dir
  } catch { /* fallback to manual */ }
}

async function run() {
  if (!sessionStore.currentSessionId || !outputDirectory.value) return
  exporting.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await exportSession(sessionStore.currentSessionId, {
      output_directory: outputDirectory.value,
      include_classes: includeClasses.value,
      include_clusters: includeClusters.value,
      include_masks: includeMasks.value,
      include_binary_masks: includeBinaryMasks.value,
      include_csv: includeCsv.value,
    })
  } catch (e: any) {
    error.value = e.message || 'Export failed'
  } finally {
    exporting.value = false
  }
}
</script>

<template>
  <div class="p-8 max-w-2xl mx-auto flex flex-col h-full">
    <div class="flex items-center justify-between mb-8 pb-4 border-b border-base-300 shrink-0">
      <h1 class="text-2xl font-bold tracking-widest uppercase">Export</h1>
    </div>

    <div v-if="!hasSession" class="alert bg-warning/10 text-warning-content border border-warning/20 rounded-[2px] font-mono tracking-tight shadow-sm font-bold">
      <AlertTriangle class="w-5 h-5 stroke-[1.5px]" />
      <span>Select a session first to export data.</span>
    </div>

    <template v-else>
      <div class="bg-base-100 shadow-sm border border-base-300 rounded-[2px] p-8 space-y-8 overflow-y-auto">
        
        <div class="space-y-2">
          <label class="block text-[10px] font-bold tracking-widest uppercase text-neutral/70">Output Directory</label>
          <div class="flex gap-2">
            <input v-model="outputDirectory" type="text" placeholder="/path/to/export" class="input input-bordered w-full rounded-[2px] font-mono text-sm focus:outline-neutral" />
            <button class="h-12 w-12 flex items-center justify-center border border-base-300 rounded-[2px] hover:bg-base-200 transition-colors text-neutral shrink-0" @click="pickDirectory">
              <FolderOpen class="w-5 h-5 stroke-[1.5px]" />
            </button>
          </div>
        </div>

        <div>
          <div class="text-[10px] font-bold tracking-widest uppercase text-neutral/50 mb-4 pb-2 border-b border-base-300/50">Include Options</div>

          <div class="grid grid-cols-2 gap-y-4 gap-x-8">
            <label class="flex items-center gap-3 cursor-pointer group">
              <input v-model="includeClasses" type="checkbox" class="checkbox checkbox-sm rounded-[2px] border-neutral/30 group-hover:border-neutral checked:bg-neutral transition-colors" />
              <span class="text-xs font-mono font-bold text-neutral">CLASSES <span class="text-neutral/40 font-normal">(images by class)</span></span>
            </label>
            <label class="flex items-center gap-3 cursor-pointer group">
              <input v-model="includeClusters" type="checkbox" class="checkbox checkbox-sm rounded-[2px] border-neutral/30 group-hover:border-neutral checked:bg-neutral transition-colors" />
              <span class="text-xs font-mono font-bold text-neutral">CLUSTERS <span class="text-neutral/40 font-normal">(images by cluster)</span></span>
            </label>
            <label class="flex items-center gap-3 cursor-pointer group">
              <input v-model="includeMasks" type="checkbox" class="checkbox checkbox-sm rounded-[2px] border-neutral/30 group-hover:border-neutral checked:bg-neutral transition-colors" />
              <span class="text-xs font-mono font-bold text-neutral">MASKS <span class="text-neutral/40 font-normal">(overlay previews)</span></span>
            </label>
            <label class="flex items-center gap-3 cursor-pointer group">
              <input v-model="includeBinaryMasks" type="checkbox" class="checkbox checkbox-sm rounded-[2px] border-neutral/30 group-hover:border-neutral checked:bg-neutral transition-colors" />
              <span class="text-xs font-mono font-bold text-neutral">BINARY MASKS <span class="text-neutral/40 font-normal">(raw 0/255 PNGs)</span></span>
            </label>
            <label class="flex items-center gap-3 cursor-pointer group">
              <input v-model="includeCsv" type="checkbox" class="checkbox checkbox-sm rounded-[2px] border-neutral/30 group-hover:border-neutral checked:bg-neutral transition-colors" />
              <span class="text-xs font-mono font-bold text-neutral">CSV <span class="text-neutral/40 font-normal">(metadata)</span></span>
            </label>
          </div>
        </div>

        <button
          class="h-10 w-full rounded-[2px] flex items-center justify-center gap-2 text-[11px] font-bold tracking-widest uppercase transition-opacity"
          :class="(outputDirectory && !exporting) ? 'bg-neutral text-neutral-content hover:opacity-80' : 'bg-base-200 text-neutral/40 cursor-not-allowed'"
          :disabled="!outputDirectory || exporting"
          @click="run"
        >
          <span v-if="exporting" class="loading loading-spinner text-neutral-content loading-sm"></span>
          <Download v-else class="w-4 h-4 stroke-[2px]" />
          {{ exporting ? 'Exporting...' : 'Export' }}
        </button>
      </div>

      <div v-if="result" class="alert bg-success/10 text-success-content border border-success/20 rounded-[2px] font-mono tracking-tight shadow-sm font-bold mt-4">
        <CheckCircle class="w-5 h-5 stroke-[1.5px]" />
        <div>
          <p class="font-bold uppercase tracking-wider text-xs">Export complete</p>
          <p class="text-xs mt-1 text-success-content/80">
            CLASSES: {{ result.classes_exported }} |
            CLUSTERS: {{ result.clusters_exported }} |
            MASKS: {{ result.masks_exported }} |
            BINARY MASKS: {{ result.binary_masks_exported }} |
            CSV ROWS: {{ result.csv_rows }}
          </p>
        </div>
      </div>

      <div v-if="error" class="alert alert-error rounded-[2px] font-mono tracking-tight mt-4 font-bold">{{ error }}</div>
    </template>
  </div>
</template>
