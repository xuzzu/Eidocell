<script setup lang="ts">
import { computed, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  ArrowLeft, ArrowRight, Check, Folder, FileImage,
  Loader2, AlertCircle, X, Play,
} from 'lucide-vue-next'
import * as importsApi from '@/api/imports'
import * as tasksApi from '@/api/tasks'
import { getPreviewStatus } from '@/api/sessions'
import { useSessionStore } from '@/stores/session'
import {
  DEFAULT_PREPROCESSING,
  type ImportCreate, type ImportOut, type ImportStatus,
  type SourceKind, type PreprocessingConfig,
  type TaskInfo, type PreviewStatus,
} from '@/types'

const props = defineProps<{ sessionId: string }>()
const router = useRouter()
const sessionStore = useSessionStore()

type Step = 'source' | 'channels' | 'locate' | 'preprocessing' | 'review' | 'progress'
const step = ref<Step>('source')

const sourceKind = ref<SourceKind>('folder')
const sourcePath = ref('')
const csvPath = ref('')
const csvFilenameCol = ref('')
const channelGrouping = ref(true)
const nChannels = ref(1)
const channelNames = ref<string[]>(['Channel 1'])

function resizeChannelNames(target: number) {
  const out = channelNames.value.slice(0, target)
  while (out.length < target) {
    out.push(`Channel ${out.length + 1}`)
  }
  channelNames.value = out
}

function onChannelCountInput(value: number) {
  const clamped = Math.max(1, Math.min(12, Math.round(Number(value) || 1)))
  nChannels.value = clamped
  resizeChannelNames(clamped)
}

const enablePreprocessing = ref(false)
const preproc = ref<PreprocessingConfig>({ ...DEFAULT_PREPROCESSING })

const previewUrl = ref<string | null>(null)

async function previewPreprocessing() {
  errorMessage.value = ''
  try {
    const payload = {
      source_kind: sourceKind.value,
      source_path: sourcePath.value.trim(),
      n_channels: nChannels.value,
      preprocessing: preproc.value
    }
    const res = await importsApi.previewPreprocessing(props.sessionId, payload)
    previewUrl.value = res.image_data
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to generate preview'
  }
}

const submitting = ref(false)
const importId = ref<string | null>(null)
const taskId = ref<string | null>(null)
const importDetail = ref<ImportOut | null>(null)
const taskInfo = ref<TaskInfo | null>(null)
const previewStatus = ref<PreviewStatus | null>(null)
const errorMessage = ref('')
let pollHandle: number | null = null

const stepIndex = computed(() => {
  switch (step.value) {
    case 'source': return 0
    case 'channels': return 1
    case 'locate': return 2
    case 'preprocessing': return 3
    case 'review': return 4
    case 'progress': return 5
  }
})

const stepLabels: Record<Step, string> = {
  source: 'Source',
  channels: 'Channels',
  locate: 'Locate',
  preprocessing: 'Preprocessing',
  review: 'Review',
  progress: 'Progress',
}

const canProceed = computed(() => {
  if (step.value === 'source') return !!sourceKind.value
  if (step.value === 'channels') {
    if (nChannels.value < 1 || nChannels.value > 12) return false
    if (channelNames.value.length !== nChannels.value) return false
    return channelNames.value.every(n => n.trim().length > 0)
  }
  if (step.value === 'locate') {
    if (!sourcePath.value) return false
    if (csvPath.value && !csvFilenameCol.value) return false
    return true
  }
  if (step.value === 'preprocessing') return true
  if (step.value === 'review') return true
  return false
})

async function pickDirectory() {
  try {
    const dir = await (window as any).ipcRenderer.invoke('select-directory')
    if (dir) sourcePath.value = dir
  } catch { /* fallback: type path */ }
}

async function pickContainerFile() {
  try {
    const ext = sourceKind.value === 'cif' ? ['cif'] : ['rif']
    const file = await (window as any).ipcRenderer.invoke('select-file', {
      name: sourceKind.value.toUpperCase(), extensions: ext,
    })
    if (file) sourcePath.value = file
  } catch { /* fallback: type path */ }
}

async function pickCsv() {
  try {
    const file = await (window as any).ipcRenderer.invoke('select-file', {
      name: 'CSV', extensions: ['csv', 'txt'],
    })
    if (file) csvPath.value = file
  } catch { /* fallback: type path */ }
}

function next() {
  if (!canProceed.value) return
  if (step.value === 'source') step.value = 'channels'
  else if (step.value === 'channels') step.value = 'locate'
  else if (step.value === 'locate') step.value = 'preprocessing'
  else if (step.value === 'preprocessing') step.value = 'review'
}

function back() {
  if (step.value === 'channels') step.value = 'source'
  else if (step.value === 'locate') step.value = 'channels'
  else if (step.value === 'preprocessing') step.value = 'locate'
  else if (step.value === 'review') step.value = 'preprocessing'
}

async function submit() {
  errorMessage.value = ''
  submitting.value = true
  const payload: ImportCreate = {
    source_kind: sourceKind.value,
    source_path: sourcePath.value.trim(),
    csv_path: csvPath.value.trim() || null,
    csv_filename_col: csvFilenameCol.value.trim() || null,
    channel_grouping: channelGrouping.value,
    n_channels: nChannels.value,
    channel_names: channelNames.value.map(n => n.trim()),
    preprocessing: enablePreprocessing.value ? preproc.value : null,
  }
  try {
    const out = await importsApi.submitImport(props.sessionId, payload)
    importId.value = out.import_id
    taskId.value = out.task_id
    step.value = 'progress'
    startPolling()
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to submit import'
  } finally {
    submitting.value = false
  }
}

function startPolling() {
  stopPolling()
  pollHandle = window.setInterval(async () => {
    if (!taskId.value || !importId.value) return
    try {
      taskInfo.value = await tasksApi.getTask(taskId.value)
      importDetail.value = await importsApi.getImport(props.sessionId, importId.value)
      const status: ImportStatus = importDetail.value.status
      if (status === 'completed') {
        // Switch to polling preview-status; previews are still being generated.
        try {
          previewStatus.value = await getPreviewStatus(props.sessionId)
        } catch { /* transient */ }
        if (previewStatus.value?.ready) {
          stopPolling()
        }
      } else if (status === 'failed' || status === 'cancelled') {
        stopPolling()
      }
    } catch {
      // ignore transient errors during polling
    }
  }, 400) as unknown as number
}

function stopPolling() {
  if (pollHandle) {
    clearInterval(pollHandle)
    pollHandle = null
  }
}

async function cancelRunning() {
  if (!importId.value) return
  try {
    await importsApi.cancelImport(props.sessionId, importId.value)
  } catch (e: any) {
    errorMessage.value = e.message || 'Failed to cancel'
  }
}

const canOpenSession = computed(() =>
  importDetail.value?.status === 'completed' && previewStatus.value?.ready === true,
)

async function openSession() {
  if (!canOpenSession.value) return
  await sessionStore.selectSession(props.sessionId)
  router.push('/workspace/gallery')
}

function leaveWizard() {
  router.push('/sessions')
}

onUnmounted(stopPolling)
</script>

<template>
  <div class="p-8 max-w-4xl mx-auto h-full flex flex-col">
    <div class="flex items-center justify-between mb-6 pb-4 border-b border-base-300 shrink-0">
      <div>
        <h1 class="text-2xl font-bold tracking-widest uppercase">Import Data</h1>
        <p class="text-[11px] font-mono text-neutral/50 mt-1 tracking-wider">SESSION {{ sessionId }}</p>
      </div>
      <button class="h-9 px-4 flex items-center gap-2 rounded-[2px] text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors" @click="leaveWizard">
        <X class="w-4 h-4" /> Cancel
      </button>
    </div>

    <!-- Step indicator -->
    <ol class="flex items-center gap-2 text-[10px] font-bold tracking-widest uppercase mb-8 shrink-0">
      <li v-for="(label, key, i) in stepLabels" :key="key" class="flex items-center gap-2">
        <span
          class="w-6 h-6 flex items-center justify-center rounded-[2px] border"
          :class="stepIndex >= i ? 'bg-neutral text-neutral-content border-neutral' : 'border-base-300 text-neutral/40'"
        >{{ i + 1 }}</span>
        <span :class="stepIndex >= i ? 'text-neutral' : 'text-neutral/40'">{{ label }}</span>
        <span v-if="i < 5" class="text-neutral/30">›</span>
      </li>
    </ol>

    <div class="flex-1 overflow-y-auto pr-2">
      <!-- STEP 1: Source kind -->
      <section v-if="step === 'source'" class="space-y-4">
        <h2 class="text-[12px] font-bold tracking-widest uppercase mb-4">Choose Source Type</h2>
        <button
          v-for="kind in (['folder','cif','rif'] as SourceKind[])"
          :key="kind"
          :class="[
            'w-full text-left p-4 rounded-[2px] border flex items-start gap-4 transition-colors',
            sourceKind === kind ? 'border-neutral bg-base-200' : 'border-base-300 hover:bg-base-200/50',
          ]"
          @click="sourceKind = kind"
        >
          <component :is="kind === 'folder' ? Folder : FileImage" class="w-6 h-6 mt-1 stroke-[1.5px]" />
          <div>
            <div class="text-sm font-bold tracking-widest uppercase">
              {{ kind === 'folder' ? 'Image Folder' : kind.toUpperCase() + ' Container' }}
            </div>
            <div class="text-[11px] font-mono text-neutral/60 mt-1">
              <template v-if="kind === 'folder'">PNG / JPG / TIFF directory, optional CSV link, optional channel-suffix grouping.</template>
              <template v-else-if="kind === 'cif'">IDEAS Compensated Image File. Multi-channel objects are extracted in one pass.</template>
              <template v-else>IDEAS Raw Image File. Same parser as CIF.</template>
            </div>
          </div>
        </button>
      </section>

      <!-- STEP 2: Channels -->
      <section v-if="step === 'channels'" class="space-y-6">
        <div>
          <h2 class="text-[12px] font-bold tracking-widest uppercase mb-1">Number of Channels</h2>
          <p class="text-[11px] font-mono text-neutral/60 mb-4">
            How many channels does each sample have? For folder imports, files are grouped by <code>_ch&lt;N&gt;</code> suffix.
            For CIF/RIF, object width is split horizontally.
          </p>
        </div>

        <div class="border border-base-300 rounded-[2px] p-5 space-y-5">
          <div class="flex items-center gap-4">
            <input
              type="range" min="1" max="12" :value="nChannels"
              @input="onChannelCountInput(($event.target as HTMLInputElement).valueAsNumber)"
              class="range range-sm range-neutral flex-1"
            />
            <div class="flex flex-col items-center justify-center min-w-[64px] px-3 py-2 bg-base-200 rounded-[2px]">
              <span class="text-2xl font-bold font-mono">{{ nChannels }}</span>
              <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/60">channels</span>
            </div>
          </div>
          <div class="flex justify-between text-[10px] font-mono text-neutral/50 px-1">
            <span>1</span><span>4</span><span>8</span><span>12</span>
          </div>
        </div>

        <div v-if="nChannels >= 1">
          <h3 class="text-[11px] font-bold tracking-widest uppercase text-neutral/70 mb-3">Channel names</h3>
          <div class="grid grid-cols-2 gap-3">
            <div v-for="(_name, idx) in channelNames" :key="idx" class="flex items-center gap-2">
              <span class="text-[10px] font-mono text-neutral/50 w-6 text-right">{{ idx + 1 }}</span>
              <input
                v-model="channelNames[idx]" type="text"
                :placeholder="`Channel ${idx + 1}`"
                class="input input-bordered input-sm rounded-[2px] flex-1 font-mono text-sm focus:outline-neutral"
              />
            </div>
          </div>
          <p class="text-[11px] font-mono text-neutral/60 mt-3">
            Used as labels throughout the app (gallery, sort dropdown, inspect, segmentation).
          </p>
        </div>
      </section>

      <!-- STEP 3: Locate -->
      <section v-if="step === 'locate'" class="space-y-6">
        <h2 class="text-[12px] font-bold tracking-widest uppercase mb-4">
          {{ sourceKind === 'folder' ? 'Locate folder & optional CSV' : `Locate ${sourceKind.toUpperCase()} file` }}
        </h2>

        <!-- Folder mode -->
        <template v-if="sourceKind === 'folder'">
          <div>
            <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Folder</span></label>
            <div class="flex gap-2">
              <input v-model="sourcePath" type="text" placeholder="/path/to/images"
                class="input input-bordered rounded-[2px] flex-1 font-mono text-sm focus:outline-neutral" />
              <button type="button" class="h-12 px-4 border border-base-300 rounded-[2px] hover:bg-base-200 transition-colors text-[11px] font-bold tracking-widest uppercase" @click="pickDirectory">Browse</button>
            </div>
          </div>
          <label class="flex items-start gap-3 text-sm cursor-pointer">
            <input type="checkbox" v-model="channelGrouping" class="checkbox checkbox-sm rounded-[2px] mt-0.5" />
            <span>
              <span class="font-bold text-[11px] tracking-widest uppercase">Detect channel suffixes</span>
              <span class="block text-[11px] font-mono text-neutral/60 mt-0.5">Group <code>name_ch1.png</code>/<code>_ch2.png</code> into a single multi-channel sample.</span>
            </span>
          </label>

          <div class="border-t border-base-300 pt-6">
            <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">CSV Attributes (optional)</span></label>
            <div class="flex gap-2">
              <input v-model="csvPath" type="text" placeholder="/path/to/metadata.csv"
                class="input input-bordered rounded-[2px] flex-1 font-mono text-sm focus:outline-neutral" />
              <button type="button" class="h-12 px-4 border border-base-300 rounded-[2px] hover:bg-base-200 transition-colors text-[11px] font-bold tracking-widest uppercase" @click="pickCsv">Browse</button>
            </div>
            <p class="text-[11px] font-mono text-neutral/60 mt-2">Numeric columns will be imported as per-sample attributes.</p>

            <label v-if="csvPath" class="label pb-1 mt-4"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Filename Column</span></label>
            <input v-if="csvPath" v-model="csvFilenameCol" type="text" placeholder="image_name"
              class="input input-bordered rounded-[2px] w-full font-mono text-sm focus:outline-neutral" />
          </div>
        </template>

        <!-- CIF / RIF mode -->
        <template v-else>
          <div>
            <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">{{ sourceKind.toUpperCase() }} File</span></label>
            <div class="flex gap-2">
              <input v-model="sourcePath" type="text" :placeholder="`/path/to/file.${sourceKind}`"
                class="input input-bordered rounded-[2px] flex-1 font-mono text-sm focus:outline-neutral" />
              <button type="button" class="h-12 px-4 border border-base-300 rounded-[2px] hover:bg-base-200 transition-colors text-[11px] font-bold tracking-widest uppercase" @click="pickContainerFile">Browse</button>
            </div>
          </div>
          <div class="bg-base-200/50 border border-base-300 rounded-[2px] p-3 flex items-center gap-3">
            <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/60">Channels per object</span>
            <span class="font-mono font-bold">{{ nChannels }}</span>
            <span class="text-[10px] font-mono text-neutral/50">(set in the Channels step)</span>
          </div>
        </template>
      </section>

      <!-- STEP 3: Preprocessing -->
      <section v-if="step === 'preprocessing'" class="space-y-6">
        <h2 class="text-[12px] font-bold tracking-widest uppercase mb-1">Preprocessing</h2>
        <p class="text-[11px] font-mono text-neutral/60 mb-4">Optional. Skip to import raw arrays as-is.</p>

        <label class="flex items-start gap-3 text-sm cursor-pointer">
          <input type="checkbox" v-model="enablePreprocessing" class="checkbox checkbox-sm rounded-[2px] mt-0.5" />
          <span class="font-bold text-[11px] tracking-widest uppercase">Apply a preprocessing pipeline</span>
        </label>

        <div v-if="enablePreprocessing" class="space-y-5 border border-base-300 rounded-[2px] p-5">
          <!-- Target shape -->
          <div>
            <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Target shape</span></label>
            <select v-model="preproc.target_shape_strategy" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
              <option value="none">No reshaping</option>
              <option value="min">Match smallest image</option>
              <option value="mean">Match average size</option>
              <option value="max">Match largest image</option>
              <option value="explicit">Specify exact shape</option>
            </select>
          </div>
          <div v-if="preproc.target_shape_strategy === 'explicit'" class="flex gap-2">
            <input
              type="number" min="1" placeholder="height"
              :value="preproc.explicit_shape?.[0] ?? ''"
              @input="preproc.explicit_shape = [Number(($event.target as HTMLInputElement).value), preproc.explicit_shape?.[1] ?? 0]"
              class="input input-bordered rounded-[2px] w-32 font-mono text-sm focus:outline-neutral"
            />
            <input
              type="number" min="1" placeholder="width"
              :value="preproc.explicit_shape?.[1] ?? ''"
              @input="preproc.explicit_shape = [preproc.explicit_shape?.[0] ?? 0, Number(($event.target as HTMLInputElement).value)]"
              class="input input-bordered rounded-[2px] w-32 font-mono text-sm focus:outline-neutral"
            />
          </div>

          <!-- Resize vs pad -->
          <div v-if="preproc.target_shape_strategy !== 'none'">
            <label class="flex items-center gap-3 text-sm cursor-pointer">
              <input type="checkbox" v-model="preproc.resize" class="checkbox checkbox-sm rounded-[2px]" />
              <span class="font-mono text-xs">Resize (otherwise pad)</span>
            </label>
            <div v-if="!preproc.resize" class="mt-3">
              <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Padding method</span></label>
              <select v-model="preproc.padding_method" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
                <option value="constant">Constant (zero)</option>
                <option value="poisson">Poisson noise (recommended for IFC)</option>
                <option value="replicate">Replicate edge</option>
              </select>
            </div>
          </div>

          <!-- Normalization -->
          <div>
            <label class="label pb-1"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Normalisation</span></label>
            <select v-model="preproc.normalize" class="select select-bordered rounded-[2px] w-full font-mono text-sm">
              <option value="none">None</option>
              <option value="zscore">Z-score (recommended)</option>
              <option value="minmax">Min-max [0,1]</option>
            </select>
          </div>
          
          <!-- Background Subtraction -->
          <div>
            <label class="flex items-center gap-3 text-sm cursor-pointer">
              <input type="checkbox" v-model="preproc.background_subtraction" class="checkbox checkbox-sm rounded-[2px]" />
              <span class="font-bold text-[11px] tracking-widest uppercase">Rolling Ball Background Subtraction</span>
            </label>
            <p v-if="preproc.background_subtraction" class="text-[11px] font-mono text-neutral/60 mt-1">
              Radius: <input type="number" min="1" v-model.number="preproc.background_subtraction_radius" class="input input-xs input-bordered rounded-[2px] w-16 font-mono ml-1" />
            </p>
          </div>

          <!-- Alignment -->
          <div>
            <label class="flex items-center gap-3 text-sm cursor-pointer">
              <input type="checkbox" v-model="preproc.align_channels" class="checkbox checkbox-sm rounded-[2px]" />
              <span class="font-bold text-[11px] tracking-widest uppercase">Sub-pixel channel alignment</span>
            </label>
            <p v-if="preproc.align_channels" class="text-[11px] font-mono text-neutral/60 mt-1">
              Reference channel index: <input type="number" min="0" v-model.number="preproc.align_reference_index" class="input input-xs input-bordered rounded-[2px] w-16 font-mono ml-1" />
            </p>
          </div>
          
          <div class="mt-4 pt-4 border-t border-base-300">
             <button type="button" class="h-8 px-4 border border-base-300 rounded-[2px] hover:bg-base-200 transition-colors text-[11px] font-bold tracking-widest uppercase flex items-center gap-2" @click="previewPreprocessing">
               Preview Image
             </button>
             <img v-if="previewUrl" :src="previewUrl" class="mt-4 max-h-48 border border-base-300" />
          </div>
        </div>
      </section>

      <!-- STEP 4: Review -->
      <section v-if="step === 'review'" class="space-y-4">
        <h2 class="text-[12px] font-bold tracking-widest uppercase mb-4">Review &amp; Start</h2>
        <dl class="grid grid-cols-3 gap-y-2 gap-x-4 text-sm font-mono">
          <dt class="text-[10px] tracking-widest uppercase text-neutral/60">Source kind</dt>
          <dd class="col-span-2">{{ sourceKind }}</dd>

          <dt class="text-[10px] tracking-widest uppercase text-neutral/60">Path</dt>
          <dd class="col-span-2 break-all">{{ sourcePath }}</dd>

          <dt class="text-[10px] tracking-widest uppercase text-neutral/60">Channels</dt>
          <dd class="col-span-2">{{ nChannels }} ({{ channelNames.join(', ') }})</dd>

          <template v-if="sourceKind === 'folder'">
            <dt class="text-[10px] tracking-widest uppercase text-neutral/60">Channel grouping</dt>
            <dd class="col-span-2">{{ channelGrouping ? 'enabled' : 'disabled' }}</dd>

            <template v-if="csvPath">
              <dt class="text-[10px] tracking-widest uppercase text-neutral/60">CSV</dt>
              <dd class="col-span-2 break-all">{{ csvPath }} ({{ csvFilenameCol }})</dd>
            </template>
          </template>

          <dt class="text-[10px] tracking-widest uppercase text-neutral/60">Preprocessing</dt>
          <dd class="col-span-2">
            <template v-if="!enablePreprocessing">skipped (raw arrays)</template>
            <template v-else>
              {{ preproc.target_shape_strategy }} shape · {{ preproc.resize ? 'resize' : 'pad ' + preproc.padding_method }} ·
              {{ preproc.normalize === 'none' ? 'no norm' : preproc.normalize }}
              <span v-if="preproc.align_channels"> · align</span>
            </template>
          </dd>
        </dl>

        <p v-if="errorMessage" class="text-error text-xs font-mono mt-4">{{ errorMessage }}</p>
      </section>

      <!-- STEP 5: Progress -->
      <section v-if="step === 'progress'" class="space-y-6">
        <h2 class="text-[12px] font-bold tracking-widest uppercase">Importing</h2>

        <div class="border border-base-300 rounded-[2px] p-5 space-y-3">
          <div class="flex items-center gap-3">
            <Loader2
              v-if="importDetail && (importDetail.status === 'loading' || importDetail.status === 'preprocessing' || importDetail.status === 'pending')"
              class="w-5 h-5 animate-spin"
            />
            <Check v-else-if="importDetail?.status === 'completed'" class="w-5 h-5 text-success" />
            <AlertCircle v-else-if="importDetail?.status === 'failed' || importDetail?.status === 'cancelled'" class="w-5 h-5 text-error" />
            <span class="text-sm font-bold tracking-widest uppercase">{{ importDetail?.status ?? 'pending' }}</span>
          </div>

          <progress
            class="progress progress-neutral w-full rounded-[2px]"
            :value="taskInfo?.percentage ?? 0" max="100"
          />
          <p class="text-[11px] font-mono text-neutral/70">{{ taskInfo?.message ?? '...' }}</p>

          <div class="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] font-mono pt-2">
            <div><span class="text-neutral/50">imported</span> {{ importDetail?.sample_count ?? 0 }}</div>
            <div><span class="text-neutral/50">skipped</span> {{ importDetail?.skipped_count ?? 0 }}</div>
          </div>
        </div>

        <!-- Phase 2: preview pre-generation. Blocks "Open Session". -->
        <div
          v-if="importDetail?.status === 'completed'"
          class="border border-base-300 rounded-[2px] p-5 space-y-3"
        >
          <div class="flex items-center gap-3">
            <Loader2 v-if="!previewStatus?.ready" class="w-5 h-5 animate-spin" />
            <Check v-else class="w-5 h-5 text-success" />
            <span class="text-sm font-bold tracking-widest uppercase">
              {{ previewStatus?.ready ? 'Previews Ready' : 'Pre-generating Previews' }}
            </span>
          </div>
          <progress
            class="progress progress-neutral w-full rounded-[2px]"
            :value="previewStatus?.progress ?? 0" max="100"
          />
          <p class="text-[11px] font-mono text-neutral/70 truncate">
            {{ previewStatus?.message ?? 'Starting...' }}
          </p>
        </div>

        <details v-if="importDetail?.errors?.length" class="border border-base-300 rounded-[2px] p-4">
          <summary class="text-[11px] font-bold tracking-widest uppercase cursor-pointer">Skipped items ({{ importDetail.errors.length }})</summary>
          <ul class="mt-3 space-y-1 text-[11px] font-mono max-h-48 overflow-y-auto">
            <li v-for="(e, i) in importDetail.errors" :key="i" class="break-all">
              <span class="text-neutral/60">{{ e.path }}</span> — {{ e.reason }}
            </li>
          </ul>
        </details>

        <div class="flex gap-2">
          <button
            v-if="importDetail && (importDetail.status === 'loading' || importDetail.status === 'preprocessing' || importDetail.status === 'pending')"
            class="h-10 px-6 rounded-[2px] border border-base-300 text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors"
            @click="cancelRunning"
          >Cancel</button>
          <button
            v-if="importDetail?.status === 'completed'"
            :disabled="!canOpenSession"
            class="h-10 px-8 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase transition-opacity"
            :class="canOpenSession ? 'hover:opacity-80' : 'opacity-30 cursor-not-allowed'"
            :title="canOpenSession ? '' : 'Waiting for preview pre-generation'"
            @click="openSession"
          >
            <span v-if="!canOpenSession" class="flex items-center gap-2">
              <Loader2 class="w-4 h-4 animate-spin" /> Waiting for previews…
            </span>
            <span v-else>Open Session</span>
          </button>
          <button
            v-if="importDetail?.status === 'failed' || importDetail?.status === 'cancelled'"
            class="h-10 px-6 rounded-[2px] border border-base-300 text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors"
            @click="leaveWizard"
          >Back to Sessions</button>
        </div>
      </section>
    </div>

    <!-- Footer nav -->
    <div v-if="step !== 'progress'" class="flex justify-between items-center mt-6 pt-4 border-t border-base-300 shrink-0">
      <button
        v-if="step !== 'source'"
        class="h-10 px-5 flex items-center gap-2 rounded-[2px] text-[11px] font-bold tracking-widest uppercase hover:bg-base-200 transition-colors"
        @click="back"
      ><ArrowLeft class="w-4 h-4" /> Back</button>
      <span v-else />

      <button
        v-if="step !== 'review'"
        :disabled="!canProceed"
        class="h-10 px-6 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase hover:opacity-80 transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
        @click="next"
      >Next <ArrowRight class="w-4 h-4" /></button>
      <button
        v-else
        :disabled="submitting"
        class="h-10 px-8 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[11px] font-bold tracking-widest uppercase hover:opacity-80 transition-opacity disabled:opacity-30"
        @click="submit"
      >
        <Loader2 v-if="submitting" class="w-4 h-4 animate-spin" />
        <Play v-else class="w-4 h-4" />
        Start Import
      </button>
    </div>
  </div>
</template>
