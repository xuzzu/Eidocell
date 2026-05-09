<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { X, Settings } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { getClassStatistics, getSessionDistributions } from '@/api/classes'
import { getClusterStatistics } from '@/api/clusters'
import type {
  ClassStatistics, ClusterStatistics, SessionDistributions,
  AttributeDistribution, AttributeStatistics,
} from '@/types'
import AttributeMiniHistogram from '@/components/classes/AttributeMiniHistogram.vue'
import { useStatsPreferences, STAT_ATTRIBUTES } from '@/composables/useStatsPreferences'

const sessionStore = useSessionStore()
const sid = computed(() => sessionStore.currentSessionId!)

const dialogRef = ref<HTMLDialogElement>()
const title = ref('')
const subtitle = ref('')
const color = ref<string | null>(null)
const sampleCount = ref(0)
const attributes = ref<AttributeStatistics[]>([])
const sessionDistributions = ref<SessionDistributions | null>(null)
const loading = ref(false)

const showSettings = ref(false)
const prefs = useStatsPreferences()

const distByName = computed<Map<string, AttributeDistribution>>(() => {
  const m = new Map<string, AttributeDistribution>()
  for (const a of sessionDistributions.value?.attributes ?? []) m.set(a.name, a)
  return m
})

const visibleAttrs = computed(() =>
  attributes.value.filter(a => prefs.isVisible(a.name)),
)

function formatAttrName(name: string): string {
  return name.replace(/_/g, ' ')
}

async function openForClass(opts: { id: string; name: string; color: string }) {
  title.value = opts.name
  subtitle.value = 'CLASS'
  color.value = opts.color
  reset()
  dialogRef.value?.showModal()
  await nextTick()
  loading.value = true
  try {
    const [stats, dists] = await Promise.all([
      getClassStatistics(sid.value, opts.id),
      sessionDistributions.value
        ? Promise.resolve(sessionDistributions.value)
        : getSessionDistributions(sid.value),
    ])
    const cs = stats as ClassStatistics
    attributes.value = cs.attributes
    sampleCount.value = cs.sample_count
    sessionDistributions.value = dists
  } catch {
    // swallow — empty display
  } finally {
    loading.value = false
  }
}

async function openForCluster(opts: { id: string; color: string; index?: number }) {
  title.value = opts.index != null ? `Cluster ${opts.index + 1}` : 'Cluster'
  subtitle.value = 'CLUSTER'
  color.value = opts.color
  reset()
  dialogRef.value?.showModal()
  await nextTick()
  loading.value = true
  try {
    const [stats, dists] = await Promise.all([
      getClusterStatistics(sid.value, opts.id),
      sessionDistributions.value
        ? Promise.resolve(sessionDistributions.value)
        : getSessionDistributions(sid.value),
    ])
    const cs = stats as ClusterStatistics
    attributes.value = cs.attributes
    sampleCount.value = cs.sample_count
    sessionDistributions.value = dists
  } catch {
    // ignore
  } finally {
    loading.value = false
  }
}

function reset() {
  attributes.value = []
  sampleCount.value = 0
  showSettings.value = false
}

function close() {
  dialogRef.value?.close()
}

defineExpose({ openForClass, openForCluster, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box max-w-3xl max-h-[85vh] flex flex-col p-6 rounded-[2px]">
      <!-- Header -->
      <div class="flex items-center gap-3 mb-4 shrink-0">
        <span
          v-if="color"
          class="w-3 h-3 rounded-[2px] shrink-0"
          :style="{ backgroundColor: color }"
        ></span>
        <div class="flex-1 min-w-0">
          <div class="text-[9px] font-bold tracking-widest uppercase text-neutral/40">{{ subtitle }} STATISTICS</div>
          <h3 class="font-bold text-lg truncate" :style="color ? { color } : {}">{{ title }}</h3>
        </div>
        <span class="text-[10px] font-mono text-neutral/50 uppercase tracking-widest">
          {{ sampleCount.toLocaleString() }} samples
        </span>
        <button
          class="h-8 w-8 flex items-center justify-center rounded-[2px] text-neutral/40 hover:text-neutral hover:bg-neutral/10 transition-colors"
          :class="{ 'bg-neutral/10 text-neutral': showSettings }"
          title="Choose visible parameters"
          @click="showSettings = !showSettings"
        >
          <Settings class="w-4 h-4 stroke-[2px]" />
        </button>
        <button class="btn btn-ghost btn-sm btn-square" @click="close">
          <X class="w-4 h-4" />
        </button>
      </div>

      <!-- Settings panel -->
      <div
        v-if="showSettings"
        class="mb-4 p-3 bg-base-200/50 border border-base-300 rounded-[2px] shrink-0"
      >
        <div class="flex items-center justify-between mb-2">
          <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/60">
            Visible parameters
          </span>
          <div class="flex gap-1">
            <button
              class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:text-neutral px-2 h-5 rounded-[2px] hover:bg-neutral/10 transition-colors"
              @click="prefs.setAll(true)"
            >
              All
            </button>
            <button
              class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:text-neutral px-2 h-5 rounded-[2px] hover:bg-neutral/10 transition-colors"
              @click="prefs.setAll(false)"
            >
              None
            </button>
          </div>
        </div>
        <div class="grid grid-cols-2 gap-1.5">
          <label
            v-for="attr in STAT_ATTRIBUTES"
            :key="attr"
            class="flex items-center gap-2 text-[10px] font-mono cursor-pointer hover:bg-neutral/5 rounded-[2px] px-2 py-1"
          >
            <input
              type="checkbox"
              class="checkbox checkbox-xs rounded-[2px]"
              :checked="prefs.isVisible(attr)"
              @change="prefs.toggle(attr)"
            />
            <span class="uppercase tracking-wider">{{ formatAttrName(attr) }}</span>
          </label>
        </div>
      </div>

      <!-- Stats grid -->
      <div class="flex-1 overflow-y-auto -mr-1 pr-1">
        <div v-if="loading" class="flex justify-center py-8">
          <span class="loading loading-spinner loading-sm text-neutral/40"></span>
        </div>
        <div v-else-if="visibleAttrs.length === 0" class="text-center py-8 text-[10px] font-mono uppercase tracking-widest text-neutral/30">
          No parameters selected
        </div>
        <div v-else class="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div
            v-for="attr in visibleAttrs"
            :key="attr.name"
            class="p-3 bg-base-100 border border-base-300 rounded-[2px]"
          >
            <div class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">
              {{ formatAttrName(attr.name) }}
            </div>
            <div v-if="attr.mean != null" class="text-base font-mono font-bold mt-1">
              {{ attr.mean.toFixed(2) }}
            </div>
            <div v-if="attr.std != null" class="text-[10px] font-mono text-neutral/40">
              &plusmn; {{ attr.std.toFixed(2) }}
            </div>
            <div v-if="attr.mean == null" class="text-[10px] font-mono text-neutral/30 mt-1">
              N/A
            </div>
            <div
              v-if="attr.min != null && attr.max != null"
              class="text-[9px] font-mono text-neutral/40 mt-0.5"
            >
              [{{ attr.min.toFixed(2) }} – {{ attr.max.toFixed(2) }}]
            </div>
            <div class="mt-2 text-neutral/70">
              <AttributeMiniHistogram
                :distribution="distByName.get(attr.name)"
                :value-marker="attr.mean"
                :marker-color="color ?? '#ef4444'"
                :width="160"
                :height="36"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <form method="dialog" class="modal-backdrop">
      <button @click="close">close</button>
    </form>
  </dialog>
</template>
