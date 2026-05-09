<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { X, Check } from 'lucide-vue-next'
import { useSessionStore } from '@/stores/session'
import { useClassesStore } from '@/stores/classes'
import { getClassStatistics, getSessionDistributions } from '@/api/classes'
import type { ClassStatistics, SessionDistributions, AttributeDistribution } from '@/types'
import { STAT_ATTRIBUTES } from '@/composables/useStatsPreferences'

const MAX_COMPARE = 6

const sessionStore = useSessionStore()
const classesStore = useClassesStore()

const dialogRef = ref<HTMLDialogElement>()
const step = ref<'select' | 'view'>('select')

const selectedClassIds = ref<Set<string>>(new Set())
const selectedAttrs = ref<Set<string>>(new Set(STAT_ATTRIBUTES))

const statsByClassId = ref<Map<string, ClassStatistics>>(new Map())
const sessionDistributions = ref<SessionDistributions | null>(null)
const loading = ref(false)

const sid = computed(() => sessionStore.currentSessionId!)

const eligibleClasses = computed(() =>
  classesStore.classes.filter(c => c.name !== 'Uncategorized'),
)

const distByName = computed<Map<string, AttributeDistribution>>(() => {
  const m = new Map<string, AttributeDistribution>()
  for (const a of sessionDistributions.value?.attributes ?? []) m.set(a.name, a)
  return m
})

const selectedClasses = computed(() =>
  Array.from(selectedClassIds.value)
    .map(id => classesStore.classes.find(c => c.id === id))
    .filter((c): c is NonNullable<typeof c> => !!c),
)

const visibleAttrs = computed(() =>
  STAT_ATTRIBUTES.filter(a => selectedAttrs.value.has(a)),
)

const canCompare = computed(() =>
  selectedClassIds.value.size >= 2 && visibleAttrs.value.length >= 1,
)

function toggleClass(id: string) {
  const next = new Set(selectedClassIds.value)
  if (next.has(id)) next.delete(id)
  else if (next.size < MAX_COMPARE) next.add(id)
  selectedClassIds.value = next
}

function toggleAttr(name: string) {
  const next = new Set(selectedAttrs.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  selectedAttrs.value = next
}

function setAllAttrs(visible: boolean) {
  selectedAttrs.value = visible ? new Set(STAT_ATTRIBUTES) : new Set()
}

function open() {
  step.value = 'select'
  selectedClassIds.value = new Set()
  statsByClassId.value = new Map()
  dialogRef.value?.showModal()
  classesStore.fetchClasses()
}

function close() {
  dialogRef.value?.close()
}

async function startCompare() {
  if (!canCompare.value) return
  loading.value = true
  try {
    const ids = Array.from(selectedClassIds.value)
    const [statsArr, dists] = await Promise.all([
      Promise.all(ids.map(id => getClassStatistics(sid.value, id))),
      sessionDistributions.value
        ? Promise.resolve(sessionDistributions.value)
        : getSessionDistributions(sid.value),
    ])
    const map = new Map<string, ClassStatistics>()
    statsArr.forEach((s, i) => map.set(ids[i], s))
    statsByClassId.value = map
    sessionDistributions.value = dists
    step.value = 'view'
    nextTick()
  } finally {
    loading.value = false
  }
}

function getStat(classId: string, attrName: string): { mean: number | null; std: number | null } {
  const s = statsByClassId.value.get(classId)
  if (!s) return { mean: null, std: null }
  const a = s.attributes.find(x => x.name === attrName)
  return { mean: a?.mean ?? null, std: a?.std ?? null }
}

// Compute marker positions for an overlaid distribution histogram
function markerPositions(attrName: string, width: number) {
  const dist = distByName.value.get(attrName)
  if (!dist || dist.bin_edges.length < 2) return []
  const lo = dist.bin_edges[0]
  const hi = dist.bin_edges[dist.bin_edges.length - 1]
  if (hi === lo) return []
  return selectedClasses.value.map(c => {
    const { mean } = getStat(c.id, attrName)
    if (mean == null) return null
    const t = (mean - lo) / (hi - lo)
    if (!isFinite(t)) return null
    return {
      x: Math.max(0, Math.min(1, t)) * width,
      color: c.color,
      name: c.name,
      mean,
    }
  }).filter((m): m is NonNullable<typeof m> => m !== null)
}

function bars(attrName: string, width: number, height: number) {
  const dist = distByName.value.get(attrName)
  if (!dist || dist.bin_counts.length === 0) return null
  const maxCount = Math.max(...dist.bin_counts)
  if (maxCount === 0) return null
  const n = dist.bin_counts.length
  const barW = width / n
  return dist.bin_counts.map((c, i) => ({
    x: i * barW,
    width: Math.max(barW - 0.5, 0.5),
    height: (c / maxCount) * height,
  }))
}

function formatAttrName(name: string): string {
  return name.replace(/_/g, ' ')
}

defineExpose({ open, close })
</script>

<template>
  <dialog ref="dialogRef" class="modal">
    <div class="modal-box max-w-5xl max-h-[90vh] flex flex-col p-6 rounded-[2px]">
      <!-- Header -->
      <div class="flex items-center gap-3 mb-4 shrink-0">
        <div class="flex-1 min-w-0">
          <div class="text-[9px] font-bold tracking-widest uppercase text-neutral/40">COMPARE</div>
          <h3 class="font-bold text-lg">Class Comparison</h3>
        </div>
        <button class="btn btn-ghost btn-sm btn-square" @click="close">
          <X class="w-4 h-4" />
        </button>
      </div>

      <!-- Step: select -->
      <div v-if="step === 'select'" class="flex-1 overflow-y-auto -mr-1 pr-1">
        <div class="grid grid-cols-2 gap-6">
          <!-- Classes -->
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/60">
                Classes
              </span>
              <span class="text-[9px] font-mono text-neutral/40">
                {{ selectedClassIds.size }} / {{ MAX_COMPARE }}
              </span>
            </div>
            <div class="space-y-1 max-h-80 overflow-y-auto border border-base-300 rounded-[2px] p-1">
              <button
                v-for="cls in eligibleClasses"
                :key="cls.id"
                class="w-full flex items-center gap-2.5 px-2 py-1.5 text-[11px] font-mono text-left rounded-[2px] transition-colors"
                :class="selectedClassIds.has(cls.id)
                  ? 'bg-neutral/10 ring-1 ring-neutral/30'
                  : (selectedClassIds.size >= MAX_COMPARE ? 'opacity-40 cursor-not-allowed' : 'hover:bg-base-200')"
                :disabled="!selectedClassIds.has(cls.id) && selectedClassIds.size >= MAX_COMPARE"
                @click="toggleClass(cls.id)"
              >
                <span class="w-3 h-3 flex items-center justify-center rounded-[2px] border border-neutral/30 shrink-0"
                  :class="selectedClassIds.has(cls.id) ? 'bg-neutral text-neutral-content' : ''"
                >
                  <Check v-if="selectedClassIds.has(cls.id)" class="w-2.5 h-2.5 stroke-[3px]" />
                </span>
                <span class="w-2.5 h-2.5 rounded-[2px] shrink-0" :style="{ backgroundColor: cls.color }"></span>
                <span class="font-bold tracking-wider uppercase truncate">{{ cls.name }}</span>
                <span class="ml-auto text-neutral/40 text-[10px]">{{ cls.sample_count }}</span>
              </button>
              <div v-if="eligibleClasses.length === 0" class="text-[10px] font-mono text-neutral/30 text-center py-4">
                No classes available
              </div>
            </div>
          </div>

          <!-- Attributes -->
          <div>
            <div class="flex items-center justify-between mb-2">
              <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/60">
                Parameters
              </span>
              <div class="flex gap-1">
                <button
                  class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:text-neutral px-2 h-5 rounded-[2px] hover:bg-neutral/10 transition-colors"
                  @click="setAllAttrs(true)"
                >All</button>
                <button
                  class="text-[9px] font-bold tracking-widest uppercase text-neutral/50 hover:text-neutral px-2 h-5 rounded-[2px] hover:bg-neutral/10 transition-colors"
                  @click="setAllAttrs(false)"
                >None</button>
              </div>
            </div>
            <div class="grid grid-cols-2 gap-1 max-h-80 overflow-y-auto border border-base-300 rounded-[2px] p-1">
              <label
                v-for="attr in STAT_ATTRIBUTES"
                :key="attr"
                class="flex items-center gap-2 text-[10px] font-mono cursor-pointer hover:bg-neutral/5 rounded-[2px] px-2 py-1"
              >
                <input
                  type="checkbox"
                  class="checkbox checkbox-xs rounded-[2px]"
                  :checked="selectedAttrs.has(attr)"
                  @change="toggleAttr(attr)"
                />
                <span class="uppercase tracking-wider">{{ formatAttrName(attr) }}</span>
              </label>
            </div>
          </div>
        </div>

        <div class="flex justify-end mt-6 gap-2">
          <button class="btn btn-ghost btn-sm rounded-[2px]" @click="close">Cancel</button>
          <button
            class="h-9 px-5 flex items-center gap-2 rounded-[2px] text-[10px] font-bold tracking-widest uppercase transition-opacity"
            :class="canCompare ? 'bg-neutral text-neutral-content hover:opacity-80' : 'bg-base-200 text-neutral/40 cursor-not-allowed'"
            :disabled="!canCompare"
            @click="startCompare"
          >
            <span v-if="loading" class="loading loading-spinner loading-xs"></span>
            Compare
          </button>
        </div>
      </div>

      <!-- Step: view -->
      <div v-else class="flex-1 overflow-y-auto -mr-1 pr-1">
        <!-- Class color legend -->
        <div class="flex flex-wrap gap-3 mb-4 pb-3 border-b border-base-300">
          <div
            v-for="cls in selectedClasses"
            :key="cls.id"
            class="flex items-center gap-1.5"
          >
            <span class="w-3 h-3 rounded-[2px]" :style="{ backgroundColor: cls.color }"></span>
            <span class="text-[10px] font-bold tracking-wider uppercase font-mono" :style="{ color: cls.color }">
              {{ cls.name }}
            </span>
            <span class="text-[9px] font-mono text-neutral/40">({{ cls.sample_count }})</span>
          </div>
        </div>

        <!-- Comparison table -->
        <div class="space-y-4">
          <div
            v-for="attr in visibleAttrs"
            :key="attr"
            class="border border-base-300 rounded-[2px] p-3 bg-base-100"
          >
            <div class="flex items-center justify-between mb-2">
              <span class="text-[10px] font-bold tracking-widest uppercase text-neutral/70">
                {{ formatAttrName(attr) }}
              </span>
            </div>

            <!-- Overlay distribution + class markers -->
            <div class="mb-3 text-neutral/60">
              <svg :width="600" :height="50" :viewBox="`0 0 600 50`" class="block w-full h-12">
                <g>
                  <rect
                    v-for="(b, i) in bars(attr, 600, 40) ?? []"
                    :key="i"
                    :x="b.x"
                    :y="40 - b.height"
                    :width="b.width"
                    :height="b.height"
                    fill="currentColor"
                    opacity="0.25"
                  />
                </g>
                <line x1="0" y1="40.5" x2="600" y2="40.5" stroke="currentColor" stroke-width="0.5" opacity="0.3" />
                <g>
                  <line
                    v-for="m in markerPositions(attr, 600)"
                    :key="m.name"
                    :x1="m.x" :y1="0" :x2="m.x" :y2="40"
                    :stroke="m.color"
                    stroke-width="2"
                  />
                </g>
              </svg>
            </div>

            <!-- Per-class stats -->
            <div class="grid gap-2" :style="{ gridTemplateColumns: `repeat(${selectedClasses.length}, minmax(0, 1fr))` }">
              <div
                v-for="cls in selectedClasses"
                :key="cls.id"
                class="p-2 bg-base-200/40 rounded-[2px] border-l-2"
                :style="{ borderLeftColor: cls.color }"
              >
                <div class="text-[9px] font-bold tracking-widest uppercase truncate" :style="{ color: cls.color }">
                  {{ cls.name }}
                </div>
                <div class="text-sm font-mono font-bold mt-0.5">
                  {{ getStat(cls.id, attr).mean != null ? getStat(cls.id, attr).mean!.toFixed(2) : 'N/A' }}
                </div>
                <div v-if="getStat(cls.id, attr).std != null" class="text-[9px] font-mono text-neutral/40">
                  &plusmn; {{ getStat(cls.id, attr).std!.toFixed(2) }}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="flex justify-end mt-6 gap-2">
          <button class="btn btn-ghost btn-sm rounded-[2px]" @click="step = 'select'">
            ← Edit selection
          </button>
          <button
            class="h-9 px-5 flex items-center gap-2 rounded-[2px] bg-neutral text-neutral-content text-[10px] font-bold tracking-widest uppercase hover:opacity-80 transition-opacity"
            @click="close"
          >
            Done
          </button>
        </div>
      </div>
    </div>

    <form method="dialog" class="modal-backdrop">
      <button @click="close">close</button>
    </form>
  </dialog>
</template>
