<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from 'vue'
import { Play, Box, BarChart3, RotateCcw, ArrowUpDown } from 'lucide-vue-next'
import { useClustersStore } from '@/stores/clusters'
import { useGalleryStore } from '@/stores/gallery'
import { getClusterSamples } from '@/api/clusters'
import { useSessionStore } from '@/stores/session'
import { useDragSelect } from '@/composables/useDragSelect'
import ClusterCard from '@/components/clusters/ClusterCard.vue'
import ClusterContextMenu from '@/components/clusters/ClusterContextMenu.vue'
import ClusteringControlPanel from '@/components/clusters/ClusteringControlPanel.vue'
import ClusterScatterPlot from '@/components/clusters/ClusterScatterPlot.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import TaskProgressBar from '@/components/common/TaskProgressBar.vue'
import SamplePreviewDialog from '@/components/common/SamplePreviewDialog.vue'
import GatingActiveBanner from '@/components/common/GatingActiveBanner.vue'
import type { ClusterSortMode } from '@/types'

const clusters = useClustersStore()
const gallery = useGalleryStore()
const sessionStore = useSessionStore()
const previewDialog = ref<InstanceType<typeof SamplePreviewDialog>>()
const gridContainer = ref<HTMLElement>()

const SORT_OPTIONS: { value: ClusterSortMode; label: string }[] = [
  { value: 'manual', label: 'Manual' },
  { value: 'count_desc', label: 'Most samples' },
  { value: 'count_asc', label: 'Fewest samples' },
  { value: 'labeled_desc', label: '% Labeled' },
  { value: 'quality_desc', label: 'Tightest (quality)' },
]

const { selectionRect } = useDragSelect(
  gridContainer,
  '[data-cluster-id]',
  'data-cluster-id',
  (ids) => clusters.setSelection(ids),
  clusters.selectedClusterIds as any,
)

// Drag-to-merge state
const draggingId = ref<string | null>(null)
const dragTargetId = ref<string | null>(null)

// Context menu state
const ctxMenu = ref({ visible: false, x: 0, y: 0 })

onMounted(() => {
  clusters.fetchClusters()
  clusters.fetchFeatureMethods()
  gallery.fetchClasses()
})

async function onPreview(clusterId: string) {
  await clusters.selectClusterForPreview(clusterId)
  const cluster = clusters.clusters.find(c => c.id === clusterId)
  const sid = sessionStore.currentSessionId!
  if (clusters.clusterSamples) {
    previewDialog.value?.open({
      title: `Cluster`,
      color: cluster?.color,
      samples: clusters.clusterSamples.items,
      total: clusters.clusterSamples.total,
      fetchMore: (offset, limit) => getClusterSamples(sid, clusterId, offset, limit),
    })
  }
}

// Drag handlers
function onDragstart(id: string) { draggingId.value = id }
function onDragend() { draggingId.value = null; dragTargetId.value = null }
function onDragenter(id: string) {
  if (draggingId.value && draggingId.value !== id) dragTargetId.value = id
}
function onDragleave(id: string) {
  if (dragTargetId.value === id) dragTargetId.value = null
}
function onDrop(targetId: string) {
  if (draggingId.value && draggingId.value !== targetId) {
    clusters.mergeClusters([draggingId.value, targetId])
  }
  draggingId.value = null
  dragTargetId.value = null
}

// Context menu handlers
function onContextmenu(e: MouseEvent, clusterId: string) {
  if (!clusters.selectedClusterIds.has(clusterId)) clusters.selectSingle(clusterId)
  ctxMenu.value = { visible: true, x: e.clientX, y: e.clientY }
}

function onContextSplit(n: number) {
  const id = Array.from(clusters.selectedClusterIds)[0]
  if (id) clusters.splitCluster(id, n)
}

// Auto-scroll to the first newly-added cluster after merge/split.
// Uses sortedClusters so the lookup respects the active sort.
watch(
  () => Array.from(clusters.recentlyAddedIds),
  (ids) => {
    if (!ids.length) return
    const ordered = clusters.sortedClusters.map(c => c.id)
    const target = ordered.find(id => ids.includes(id))
    if (!target) return
    nextTick(() => {
      const el = gridContainer.value?.querySelector(`[data-cluster-id="${target}"]`)
      ;(el as HTMLElement | null)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    })
  },
)
</script>

<template>
  <div class="flex h-full relative overflow-hidden bg-base-200">
    <!-- Left panel: Pipeline controls -->
    <div class="w-80 min-w-[320px] bg-base-100 border-r border-base-300 p-6 flex flex-col gap-6 overflow-y-auto z-10 shadow-sm shrink-0">
      <h2 class="text-[14px] font-bold tracking-widest uppercase">Clustering Pipeline</h2>

      <ClusteringControlPanel />

      <div class="space-y-3 pt-2">
        <!-- Run button -->
        <button
          class="h-10 w-full rounded-[2px] flex items-center justify-center gap-2 text-[11px] font-bold tracking-widest uppercase transition-opacity"
          :class="clusters.loading ? 'bg-base-200 text-neutral/40' : 'bg-neutral text-neutral-content hover:opacity-80'"
          :disabled="clusters.loading"
          @click="clusters.runClusteringPipeline()"
        >
          <span v-if="clusters.loading" class="loading loading-spinner loading-sm"></span>
          <Play v-else class="w-4 h-4 stroke-[2px]" />
          Run Pipeline
        </button>

        <!-- Reset button -->
        <button
          class="h-8 w-full rounded-[2px] flex items-center justify-center gap-2 text-[10px] font-bold tracking-widest uppercase transition-colors"
          :class="(clusters.loading || clusters.clusters.length === 0) ? 'text-neutral/30' : 'text-neutral/70 hover:bg-neutral/10 hover:text-neutral'"
          :disabled="clusters.loading || clusters.clusters.length === 0"
          @click="clusters.clearAll()"
        >
          <RotateCcw class="w-3 h-3 stroke-[2px]" />
          Reset Defaults
        </button>
      </div>

      <!-- Task progress -->
      <TaskProgressBar :task-id="clusters.taskId" @complete="clusters.onPipelineComplete" />

      <!-- Scatter plot toggle -->
      <div class="border-t border-base-300 pt-6 mt-2">
        <label class="flex items-center gap-3 cursor-pointer group">
          <input
            v-model="clusters.showScatterPlot"
            type="checkbox"
            class="toggle toggle-sm rounded-full border-neutral/30 bg-base-100 [--tglbg:theme(colors.neutral)] checked:bg-neutral checked:border-neutral checked:[--tglbg:theme(colors.base-100)] transition-all"
          />
          <span class="text-[11px] font-bold tracking-widest uppercase text-neutral group-hover:text-neutral">2D Projection</span>
          <BarChart3 class="w-4 h-4 text-neutral/40 ml-auto group-hover:text-neutral transition-colors" />
        </label>
      </div>
    </div>

    <!-- Right panel: Results -->
    <div class="flex-1 flex flex-col overflow-hidden">
      <GatingActiveBanner />
      <!-- Scatter plot (toggleable) -->
      <div
        v-if="clusters.showScatterPlot"
        class="h-[300px] border-b border-base-300 bg-base-100 p-2 shrink-0 relative z-0"
      >
        <ClusterScatterPlot />
      </div>

      <!-- Cluster cards -->
      <div ref="gridContainer" class="flex-1 p-6 overflow-y-auto relative select-none">
        <div
          v-if="selectionRect"
          class="absolute border border-primary/40 bg-primary/10 pointer-events-none z-20"
          :style="{
            left: selectionRect.x + 'px',
            top: selectionRect.y + 'px',
            width: selectionRect.width + 'px',
            height: selectionRect.height + 'px',
          }"
        />

        <EmptyState
          v-if="clusters.clusters.length === 0 && !clusters.loading"
          :icon="Box"
          title="No clusters yet"
          description="CONFIGURE THE PIPELINE AND CLICK RUN TO GROUP SIMILAR SAMPLES"
        />

        <template v-else>
          <!-- Toolbar: sort -->
          <div class="flex items-center gap-3 mb-4">
            <label class="flex items-center gap-2 text-[10px] font-bold tracking-widest uppercase text-neutral/60">
              <ArrowUpDown class="w-3 h-3 stroke-[2px]" />
              Sort
            </label>
            <select
              v-model="clusters.sortMode"
              class="select select-xs select-bordered rounded-[2px] text-[11px] uppercase tracking-wider font-mono"
            >
              <option v-for="opt in SORT_OPTIONS" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
            <span class="text-[10px] text-neutral/40 font-mono ml-auto">
              {{ clusters.clusters.length }} clusters
            </span>
          </div>

          <!-- TransitionGroup wraps the grid for FLIP reflow on merge/split.
               Note: TransitionGroup measures bounding rects — incompatible with
               virtualization should this grid ever be virtualized. -->
          <TransitionGroup
            name="cluster"
            tag="div"
            class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 auto-rows-max"
          >
            <ClusterCard
              v-for="cluster in clusters.sortedClusters"
              :key="cluster.id"
              :cluster="cluster"
              :is-single-selected="clusters.selectedClusterIds.size === 1 && clusters.selectedClusterIds.has(cluster.id)"
              :is-drag-target="dragTargetId === cluster.id"
              :is-dragging="draggingId === cluster.id"
              :is-recently-added="clusters.recentlyAddedIds.has(cluster.id)"
              @preview="onPreview(cluster.id)"
              @split="(id, n) => clusters.splitCluster(id, n)"
              @contextmenu="onContextmenu"
              @dragstart="onDragstart"
              @dragend="onDragend"
              @dragenter="onDragenter"
              @dragleave="onDragleave"
              @drop="onDrop"
            />
          </TransitionGroup>
        </template>
      </div>
    </div>

    <!-- Context menu -->
    <ClusterContextMenu
      v-if="ctxMenu.visible"
      :x="ctxMenu.x"
      :y="ctxMenu.y"
      :selected-ids="Array.from(clusters.selectedClusterIds)"
      :classes="gallery.classes"
      @view="onPreview(Array.from(clusters.selectedClusterIds)[0])"
      @split="onContextSplit"
      @merge="clusters.mergeClusters(Array.from(clusters.selectedClusterIds))"
      @assign-class="(id) => clusters.assignToClass(Array.from(clusters.selectedClusterIds), id)"
      @close="ctxMenu.visible = false"
    />

    <SamplePreviewDialog ref="previewDialog" />
  </div>
</template>

<style scoped>
.cluster-enter-active {
  transition: opacity 250ms ease-out, transform 250ms ease-out;
}
.cluster-leave-active {
  transition: opacity 200ms ease-in, transform 200ms ease-in;
  position: absolute;
}
.cluster-enter-from,
.cluster-leave-to {
  opacity: 0;
  transform: scale(0.95);
}
.cluster-move {
  transition: transform 300ms ease;
}
</style>
