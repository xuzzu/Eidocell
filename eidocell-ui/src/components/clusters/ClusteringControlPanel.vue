<script setup lang="ts">
import { computed, watch } from 'vue'
import { useClustersStore } from '@/stores/clusters'
import { useGalleryStore } from '@/stores/gallery'
import type { ClusteringScopeMode } from '@/types'

const clusters = useClustersStore()
const gallery = useGalleryStore()

const clusteringMethods = [
  { value: 'kmeans', label: 'K-Means' },
  { value: 'evoc', label: 'EVoC (auto)' },
]

const dimMethods = [
  { value: 'pca', label: 'PCA' },
  { value: 'umap', label: 'UMAP' },
  { value: null, label: 'None' },
]

const scopeOptions: { value: ClusteringScopeMode; label: string }[] = [
  { value: 'all', label: 'All active samples' },
  { value: 'unlabeled', label: 'Unlabeled only' },
  { value: 'class', label: 'Specific class' },
  { value: 'samples', label: 'Selected samples' },
]

const currentDimMethod = computed({
  get: () => clusters.dimReductionMethod,
  set: (val: string | null) => clusters.updateDimReductionParams(val),
})

const currentClusteringMethod = computed({
  get: () => clusters.clusteringMethod,
  set: (val: string) => clusters.updateClusteringMethod(val),
})

const isEvoc = computed(() => clusters.clusteringMethod === 'evoc')

const selectableClasses = computed(() =>
  gallery.classes.filter(c => c.name !== 'Uncategorized'),
)

const selectedSampleCount = computed(() => gallery.selectedIds.size)

// Keep scope payload tidy: clear class_id when not in 'class' mode, etc.
watch(
  () => clusters.scope.mode,
  (mode) => {
    if (mode !== 'class') clusters.scope.class_id = null
    if (mode === 'samples') {
      clusters.scope.sample_ids = Array.from(gallery.selectedIds)
    } else {
      clusters.scope.sample_ids = null
    }
  },
)

// When the user picks "Selected samples", capture the current selection into
// scope. We snapshot at scope time rather than reactively on every gallery
// change so the user can pick a snapshot then return to clusters view.
function refreshSelectedSnapshot() {
  clusters.scope.sample_ids = Array.from(gallery.selectedIds)
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <!-- Scope -->
    <div class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Scope</span>
      </label>
      <select
        v-model="clusters.scope.mode"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option v-for="opt in scopeOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
      </select>

      <div
        v-if="clusters.scope.mode === 'class'"
        class="mt-3 flex flex-col gap-2 p-3 bg-base-100 border border-base-300 rounded-[2px]"
      >
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Class</span>
        <select
          v-model="clusters.scope.class_id"
          class="select select-bordered select-xs rounded-[2px] w-full font-mono text-[11px] focus:outline-neutral"
        >
          <option :value="null" disabled>— pick a class —</option>
          <option v-for="c in selectableClasses" :key="c.id" :value="c.id">{{ c.name }}</option>
        </select>
      </div>

      <div
        v-if="clusters.scope.mode === 'samples'"
        class="mt-3 flex items-center justify-between gap-3 p-3 bg-base-100 border border-base-300 rounded-[2px]"
      >
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">
          {{ selectedSampleCount }} selected
        </span>
        <button
          class="text-[9px] font-bold tracking-widest uppercase text-neutral/60 hover:text-neutral underline-offset-2 hover:underline"
          @click="refreshSelectedSnapshot"
        >
          Refresh
        </button>
      </div>

      <p
        v-if="clusters.scope.mode !== 'all'"
        class="mt-2 text-[9px] text-neutral/40 tracking-widest uppercase"
      >
        Will replace existing clusters
      </p>
    </div>

    <!-- Feature Extractor -->
    <div class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Feature Extractor</span>
      </label>
      <select
        v-model="clusters.featureMethod"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option
          v-for="m in clusters.featureMethods"
          :key="m.id"
          :value="m.id"
        >
          {{ m.name }} ({{ m.feature_dim }}d)
        </option>
      </select>
    </div>

    <!-- Clustering Method -->
    <div class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Clustering Method</span>
      </label>
      <select
        v-model="currentClusteringMethod"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option
          v-for="cm in clusteringMethods"
          :key="cm.value"
          :value="cm.value"
        >
          {{ cm.label }}
        </option>
      </select>
    </div>

    <!-- Dimensionality Reduction (hidden for EVoC) -->
    <div v-if="!isEvoc" class="form-control">
      <label class="label pb-2">
        <span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">Dimensionality Reduction</span>
      </label>
      <select
        v-model="currentDimMethod"
        class="select select-bordered rounded-[2px] w-full font-mono text-xs focus:outline-neutral"
      >
        <option
          v-for="dm in dimMethods"
          :key="String(dm.value)"
          :value="dm.value"
        >
          {{ dm.label }}
        </option>
      </select>

      <!-- PCA params -->
      <div v-if="clusters.dimReductionMethod === 'pca'" class="mt-4 flex flex-col gap-3 p-3 bg-base-100 border border-base-300 rounded-[2px]">
        <div class="flex items-center justify-between gap-3">
          <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Variance Ratio</span>
          <input
            :value="clusters.dimReductionParams.n_components"
            @input="clusters.dimReductionParams.n_components = parseFloat(($event.target as HTMLInputElement).value) || 0.95"
            type="number"
            min="0.01"
            max="0.99"
            step="0.01"
            class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
          />
        </div>
      </div>

      <!-- UMAP params -->
      <div v-if="clusters.dimReductionMethod === 'umap'" class="mt-4 flex flex-col gap-3 p-3 bg-base-100 border border-base-300 rounded-[2px]">
        <div class="flex items-center justify-between gap-3">
          <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Neighbors</span>
          <input
            :value="clusters.dimReductionParams.n_neighbors"
            @input="clusters.dimReductionParams.n_neighbors = parseInt(($event.target as HTMLInputElement).value) || 15"
            type="number"
            min="2"
            max="200"
            class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
          />
        </div>
        <div class="flex items-center justify-between gap-3">
          <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Min Dist</span>
          <input
            :value="clusters.dimReductionParams.min_dist"
            @input="clusters.dimReductionParams.min_dist = parseFloat(($event.target as HTMLInputElement).value) || 0.1"
            type="number"
            min="0.0"
            max="0.99"
            step="0.01"
            class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
          />
        </div>
        <div class="flex items-center justify-between gap-3">
          <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Components</span>
          <input
            :value="clusters.dimReductionParams.n_components"
            @input="clusters.dimReductionParams.n_components = parseInt(($event.target as HTMLInputElement).value) || 10"
            type="number"
            min="2"
            max="100"
            class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
          />
        </div>
      </div>
    </div>

    <!-- K-Means cluster count (hidden for EVoC) -->
    <div v-if="!isEvoc" class="form-control">
      <div class="flex items-center justify-between pb-2">
        <label class="label p-0"><span class="label-text font-bold text-[10px] tracking-widest uppercase text-neutral/70">K-Means Clusters</span></label>
        <div class="text-xs font-mono text-neutral font-bold">{{ clusters.nClusters }}</div>
      </div>
      <div class="flex items-center gap-3">
        <input
          v-model.number="clusters.nClusters"
          type="range"
          min="2"
          max="100"
          class="range range-xs range-neutral flex-1 transition-none"
        />
      </div>
    </div>

    <!-- EVoC params -->
    <div v-if="isEvoc" class="flex flex-col gap-3 p-3 bg-base-100 border border-base-300 rounded-[2px]">
      <div class="text-[9px] font-bold tracking-widest uppercase text-neutral/40 mb-1">EVoC Parameters</div>

      <div class="flex items-center justify-between gap-3">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Min Cluster Size</span>
        <input
          :value="clusters.clusteringParams.min_cluster_size"
          @input="clusters.clusteringParams.min_cluster_size = parseInt(($event.target as HTMLInputElement).value) || 5"
          type="number"
          min="2"
          max="100"
          class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
        />
      </div>

      <div class="flex items-center justify-between gap-3">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Min Samples</span>
        <input
          :value="clusters.clusteringParams.min_samples"
          @input="clusters.clusteringParams.min_samples = parseInt(($event.target as HTMLInputElement).value) || 5"
          type="number"
          min="1"
          max="100"
          class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
        />
      </div>

      <div class="flex items-center justify-between gap-3">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">N Neighbors</span>
        <input
          :value="clusters.clusteringParams.n_neighbors"
          @input="clusters.clusteringParams.n_neighbors = parseInt(($event.target as HTMLInputElement).value) || 15"
          type="number"
          min="2"
          max="200"
          class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
        />
      </div>

      <div class="flex items-center justify-between gap-3">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Neighbor Scale</span>
        <input
          :value="clusters.clusteringParams.neighbor_scale"
          @input="clusters.clusteringParams.neighbor_scale = parseFloat(($event.target as HTMLInputElement).value) || 1.0"
          type="number"
          min="0.1"
          max="10"
          step="0.1"
          class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
        />
      </div>

      <div class="flex items-center justify-between gap-3">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Noise Level</span>
        <input
          :value="clusters.clusteringParams.noise_level"
          @input="clusters.clusteringParams.noise_level = parseFloat(($event.target as HTMLInputElement).value) || 0.5"
          type="number"
          min="0.0"
          max="1.0"
          step="0.05"
          class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral"
        />
      </div>

      <div class="border-t border-base-300 pt-3 flex items-center justify-between gap-3">
        <span class="text-[9px] font-bold tracking-widest uppercase text-neutral/50">Approx Clusters</span>
        <input
          :value="clusters.clusteringParams.approx_n_clusters ?? ''"
          @input="clusters.clusteringParams.approx_n_clusters = ($event.target as HTMLInputElement).value ? parseInt(($event.target as HTMLInputElement).value) || null : null"
          type="number"
          min="2"
          max="500"
          placeholder="auto"
          class="input input-bordered input-xs rounded-[2px] w-20 text-right font-mono focus:outline-neutral placeholder:text-neutral/30"
        />
      </div>
    </div>
  </div>
</template>
