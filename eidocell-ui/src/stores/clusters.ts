import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useSessionStore } from './session'
import * as clustersApi from '@/api/clusters'
import * as featuresApi from '@/api/features'
import type {
  ClusterOut, ClusterEmbeddingPoint, ClusteringScope, ClusterSortMode,
  FeatureExtractionMethod, SamplePage,
} from '@/types'

const PULSE_MS = 1500

export const useClustersStore = defineStore('clusters', () => {
  const sessionStore = useSessionStore()

  // Cluster data
  const clusters = ref<ClusterOut[]>([])
  const embeddings = ref<ClusterEmbeddingPoint[]>([])
  const loading = ref(false)
  const selectedClusterIds = ref<Set<string>>(new Set())

  // Recently mutated IDs — drive the merge/split pulse + auto-scroll.
  // Stored as plain Sets and replaced (not mutated) so Vue reactivity tracks them.
  const recentlyAddedIds = ref<Set<string>>(new Set())
  const recentlyRemovedIds = ref<Set<string>>(new Set())

  // Sorting
  const sortMode = ref<ClusterSortMode>('manual')

  // Pipeline parameters
  const featureMethod = ref('mobilenetv3')
  const dimReductionMethod = ref<string | null>('pca')
  const dimReductionParams = ref<Record<string, unknown>>({ n_components: 0.95 })
  const clusteringMethod = ref('kmeans')
  const nClusters = ref(10)
  const clusteringParams = ref<Record<string, unknown>>({})
  const scope = ref<ClusteringScope>({ mode: 'all', class_id: null, sample_ids: null })

  // Available methods (fetched from API)
  const featureMethods = ref<FeatureExtractionMethod[]>([])

  // Task tracking
  const taskId = ref<string | null>(null)

  // Scatter plot toggle
  const showScatterPlot = ref(false)

  // Cluster sample preview
  const selectedPreviewClusterId = ref<string | null>(null)
  const clusterSamples = ref<SamplePage | null>(null)

  // ── Sorted view ─────────────────────────────────────────────────────
  const sortedClusters = computed(() => {
    const arr = [...clusters.value]
    const mode = sortMode.value
    if (mode === 'manual') return arr

    const safePct = (c: ClusterOut) =>
      c.sample_count > 0 ? c.labeled_count / c.sample_count : 0

    if (mode === 'count_desc') {
      arr.sort((a, b) => b.sample_count - a.sample_count)
    } else if (mode === 'count_asc') {
      arr.sort((a, b) => a.sample_count - b.sample_count)
    } else if (mode === 'labeled_desc') {
      arr.sort((a, b) => safePct(b) - safePct(a))
    } else if (mode === 'quality_desc') {
      // Lower mean-distance = tighter; nulls last.
      arr.sort((a, b) => {
        const av = a.quality_score ?? Number.POSITIVE_INFINITY
        const bv = b.quality_score ?? Number.POSITIVE_INFINITY
        return av - bv
      })
    }
    return arr
  })

  function $reset() {
    clusters.value = []
    embeddings.value = []
    loading.value = false
    selectedClusterIds.value.clear()
    taskId.value = null
    showScatterPlot.value = false
    selectedPreviewClusterId.value = null
    clusterSamples.value = null
    recentlyAddedIds.value = new Set()
    recentlyRemovedIds.value = new Set()
    sortMode.value = 'manual'
    scope.value = { mode: 'all', class_id: null, sample_ids: null }
  }

  // ── Recent-IDs animation tracking ──────────────────────────────────

  function _markRecentlyAdded(ids: string[]) {
    if (!ids.length) return
    const next = new Set(recentlyAddedIds.value)
    ids.forEach(id => next.add(id))
    recentlyAddedIds.value = next
    setTimeout(() => {
      const after = new Set(recentlyAddedIds.value)
      ids.forEach(id => after.delete(id))
      recentlyAddedIds.value = after
    }, PULSE_MS)
  }

  function _markRecentlyRemoved(ids: string[]) {
    if (!ids.length) return
    const next = new Set(recentlyRemovedIds.value)
    ids.forEach(id => next.add(id))
    recentlyRemovedIds.value = next
    setTimeout(() => {
      const after = new Set(recentlyRemovedIds.value)
      ids.forEach(id => after.delete(id))
      recentlyRemovedIds.value = after
    }, PULSE_MS)
  }

  // ── API actions ────────────────────────────────────────────────────

  async function fetchFeatureMethods() {
    if (!sessionStore.currentSessionId) return
    featureMethods.value = await featuresApi.listFeatureMethods(sessionStore.currentSessionId)
  }

  async function fetchClusters() {
    if (!sessionStore.currentSessionId) return
    clusters.value = await clustersApi.listClusters(sessionStore.currentSessionId)
  }

  async function runClusteringPipeline() {
    if (!sessionStore.currentSessionId) return
    loading.value = true
    embeddings.value = []
    try {
      const isEvoc = clusteringMethod.value === 'evoc'
      const evocApproxN = isEvoc ? (clusteringParams.value.approx_n_clusters as number | null ?? null) : null
      // Strip approx_n_clusters from clustering_params — it's passed as the top-level n_clusters
      const { approx_n_clusters: _dropped, ...evocParams } = clusteringParams.value as any
      const result = await clustersApi.runClusteringPipeline(sessionStore.currentSessionId, {
        feature_method: featureMethod.value,
        dim_reduction_method: isEvoc ? null : dimReductionMethod.value,
        dim_reduction_params: isEvoc ? {} : dimReductionParams.value,
        clustering_method: clusteringMethod.value,
        n_clusters: isEvoc ? evocApproxN : nClusters.value,
        clustering_params: isEvoc ? evocParams : clusteringParams.value,
        scope: scope.value,
      })
      taskId.value = result.task_id
    } catch {
      loading.value = false
    }
  }

  function onPipelineComplete(task: any) {
    loading.value = false
    if (task.result) {
      const previousIds = new Set(clusters.value.map(c => c.id))
      const nextClusters = task.result.clusters as ClusterOut[]
      clusters.value = nextClusters
      embeddings.value = task.result.embeddings as ClusterEmbeddingPoint[]
      const newIds = nextClusters.map(c => c.id).filter(id => !previousIds.has(id))
      _markRecentlyAdded(newIds)
    }
    taskId.value = null
  }

  function onPipelineFailed() {
    loading.value = false
    taskId.value = null
  }

  async function splitCluster(clusterId: string, nSubClusters: number) {
    if (!sessionStore.currentSessionId) return
    const result = await clustersApi.splitCluster(sessionStore.currentSessionId, clusterId, {
      n_sub_clusters: nSubClusters,
    })
    // Local diff: remove parent, append children.
    clusters.value = [
      ...clusters.value.filter(c => c.id !== result.deleted_cluster_id),
      ...result.new_clusters,
    ]
    selectedClusterIds.value.delete(result.deleted_cluster_id)
    _markRecentlyRemoved([result.deleted_cluster_id])
    _markRecentlyAdded(result.new_clusters.map(c => c.id))
  }

  async function mergeClusters(clusterIds: string[]) {
    if (!sessionStore.currentSessionId) return
    const result = await clustersApi.mergeClusters(sessionStore.currentSessionId, {
      cluster_ids: clusterIds,
    })
    const deleted = new Set(result.deleted_cluster_ids)
    clusters.value = [
      ...clusters.value.filter(c => !deleted.has(c.id)),
      result.new_cluster,
    ]
    selectedClusterIds.value.clear()
    _markRecentlyRemoved(result.deleted_cluster_ids)
    _markRecentlyAdded([result.new_cluster.id])
  }

  async function assignToClass(clusterIds: string[], classId: string) {
    if (!sessionStore.currentSessionId) return
    await clustersApi.assignClustersToClass(sessionStore.currentSessionId, {
      cluster_ids: clusterIds,
      class_id: classId,
    })
    selectedClusterIds.value.clear()
    await fetchClusters()
  }

  async function deleteCluster(clusterId: string) {
    if (!sessionStore.currentSessionId) return
    await clustersApi.deleteCluster(sessionStore.currentSessionId, clusterId)
    selectedClusterIds.value.delete(clusterId)
    clusters.value = clusters.value.filter(c => c.id !== clusterId)
    _markRecentlyRemoved([clusterId])
  }

  async function clearAll() {
    if (!sessionStore.currentSessionId) return
    await clustersApi.clearClusters(sessionStore.currentSessionId)
    clusters.value = []
    embeddings.value = []
    selectedClusterIds.value.clear()
  }

  function toggleSelection(id: string) {
    if (selectedClusterIds.value.has(id)) {
      selectedClusterIds.value.delete(id)
    } else {
      selectedClusterIds.value.add(id)
    }
  }

  function selectSingle(id: string) {
    selectedPreviewClusterId.value = null
    selectedClusterIds.value.clear()
    selectedClusterIds.value.add(id)
  }

  function setSelection(ids: string[]) {
    selectedPreviewClusterId.value = null
    selectedClusterIds.value = new Set(ids)
  }

  async function selectClusterForPreview(clusterId: string) {
    if (selectedPreviewClusterId.value === clusterId) {
      selectedPreviewClusterId.value = null
      clusterSamples.value = null
      return
    }
    selectedPreviewClusterId.value = clusterId
    if (!sessionStore.currentSessionId) return
    clusterSamples.value = await clustersApi.getClusterSamples(
      sessionStore.currentSessionId, clusterId, 0, 100,
    )
  }

  function updateClusteringMethod(method: string) {
    clusteringMethod.value = method
    if (method === 'evoc') {
      clusteringParams.value = { min_cluster_size: 5, min_samples: 5, n_neighbors: 15, neighbor_scale: 1.0, noise_level: 0.5 }
    } else {
      clusteringParams.value = {}
    }
  }

  function updateDimReductionParams(method: string | null) {
    dimReductionMethod.value = method
    if (method === 'pca') {
      dimReductionParams.value = { n_components: 0.95 }
    } else if (method === 'umap') {
      dimReductionParams.value = { n_neighbors: 15, min_dist: 0.1, n_components: 10 }
    } else {
      dimReductionParams.value = {}
    }
  }

  return {
    clusters, sortedClusters, embeddings, loading, selectedClusterIds,
    recentlyAddedIds, recentlyRemovedIds, sortMode, scope,
    featureMethod, dimReductionMethod, dimReductionParams,
    clusteringMethod, nClusters, clusteringParams,
    featureMethods, taskId, showScatterPlot,
    selectedPreviewClusterId, clusterSamples,
    $reset,
    fetchFeatureMethods, fetchClusters,
    runClusteringPipeline, onPipelineComplete, onPipelineFailed,
    splitCluster, mergeClusters,
    assignToClass, deleteCluster, clearAll, toggleSelection, selectSingle, setSelection,
    updateClusteringMethod, updateDimReductionParams, selectClusterForPreview,
  }
})
