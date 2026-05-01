import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useSessionStore } from './session'
import * as clustersApi from '@/api/clusters'
import * as featuresApi from '@/api/features'
import type { ClusterOut, ClusterEmbeddingPoint, FeatureExtractionMethod, SamplePage } from '@/types'

export const useClustersStore = defineStore('clusters', () => {
  const sessionStore = useSessionStore()

  // Cluster data
  const clusters = ref<ClusterOut[]>([])
  const embeddings = ref<ClusterEmbeddingPoint[]>([])
  const loading = ref(false)
  const selectedClusterIds = ref<Set<string>>(new Set())

  // Pipeline parameters
  const featureMethod = ref('mobilenetv3')
  const dimReductionMethod = ref<string | null>('pca')
  const dimReductionParams = ref<Record<string, unknown>>({ n_components: 0.95 })
  const clusteringMethod = ref('kmeans')
  const nClusters = ref(10)
  const clusteringParams = ref<Record<string, unknown>>({})

  // Available methods (fetched from API)
  const featureMethods = ref<FeatureExtractionMethod[]>([])

  // Task tracking
  const taskId = ref<string | null>(null)

  // Scatter plot toggle
  const showScatterPlot = ref(false)

  // Cluster sample preview
  const selectedPreviewClusterId = ref<string | null>(null)
  const clusterSamples = ref<SamplePage | null>(null)

  function $reset() {
    clusters.value = []
    embeddings.value = []
    loading.value = false
    selectedClusterIds.value.clear()
    taskId.value = null
    showScatterPlot.value = false
    selectedPreviewClusterId.value = null
    clusterSamples.value = null
  }

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
      })
      taskId.value = result.task_id
    } catch {
      loading.value = false
    }
  }

  function onPipelineComplete(task: any) {
    loading.value = false
    if (task.result) {
      clusters.value = task.result.clusters as ClusterOut[]
      embeddings.value = task.result.embeddings as ClusterEmbeddingPoint[]
    }
    taskId.value = null
  }

  function onPipelineFailed() {
    loading.value = false
    taskId.value = null
  }

  async function splitCluster(clusterId: string, nSubClusters: number) {
    if (!sessionStore.currentSessionId) return
    await clustersApi.splitCluster(sessionStore.currentSessionId, clusterId, {
      n_sub_clusters: nSubClusters,
    })
    await fetchClusters()
  }

  async function mergeClusters(clusterIds: string[]) {
    if (!sessionStore.currentSessionId) return
    await clustersApi.mergeClusters(sessionStore.currentSessionId, {
      cluster_ids: clusterIds,
    })
    selectedClusterIds.value.clear()
    await fetchClusters()
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
    await fetchClusters()
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
    clusters, embeddings, loading, selectedClusterIds,
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
