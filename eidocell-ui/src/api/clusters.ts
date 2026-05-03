import { apiGet, apiPost, apiDelete, imageUrl } from './client'
import type {
  ClusterOut, ClusteringResult, RunClusteringRequest,
  RunClusteringPipelineRequest,
  SplitClusterRequest, SplitClusterResult,
  MergeClustersRequest, MergeClustersResult,
  AssignClustersToClassRequest,
  SamplePage,
} from '@/types'

export const runClustering = (sid: string, data: RunClusteringRequest) =>
  apiPost<ClusteringResult>(`/sessions/${sid}/clusters/run`, data)

export const runClusteringPipeline = (sid: string, data: RunClusteringPipelineRequest) =>
  apiPost<{ task_id: string }>(`/sessions/${sid}/clusters/run-pipeline`, data)

export const listClusters = (sid: string) =>
  apiGet<ClusterOut[]>(`/sessions/${sid}/clusters/`)

export const clearClusters = (sid: string) =>
  apiDelete(`/sessions/${sid}/clusters/`)

export const getClusterSamples = (sid: string, clusterId: string, offset = 0, limit = 100) =>
  apiGet<SamplePage>(`/sessions/${sid}/clusters/${clusterId}/samples?offset=${offset}&limit=${limit}`)

export const clusterPreviewUrl = (sid: string, clusterId: string) =>
  imageUrl(`/sessions/${sid}/clusters/${clusterId}/preview`)

export const deleteCluster = (sid: string, clusterId: string) =>
  apiDelete(`/sessions/${sid}/clusters/${clusterId}`)

export const splitCluster = (sid: string, clusterId: string, data: SplitClusterRequest) =>
  apiPost<SplitClusterResult>(`/sessions/${sid}/clusters/${clusterId}/split`, data)

export const mergeClusters = (sid: string, data: MergeClustersRequest) =>
  apiPost<MergeClustersResult>(`/sessions/${sid}/clusters/merge`, data)

export const assignClustersToClass = (sid: string, data: AssignClustersToClassRequest) =>
  apiPost<{ updated: number }>(`/sessions/${sid}/clusters/assign-class`, data)
