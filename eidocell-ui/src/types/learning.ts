export interface TrainModelRequest {
  name: string
  classifier_type?: string
  feature_source?: string
  params?: Record<string, unknown>
}

export interface TrainModelResult {
  model_id: string
  name: string
  classifier_type: string
  feature_source: string
  n_classes: number
  n_training_samples: number
  accuracy: number | null
  label_mapping: Record<string, string>
}

export interface RunInferenceRequest {
  model_id: string
}

export interface InferenceResult {
  total_predicted: number
  classes_created: string[]
  assignments: Record<string, number>
}

export interface ApplyModelRequest {
  target_session_id: string
}

export interface TrainedModelOut {
  id: string
  name: string
  classifier_type: string
  feature_source: string
  feature_dim: number
  n_classes: number
  n_training_samples: number
  accuracy: number | null
  label_mapping: Record<string, string>
  session_id: string
  created_at: string
}
