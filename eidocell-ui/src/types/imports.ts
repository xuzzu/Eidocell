export type SourceKind = 'folder' | 'cif' | 'rif'
export type TargetShapeStrategy =
  | 'none' | 'min' | 'max' | 'mean' | 'explicit' | 'per_image_square'
export type NormalizeStrategy = 'none' | 'minmax' | 'zscore'
export type PaddingMethod = 'constant' | 'poisson' | 'replicate'
export type PostResizeStrategy =
  | 'none' | 'explicit' | 'min_longest' | 'max_longest' | 'mean_longest'

export type ImportStatus =
  | 'pending'
  | 'loading'
  | 'preprocessing'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface PreprocessingConfig {
  target_shape_strategy: TargetShapeStrategy
  explicit_shape?: [number, number] | null
  padding_method: PaddingMethod
  resize: boolean
  normalize: NormalizeStrategy
  normalize_per_channel: boolean
  align_channels: boolean
  align_reference_index: number
  align_upsample_factor: number
  compensation_matrix?: number[][] | null
  post_resize_strategy: PostResizeStrategy
  post_resize_value: number | null
}

export interface ImportCreate {
  source_kind: SourceKind
  source_path: string
  csv_path?: string | null
  csv_filename_col?: string | null
  channel_grouping: boolean
  n_channels: number
  channel_names?: string[] | null
  /** @deprecated use n_channels */
  cif_n_channels?: number
  preprocessing?: PreprocessingConfig | null
}

export interface ImportSubmitOut {
  import_id: string
  task_id: string
}

export interface ImportError {
  path: string
  reason: string
}

export interface ImportOut {
  id: string
  session_id: string
  created_at: string
  source_kind: SourceKind
  source_path: string
  csv_path: string | null
  csv_filename_col: string | null
  channel_grouping: boolean
  preprocessing_config: Record<string, unknown> | null
  status: ImportStatus
  task_id: string | null
  sample_count: number
  skipped_count: number
  errors: ImportError[] | null
}

export const DEFAULT_PREPROCESSING: PreprocessingConfig = {
  target_shape_strategy: 'mean',
  explicit_shape: null,
  padding_method: 'poisson',
  resize: false,
  normalize: 'zscore',
  normalize_per_channel: true,
  align_channels: false,
  align_reference_index: 0,
  align_upsample_factor: 10,
  compensation_matrix: null,
  post_resize_strategy: 'none',
  post_resize_value: null,
}
