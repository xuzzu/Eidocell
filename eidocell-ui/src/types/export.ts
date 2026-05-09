export interface ExportRequest {
  output_directory: string
  include_classes?: boolean
  include_clusters?: boolean
  include_masks?: boolean
  include_binary_masks?: boolean
  include_csv?: boolean
}

export interface ExportResult {
  output_directory: string
  classes_exported: number
  clusters_exported: number
  masks_exported: number
  binary_masks_exported: number
  csv_rows: number
}
