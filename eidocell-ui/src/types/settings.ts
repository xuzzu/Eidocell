export interface AppSettings {
  theme: string
  thumbnail_quality: number
  images_per_collage: number
  default_segmentation_method: string
  default_feature_method: string
  default_dim_reduction_method: string
}

export interface AppSettingsUpdate {
  theme?: string
  thumbnail_quality?: number
  images_per_collage?: number
  default_segmentation_method?: string
  default_feature_method?: string
  default_dim_reduction_method?: string
}
