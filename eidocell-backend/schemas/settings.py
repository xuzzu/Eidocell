from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    theme: str = "dark"
    thumbnail_quality: int = Field(75, ge=1, le=100)
    images_per_collage: int = Field(25, ge=1, le=50)
    default_segmentation_method: str = "otsu_intensity"
    default_feature_method: str = "morphological"
    default_dim_reduction_method: str = "pca"


class AppSettingsUpdate(BaseModel):
    theme: str | None = None
    thumbnail_quality: int | None = Field(None, ge=1, le=100)
    images_per_collage: int | None = Field(None, ge=1, le=50)
    default_segmentation_method: str | None = None
    default_feature_method: str | None = None
    default_dim_reduction_method: str | None = None
