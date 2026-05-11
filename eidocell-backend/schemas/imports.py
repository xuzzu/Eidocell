from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


SourceKind = Literal["folder", "cif", "rif"]
TargetShapeStrategy = Literal["none", "min", "max", "mean", "explicit"]
NormalizeStrategy = Literal["none", "minmax", "zscore"]
PaddingMethod = Literal["constant", "poisson", "replicate"]


class PreprocessingConfig(BaseModel):
    target_shape_strategy: TargetShapeStrategy = "none"
    explicit_shape: list[int] | None = None  # [H, W]
    padding_method: PaddingMethod = "constant"
    resize: bool = False
    normalize: NormalizeStrategy = "none"
    normalize_per_channel: bool = True
    background_subtraction: bool = False
    background_subtraction_radius: int = 50
    background_subtraction_per_channel: bool = True
    align_channels: bool = False
    align_reference_index: int = 0
    align_upsample_factor: int = 10
    compensation_matrix: list[list[float]] | None = None


class ImportCreate(BaseModel):
    source_kind: SourceKind
    source_path: str
    csv_path: str | None = None
    csv_filename_col: str | None = None
    channel_grouping: bool = True
    # Unified channel count for all import formats (folder grouping, CIF/RIF tiling).
    n_channels: int = Field(default=1, ge=1, le=12)
    channel_names: list[str] | None = None
    # Deprecated: kept as an alias for back-compat. If provided and n_channels is
    # the default 1, it is promoted to n_channels.
    cif_n_channels: int = 1
    preprocessing: PreprocessingConfig | None = None  # null = skip preprocessing

    @field_validator("channel_names")
    @classmethod
    def _strip_blank_names(cls, v):
        if v is None:
            return None
        return [str(s).strip() for s in v]

    def resolved_n_channels(self) -> int:
        """Return the effective channel count, honoring the legacy cif_n_channels alias."""
        if self.n_channels != 1:
            return self.n_channels
        return max(1, int(self.cif_n_channels))


class ImportSubmitOut(BaseModel):
    import_id: str
    task_id: str


class ImportError(BaseModel):
    path: str
    reason: str


class ImportOut(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    source_kind: str
    source_path: str
    csv_path: str | None
    csv_filename_col: str | None
    channel_grouping: bool
    preprocessing_config: dict | None
    status: str
    task_id: str | None
    sample_count: int
    skipped_count: int
    errors: list[dict] | None

    model_config = {"from_attributes": True}


class PreviewRequest(BaseModel):
    source_kind: SourceKind
    source_path: str
    n_channels: int = Field(default=1, ge=1, le=12)
    cif_n_channels: int = 1  # legacy alias
    preprocessing: PreprocessingConfig

    def resolved_n_channels(self) -> int:
        if self.n_channels != 1:
            return self.n_channels
        return max(1, int(self.cif_n_channels))
