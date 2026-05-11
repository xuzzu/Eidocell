from pydantic import BaseModel


class RunSegmentationRequest(BaseModel):
    method: str = "otsu_intensity"
    params: dict = {}
    channel_index: int = 0


class SegmentationResult(BaseModel):
    processed: int
    failed: int
    total: int


class RunSegmentationPreviewRequest(BaseModel):
    method: str
    params: dict = {}
    sample_ids: list[str]
    channel_index: int = 0


class SegmentationMethodParam(BaseModel):
    name: str
    label: str
    min: int | float
    max: int | float
    default: int | float
    step: int | float


class SegmentationMethod(BaseModel):
    id: str
    name: str
    params: list[SegmentationMethodParam]
