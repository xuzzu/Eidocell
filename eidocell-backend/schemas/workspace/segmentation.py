from pydantic import BaseModel


class RunSegmentationRequest(BaseModel):
    method: str = "otsu_intensity"
    params: dict = {}


class SegmentationResult(BaseModel):
    processed: int
    failed: int
    total: int


class RunSegmentationPreviewRequest(BaseModel):
    method: str
    params: dict = {}
    sample_ids: list[str]


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
